import os
import re
import json
import csv
import uuid
import base64
import random
import zipfile
import sqlite3
import threading
import traceback
import requests
from io import BytesIO
from datetime import datetime, date

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file, abort
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from elevenlabs import ElevenLabs
from openai import OpenAI

from generate_tts import generate_storybook_audio


app = Flask(__name__, static_folder="/var/data/static")

app.secret_key = os.getenv("FLASK_SECRET_KEY", "secret_key_here")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY if OPENAI_API_KEY else None)
client_api = ElevenLabs(api_key=ELEVENLABS_API_KEY if ELEVENLABS_API_KEY else None)

STORY_JSON_PATH = "saved_data/story.json"
IMAGES_JSON_PATH = "saved_data/images.json"

MOUNT_BASE = "/var/data/static"
IMAGE_DIR = os.path.join(MOUNT_BASE, "images")
AUDIO_DIR = os.path.join(MOUNT_BASE, "audio")
PDF_DIR = os.path.join(MOUNT_BASE, "pdf")

DB_PATH = os.getenv("DB_PATH", "/var/data/projectgpt.db")

app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", "587"))
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME", "projectgptgpt@gmail.com")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD", "")
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_DEFAULT_SENDER", app.config["MAIL_USERNAME"])

mail = Mail(app)

for d in [IMAGE_DIR, AUDIO_DIR, PDF_DIR, os.path.dirname(STORY_JSON_PATH) or ".", os.path.dirname(IMAGES_JSON_PATH) or "."]:
    os.makedirs(d, exist_ok=True)

available_voices = {
    "Narrator": "MltcMkX8tlDeUdYq1uCd",
    "Expressive narrator": "zv0Q6YuQUa0P3IK62XgN",
    "Old man narrator": "B52raBK48m23qWYbwchQ",
    "Young girl": "BlgEcC0TfWpBak7FmvHW",
    "Young boy": "v9LgF91V36LGgbLX3iHW",
    "Old lady": "7NsaqHdLuKNFvEfjpUno",
    "Cowboy": "ruirxsoakN0GWmGNIo04",
    "Woman": "flHkNRp1BlvT73UL6gyz",
    "Young British girl": "nDJIICjR9zfJExIFeSCN",
    "Goofy man": "XsmrVB66q3D4TaXVaWNF",
    "Young teenage boy": "IigRH4ZsY7dfxk9VRn2r",
    "Old man": "BBfN7Spa3cqLPH1xAS22",
    "Villainous man": "dG7SBJDxDoZkQUrwvqrD",
}

CSV_PATH = os.getenv("CSV_PATH", "userlist.csv")


def slugify(text):
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


def as_str(v):
    return "" if v is None else str(v).strip()


def normalize_newlines(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\\\n", "\n").replace("\\n", "\n")
    return s


def _first_image_by_title(title: str):
    if not title:
        return None, None
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT id, image_json
            FROM user_stories
            WHERE title = ? COLLATE NOCASE
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (title.strip(),),
        )
        row = c.fetchone()
    if not row:
        return None, None
    story_id, image_json = row[0], row[1]
    try:
        imgs = json.loads(image_json or "[]")
    except Exception:
        imgs = []
    if not imgs:
        return None, story_id
    img = imgs[0]
    if img.startswith("/var/data/static/"):
        img = img.replace("/var/data/static/", "/static/")
    elif img.startswith("static/"):
        img = "/" + img
    return img, story_id


def load_people():
    people = {}
    if not os.path.exists(CSV_PATH):
        return people
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            username = (row.get("username") or "").strip()
            uid = (row.get("uid") or "").strip()
            if not uid:
                base = username.split("@")[0] if "@" in username else username
                uid = slugify(base or name)
            base = username.split("@")[0] if "@" in username else username
            slug = slugify(base or name)

            def val(k):
                return (row.get(k) or "").strip()

            intro_norm = normalize_newlines(val("introduction"))
            t1, t2, t3 = val("story1_title"), val("story2_title"), val("story3_title")
            img1, story1_id = _first_image_by_title(t1)
            img2, story2_id = _first_image_by_title(t2)
            img3, story3_id = _first_image_by_title(t3)

            if not img1 and val("story1_id"):
                img1 = f"/static/images/raw_story_{val('story1_id')}_page_1.png"
            if not img2 and val("story2_id"):
                img2 = f"/static/images/raw_story_{val('story2_id')}_page_1.png"
            if not img3 and val("story3_id"):
                img3 = f"/static/images/raw_story_{val('story3_id')}_page_1.png"

            people[uid] = {
                "uid": uid,
                "slug": slug,
                "name": name,
                "team": val("team"),
                "introduction": intro_norm,
                "img_url": f"/static/student_photo/{val('img_name')}",
                "stories": [
                    {
                        "id": story1_id or val("story1_id"),
                        "title": t1,
                        "img": img1,
                        "pdf": val("story1_pdf"),
                        "mp3": f"/audio/storybook_{story1_id}.mp3" if story1_id else "",
                    },
                    {
                        "id": story2_id or val("story2_id"),
                        "title": t2,
                        "img": img2,
                        "pdf": val("story2_pdf"),
                        "mp3": f"/audio/storybook_{story2_id}.mp3" if story2_id else "",
                    },
                    {
                        "id": story3_id or val("story3_id"),
                        "title": t3,
                        "img": img3,
                        "pdf": val("story3_pdf"),
                        "mp3": f"/audio/storybook_{story3_id}.mp3" if story3_id else "",
                    },
                ],
            }
    return people


def ensure_summary_quote_in_story_json(story_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT story_json FROM user_stories WHERE id=?", (story_id,))
        row = cur.fetchone()
        if not row:
            return {"summary": None, "quote": None}
        try:
            data = json.loads(row["story_json"] or "[]")
        except Exception:
            data = []
        if isinstance(data, dict):
            pages = data.get("pages") or []
            summary = (data.get("summary") or "").strip()
            quote = (data.get("quote") or "").strip()
        else:
            pages, summary, quote = data, "", ""
        if summary and quote:
            return {"summary": summary, "quote": quote}
        full_text = "\n".join(pages)[:12000] if pages else ""
        if not full_text:
            return {"summary": None, "quote": None}
        prompt = f"""
Read this children's story and produce a concise one-paragraph summary (3–5 sentences)
and a short, memorable quote (<= 12 words). Return STRICT JSON:
{{"summary":"...","quote":"..."}}

Story:
\"\"\"{full_text}\"\"\"
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", text, re.S)
            j = json.loads(m.group(0)) if m else {}
            summary = (j.get("summary") or "").strip()
            quote = (j.get("quote") or "").strip().strip('"“”')
        except Exception:
            summary = ""
            quote = ""
        if not summary:
            summary = "A short, child-friendly summary will be available soon."
        if not quote:
            quote = "Small steps make big courage."
        new_obj = {"pages": pages, "summary": summary, "quote": f"“{quote}”"}
        cur.execute("UPDATE user_stories SET story_json=? WHERE id=?", (json.dumps(new_obj), story_id))
        conn.commit()
        return {"summary": summary, "quote": f"“{quote}”"}


def find_story_by_id_or_title(story_id, title):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        sid_str = as_str(story_id)
        if sid_str:
            try:
                sid_int = int(sid_str)
                cur.execute("SELECT * FROM user_stories WHERE id=?", (sid_int,))
                row = cur.fetchone()
                if row:
                    return row
            except ValueError:
                pass
        title_str = as_str(title)
        if title_str:
            cur.execute(
                "SELECT * FROM user_stories WHERE title LIKE ? ORDER BY created_at DESC LIMIT 1",
                (f"%{title_str}%",),
            )
            row = cur.fetchone()
            if row:
                return row
        return None


PEOPLE = load_people()


@app.route("/<uid>")
def people_detail(uid):
    person = PEOPLE.get(uid)
    if not person:
        abort(404)
    enriched = []
    for s in person.get("stories", []):
        title = as_str(s.get("title"))
        sid = as_str(s.get("id"))
        row = find_story_by_id_or_title(sid, title)

        summary = s.get("summary")
        quote = s.get("quote")
        img = s.get("img")

        topic = (s.get("topic") or "").strip()
        grade = (s.get("grade") or "").strip()
        field = (s.get("field") or "").strip()

        if row:
            res = ensure_summary_quote_in_story_json(row["id"])
            if res:
                summary = summary or res["summary"]
                quote = quote or res["quote"]
            field = (row["field"] or field or "").strip()
            topic = (row["topic"] or topic or "").strip()
            grade = (row["grade"] or grade or "").strip()

        enriched.append(
            {
                "id": row["id"] if row else sid,
                "title": title,
                "img": img,
                "pdf": s.get("pdf"),
                "mp3": s.get("mp3"),
                "summary": summary,
                "quote": quote,
                "field": field,
                "topic": topic,
                "grade": grade,
            }
        )

    person = dict(person)
    person["stories"] = enriched
    team = (person.get("team") or "").strip()
    special_teams = {"Software Engineering", "LLM & Processing", "Artificial Intelligence"}
    if team in special_teams:
        return render_template("personal2.html", name=person.get("name", ""), teamname=team, person=person)
    return render_template("personal.html", person=person)


@app.route("/check_audio/<int:story_id>")
def check_audio(story_id):
    audio_path = f"/var/data/static/audio/storybook_{story_id}.mp3"
    return jsonify({"ready": os.path.exists(audio_path)})


@app.route("/audio_status/<int:story_id>")
def audio_status(story_id):
    return render_template("audio_status.html", story_id=story_id)


def extract_first_json(text):
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
    matches = json_pattern.findall(text or "")
    for m in matches:
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON found in GPT output.")


@app.route("/generate_audio/<int:story_id>")
def generate_audio_by_id(story_id):
    user_id = session.get("user_id")
    if not user_id:
        flash("Please log in to download the audiobook.", "warning")
        return redirect(url_for("login"))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT story_json FROM user_stories WHERE id = ? AND user_id = ?",
            (story_id, user_id),
        )
        row = cursor.fetchone()
        if not row:
            flash("Story not found.", "danger")
            return redirect(url_for("profile"))

        try:
            pages = json.loads(row[0])
        except Exception:
            pages = []

        if isinstance(pages, dict):
            base_pages = pages.get("pages") or []
        elif isinstance(pages, list):
            base_pages = pages
        else:
            base_pages = []

        title = base_pages[0].split("\n")[0] if base_pages else "Untitled"

        def run_audio_generation():
            try:
                output_path = generate_storybook_audio(base_pages, title, story_id)
                audiobook_rel_path = output_path.replace("/var/data/static/", "/var/data/static/")
                with sqlite3.connect(DB_PATH) as conn_inner:
                    cursor_inner = conn_inner.cursor()
                    cursor_inner.execute(
                        """
                        UPDATE user_stories SET audiobook_path = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (audiobook_rel_path, story_id, user_id),
                    )
                    conn_inner.commit()
            except Exception as e:
                print(f"[AUDIO ERROR] story_id {story_id} — {e}")

        threading.Thread(target=run_audio_generation).start()
        flash("Audio is being generated in the background. Please refresh after 1–2 minutes.", "info")
        return redirect(url_for("profile"))


@app.route("/play_audio/<int:story_id>")
def play_audio(story_id):
    audio_path = f"/var/data/static/audio/storybook_{story_id}.mp3"
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/mpeg", as_attachment=False)
    flash("Audiobook not found.", "danger")
    return redirect(url_for("profile"))


@app.route("/generate_quiz/<int:story_id>", methods=["POST"])
def generate_quiz(story_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT COUNT(*) FROM user_story_quiz WHERE user_id = ? AND story_id = ?",
            (user_id, story_id),
        )
        count = c.fetchone()[0]
        if count > 0:
            flash("Quiz already generated for this story.", "info")
            return redirect(url_for("view_quiz", story_id=story_id))

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT story_json FROM user_stories WHERE id = ?", (story_id,))
            row = c.fetchone()

        if not row:
            flash("Story not found.", "danger")
            return redirect(url_for("profile"))

        story_content = row[0]

        prompt = f"""
Generate 10 multiple choice quiz questions based on the following children's story:

\"\"\"{story_content}\"\"\"

Format:
[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "answer": "A"
  }},
  ...
]

Only return valid JSON.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        quiz_text = (response.choices[0].message.content or "").strip()

        try:
            questions = json.loads(quiz_text)
        except json.JSONDecodeError:
            from ast import literal_eval

            questions = literal_eval(quiz_text)

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            for q in questions:
                c.execute(
                    """
                    INSERT INTO user_story_quiz (user_id, story_id, question, options, correct_answer)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, story_id, q["question"], json.dumps(q["options"]), q["answer"]),
                )
            conn.commit()

        flash("Quiz generated successfully.", "success")
        return redirect(url_for("view_quiz", story_id=story_id))
    except Exception as e:
        print("Quiz generation error:", e)
        flash("Failed to generate quiz.", "danger")
        return redirect(url_for("profile"))


@app.route("/quiz/<int:story_id>", methods=["GET", "POST"])
def view_quiz(story_id):
    user_id = session.get("user_id")
    if not user_id:
        flash("Please log in to view the quiz.", "warning")
        return redirect(url_for("login"))

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, question, options, correct_answer, selected_answer, is_correct
            FROM user_story_quiz
            WHERE user_id = ? AND story_id = ?
            ORDER BY id
            """,
            (user_id, story_id),
        )
        rows = cursor.fetchall()

        if not rows:
            flash("Quiz not available. Please generate it first.", "info")
            return redirect(url_for("profile"))

        if request.method == "POST":
            correct_count = 0
            total = len(rows)

            for i, row in enumerate(rows):
                user_ans = request.form.get(f"q{i}")
                if user_ans is None:
                    cursor.execute(
                        """
                        UPDATE user_story_quiz
                        SET selected_answer = NULL, is_correct = NULL
                        WHERE id = ? AND user_id = ?
                        """,
                        (row["id"], user_id),
                    )
                    continue

                is_correct = 1 if user_ans == row["correct_answer"] else 0
                correct_count += is_correct

                cursor.execute(
                    """
                    UPDATE user_story_quiz
                    SET selected_answer = ?, is_correct = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (user_ans, is_correct, row["id"], user_id),
                )

            conn.commit()
            flash(f"You got {correct_count}/{total} correct.", "success")

            cursor.execute(
                """
                SELECT id, question, options, correct_answer, selected_answer, is_correct
                FROM user_story_quiz
                WHERE user_id = ? AND story_id = ?
                ORDER BY id
                """,
                (user_id, story_id),
            )
            rows = cursor.fetchall()

        questions, options, selected, correct = [], [], [], []
        for row in rows:
            questions.append(row["question"])
            options.append(json.loads(row["options"]))
            correct.append(row["correct_answer"])
            selected.append(row["selected_answer"] if row["selected_answer"] else None)

        return render_template(
            "quiz_detail.html",
            question=questions,
            options=options,
            story_id=story_id,
            selected=selected,
            correct=correct,
        )


@app.route("/api/quiz_submit/<int:story_id>", methods=["POST"])
def api_quiz_submit(story_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    data = request.get_json(silent=True) or {}
    answers = data.get("answers")
    if not isinstance(answers, list):
        return jsonify({"ok": False, "error": "bad_payload"}), 400

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, correct_answer
            FROM user_story_quiz
            WHERE user_id = ? AND story_id = ?
            ORDER BY id
            """,
            (user_id, story_id),
        )
        rows = cur.fetchall()

        if not rows:
            return jsonify({"ok": False, "error": "no_quiz"}), 404

        n = min(len(rows), len(answers))
        correct_count = 0
        total = n

        for i in range(n):
            row = rows[i]
            user_ans = answers[i] if answers[i] in ("A", "B", "C", "D") else None
            is_correct = None
            if user_ans is not None:
                is_correct = 1 if user_ans == row["correct_answer"] else 0
                correct_count += (is_correct or 0)

            cur.execute(
                """
                UPDATE user_story_quiz
                SET selected_answer = ?, is_correct = ?
                WHERE id = ? AND user_id = ?
                """,
                (user_ans, is_correct, row["id"], user_id),
            )

        conn.commit()

    return jsonify({"ok": True, "correct": correct_count, "total": total})


@app.route("/daily_quiz", methods=["GET", "POST"])
def daily_quiz():
    if "user_id" not in session:
        flash("Please log in to take your daily quiz.", "warning")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    today = date.today().isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT question, options, correct_answer, selected_answer, is_correct "
            "FROM user_daily_quiz WHERE user_id = ? AND quiz_date = ?",
            (user_id, today),
        )
        row = cursor.fetchone()

        if request.method == "POST" and row:
            try:
                user_answers = [request.form.get(f"q{i}") for i in range(10)]
                correct_answers = json.loads(row[2])
                results = [ua == ca for ua, ca in zip(user_answers, correct_answers)]
                correct_count = sum(results)

                cursor.execute(
                    """
                    UPDATE user_daily_quiz
                    SET selected_answer = ?, is_correct = ?
                    WHERE user_id = ? AND quiz_date = ?
                    """,
                    (json.dumps(user_answers), correct_count, user_id, today),
                )
                conn.commit()
                flash(f"You got {correct_count}/10 correct!", "info")
                return redirect(url_for("daily_quiz"))
            except Exception as e:
                print("Error processing submission:", e)
                flash("Error submitting answers.", "danger")
                return redirect(url_for("daily_quiz"))

        if row:
            try:
                question = json.loads(row[0])
                options = json.loads(row[1])
                correct = json.loads(row[2])
                selected = json.loads(row[3]) if row[3] else []
                return render_template(
                    "daily_quiz.html",
                    question=question,
                    options=options,
                    answered=bool(row[3]),
                    selected=selected,
                    correct=correct,
                )
            except Exception as e:
                print("Error parsing stored quiz JSON:", e)

        cursor.execute(
            "SELECT story_json FROM user_stories WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        )
        latest = cursor.fetchone()

        if not latest:
            flash("Please generate a storybook before taking quizzes.", "info")
            return redirect(url_for("generate"))

        try:
            pages_raw = json.loads(latest[0])
        except Exception:
            pages_raw = []

        if isinstance(pages_raw, dict):
            pages = pages_raw.get("pages") or []
        elif isinstance(pages_raw, list):
            pages = pages_raw
        else:
            pages = []

        context = " ".join(pages)

        prompt = f"""
Generate 10 multiple choice quiz questions based on the following children's story:

\"\"\"{context}\"\"\"

Format:
[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "answer": "A"
  }},
  ...
]

Only return valid JSON.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
            )
            content = (response.choices[0].message.content or "").strip()
            quiz_data = json.loads(content)

            if not isinstance(quiz_data, list) or len(quiz_data) != 10:
                raise ValueError("Invalid quiz data structure")

            questions = [q["question"] for q in quiz_data]
            options = [q["options"] for q in quiz_data]
            answers = [q["answer"] for q in quiz_data]

            cursor.execute(
                """
                INSERT INTO user_daily_quiz (user_id, quiz_date, question, options, correct_answer)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, today, json.dumps(questions), json.dumps(options), json.dumps(answers)),
            )
            conn.commit()

            return render_template(
                "daily_quiz.html",
                question=questions,
                options=options,
                answered=False,
                correct=answers,
            )
        except Exception as e:
            print("GPT quiz generation error:", e)
            flash("Unable to generate quiz today. Try again later.", "danger")
            return redirect(url_for("index"))


@app.route("/generate_audio", methods=["GET", "POST"])
def generate_audio():
    try:
        with open(STORY_JSON_PATH) as f:
            pages = json.load(f)
        title = pages[0].split("\n")[0] if pages else "Unavailable"
        _ = generate_storybook_audio(pages, title)
        return send_file("storybook.mp3", as_attachment=True, download_name="storybook_audio.mp3")
    except Exception as e:
        return f"Audio generation failed: {e}"


def save_image_to_db(story_id, page_number, image_path):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE user_stories
                SET image_progress = ?
                WHERE id = ?
                """,
                (page_number, story_id),
            )
            conn.commit()
    except Exception as e:
        print(f"[DB ERROR] Could not update image progress for page {page_number}: {e}")


def split_text(text, max_chars=50):
    words = (text or "").split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + (1 if current else 0) <= max_chars:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


@app.route("/logout")
def logout():
    session.clear()
    return render_template("index2.html")


def get_recommended_topics(history_list):
    prompt = (
        "You are a children's story assistant. Based on the user's previous stories below:\n"
        + "\n".join(f"- {item}" for item in history_list)
        + "\n\nRecommend 3 new story topics in the following exact format:\n"
        "Field: <subject>\n"
        "Topic: <new topic>\n"
        "Reason: <why it's relevant based on past topics>\n\n"
        "Return exactly 3 sets, separated by a blank line. No extra commentary."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[RECS] OpenAI error: {e}")
        return []

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    recs = []
    for block in blocks:
        field = topic = reason = ""
        for line in block.splitlines():
            low = line.lower()
            if low.startswith("field:"):
                field = line.split(":", 1)[1].strip()
            elif low.startswith("topic:"):
                topic = line.split(":", 1)[1].strip()
            elif low.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
        if field and topic:
            recs.append({"field": field, "topic": topic, "reason": reason})
    return recs


@app.route("/view_story/<int:story_id>")
def view_story(story_id):
    user_email = session.get("username")
    if not user_email:
        return redirect(url_for("login"))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_email,))
        user_row = cursor.fetchone()
        if not user_row:
            return "User not found", 404
        user_id = user_row[0]
        cursor.execute(
            "SELECT title, story_json, image_json FROM user_stories WHERE id = ? AND user_id = ?",
            (story_id, user_id),
        )
        row = cursor.fetchone()

    if not row:
        return "Story not found or access denied", 403

    title = row[0]
    try:
        data = json.loads(row[1]) if isinstance(row[1], str) else row[1]
    except Exception:
        data = []

    if isinstance(data, dict):
        pages = data.get("pages", [])
    elif isinstance(data, list):
        pages = data
    else:
        pages = []

    try:
        images = json.loads(row[2] or "[]")
    except Exception:
        images = []

    return render_template("story_detail.html", title=title, pages=pages, images=images, story_id=story_id)


@app.route("/storybook")
def storybook():
    pages, images = [], []
    if os.path.exists(STORY_JSON_PATH):
        with open(STORY_JSON_PATH) as f:
            pages = json.load(f)
    if os.path.exists(IMAGES_JSON_PATH):
        with open(IMAGES_JSON_PATH) as f:
            images = json.load(f)
    return render_template("story_detail.html", pages=pages, images=images)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,))
            result = cursor.fetchone()

        if result and check_password_hash(result[2], password):
            session["user_id"] = result[0]
            session["username"] = result[1]
            flash("Login successful! Welcome back.", "success")
            return redirect(url_for("profile"))
        flash("Invalid credentials. Please try again.", "danger")

    return render_template("login.html")


@app.route("/")
def index():
    return render_template("index3.html")


@app.route("/newindex")
def new_index():
    return render_template("index2.html")


@app.route("/download_all_images")
def download_all_images():
    image_folder = IMAGE_DIR
    if not os.path.exists(image_folder):
        abort(404, description="Image folder not found.")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename in os.listdir(image_folder):
            abs_path = os.path.join(image_folder, filename)
            if os.path.isfile(abs_path):
                with open(abs_path, "rb") as f:
                    zip_file.writestr(filename, f.read())

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name="all_images.zip")


def split_story_into_pages(text):
    title_match = re.search(r'^\s*Title:\s*"(.+?)"', text or "", re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Untitled"
    page_pattern = r"Page\s+\d+:?\s+(.*?)(?=Page\s+\d+|$)"
    matches = re.findall(page_pattern, text or "", re.DOTALL)
    pages = [match.strip() for match in matches]
    return [title] + pages


def generate_storybook_page_with_text(initialfile, page, result):
    img = Image.open(initialfile).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("RobotoMono-Regular.ttf", size=36)

    max_width = img.width - 80
    lines = []
    words = (page or "").split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        w, _ = draw.textsize(test_line, font=font)
        if w <= max_width:
            line = test_line
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    y = img.height - (len(lines) * 45) - 30
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += 45
    img.save(result)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def calculate_position(image_path):
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {
                        "type": "text",
                        "text": "Where should i overlay this text to make it readable and not block important parts? Reply with top-left, top-center, center, bottom-left, etc.",
                    },
                ],
            }
        ],
        max_tokens=20,
    )
    return (response.choices[0].message.content or "").strip()


@app.route("/download_pdf2/<int:story_id>")
def download_pdf2(story_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT story_json, image_json, title FROM user_stories WHERE id = ? AND user_id = ?",
            (story_id, user_id),
        )
        row = cursor.fetchone()

    if not row:
        return "Story not found or access denied", 403

    try:
        pages = json.loads(row[0] or "[]")
    except Exception:
        pages = []
    try:
        images = json.loads(row[1] or "[]")
    except Exception:
        images = []
    title = row[2] or "Untitled"

    if isinstance(pages, dict):
        pages = pages.get("pages") or []

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=landscape(letter))
    width, height = landscape(letter)
    padding = 40

    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawCentredString(width / 2, height / 2 + 20, title)
    pdf.setFont("Helvetica", 16)
    pdf.drawCentredString(width / 2, height / 2 - 20, "A Storybook Adventure")
    pdf.showPage()

    for page_num, (text, img_path) in enumerate(zip(pages, images), start=1):
        left_margin = padding
        middle_gap = 30

        left_width = (width - 2 * padding - middle_gap) * 0.5
        right_width = left_width
        right_x = left_margin + left_width + middle_gap

        pdf.setFont("Helvetica", 14)
        x = left_margin
        line_height = 20
        lines = split_text(text, 50)
        total_text_height = len(lines) * line_height
        y = (height + total_text_height) / 2

        for line in lines:
            if y < padding:
                break
            pdf.drawString(x, y, line)
            y -= line_height

        try:
            if img_path.startswith("/"):
                img_path_abs = "/var/data/" + img_path[1:]
            else:
                img_path_abs = img_path
            reader = ImageReader(img_path_abs)
            img_width, img_height = reader.getSize()
            img_aspect = img_width / img_height if img_height else 1.0
            box_height = height - 2 * padding
            box_width = right_width
            box_aspect = box_width / box_height if box_height else 1.0

            if img_aspect > box_aspect:
                draw_width = box_width
                draw_height = draw_width / img_aspect
            else:
                draw_height = box_height
                draw_width = draw_height * img_aspect

            draw_x = right_x + (right_width - draw_width) / 2
            draw_y = padding + (box_height - draw_height) / 2

            pdf.drawImage(reader, draw_x, draw_y, width=draw_width, height=draw_height, preserveAspectRatio=True, mask="auto")
        except Exception as e:
            print(f"[Image error] Page {page_num}: {e}")

        pdf.setFont("Helvetica-Oblique", 10)
        pdf.drawRightString(width - padding, padding / 2, f"Page {page_num}")
        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"storybook_{story_id}.pdf", mimetype="application/pdf")


@app.route("/download_pdf/<int:story_id>")
def download_pdf(story_id):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT story_json, image_json, title FROM user_stories WHERE id = ?", (story_id,))
        row = cursor.fetchone()

    if not row:
        return "Story not found or access denied", 403

    try:
        pages = json.loads(row[0] or "[]")
    except Exception:
        pages = []
    try:
        images = json.loads(row[1] or "[]")
    except Exception:
        images = []
    title = row[2] or "Untitled"

    if isinstance(pages, dict):
        pages = pages.get("pages") or []

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=landscape(letter))
    width, height = landscape(letter)
    padding = 40

    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawCentredString(width / 2, height / 2 + 20, title)
    pdf.setFont("Helvetica", 16)
    pdf.drawCentredString(width / 2, height / 2 - 20, "A Storybook Adventure")
    pdf.showPage()

    for page_num, (text, img_path) in enumerate(zip(pages, images), start=1):
        left_margin = padding
        middle_gap = 30

        left_width = (width - 2 * padding - middle_gap) * 0.5
        right_width = left_width
        right_x = left_margin + left_width + middle_gap

        pdf.setFont("Helvetica", 14)
        x = left_margin
        line_height = 20
        lines = split_text(text, 50)
        total_text_height = len(lines) * line_height
        y = (height + total_text_height) / 2

        for line in lines:
            if y < padding:
                break
            pdf.drawString(x, y, line)
            y -= line_height

        try:
            if img_path.startswith("/"):
                img_path_abs = "/var/data/" + img_path[1:]
            else:
                img_path_abs = img_path
            reader = ImageReader(img_path_abs)
            img_width, img_height = reader.getSize()
            img_aspect = img_width / img_height if img_height else 1.0
            box_height = height - 2 * padding
            box_width = right_width
            box_aspect = box_width / box_height if box_height else 1.0

            if img_aspect > box_aspect:
                draw_width = box_width
                draw_height = draw_width / img_aspect
            else:
                draw_height = box_height
                draw_width = draw_height * img_aspect

            draw_x = right_x + (right_width - draw_width) / 2
            draw_y = padding + (box_height - draw_height) / 2

            pdf.drawImage(reader, draw_x, draw_y, width=draw_width, height=draw_height, preserveAspectRatio=True, mask="auto")
        except Exception as e:
            print(f"[Image error] Page {page_num}: {e}")

        pdf.setFont("Helvetica-Oblique", 10)
        pdf.drawRightString(width - padding, padding / 2, f"Page {page_num}")
        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"storybook_{story_id}.pdf", mimetype="application/pdf")


@app.route("/download_pdf3/<int:story_id>")
def download_pdf3(story_id):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT story_json, image_json, title FROM user_stories WHERE id = ?", (story_id,))
        row = cursor.fetchone()

    if not row:
        return "Story not found or access denied", 403

    try:
        pages = json.loads(row[0] or "[]")
    except Exception:
        pages = []
    try:
        images = json.loads(row[1] or "[]")
    except Exception:
        images = []
    title = row[2] or "Untitled"

    if isinstance(pages, dict):
        pages = pages.get("pages") or []

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=landscape(letter))
    width, height = landscape(letter)
    padding = 40

    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawCentredString(width / 2, height / 2 + 20, title)
    pdf.setFont("Helvetica", 16)
    pdf.drawCentredString(width / 2, height / 2 - 20, "A Storybook Adventure")
    pdf.showPage()

    for page_num, (text, img_path) in enumerate(zip(pages, images), start=1):
        left_margin = padding
        middle_gap = 30

        left_width = (width - 2 * padding - middle_gap) * 0.5
        right_width = left_width
        right_x = left_margin + left_width + middle_gap

        pdf.setFont("Helvetica", 14)
        x = left_margin
        line_height = 20
        lines = split_text(text, 50)
        total_text_height = len(lines) * line_height
        y = (height + total_text_height) / 2

        for line in lines:
            if y < padding:
                break
            pdf.drawString(x, y, line)
            y -= line_height

        try:
            if img_path.startswith("/"):
                img_path_abs = "/var/data/" + img_path[1:]
            else:
                img_path_abs = img_path
            reader = ImageReader(img_path_abs)
            img_width, img_height = reader.getSize()
            img_aspect = img_width / img_height if img_height else 1.0
            box_height = height - 2 * padding
            box_width = right_width
            box_aspect = box_width / box_height if box_height else 1.0

            if img_aspect > box_aspect:
                draw_width = box_width
                draw_height = draw_width / img_aspect
            else:
                draw_height = box_height
                draw_width = draw_height * img_aspect

            draw_x = right_x + (right_width - draw_width) / 2
            draw_y = padding + (box_height - draw_height) / 2

            pdf.drawImage(reader, draw_x, draw_y, width=draw_width, height=draw_height, preserveAspectRatio=True, mask="auto")
        except Exception as e:
            print(f"[Image error] Page {page_num}: {e}")

        pdf.setFont("Helvetica-Oblique", 10)
        pdf.drawRightString(width - padding, padding / 2, f"Page {page_num}")
        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"storybook_{story_id}.pdf", mimetype="application/pdf")


def fix_orientation(path):
    img = Image.open(path)
    if img.width > img.height:
        img = img.rotate(90, expand=True)
        img.save(path)


@app.route("/download-backup")
def download_backup():
    return send_file("backup.zip", as_attachment=True)


def generate_images_for_pages(pages, story_id, character_identity, title=None):
    image_paths = []
    for i, page in enumerate(pages):
        if i == 0 and title:
            prompt = f"""
Children's storybook cover in portrait layout (vertical format).
Use warm watercolor and soft digital painting style.
Title: '{title}' should be clearly written at the top center in beautiful child-friendly font.
Do not include any other text. Focus on a visually engaging cover that matches the story theme.
Character identity: {character_identity}.
Page scene: {page}
"""
        else:
            prompt = f"""
Children's storybook illustration in portrait layout (vertical format).
Do NOT use landscape layout.
Use warm watercolor and soft digital painting style.
Do not include any text or title.
This is the full story: {pages}
Page scene: {page}
"""
        try:
            result = client.images.generate(model="gpt-image-1", prompt=prompt, quality="standard")
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            filename_abs = f"{IMAGE_DIR}/raw_story_{story_id}_page_{i + 1}.png"
            image_web_path = f"/static/images/raw_story_{story_id}_page_{i + 1}.png"
            with open(filename_abs, "wb") as f:
                f.write(image_bytes)
            save_image_to_db(story_id, i + 1, image_web_path)
            image_paths.append(image_web_path)
        except Exception as e:
            print(f"[ERROR] Image generation failed on page {i+1}: {e}")
            image_paths.append("/static/images/default.jpg")
    return image_paths


def generate_story_background(user_id, story_id, field, topic, grade, age, gender):
    try:
        prompt = (
            f"Write a creative, engaging, and age-appropriate storybook for a {gender} student in grade {grade} "
            f"(approximately age {age}). The story should be exactly 10 pages long and center around the topic of '{topic}' "
            f"in the field of {field}.\n\n"
            f"The storybook must:\n"
            f"- Have a title which must be very concise at the beginning. \n"
            f"- Educate the student clearly about the topic, integrating key concepts naturally into the storyline.\n"
            f"- Use vocabulary and sentence structures appropriate for their reading level.\n"
            f"- Follow a logically structured narrative with a beginning, middle, and end.\n"
            f"- Include a relatable main character and possibly supporting characters.\n"
            f"- Use dialogue, metaphors, or scenarios that help students grasp complex ideas.\n"
            f"- Title must be a short phrase (1–5 words), NOT a full sentence. ex) Title: 'The Amazing Coin Collector'\n"
            f"- Do NOT include character names, quotes, or full story lines in the title.\n"
            f"Use exactly:\n"
            f"- 1 sentence per page if age < 8\n"
            f"- 2 sentences per page if 8 ≤ age ≤ 9\n"
            f"- 3 sentences per page if age ≥ 10\n\n"
            f"VERY IMPORTANT:\n"
            f"- Follow MUST this exact format:\n"
            f"EXAMPLE:\n\n"
            f'Title: "The Amazing Coin Collector"\n'
            f'Page 1\n...\n'
            f'Page 2\n...\n'
            f"...continue this pattern up to Page 11."
        )

        story = ""
        pages = []
        for _ in range(5):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a friendly AI that tells helpful, age-appropriate stories to children."},
                        {"role": "user", "content": prompt},
                    ],
                )
                story = (response.choices[0].message.content or "").strip()
                if not story.lower().strip().startswith("title:"):
                    continue
                pages = split_story_into_pages(story)
                break
            except Exception:
                continue

        title = pages[0].split("\n")[0].strip() if pages else "Untitled"

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM user_stories WHERE id = ?", (story_id,))
            if not cursor.fetchone():
                cursor.execute(
                    """
                    INSERT INTO user_stories (id, user_id, title, field, topic, grade, story_json, image_json, image_progress, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (story_id, user_id, title, field, topic, grade, json.dumps(pages), json.dumps([]), 0, created_at),
                )
            else:
                cursor.execute("UPDATE user_stories SET title=?, story_json=? WHERE id=?", (title, json.dumps(pages), story_id))
            conn.commit()

        identity_prompt = f"""
Extract the main character's appearance for consistent illustrations from this story excerpt:
\"\"\"{pages[0] if pages else ""}\"\"\"

Describe them in 1 sentence (appearance, clothes, mood). Do not include names or plot.
"""
        try:
            identity_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": identity_prompt}],
                max_tokens=60,
            )
            character_identity = (identity_response.choices[0].message.content or "").strip()
        except Exception:
            character_identity = ""

        image_paths = []
        for i, page in enumerate(pages):
            try:
                image_prompt = (
                    f"Children's storybook illustration in portrait layout. "
                    f"Watercolor and soft digital painting style. No text. Full story: {pages} "
                    f"Page scene: {page}. Character style: {character_identity}."
                )
                result = client.images.generate(model="gpt-image-1", prompt=image_prompt)
                image_base64 = result.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)

                filename_abs = os.path.join(IMAGE_DIR, f"raw_story_{story_id}_page_{i + 1}.png")
                image_path = f"/static/images/raw_story_{story_id}_page_{i + 1}.png"

                with open(filename_abs, "wb") as f:
                    f.write(image_bytes)
            except Exception:
                image_path = "/static/images/default.jpg"

            image_paths.append(image_path)

            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT image_json FROM user_stories WHERE id=?", (story_id,))
                row = cursor.fetchone()
                try:
                    current = json.loads(row[0]) if row and row[0] else []
                except Exception:
                    current = []
                current.append(image_path)
                cursor.execute("UPDATE user_stories SET image_json=?, image_progress=? WHERE id=?", (json.dumps(current), len(current), story_id))
                conn.commit()

        with open(STORY_JSON_PATH, "w") as f:
            json.dump(pages, f)
        with open(IMAGES_JSON_PATH, "w") as f:
            json.dump(image_paths, f)
    except Exception as e:
        print(f"[ERROR] Failed to generate story for story_id {story_id}: {e}")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE user_stories SET story_json=? WHERE id=?", (json.dumps([f"Error generating story: {str(e)}"]), story_id))
            conn.commit()


@app.route("/generate", methods=["GET", "POST"])
def generate():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        field = request.form.get("field", "")
        topic = request.form.get("topic", "")
        grade = request.form.get("grade", "")
        age = request.form.get("age", "")
        gender = request.form.get("gender", "")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(id) FROM user_stories")
            max_id = cursor.fetchone()[0] or 0
            story_id = max_id + 1

        threading.Thread(target=generate_story_background, args=(session["user_id"], story_id, field, topic, grade, age, gender)).start()
        return redirect(url_for("generate_wait", story_id=story_id))

    return render_template("generate.html")


@app.route("/generate_wait/<story_id>")
def generate_wait(story_id):
    return render_template("generate_wait.html", story_id=story_id)


@app.route("/profile")
def profile():
    if "user_id" not in session:
        flash("Please log in to view your profile.", "warning")
        return redirect(url_for("login"))

    user_id = session["user_id"]

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        user_info = cursor.fetchone() or ("",)

        cursor.execute(
            "SELECT id, title, field, topic, grade, created_at, image_json, audiobook_path "
            "FROM user_stories WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        story_rows = cursor.fetchall()

        stories = []
        for row in story_rows:
            story_id, title, field, topic, grade, created_at, image_json, audiobook_path = row
            try:
                images = json.loads(image_json or "[]")
                cover_image = images[0] if images else None
            except Exception:
                cover_image = None
            stories.append((story_id, title, field, topic, grade, created_at, cover_image, audiobook_path))

        cursor.execute("SELECT recommendations, last_story_id FROM user_recommendations WHERE user_id = ?", (user_id,))
        rec_row = cursor.fetchone()

        recommended_topics = []
        if stories:
            latest_story_id = stories[0][0]
            needs_refresh = (rec_row is None) or (str(rec_row[1]) != str(latest_story_id))
            if needs_refresh:
                recent_history = [f"{s[2]} - {s[3]}" for s in stories[:5]]
                recommended_topics = get_recommended_topics(recent_history) or []
                json_recs = json.dumps(recommended_topics, ensure_ascii=False)
                cursor.execute(
                    """
                    INSERT INTO user_recommendations (user_id, recommendations, last_story_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        recommendations = excluded.recommendations,
                        last_story_id = excluded.last_story_id
                    """,
                    (user_id, json_recs, latest_story_id),
                )
                conn.commit()
            else:
                try:
                    recommended_topics = json.loads(rec_row[0]) if rec_row and rec_row[0] else []
                except Exception:
                    recommended_topics = []
        else:
            recommended_topics = []

        from collections import Counter

        fields = [story[2] for story in stories]
        field_counts = Counter(fields)
        chart1_labels = list(field_counts.keys())
        chart1_values = list(field_counts.values())

        quiz_scores_by_field = {}
        for story in stories:
            sid, _, field, _, _, _, _, _ = story
            cursor.execute(
                "SELECT is_correct FROM user_story_quiz WHERE story_id = ? AND user_id = ?",
                (sid, user_id),
            )
            scores = [r[0] for r in cursor.fetchall()]
            if scores:
                quiz_scores_by_field.setdefault(field, []).extend(scores)

        avg_scores = {
            f: round(sum(x for x in v if x is not None) / len([x for x in v if x is not None]), 2)
            for f, v in quiz_scores_by_field.items()
            if any(x is not None for x in v)
        }
        chart2_labels = list(avg_scores.keys())
        chart2_values = list(avg_scores.values())

    return render_template(
        "profile.html",
        username=user_info[0],
        stories=stories,
        recommended=recommended_topics,
        chart1_labels=json.dumps(chart1_labels),
        chart1_values=json.dumps(chart1_values),
        chart2_labels=json.dumps(chart2_labels),
        chart2_values=json.dumps(chart2_values),
    )


@app.route("/contact", methods=["POST"])
def contact():
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")

    body = f"""
New collaboration inquiry from Lost & Found system.

From:
- Name: {name}
- Email: {email}

Message:
{message}
"""

    try:
        msg = Message(subject="[Collaboration Inquiry] New Message", recipients=[app.config["MAIL_USERNAME"]], body=body)
        mail.send(msg)
        flash("Your message has been sent successfully.", "email_sent")
    except Exception as e:
        print(f"Email send error: {e}")
        flash("Failed to send your message. Please try again later.", "danger")

    return redirect(url_for("index"))


@app.route("/generate_status/<int:story_id>")
def generate_status(story_id):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT image_progress FROM user_stories WHERE id = ?", (story_id,))
        row = c.fetchone()
    done = row[0] if row else 0
    return {"done": done, "total": 9}


@app.route("/download-db")
def download_db():
    db_path = DB_PATH
    if os.path.exists(db_path):
        return send_file(db_path, as_attachment=True)
    abort(404, description="Database file not found")


@app.route("/download_audio/<int:story_id>")
def download_audio(story_id):
    audio_path = f"/var/data/static/audio/storybook_{story_id}.mp3"
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/mpeg", as_attachment=True, download_name=f"storybook_{story_id}.mp3")
    flash("Audiobook not found.", "danger")
    return redirect(url_for("profile"))


@app.template_filter("datetimeformat")
def datetimeformat(value):
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%b %d")
    except Exception:
        return value


if __name__ == "__main__":
    app.run(debug=True)
