# ProjectGPT — AI Storybook & Quiz Platform (School Edition)

ProjectGPT is a Flask-based web application used by real students in a school setting to generate **personalized children’s storybooks**, **illustrations**, **audiobooks**, and **comprehension quizzes**. Students can log in, create storybooks by selecting a **field/topic/grade/age/gender**, and then view/download their content as images, PDF, and audio. The platform also supports quiz generation per story and a daily quiz experience.

> **Privacy & Safety Note (Important)**  
> This repository contains **only selected code snippets** from the real production project.  
> Because the live system is used by real students, we intentionally removed:
> - private keys / tokens / credentials  
> - student-identifying data sources  
> - internal deployment paths and school infrastructure details  
> - any code that could expose sensitive data or system access  
>
> The shared code is meant for **portfolio / educational / reference** purposes, not as a drop-in production deployment.

---

## Key Features

### Storybook Generation
- Generates an **age-appropriate story** in a strict page-based format (Title + Pages)
- Uses OpenAI Chat Completions to create educational narratives from student-selected inputs:
  - `field`, `topic`, `grade`, `age`, `gender`
- Splits story text into pages and stores it in SQLite (`user_stories.story_json`)

### Illustration Generation (Per Page)
- Creates portrait-format children’s book illustrations using OpenAI image generation (`gpt-image-1`)
- Tracks **image generation progress** in DB (`image_progress`) for live loading screens
- Saves generated images to a static directory and stores **web paths** in DB (`image_json`)

### Audiobook Generation (Background Thread)
- Generates MP3 audiobooks using ElevenLabs (via a helper like `generate_storybook_audio`)
- Runs in a background thread so the UI stays responsive
- Provides endpoints to check readiness:
  - `/check_audio/<story_id>`
  - `/play_audio/<story_id>`
  - `/download_audio/<story_id>`

### PDF Export
- Generates downloadable **storybook PDFs** using ReportLab
- Includes:
  - Title page
  - Text + illustration layout per page
  - Page numbers
- Example endpoints:
  - `/download_pdf/<story_id>`
  - `/download_pdf2/<story_id>`

### Quiz System (Per Story + Daily Quiz)
- Generates 10 multiple-choice questions per story using OpenAI
- Stores questions/options/answers in SQLite (`user_story_quiz`)
- Supports:
  - Web form quiz submit
  - API-style submit (`/api/quiz_submit/<story_id>`)
  - Daily quiz auto-generated from the most recent story (`/daily_quiz`)

### Student Showcase Pages
- Loads a class roster from a CSV (`userlist.csv`) and renders student profile pages
- Shows their story covers + generated metadata such as:
  - topic / field / grade
  - summary + quote cached into DB for reuse

---

## Tech Stack

- **Backend:** Python, Flask
- **Database:** SQLite
- **AI (Text):** OpenAI Chat Completions
- **AI (Images):** OpenAI Image Generation (`gpt-image-1`)
- **TTS:** ElevenLabs
- **PDF:** ReportLab
- **Frontend:** Jinja2 templates + static assets

---

## High-Level Architecture

1. Student logs in (SQLite users table)
2. Student requests a new storybook via `/generate`
3. Server spawns a background thread:
   - Generates story pages 
   - Extracts a consistent character description 
   - Generates illustrations per page 
   - Writes progress to DB 
4. UI polls `/generate_status/<story_id>` until done
5. Student can:
   - View story pages and images
   - Download PDF
   - Generate audiobook (threaded)
   - Generate quizzes and track results

---


