# Margin — deployable app (Vite + React + Supabase)

A complete, deployable version of the resume-review tool: React frontend
(Vercel-ready) + Supabase for both the database and the AI pipeline (via an
Edge Function). No Anthropic API key is required to deploy — the pipeline
runs in an honest offline/heuristic mode until you add one.

## What "no API key yet" actually means

You asked to deploy without wiring up an API key for now. Here's exactly
what that gets you:

- **Stages I–IV** (parse resume, parse job, gap analysis, ATS scoring) run
  using real classical NLP — regex-based section parsing, TF-IDF + cosine
  similarity, keyword-coverage counting. These are genuine computed numbers,
  not fake ones, just not AI judgment.
- **Stage V** (the AI-rewritten resume) is honestly gated: without a key it
  returns a clear message instead of pretending to generate a rewrite it
  can't actually do.
- The UI shows an amber banner whenever a result came from heuristic mode,
  so it's never presented as if it were the real AI analysis.
- The moment you add `ANTHROPIC_API_KEY` as a Supabase secret, every stage
  switches to real Claude calls — **no frontend code changes, no
  redeploy of the site needed**, just add the secret.

## Deployment steps

### 1. Create a Supabase project
Go to [supabase.com](https://supabase.com) → New project. Note your Project
URL and anon public key from **Project → Settings → API**.

### 2. Create the database table
**Project → SQL Editor → New query**, paste the contents of
`supabase/schema.sql`, and run it.

### 3. Deploy the Edge Function
Install the Supabase CLI, then from this project's root:
```bash
supabase login
supabase link --project-ref your-project-ref
supabase functions deploy pipeline --no-verify-jwt
```
`--no-verify-jwt` is used because the frontend calls this function with the
anon key directly rather than a logged-in user's JWT (there's no login
system yet — see Limitations below).

### 4. Deploy the frontend to Vercel
Push this folder to a GitHub repo, then in Vercel: **New Project → import
the repo**. Vercel auto-detects Vite. Before the first deploy, add these
under **Project → Settings → Environment Variables**:
```
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-public-key
```
Deploy. That's it — the site is live, fully clickable, running in
heuristic mode.

### 5. Add the real AI later (whenever you're ready)
```bash
supabase secrets set ANTHROPIC_API_KEY=sk-ant-...
```
No redeploy of the frontend needed — the next review you run will
automatically use real Claude calls instead of heuristics.

## Local development
```bash
npm install
cp .env.example .env   # fill in your Supabase URL + anon key
npm run dev
```

## Tested before delivery
- `npm run build` — a real production build was run and verified to
  succeed (Vite/Rollup resolves every import and fails on invalid syntax,
  so a successful build is a genuine correctness check, not just a syntax
  check).
- The built app was served locally and its JS bundle confirmed to contain
  the Supabase client code.
- The Edge Function (`supabase/functions/pipeline/index.ts`) was run
  standalone with a mocked Deno runtime against a realistic sample resume
  and job description, exercising all 5 stages, the hybrid-score stage,
  CORS preflight, and error handling — several real bugs (a job posting's
  "REQUIREMENTS" header not being recognized, a "RESPONSIBILITIES" section
  leaking into the skills list, duplicate matched skills) were caught and
  fixed this way before delivery.
- The "live" code path (when a key *is* set) was tested with a deliberately
  invalid key to confirm it calls the real Anthropic API and handles
  errors cleanly, rather than only ever being tested through the mock path.

## Known limitations
- **No authentication.** The Supabase RLS policies in `schema.sql` allow
  anyone with your anon key (i.e. anyone visiting your deployed site) to
  read, insert, and delete review rows. Fine for a personal demo or PBL
  submission; add Supabase Auth + user-scoped policies before treating
  this as a real multi-user product.
- **PDF resumes aren't parsed** — no PDF-parsing library is available in
  the browser build; upload `.docx`/`.txt` or paste text instead.
- **Heuristic mode is intentionally limited.** It cannot judge relevance
  or rewrite text — only Claude can. That's a deliberate honesty
  boundary, not a bug to "improve" by faking it.
- The Edge Function runs all 5 stages per request from the frontend
  sequentially (5 separate calls) — normal for a demo; a very high-traffic
  deployment might want to batch or cache more aggressively.
