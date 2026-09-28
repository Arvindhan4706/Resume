-- Run this in the Supabase SQL editor (Project -> SQL Editor -> New query)
-- before deploying the frontend. Creates the reviews table used for history.

create extension if not exists "pgcrypto";

create table if not exists reviews (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  job_title text,
  role_label text,
  match_score integer,
  ats_score integer,
  resume_text text,
  job_text text,
  resume_parsed jsonb,
  job_parsed jsonb,
  gap_analysis jsonb,
  ats_analysis jsonb,
  optimization jsonb,
  hybrid jsonb
);

alter table reviews enable row level security;

-- No login/auth system is wired up yet, so these policies allow anyone with
-- the anon key (i.e. anyone who has your deployed site open) to read, write,
-- and delete review rows. That's fine for a personal demo/PBL submission,
-- but before treating this as a real multi-user product, replace these with
-- policies scoped to auth.uid() and add Supabase Auth on the frontend.
create policy "anon can insert reviews"
  on reviews for insert
  to anon
  with check (true);

create policy "anon can select reviews"
  on reviews for select
  to anon
  using (true);

create policy "anon can delete reviews"
  on reviews for delete
  to anon
  using (true);
