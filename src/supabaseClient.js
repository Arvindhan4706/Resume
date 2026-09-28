import { createClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!url || !anonKey) {
  // Don't throw at import time — let the UI surface a clear message instead
  // of a blank white screen if env vars weren't set before deploying.
  console.warn(
    "Supabase env vars are missing. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY " +
      "(see .env.example) before the pipeline or history features will work."
  );
}

export const supabase = url && anonKey ? createClient(url, anonKey) : null;
export const supabaseConfigured = !!(url && anonKey);
export const SUPABASE_URL = url;
export const SUPABASE_ANON_KEY = anonKey;
