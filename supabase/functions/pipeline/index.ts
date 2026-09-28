// supabase/functions/pipeline/index.ts
//
// Single edge function handling all 5 pipeline stages. POST body:
//   { stage: "resumeParse"|"jobParse"|"gapAnalysis"|"ats"|"optimize", payload: {...} }
//
// If the ANTHROPIC_API_KEY secret is set (supabase secrets set ANTHROPIC_API_KEY=...),
// each stage calls Claude for real. If it is NOT set, each stage falls back to a
// deterministic, classical (non-AI) heuristic so the deployed app is fully
// clickable end-to-end before you add a key — every response is tagged
// { data, mock } so the frontend can show when it's running in heuristic mode.
//
// The one stage that fundamentally requires generation — optimize — is honestly
// gated behind having a real key rather than faking a rewrite.

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
  });
}

// ---------------------------------------------------------------------------
// Classical NLP helpers (same approach as the frontend's nlp.js) — used both
// for the always-on hybrid scoring and as the basis of heuristic mock mode.
// ---------------------------------------------------------------------------
const STOPWORDS = new Set([
  "the", "and", "a", "an", "to", "of", "in", "on", "for", "with", "is", "are", "as", "by",
  "at", "or", "be", "this", "that", "it", "from", "will", "your", "you", "we", "our", "have",
  "has", "had", "was", "were", "their", "they", "them", "but", "not", "can", "also", "into",
  "than", "then", "so", "such", "which", "who", "what", "when", "where", "how", "all", "any",
  "each", "other", "more", "most", "some", "no", "if", "about", "up", "out", "over", "under",
  "again", "further", "only", "own", "same", "too", "very", "just", "should", "now",
]);

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9+.#]+/g) || []).filter(
    (t) => t.length > 2 && !STOPWORDS.has(t)
  );
}

function termFreq(tokens: string[]): Record<string, number> {
  const tf: Record<string, number> = {};
  for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
  return tf;
}

function cosineSimilarity(a: string, b: string): number {
  const rTF = termFreq(tokenize(a));
  const jTF = termFreq(tokenize(b));
  const vocab = new Set([...Object.keys(rTF), ...Object.keys(jTF)]);
  let dot = 0, rMag = 0, jMag = 0;
  for (const term of vocab) {
    const rtf = rTF[term] || 0;
    const jtf = jTF[term] || 0;
    const df = (rtf > 0 ? 1 : 0) + (jtf > 0 ? 1 : 0);
    const idf = Math.log(2 / df) + 1;
    const rw = rtf * idf, jw = jtf * idf;
    dot += rw * jw; rMag += rw * rw; jMag += jw * jw;
  }
  if (rMag === 0 || jMag === 0) return 0;
  return dot / (Math.sqrt(rMag) * Math.sqrt(jMag));
}

function keywordCoverage(text: string, keywords: string[]): number {
  const list = (keywords || []).filter(Boolean);
  if (list.length === 0) return 0;
  const lower = text.toLowerCase();
  const found = list.filter((k) => lower.includes(String(k).toLowerCase()));
  return Math.round((found.length / list.length) * 100);
}

function topKeywords(text: string, n: number): string[] {
  const tf = termFreq(tokenize(text));
  return Object.entries(tf)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([term]) => term);
}

// ---------------------------------------------------------------------------
// Heuristic (offline/mock) stage implementations — real classical techniques,
// honestly limited where genuine language understanding is required.
// ---------------------------------------------------------------------------
const SECTION_PATTERNS: Record<string, RegExp> = {
  summary: /^(summary|objective|profile)\b/i,
  education: /^education\b/i,
  skills: /^(skills|technical skills|core competencies|requirements|required skills|qualifications)\b/i,
  experience: /^(experience|work experience|employment)\b/i,
  responsibilities: /^(responsibilities|duties|what you.ll do|role overview)\b/i,
  projects: /^projects\b/i,
  certifications: /^certifications?\b/i,
  achievements: /^(achievements|accomplishments)\b/i,
};

function splitIntoSections(text: string): Record<string, string[]> {
  const lines = text.split(/\r?\n/).map((l) => l.trim());
  const sections: Record<string, string[]> = {};
  let current: string | null = null;
  for (const line of lines) {
    if (!line) continue;
    const matched = Object.entries(SECTION_PATTERNS).find(([, re]) => re.test(line));
    if (matched) {
      current = matched[0];
      sections[current] = sections[current] || [];
      continue;
    }
    if (current) {
      sections[current] = sections[current] || [];
      sections[current].push(line);
    }
  }
  return sections;
}

function splitList(lines: string[]): string[] {
  return lines
    .join(", ")
    .split(/[,•\u2022]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 1)
    .slice(0, 12);
}

function heuristicResumeParse(resumeText: string) {
  const sections = splitIntoSections(resumeText);
  const firstParagraph = resumeText.split(/\r?\n\s*\r?\n/)[0]?.trim().slice(0, 400) || "";
  return {
    professionalSummary: (sections.summary || []).join(" ") || firstParagraph,
    education: (sections.education || []).slice(0, 3),
    skills: splitList(sections.skills || []).slice(0, 8),
    technicalSkills: splitList(sections.skills || []).slice(0, 12),
    experience: (sections.experience || []).slice(0, 4).map((line) => ({
      title: line, org: "", duration: "", highlights: [],
    })),
    projects: (sections.projects || []).slice(0, 5).map((line) => ({
      name: line, tech: "", highlight: "",
    })),
    certifications: (sections.certifications || []).slice(0, 5),
    achievements: (sections.achievements || []).slice(0, 4),
    _heuristicSectionsFound: Object.keys(sections),
  };
}

function heuristicJobParse(jobText: string) {
  const sections = splitIntoSections(jobText);
  const firstLine = jobText.split(/\r?\n/).map((l) => l.trim()).find(Boolean) || "";
  const lower = jobText.toLowerCase();

  let roleCategory = "software";
  let roleLabel = "Software / IT role";
  if (/embedded|circuit|pcb|firmware|microcontroller|hardware/.test(lower)) {
    roleCategory = "hardware"; roleLabel = "Hardware / Electronics role";
  } else if (/machine learning|deep learning|tensorflow|pytorch|nlp|computer vision|data scien/.test(lower)) {
    roleCategory = "ai_ml"; roleLabel = "AI / ML / Data Science role";
  }

  let seniority = "";
  if (/senior|lead|principal/.test(lower)) seniority = "Senior";
  else if (/entry.level|junior|graduate|fresher|intern/.test(lower)) seniority = "Entry-level";

  return {
    jobTitle: firstLine.slice(0, 100),
    roleCategory,
    roleLabel,
    domain: "",
    seniority,
    requiredSkills: splitList(sections.skills || sections.qualifications || []).slice(0, 8),
    preferredSkills: [],
    technologies: splitList(sections.skills || []).slice(0, 8),
    responsibilities: (sections.responsibilities || sections.experience || []).slice(0, 6),
    qualifications: (sections.education || sections.certifications || []).slice(0, 5),
    keywords: topKeywords(jobText, 10),
  };
}

function heuristicGapAnalysis(resumeText: string, jobText: string, jobParsed: any) {
  const seen = new Set<string>();
  const candidates = [...(jobParsed.requiredSkills || []), ...(jobParsed.technologies || [])].filter((s) => {
    const key = String(s).toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  const lowerResume = resumeText.toLowerCase();
  const matchingSkills: string[] = [];
  const missingSkills: any[] = [];
  for (const skill of candidates) {
    if (!skill) continue;
    if (lowerResume.includes(String(skill).toLowerCase())) matchingSkills.push(skill);
    else missingSkills.push({
      skill,
      status: "not_mentioned",
      note: "Not found verbatim in the resume text — heuristic mode can only check for presence, not judge whether you actually have it.",
    });
  }
  const recommendations = missingSkills.slice(0, 8).map((m) => ({
    type: "ADD",
    section: "Skills",
    detail: `Add "${m.skill}" if you have relevant experience with it.`,
    reason: "Listed in the job posting but not found in your resume text.",
  }));
  return {
    matchScore: Math.round(cosineSimilarity(resumeText, jobText) * 100),
    matchingSkills: matchingSkills.slice(0, 10),
    missingSkills: missingSkills.slice(0, 8),
    unnecessarySkills: [],
    emphasizeSkills: matchingSkills.slice(0, 6),
    recommendations,
  };
}

function heuristicAts(resumeText: string, jobParsed: any) {
  const keywords = [...(jobParsed.keywords || []), ...(jobParsed.requiredSkills || [])];
  const coverage = keywordCoverage(resumeText, keywords);
  const titleWords = tokenize(jobParsed.jobTitle || "");
  const titleAlignment = titleWords.length ? keywordCoverage(resumeText, titleWords) : 0;
  const wordCount = resumeText.trim().split(/\s+/).length;
  const formattingScore = wordCount > 150 && wordCount < 1400 ? 85 : 60;
  return {
    atsScore: coverage,
    keywordCoverage: coverage,
    titleAlignment,
    formattingScore,
    missingKeywords: (jobParsed.keywords || []).filter(
      (k: string) => !resumeText.toLowerCase().includes(String(k).toLowerCase())
    ).slice(0, 8),
    formattingNotes: [
      "Running in offline heuristic mode — this is a keyword/overlap estimate, not a full AI read.",
    ],
    explanation:
      "Computed from keyword and title-word overlap only (offline/heuristic mode). Add your Anthropic API key as a Supabase secret for a full AI-driven ATS evaluation.",
  };
}

function computeHybrid(resumeText: string, jobText: string, jobParsed: any, llmMatchScore: number) {
  const classicalScore = Math.round(cosineSimilarity(resumeText, jobText) * 100);
  const classicalKeywordCoverage = keywordCoverage(resumeText, [
    ...(jobParsed.keywords || []),
    ...(jobParsed.requiredSkills || []),
  ]);
  const blendedMatch = Math.round(0.55 * llmMatchScore + 0.45 * classicalScore);
  return { classicalScore, classicalKeywordCoverage, blendedMatch };
}

// ---------------------------------------------------------------------------
// Real Claude calls (used once ANTHROPIC_API_KEY is set)
// ---------------------------------------------------------------------------
function cleanJSON(raw: string) {
  let t = raw.trim();
  t = t.replace(/^```json\s*/i, "").replace(/^```\s*/, "").replace(/```\s*$/, "");
  try { return JSON.parse(t); } catch (e) {
    const match = t.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);
    throw e;
  }
}

async function callClaude(prompt: string, apiKey: string) {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!response.ok) throw new Error(`Claude API error ${response.status}: ${await response.text()}`);
  const data = await response.json();
  const text = (data.content || []).filter((b: any) => b.type === "text").map((b: any) => b.text).join("\n");
  if (!text) throw new Error("Claude API returned an empty response.");
  return cleanJSON(text);
}

function promptResumeParse(resumeText: string) {
  return (
    "You are a resume parsing engine. Extract structured data from the RESUME below.\n" +
    "Respond with ONLY valid JSON, no markdown fences, no commentary, matching exactly:\n" +
    '{\n  "professionalSummary": "string, empty if none present",\n  "education": ["up to 3 short strings"],\n  "skills": ["up to 8"],\n  "technicalSkills": ["up to 12"],\n  "experience": [{"title":"","org":"","duration":"","highlights":["up to 2"]}],\n  "projects": [{"name":"","tech":"","highlight":""}],\n  "certifications": ["up to 5"],\n  "achievements": ["up to 4"]\n}\n' +
    "Rules: extract only what is actually present, never invent content. Limit experience to 4 and projects to 5.\n\n" +
    'RESUME:\n"""' + resumeText + '"""'
  );
}
function promptJobParse(jobText: string) {
  return (
    "You are a job description parsing engine. Extract structured requirements from the JOB DESCRIPTION below.\n" +
    "Respond with ONLY valid JSON, no markdown fences, no commentary, matching exactly:\n" +
    '{\n  "jobTitle": "string",\n  "roleCategory": "software" | "hardware" | "ai_ml" | "other",\n  "roleLabel": "short human label",\n  "domain": "string",\n  "seniority": "string, empty if unclear",\n  "requiredSkills": ["up to 8"],\n  "preferredSkills": ["up to 6"],\n  "technologies": ["up to 8"],\n  "responsibilities": ["up to 6"],\n  "qualifications": ["up to 5"],\n  "keywords": ["up to 10"]\n}\n\n' +
    'JOB DESCRIPTION:\n"""' + jobText + '"""'
  );
}
function promptGapAnalysis(resumeText: string, jobText: string, resumeParsed: any, jobParsed: any) {
  return (
    "You are a resume/job-fit analyst. Compare the parsed RESUME DATA against the parsed JOB DATA (raw text included for grounding).\n" +
    "Respond with ONLY valid JSON, no markdown fences, no commentary, matching exactly:\n" +
    '{\n  "matchScore": 0-100,\n  "matchingSkills": ["up to 10"],\n  "missingSkills": [{"skill":"","status":"not_mentioned"|"genuinely_missing","note":""}],\n  "unnecessarySkills": ["up to 6"],\n  "emphasizeSkills": ["up to 6"],\n  "recommendations": [{"type":"ADD"|"REMOVE"|"MODIFY"|"KEEP"|"EMPHASIZE","section":"","detail":"","reason":""}]\n}\n' +
    "Rules: never invent skills or experience the candidate does not have. Limit missingSkills to 8, recommendations to 10.\n\n" +
    "PARSED RESUME DATA:\n" + JSON.stringify(resumeParsed) + "\n\nPARSED JOB DATA:\n" + JSON.stringify(jobParsed) +
    '\n\nRAW RESUME:\n"""' + resumeText + '"""\n\nRAW JOB DESCRIPTION:\n"""' + jobText + '"""'
  );
}
function promptAts(resumeText: string, jobParsed: any, gapAnalysis: any) {
  return (
    "You are an ATS compatibility evaluator. Score the RESUME against the JOB DATA below.\n" +
    "Respond with ONLY valid JSON, no markdown fences, no commentary, matching exactly:\n" +
    '{\n  "atsScore": 0-100,\n  "keywordCoverage": 0-100,\n  "titleAlignment": 0-100,\n  "formattingScore": 0-100,\n  "missingKeywords": ["up to 8"],\n  "formattingNotes": ["up to 4"],\n  "explanation": "2-3 sentences"\n}\n\n' +
    "PARSED JOB DATA:\n" + JSON.stringify(jobParsed) +
    "\n\nGAP ANALYSIS SO FAR:\n" + JSON.stringify({ matchingSkills: gapAnalysis.matchingSkills, missingSkills: gapAnalysis.missingSkills }) +
    '\n\nRAW RESUME TEXT:\n"""' + resumeText + '"""'
  );
}
function promptOptimize(resumeText: string, jobText: string, resumeParsed: any, jobParsed: any, gapAnalysis: any, atsAnalysis: any) {
  return (
    "You are a precise resume editor. Rewrite the candidate's resume to be optimized specifically for this job.\n" +
    "Respond with ONLY valid JSON, no markdown fences, matching:\n" +
    '{\n  "optimizedSummary": "2-3 sentences",\n  "optimizedResume": "plain text, under ~450 words",\n  "changes": [{"type":"ADD"|"REMOVE"|"MODIFY"|"KEEP"|"EMPHASIZE","note":""}]\n}\n' +
    "Rules: never fabricate skills, titles, employers, dates, or achievements absent from the ORIGINAL RESUME. Limit changes to 10.\n\n" +
    "PARSED RESUME:\n" + JSON.stringify(resumeParsed) + "\nPARSED JOB:\n" + JSON.stringify(jobParsed) +
    "\nGAP ANALYSIS:\n" + JSON.stringify(gapAnalysis) +
    "\nATS ANALYSIS:\n" + JSON.stringify({ missingKeywords: atsAnalysis.missingKeywords, formattingNotes: atsAnalysis.formattingNotes }) +
    '\n\nORIGINAL RESUME:\n"""' + resumeText + '"""\n\nJOB DESCRIPTION:\n"""' + jobText + '"""'
  );
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: CORS_HEADERS });
  if (req.method !== "POST") return json({ error: "POST only" }, 405);

  let body: any;
  try {
    body = await req.json();
  } catch {
    return json({ error: "Invalid JSON body." }, 400);
  }

  const { stage, payload } = body || {};
  const apiKey = Deno.env.get("ANTHROPIC_API_KEY");
  const live = !!apiKey;

  try {
    if (stage === "resumeParse") {
      const data = live
        ? await callClaude(promptResumeParse(payload.resumeText), apiKey!)
        : heuristicResumeParse(payload.resumeText);
      return json({ data, mock: !live });
    }

    if (stage === "jobParse") {
      const data = live
        ? await callClaude(promptJobParse(payload.jobText), apiKey!)
        : heuristicJobParse(payload.jobText);
      return json({ data, mock: !live });
    }

    if (stage === "gapAnalysis") {
      const data = live
        ? await callClaude(promptGapAnalysis(payload.resumeText, payload.jobText, payload.resumeParsed, payload.jobParsed), apiKey!)
        : heuristicGapAnalysis(payload.resumeText, payload.jobText, payload.jobParsed);
      return json({ data, mock: !live });
    }

    if (stage === "ats") {
      const data = live
        ? await callClaude(promptAts(payload.resumeText, payload.jobParsed, payload.gapAnalysis), apiKey!)
        : heuristicAts(payload.resumeText, payload.jobParsed);
      return json({ data, mock: !live });
    }

    if (stage === "optimize") {
      if (!live) {
        return json({
          data: {
            error: true,
            message: "AI-optimized rewriting needs a real language model. Add ANTHROPIC_API_KEY as a Supabase Edge Function secret, then run this review again.",
          },
          mock: true,
        });
      }
      const data = await callClaude(
        promptOptimize(payload.resumeText, payload.jobText, payload.resumeParsed, payload.jobParsed, payload.gapAnalysis, payload.atsAnalysis),
        apiKey!
      );
      return json({ data, mock: false });
    }

    if (stage === "hybrid") {
      const data = computeHybrid(payload.resumeText, payload.jobText, payload.jobParsed, payload.llmMatchScore);
      return json({ data, mock: false });
    }

    return json({ error: `Unknown stage: ${stage}` }, 400);
  } catch (err) {
    return json({ error: err instanceof Error ? err.message : String(err) }, 500);
  }
});
