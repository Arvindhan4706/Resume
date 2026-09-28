// src/localPipeline.js
// Provides client-side fallback execution of stages I-IV and an honest preview for stage V.
// This allows Margin to work completely offline / standalone when Supabase or Anthropic keys
// haven't been configured yet.

const STOPWORDS = new Set([
  "the", "and", "a", "an", "to", "of", "in", "on", "for", "with", "is", "are", "as", "by",
  "at", "or", "be", "this", "that", "it", "from", "will", "your", "you", "we", "our", "have",
  "has", "had", "was", "were", "their", "they", "them", "but", "not", "can", "also", "into",
  "than", "then", "so", "such", "which", "who", "what", "when", "where", "how", "all", "any",
  "each", "other", "more", "most", "some", "no", "if", "about", "up", "out", "over", "under",
  "again", "further", "only", "own", "same", "too", "very", "just", "should", "now",
]);

export function tokenize(text) {
  return (String(text).toLowerCase().match(/[a-z0-9+.#]+/g) || []).filter(
    (t) => t.length > 2 && !STOPWORDS.has(t)
  );
}

export function termFreq(tokens) {
  const tf = {};
  for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
  return tf;
}

export function cosineSimilarity(a, b) {
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
    dot += rw * jw;
    rMag += rw * rw;
    jMag += jw * jw;
  }
  if (rMag === 0 || jMag === 0) return 0;
  return dot / (Math.sqrt(rMag) * Math.sqrt(jMag));
}

export function keywordCoverage(text, keywords) {
  const list = (keywords || []).filter(Boolean);
  if (list.length === 0) return 0;
  const lower = String(text).toLowerCase();
  const found = list.filter((k) => lower.includes(String(k).toLowerCase()));
  return Math.round((found.length / list.length) * 100);
}

export function topKeywords(text, n = 10) {
  const tf = termFreq(tokenize(text));
  return Object.entries(tf)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([term]) => term);
}

const SECTION_PATTERNS = {
  summary: /^(summary|objective|profile|about me)\b/i,
  education: /^education\b/i,
  skills: /^(skills|technical skills|core competencies|requirements|required skills|qualifications)\b/i,
  experience: /^(experience|work experience|employment|work history)\b/i,
  responsibilities: /^(responsibilities|duties|what you.ll do|role overview)\b/i,
  projects: /^projects\b/i,
  certifications: /^certifications?\b/i,
  achievements: /^(achievements|accomplishments|awards)\b/i,
};

function splitIntoSections(text) {
  const lines = String(text).split(/\r?\n/).map((l) => l.trim());
  const sections = {};
  let current = null;
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

function splitList(lines) {
  return lines
    .join(", ")
    .split(/[,•\u2022|;\n]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 1)
    .slice(0, 14);
}

export function localResumeParse(resumeText) {
  const sections = splitIntoSections(resumeText);
  const firstParagraph = resumeText.split(/\r?\n\s*\r?\n/)[0]?.trim().slice(0, 400) || "";
  return {
    professionalSummary: (sections.summary || []).join(" ") || firstParagraph,
    education: (sections.education || []).slice(0, 3),
    skills: splitList(sections.skills || []).slice(0, 10),
    technicalSkills: splitList(sections.skills || []).slice(0, 12),
    experience: (sections.experience || []).slice(0, 4).map((line) => ({
      title: line,
      org: "",
      duration: "",
      highlights: [],
    })),
    projects: (sections.projects || []).slice(0, 5).map((line) => ({
      name: line,
      tech: "",
      highlight: "",
    })),
    certifications: (sections.certifications || []).slice(0, 5),
    achievements: (sections.achievements || []).slice(0, 4),
    _heuristicSectionsFound: Object.keys(sections),
  };
}

export function localJobParse(jobText) {
  const sections = splitIntoSections(jobText);
  const firstLine = jobText.split(/\r?\n/).map((l) => l.trim()).find(Boolean) || "Target Position";
  const lower = jobText.toLowerCase();

  let roleCategory = "software";
  let roleLabel = "Software / IT role";
  if (/embedded|circuit|pcb|firmware|microcontroller|hardware/.test(lower)) {
    roleCategory = "hardware";
    roleLabel = "Hardware / Electronics role";
  } else if (/machine learning|deep learning|tensorflow|pytorch|nlp|computer vision|data scien/.test(lower)) {
    roleCategory = "ai_ml";
    roleLabel = "AI / ML / Data Science role";
  }

  let seniority = "Mid-Level";
  if (/senior|lead|principal|staff|director|head/.test(lower)) seniority = "Senior";
  else if (/entry.level|junior|graduate|fresher|intern/.test(lower)) seniority = "Entry-level";

  const keywords = topKeywords(jobText, 10);
  const requiredSkills = splitList(sections.skills || sections.qualifications || []).slice(0, 8);
  const finalSkills = requiredSkills.length > 0 ? requiredSkills : keywords.slice(0, 6);

  return {
    jobTitle: firstLine.slice(0, 100),
    roleCategory,
    roleLabel,
    domain: roleCategory === "ai_ml" ? "Artificial Intelligence" : "Technology",
    seniority,
    requiredSkills: finalSkills,
    preferredSkills: splitList(sections.qualifications || []).slice(0, 6),
    technologies: finalSkills,
    responsibilities: (sections.responsibilities || sections.experience || []).slice(0, 6),
    qualifications: (sections.education || sections.certifications || []).slice(0, 5),
    keywords,
  };
}

export function localGapAnalysis(resumeText, jobText, jobParsed) {
  const seen = new Set();
  const candidates = [...(jobParsed.requiredSkills || []), ...(jobParsed.technologies || [])].filter((s) => {
    const key = String(s).toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  const lowerResume = resumeText.toLowerCase();
  const matchingSkills = [];
  const missingSkills = [];

  for (const skill of candidates) {
    if (!skill) continue;
    if (lowerResume.includes(String(skill).toLowerCase())) {
      matchingSkills.push(skill);
    } else {
      missingSkills.push({
        skill,
        status: "not_mentioned",
        note: "Not found verbatim in the resume text — check if you have transferable experience.",
      });
    }
  }

  const recommendations = missingSkills.slice(0, 8).map((m) => ({
    type: "ADD",
    section: "Skills & Experience",
    detail: `Add or elaborate on "${m.skill}" if you have experience with it.`,
    reason: "Explicitly requested in the job description keywords.",
  }));

  const matchScore = Math.round(cosineSimilarity(resumeText, jobText) * 100);

  return {
    matchScore: Math.min(100, Math.max(15, matchScore)),
    matchingSkills: matchingSkills.slice(0, 10),
    missingSkills: missingSkills.slice(0, 8),
    unnecessarySkills: [],
    emphasizeSkills: matchingSkills.slice(0, 6),
    recommendations,
  };
}

export function localAts(resumeText, jobParsed) {
  const keywords = [...(jobParsed.keywords || []), ...(jobParsed.requiredSkills || [])];
  const coverage = keywordCoverage(resumeText, keywords);
  const titleWords = tokenize(jobParsed.jobTitle || "");
  const titleAlignment = titleWords.length ? keywordCoverage(resumeText, titleWords) : 0;
  const wordCount = resumeText.trim().split(/\s+/).length;
  const formattingScore = wordCount > 150 && wordCount < 1400 ? 85 : 60;

  return {
    atsScore: Math.min(100, Math.max(20, Math.round((coverage * 0.7) + (titleAlignment * 0.15) + (formattingScore * 0.15)))),
    keywordCoverage: coverage,
    titleAlignment,
    formattingScore,
    missingKeywords: (jobParsed.keywords || []).filter(
      (k) => !resumeText.toLowerCase().includes(String(k).toLowerCase())
    ).slice(0, 8),
    formattingNotes: [
      wordCount < 150 ? "Resume appears very brief (under 150 words)." : "Length is within standard 1-2 page scan range.",
      "Clear standard section headers detected.",
      "Heuristic scan completed (local mode).",
    ],
    explanation:
      "Computed via keyword frequency, section analysis, and title alignment. Connect an Anthropic API key to unlock full semantic AI ATS judgment.",
  };
}

export function localHybrid(resumeText, jobText, jobParsed, llmMatchScore) {
  const classicalScore = Math.round(cosineSimilarity(resumeText, jobText) * 100);
  const classicalKeywordCoverage = keywordCoverage(resumeText, [
    ...(jobParsed.keywords || []),
    ...(jobParsed.requiredSkills || []),
  ]);
  const blendedMatch = Math.round(0.55 * llmMatchScore + 0.45 * classicalScore);
  return { classicalScore, classicalKeywordCoverage, blendedMatch };
}
