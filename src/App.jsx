import { useState, useRef } from "react";
import mammoth from "mammoth";
import {
  Upload,
  FileText,
  ArrowRight,
  Download,
  Copy,
  Check,
  AlertCircle,
  RotateCcw,
  Clock,
  Printer,
  Sparkles,
} from "lucide-react";
import { supabase, supabaseConfigured, SUPABASE_URL, SUPABASE_ANON_KEY } from "./supabaseClient.js";
import { extractTextFromPdf } from "./pdfExtractor.js";
import {
  localResumeParse,
  localJobParse,
  localGapAnalysis,
  localAts,
  localHybrid,
} from "./localPipeline.js";
import { SAMPLE_RESUME, SAMPLE_JOB } from "./sampleData.js";
import {
  getStoredReviews,
  saveReviewToStorage,
  getStoredReviewById,
  deleteStoredReview,
  clearAllStoredReviews,
} from "./browserStorage.js";

const STEPS = [
  { key: "resume", label: "The resume" },
  { key: "job", label: "The role" },
  { key: "review", label: "The reveal" },
];

const ACTS = [
  { key: "resumeParsed", roman: "I", title: "Parsing the resume" },
  { key: "jobParsed", roman: "II", title: "Parsing the role" },
  { key: "gapAnalysis", roman: "III", title: "Matching skills, finding gaps" },
  { key: "atsAnalysis", roman: "IV", title: "Scoring for ATS" },
  { key: "optimization", roman: "V", title: "Writing the optimized cut" },
];

const TAG_COLORS = {
  ADD: "var(--teal-green)",
  REMOVE: "var(--flare)",
  MODIFY: "var(--teal)",
  KEEP: "var(--text-muted)",
  EMPHASIZE: "var(--amber)",
};

// ---------------------------------------------------------------------------
// Calls the Supabase Edge Function that runs each pipeline stage. If Supabase
// is not configured, or if the edge function is offline, it gracefully falls
// back to client-side local NLP heuristics so the entire app works smoothly.
// ---------------------------------------------------------------------------
async function callStage(stage, payload) {
  if (supabaseConfigured) {
    try {
      const res = await fetch(`${SUPABASE_URL}/functions/v1/pipeline`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
          apikey: SUPABASE_ANON_KEY,
        },
        body: JSON.stringify({ stage, payload }),
      });
      const body = await res.json().catch(() => null);
      if (res.ok && body && body.data) {
        return body; // { data, mock }
      }
    } catch (e) {
      console.warn(`Supabase edge function for "${stage}" unavailable, using local NLP.`, e);
    }
  }

  // Client-side local NLP fallback
  if (stage === "resumeParse") {
    return { data: localResumeParse(payload.resumeText), mock: true };
  }
  if (stage === "jobParse") {
    return { data: localJobParse(payload.jobText), mock: true };
  }
  if (stage === "gapAnalysis") {
    return {
      data: localGapAnalysis(payload.resumeText, payload.jobText, payload.jobParsed),
      mock: true,
    };
  }
  if (stage === "ats") {
    return { data: localAts(payload.resumeText, payload.jobParsed), mock: true };
  }
  if (stage === "optimize") {
    const skillsToEmphasize = (payload.gapAnalysis?.matchingSkills || []).slice(0, 4).join(", ");
    return {
      data: {
        optimizedSummary: `Targeted for ${payload.jobParsed?.jobTitle || "the position"}: Highlights competencies in ${skillsToEmphasize || "core domain areas"}.`,
        optimizedResume: payload.resumeText,
        changes: [
          {
            type: "EMPHASIZE",
            note: `Targeted relevant skills (${skillsToEmphasize || "key requirements"}) matching the job description.`,
          },
          {
            type: "KEEP",
            note: "Preserved original candidate experience without fabricating credentials.",
          },
        ],
      },
      mock: true,
    };
  }
  return { data: {}, mock: true };
}

// ---------------------------------------------------------------------------
// Classical NLP layer — TF-IDF + cosine similarity, computed locally, no AI.
// Mirrors nlp logic in supabase/functions/pipeline/index.ts. Used here to
// blend with the model's own match score client-side (no extra network call
// needed since both texts are already in memory).
// ---------------------------------------------------------------------------
const STOPWORDS = new Set([
  "the", "and", "a", "an", "to", "of", "in", "on", "for", "with", "is", "are", "as", "by",
  "at", "or", "be", "this", "that", "it", "from", "will", "your", "you", "we", "our", "have",
  "has", "had", "was", "were", "their", "they", "them", "but", "not", "can", "also", "into",
  "than", "then", "so", "such", "which", "who", "what", "when", "where", "how", "all", "any",
  "each", "other", "more", "most", "some", "no", "if", "about", "up", "out", "over", "under",
  "again", "further", "only", "own", "same", "too", "very", "just", "should", "now",
]);

function tokenize(text) {
  return (String(text).toLowerCase().match(/[a-z0-9+.#]+/g) || []).filter(
    (t) => t.length > 2 && !STOPWORDS.has(t)
  );
}

function termFreq(tokens) {
  const tf = {};
  tokens.forEach((t) => {
    tf[t] = (tf[t] || 0) + 1;
  });
  return tf;
}

function cosineSimilarity(resumeText, jobText) {
  const rTF = termFreq(tokenize(resumeText));
  const jTF = termFreq(tokenize(jobText));
  const vocab = new Set([...Object.keys(rTF), ...Object.keys(jTF)]);
  let dot = 0;
  let rMag = 0;
  let jMag = 0;
  vocab.forEach((term) => {
    const rtf = rTF[term] || 0;
    const jtf = jTF[term] || 0;
    const df = (rtf > 0 ? 1 : 0) + (jtf > 0 ? 1 : 0);
    const idf = Math.log(2 / df) + 1;
    const rw = rtf * idf;
    const jw = jtf * idf;
    dot += rw * jw;
    rMag += rw * rw;
    jMag += jw * jw;
  });
  if (rMag === 0 || jMag === 0) return 0;
  return dot / (Math.sqrt(rMag) * Math.sqrt(jMag));
}

function keywordCoverage(resumeText, keywords) {
  const list = (keywords || []).filter(Boolean);
  if (list.length === 0) return 0;
  const lower = resumeText.toLowerCase();
  const found = list.filter((k) => lower.includes(String(k).toLowerCase()));
  return Math.round((found.length / list.length) * 100);
}

function computeHybrid(resumeText, jobText, jobParsed, llmMatchScore) {
  const classicalScore = Math.round(cosineSimilarity(resumeText, jobText) * 100);
  const classicalKeywordCoverage = keywordCoverage(resumeText, [
    ...(jobParsed.keywords || []),
    ...(jobParsed.requiredSkills || []),
  ]);
  const blendedMatch = Math.round(0.55 * llmMatchScore + 0.45 * classicalScore);
  return { classicalScore, classicalKeywordCoverage, blendedMatch };
}

function escapeHtml(str) {
  return String(str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatDate(value) {
  const d = new Date(value);
  return (
    d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    ", " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
  );
}

function highlightText(text, emphasizeList, unnecessaryList) {
  if (!text) return null;
  const marks = [];
  (emphasizeList || []).forEach((s) => marks.push({ term: s, kind: "emphasize" }));
  (unnecessaryList || []).forEach((s) => marks.push({ term: s, kind: "remove" }));
  const validMarks = marks.filter((m) => m.term && m.term.length > 2);
  if (validMarks.length === 0) return text;

  let pieces = [{ text, mark: null }];
  validMarks.forEach((m) => {
    const next = [];
    pieces.forEach((p) => {
      if (p.mark) {
        next.push(p);
        return;
      }
      const idx = p.text.toLowerCase().indexOf(m.term.toLowerCase());
      if (idx === -1) {
        next.push(p);
      } else {
        const before = p.text.slice(0, idx);
        const match = p.text.slice(idx, idx + m.term.length);
        const after = p.text.slice(idx + m.term.length);
        if (before) next.push({ text: before, mark: null });
        next.push({ text: match, mark: m.kind });
        if (after) next.push({ text: after, mark: null });
      }
    });
    pieces = next;
  });

  return pieces.map((p, i) => {
    if (!p.mark) return <span key={i}>{p.text}</span>;
    if (p.mark === "emphasize") {
      return (
        <mark key={i} className="mark-emphasize">
          {p.text}
        </mark>
      );
    }
    return (
      <span key={i} className="mark-remove">
        {p.text}
      </span>
    );
  });
}

function Iris({ size = 100 }) {
  const blades = Array.from({ length: 8 });
  return (
    <div className="iris-wrap" style={{ width: size, height: size }}>
      <div className="iris-glow" />
      <svg viewBox="0 0 120 120" className="iris-svg">
        <g className="iris-group">
          {blades.map((_, i) => (
            <rect
              key={i}
              x="56"
              y="4"
              width="8"
              height="50"
              rx="2"
              className="iris-blade"
              transform={`rotate(${i * 45} 60 60)`}
            />
          ))}
          <circle cx="60" cy="60" r="9" className="iris-core" />
        </g>
      </svg>
    </div>
  );
}

function LightMeter({ label, value, color, glow, caption, children }) {
  const segments = 24;
  const filled = Math.round((value / 100) * segments);
  return (
    <div className="meter">
      <span className="meter-label">{label}</span>
      <div className="meter-number" style={{ color, textShadow: "0 0 22px " + glow }}>
        {value}
      </div>
      <div className="meter-lights">
        {Array.from({ length: segments }).map((_, i) => (
          <span
            key={i}
            className="light"
            style={i < filled ? { background: color, boxShadow: "0 0 6px " + glow } : {}}
          />
        ))}
      </div>
      {caption && <p className="meter-caption">{caption}</p>}
      {children}
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div className="ministat">
      <div className="ministat-head">
        <span>{label}</span>
        <span>{value}</span>
      </div>
      <div className="ministat-track">
        <div className="ministat-fill" style={{ width: value + "%" }} />
      </div>
    </div>
  );
}

function Tag({ type }) {
  return (
    <span className="tag" style={{ color: TAG_COLORS[type], borderColor: TAG_COLORS[type] }}>
      {type}
    </span>
  );
}

export default function App() {
  const [step, setStep] = useState("resume");
  const [resumeText, setResumeText] = useState("");
  const [jobText, setJobText] = useState("");
  const [fileName, setFileName] = useState("");
  const [fileError, setFileError] = useState("");

  const [running, setRunning] = useState(false);
  const [actIndex, setActIndex] = useState(0);
  const [pipelineError, setPipelineError] = useState("");
  const [resumeParsed, setResumeParsed] = useState(null);
  const [jobParsed, setJobParsed] = useState(null);
  const [gapAnalysis, setGapAnalysis] = useState(null);
  const [atsAnalysis, setAtsAnalysis] = useState(null);
  const [optimization, setOptimization] = useState(null);
  const [hybrid, setHybrid] = useState(null);
  const [mockMode, setMockMode] = useState(false);

  const [showHistory, setShowHistory] = useState(false);
  const [historyList, setHistoryList] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");

  const [activeTab, setActiveTab] = useState("original");
  const [copied, setCopied] = useState(false);
  const [downloadError, setDownloadError] = useState("");
  const fileInputRef = useRef(null);

  const stepIndex = STEPS.findIndex((s) => s.key === step);
  const pipelineDone = !!(resumeParsed && jobParsed && gapAnalysis && atsAnalysis && optimization && hybrid);

  function resetAll() {
    setStep("resume");
    setResumeText("");
    setJobText("");
    setFileName("");
    setFileError("");
    setRunning(false);
    setActIndex(0);
    setPipelineError("");
    setResumeParsed(null);
    setJobParsed(null);
    setGapAnalysis(null);
    setAtsAnalysis(null);
    setOptimization(null);
    setHybrid(null);
    setMockMode(false);
    setActiveTab("original");
    setShowHistory(false);
  }

  async function handleFile(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setFileError("");
    const lower = file.name.toLowerCase();
    try {
      if (lower.endsWith(".docx")) {
        const buf = await file.arrayBuffer();
        const result = await mammoth.extractRawText({ arrayBuffer: buf });
        setResumeText(result.value.trim());
        setFileName(file.name);
      } else if (lower.endsWith(".txt") || lower.endsWith(".md")) {
        const text = await file.text();
        setResumeText(text.trim());
        setFileName(file.name);
      } else if (lower.endsWith(".pdf")) {
        try {
          const buf = await file.arrayBuffer();
          const extracted = await extractTextFromPdf(buf);
          if (!extracted) {
            setFileError(
              "Could not extract readable text from this PDF (it might be a scanned image). Please paste the text directly."
            );
          } else {
            setResumeText(extracted);
            setFileName(file.name);
          }
        } catch (pdfErr) {
          setFileError("Failed to parse PDF. Please upload a .docx, .txt, or paste the text directly.");
        }
      } else {
        setFileError("Unsupported file type. Use .pdf, .docx, .txt, or paste your resume directly.");
      }
    } catch (err) {
      setFileError("Couldn't read that file. Try pasting your resume text instead.");
    }
  }

  function handleLoadSampleResume() {
    setResumeText(SAMPLE_RESUME.trim());
    setFileName("sample_resume_alex_morgan.txt");
    setFileError("");
  }

  function handleLoadSampleJob() {
    setJobText(SAMPLE_JOB.trim());
  }

  function saveHistory(rp, jp, gap, ats, opt, hyb) {
    const reviewData = {
      job_title: jp.jobTitle,
      role_label: jp.roleLabel,
      match_score: hyb.blendedMatch,
      ats_score: ats.atsScore,
      resume_text: resumeText,
      job_text: jobText,
      resume_parsed: rp,
      job_parsed: jp,
      gap_analysis: gap,
      ats_analysis: ats,
      optimization: opt,
      hybrid: hyb,
    };

    // 1. Primary prototype storage: User's browser space (localStorage)
    saveReviewToStorage(reviewData);

    // 2. Optional cloud backup if Supabase is connected
    if (supabase) {
      supabase.from("reviews").insert(reviewData).catch(() => {});
    }
  }

  async function loadHistory() {
    setHistoryLoading(true);
    setHistoryError("");

    // Read directly from user's browser storage
    const localReviews = getStoredReviews();
    if (localReviews.length > 0) {
      setHistoryList(
        localReviews.map((r) => ({
          id: r.id,
          timestamp: r.created_at,
          jobTitle: r.job_title,
          roleLabel: r.role_label,
          matchScore: r.match_score,
          atsScore: r.ats_score,
        }))
      );
      setHistoryLoading(false);
      return;
    }

    // Fallback to Supabase if local is empty and Supabase is configured
    if (supabase) {
      try {
        const { data, error } = await supabase
          .from("reviews")
          .select("id, created_at, job_title, role_label, match_score, ats_score")
          .order("created_at", { ascending: false });
        if (!error && data) {
          setHistoryList(
            data.map((r) => ({
              id: r.id,
              timestamp: r.created_at,
              jobTitle: r.job_title,
              roleLabel: r.role_label,
              matchScore: r.match_score,
              atsScore: r.ats_score,
            }))
          );
        }
      } catch (e) {
        // Fallback silently
      }
    } else {
      setHistoryList([]);
    }

    setHistoryLoading(false);
  }

  function openHistory() {
    setShowHistory(true);
    loadHistory();
  }

  async function reopenReview(id) {
    // 1. Check user's browser storage
    let data = getStoredReviewById(id);

    // 2. Check Supabase if not found in browser storage
    if (!data && supabase) {
      try {
        const res = await supabase.from("reviews").select("*").eq("id", id).single();
        if (res.data) data = res.data;
      } catch (e) {
        // Fallback
      }
    }

    if (!data) {
      setHistoryError("Couldn't find that review in browser storage.");
      return;
    }

    setResumeText(data.resume_text || "");
    setJobText(data.job_text || "");
    setResumeParsed(data.resume_parsed);
    setJobParsed(data.job_parsed);
    setGapAnalysis(data.gap_analysis);
    setAtsAnalysis(data.ats_analysis);
    setOptimization(data.optimization);
    setHybrid(data.hybrid);
    setActIndex(5);
    setRunning(false);
    setPipelineError("");
    setActiveTab("original");
    setStep("review");
    setShowHistory(false);
  }

  async function deleteReview(id) {
    // Delete from user's browser storage
    deleteStoredReview(id);
    setHistoryList((prev) => prev.filter((h) => h.id !== id));

    // Also delete from Supabase if connected
    if (supabase) {
      supabase.from("reviews").delete().eq("id", id).catch(() => {});
    }
  }

  function handleClearAllHistory() {
    clearAllStoredReviews();
    setHistoryList([]);
  }

  async function runPipeline() {
    setStep("review");
    setRunning(true);
    setPipelineError("");
    setResumeParsed(null);
    setJobParsed(null);
    setGapAnalysis(null);
    setAtsAnalysis(null);
    setOptimization(null);
    setHybrid(null);
    setMockMode(false);
    setActIndex(0);

    try {
      const rpRes = await callStage("resumeParse", { resumeText });
      const rp = rpRes.data;
      setResumeParsed(rp);
      if (rpRes.mock) setMockMode(true);
      setActIndex(1);

      const jpRes = await callStage("jobParse", { jobText });
      const jp = jpRes.data;
      setJobParsed(jp);
      if (jpRes.mock) setMockMode(true);
      setActIndex(2);

      const gapRes = await callStage("gapAnalysis", { resumeText, jobText, resumeParsed: rp, jobParsed: jp });
      const gap = gapRes.data;
      setGapAnalysis(gap);
      if (gapRes.mock) setMockMode(true);
      setActIndex(3);

      const atsRes = await callStage("ats", { resumeText, jobParsed: jp, gapAnalysis: gap });
      const ats = atsRes.data;
      setAtsAnalysis(ats);
      if (atsRes.mock) setMockMode(true);
      setActIndex(4);

      const hyb = computeHybrid(resumeText, jobText, jp, gap.matchScore);
      setHybrid(hyb);

      let opt;
      try {
        const optRes = await callStage("optimize", {
          resumeText, jobText, resumeParsed: rp, jobParsed: jp, gapAnalysis: gap, atsAnalysis: ats,
        });
        opt = optRes.data;
        if (optRes.mock) setMockMode(true);
        setOptimization(opt);
      } catch (errOpt) {
        opt = {
          error: true,
          message: "The optimized rewrite couldn't be generated. Everything above it is still complete.",
        };
        setOptimization(opt);
      }
      setActIndex(5);
      saveHistory(rp, jp, gap, ats, opt, hyb);
    } catch (err) {
      setPipelineError(err.message || "Something went wrong partway through the pipeline.");
    } finally {
      setRunning(false);
    }
  }

  function handleCopy() {
    if (!optimization || !optimization.optimizedResume) return;
    try {
      navigator.clipboard.writeText(optimization.optimizedResume);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch (e) {
      setDownloadError("Couldn't access the clipboard in this preview.");
    }
  }

  function handleDownload() {
    if (!optimization || !optimization.optimizedResume) return;
    try {
      const blob = new Blob([optimization.optimizedResume], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "optimized-resume.txt";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      setDownloadError("Download isn't available — use Copy instead.");
    }
  }

  function handleDownloadWord() {
    if (!optimization || !optimization.optimizedResume) return;
    try {
      const html =
        "<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>" +
        "<head><meta charset='utf-8'><title>Optimized Resume</title></head>" +
        '<body style="font-family:Calibri, Arial, sans-serif; font-size:11pt; line-height:1.5; color:#111;">' +
        (optimization.optimizedSummary
          ? '<p style="font-style:italic;">' + escapeHtml(optimization.optimizedSummary) + "</p>"
          : "") +
        '<pre style="white-space:pre-wrap; font-family:Calibri, Arial, sans-serif; font-size:11pt;">' +
        escapeHtml(optimization.optimizedResume) +
        "</pre></body></html>";
      const blob = new Blob(["\ufeff", html], { type: "application/msword" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "optimized-resume.doc";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      setDownloadError("Word export isn't available — try Copy or PDF instead.");
    }
  }

  function handlePrint() {
    window.print();
  }

  const canGoToJob = resumeText.trim().length > 40;
  const canAnalyze = jobText.trim().length > 40;
  const sceneNumber = String(stepIndex + 1).padStart(2, "0");

  const actData = { resumeParsed, jobParsed, gapAnalysis, atsAnalysis, optimization };
  function actSummary(key, data) {
    if (!data) return "";
    if (key === "resumeParsed") return (data.experience || []).length + " roles, " + (data.projects || []).length + " projects found";
    if (key === "jobParsed") return (data.requiredSkills || []).length + " required skills identified";
    if (key === "gapAnalysis") return (data.matchingSkills || []).length + " matches, " + (data.missingSkills || []).length + " gaps found";
    if (key === "atsAnalysis") return "ATS score " + data.atsScore;
    if (key === "optimization") return data.error ? "skipped" : (data.changes || []).length + " changes made";
    return "";
  }

  return (
    <div className="margin-app">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, #root { height: 100%; }

        .margin-app {
          --stage: #0A0B0D;
          --panel: #16181C;
          --panel-raised: #1D2024;
          --hairline: rgba(255,255,255,0.10);
          --text: #F2EFE9;
          --text-muted: #93928C;
          --amber: #E8A548;
          --amber-glow: rgba(232,165,72,0.45);
          --amber-bg: rgba(232,165,72,0.16);
          --teal: #4FA8C9;
          --teal-glow: rgba(79,168,201,0.45);
          --teal-green: #6FB98F;
          --flare: #D6785A;
          --flare-bg: rgba(214,120,90,0.18);
          font-family: 'IBM Plex Mono', monospace;
          background: var(--stage);
          color: var(--text);
          min-height: 100vh;
          position: relative;
          box-sizing: border-box;
        }
        .margin-app * { box-sizing: border-box; }

        .grain {
          position: fixed; inset: 0; pointer-events: none; z-index: 5; opacity: 0.05; mix-blend-mode: overlay;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
        }
        .letterbox { position: fixed; left: 0; right: 0; height: 30px; background: #000; z-index: 6; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; font-size: 11px; letter-spacing: 0.08em; color: var(--text-muted); }
        .letterbox-top { top: 0; }
        .letterbox-bottom { bottom: 0; }
        .letterbox .wordmark { font-family: 'Bebas Neue', sans-serif; font-size: 15px; letter-spacing: 0.12em; color: var(--text); }
        .letterbox .scene-count { color: var(--amber); }

        .shell { max-width: 1020px; margin: 0 auto; padding: 70px 24px 90px; position: relative; z-index: 1; }

        .masthead { margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 14px; }
        .masthead .title-block h1 { font-family: 'Bebas Neue', sans-serif; font-size: 64px; line-height: 0.9; margin: 0; letter-spacing: 0.02em; }
        .masthead .title-block p { margin: 10px 0 0; color: var(--text-muted); font-size: 13px; max-width: 46ch; line-height: 1.6; }
        .masthead-actions { display: flex; gap: 8px; flex-wrap: wrap; }
        .reset-link { display: inline-flex; align-items: center; gap: 6px; background: none; border: 1px solid var(--hairline); color: var(--text-muted); font-size: 12px; cursor: pointer; font-family: inherit; padding: 8px 12px; }
        .reset-link:hover { color: var(--text); border-color: var(--text-muted); }

        .config-warning, .mock-banner { display: flex; gap: 10px; align-items: flex-start; border: 1px solid var(--amber); background: var(--amber-bg); color: var(--amber); font-size: 12.5px; padding: 12px 16px; margin-bottom: 30px; line-height: 1.6; }

        .steps { display: flex; gap: 36px; margin-bottom: 44px; padding-bottom: 24px; border-bottom: 1px solid var(--hairline); }
        .step-item { display: flex; flex-direction: column; gap: 4px; }
        .step-num { font-family: 'Bebas Neue', sans-serif; font-size: 30px; color: var(--hairline); line-height: 1; }
        .step-item.active .step-num { color: var(--amber); }
        .step-item.done .step-num { color: var(--text-muted); }
        .step-label { font-size: 12px; color: var(--text-muted); }
        .step-item.active .step-label { color: var(--text); }

        .panel { background: var(--panel); border: 1px solid var(--hairline); padding: 36px; position: relative; }
        .panel h2 { font-family: 'Bebas Neue', sans-serif; font-size: 30px; letter-spacing: 0.01em; margin: 0 0 8px; font-weight: normal; }
        .panel .hint { color: var(--text-muted); font-size: 13px; margin: 0 0 24px; max-width: 62ch; line-height: 1.6; }

        textarea { width: 100%; min-height: 260px; resize: vertical; border: 1px solid var(--hairline); background: #101215; padding: 16px; font-family: 'IBM Plex Mono', monospace; font-size: 13.5px; line-height: 1.7; color: var(--text); }
        textarea::placeholder { color: var(--text-muted); }
        textarea:focus { outline: 1px solid var(--amber); outline-offset: 1px; }

        .upload-row { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; flex-wrap: wrap; }
        .upload-btn { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--text); background: transparent; padding: 10px 16px; font-size: 12.5px; cursor: pointer; font-family: inherit; color: var(--text); }
        .upload-btn:hover { background: var(--text); color: var(--stage); }
        .file-name { font-size: 12.5px; color: var(--text-muted); }
        .file-error, .analysis-error { display: flex; gap: 8px; align-items: flex-start; color: var(--flare); font-size: 13px; margin-top: 10px; }

        .divider-or { display: flex; align-items: center; gap: 12px; color: var(--text-muted); font-size: 11.5px; margin: 8px 0 16px; }
        .divider-or::before, .divider-or::after { content: ''; flex: 1; height: 1px; background: var(--hairline); }

        .actions-row { display: flex; justify-content: flex-end; margin-top: 26px; }
        .primary-btn { display: inline-flex; align-items: center; gap: 8px; background: var(--amber); color: #16130A; border: none; padding: 14px 24px; font-size: 13.5px; cursor: pointer; font-family: inherit; font-weight: 600; }
        .primary-btn:disabled { background: var(--panel-raised); color: var(--text-muted); cursor: not-allowed; }
        .primary-btn:not(:disabled):hover { box-shadow: 0 0 24px var(--amber-glow); }

        .acts-wrap { padding: 20px 6px; }
        .act-row { display: flex; align-items: center; gap: 22px; padding: 18px 0; border-bottom: 1px solid var(--hairline); }
        .act-row:last-child { border-bottom: none; }
        .act-roman { font-family: 'Bebas Neue', sans-serif; font-size: 26px; width: 44px; flex-shrink: 0; color: var(--hairline); }
        .act-row.done .act-roman { color: var(--teal-green); }
        .act-row.running .act-roman { color: var(--amber); }
        .act-body { flex: 1; }
        .act-title { font-size: 14px; color: var(--text-muted); }
        .act-row.active .act-title, .act-row.done .act-title { color: var(--text); }
        .act-result { font-size: 12px; color: var(--teal-green); margin-top: 4px; }
        .act-icon { width: 26px; flex-shrink: 0; display: flex; justify-content: center; }

        .analyzing-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px 10px; text-align: center; gap: 18px; }
        .analyzing-wrap h2 { font-family: 'Bebas Neue', sans-serif; font-size: 24px; margin: 0; font-weight: normal; letter-spacing: 0.02em; }
        .analyzing-wrap p { color: var(--text-muted); font-size: 13px; margin: 0; }

        .iris-wrap { position: relative; }
        .iris-glow { position: absolute; inset: -24px; border-radius: 50%; background: radial-gradient(circle, var(--amber-glow), transparent 70%); filter: blur(6px); animation: irisGlow 2.6s ease-in-out infinite; }
        .iris-svg { width: 100%; height: 100%; position: relative; }
        .iris-group { transform-origin: 60px 60px; animation: irisTurn 2.6s ease-in-out infinite; }
        .iris-blade { fill: var(--amber); }
        .iris-core { fill: var(--stage); stroke: var(--amber); stroke-width: 1.5; }
        @keyframes irisTurn { 0%, 100% { transform: scale(0.7) rotate(0deg); } 50% { transform: scale(1) rotate(22deg); } }
        @keyframes irisGlow { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }

        .result-head { margin-bottom: 30px; }
        .role-badge { display: inline-block; border: 1px solid var(--amber); color: var(--amber); font-size: 12px; padding: 4px 11px; margin-bottom: 10px; }
        .result-head h2 { font-family: 'Bebas Neue', sans-serif; font-size: 40px; margin: 0; letter-spacing: 0.01em; font-weight: normal; }

        .method-note { font-size: 12px; color: var(--text-muted); line-height: 1.7; margin: -14px 0 24px; max-width: 78ch; }

        .meters { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-bottom: 20px; align-items: start; }
        .meter { background: var(--panel); border: 1px solid var(--hairline); padding: 24px; }
        .meter-label { font-size: 12px; color: var(--text-muted); display: block; margin-bottom: 10px; }
        .meter-number { font-family: 'Bebas Neue', sans-serif; font-size: 54px; line-height: 1; margin-bottom: 14px; }
        .meter-lights { display: flex; gap: 3px; }
        .light { flex: 1; height: 9px; background: rgba(255,255,255,0.07); }
        .meter-caption { margin: 12px 0 0; font-size: 12px; color: var(--text-muted); line-height: 1.6; }

        .ministat { margin-top: 14px; }
        .ministat-head { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
        .ministat-track { height: 4px; background: rgba(255,255,255,0.08); }
        .ministat-fill { height: 100%; background: var(--teal); }

        .grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 28px; align-items: start; }
        @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } .meters { grid-template-columns: 1fr; } .masthead .title-block h1 { font-size: 46px; } }

        .tabs { display: flex; gap: 2px; margin-bottom: 16px; border-bottom: 1px solid var(--hairline); flex-wrap: wrap; }
        .tab-btn { background: none; border: none; padding: 10px 4px; margin-right: 24px; font-size: 13px; color: var(--text-muted); cursor: pointer; font-family: inherit; border-bottom: 2px solid transparent; position: relative; top: 1px; }
        .tab-btn.active { color: var(--text); border-bottom-color: var(--amber); }

        .manuscript { background: var(--panel); border: 1px solid var(--hairline); padding: 26px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; line-height: 1.85; white-space: pre-wrap; max-height: 560px; overflow-y: auto; position: relative; }
        .manuscript .summary-lead { color: var(--amber); margin: 0 0 18px; border-left: 2px solid var(--amber); padding-left: 14px; white-space: normal; }
        .mark-emphasize { background: var(--amber-bg); color: var(--amber); padding: 0 2px; }
        .mark-remove { background: var(--flare-bg); color: var(--flare); text-decoration: line-through; padding: 0 2px; }

        .extract-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0; background: var(--panel); border: 1px solid var(--hairline); max-height: 560px; overflow-y: auto; }
        .extract-col { padding: 24px; }
        .extract-col:first-child { border-right: 1px solid var(--hairline); }
        .extract-col h4 { font-family: 'Bebas Neue', sans-serif; font-size: 17px; margin: 0 0 16px; font-weight: normal; letter-spacing: 0.02em; color: var(--amber); }
        .extract-block { margin-bottom: 16px; }
        .extract-label { font-size: 11px; color: var(--text-muted); display: block; margin-bottom: 6px; }
        .extract-entry { font-size: 12.5px; margin-bottom: 10px; line-height: 1.6; }
        .extract-entry strong { color: var(--text); }
        .extract-entry .muted { color: var(--text-muted); font-size: 11.5px; }
        .extract-entry p { margin: 2px 0 0; color: var(--text-muted); font-size: 12px; }
        .extract-summary { font-size: 12.5px; color: var(--text-muted); line-height: 1.7; margin: 0 0 16px; }
        @media (max-width: 760px) { .extract-grid { grid-template-columns: 1fr; } .extract-col:first-child { border-right: none; border-bottom: 1px solid var(--hairline); } }

        .opt-loading { display: flex; align-items: center; gap: 10px; color: var(--text-muted); font-size: 13px; padding: 30px 0; }

        .action-row { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
        .ghost-btn { display: inline-flex; align-items: center; gap: 7px; background: transparent; border: 1px solid var(--text); padding: 9px 15px; font-size: 12.5px; cursor: pointer; font-family: inherit; color: var(--text); }
        .ghost-btn:hover { background: var(--text); color: var(--stage); }
        .ghost-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .dl-error { font-size: 12px; color: var(--flare); margin-top: 8px; }

        .side-section { margin-bottom: 28px; }
        .side-section h3 { font-family: 'Bebas Neue', sans-serif; font-size: 18px; margin: 0 0 14px; border-bottom: 1px solid var(--hairline); padding-bottom: 10px; font-weight: normal; letter-spacing: 0.02em; }
        .skill-chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
        .skill-chip { font-size: 12px; border: 1px solid var(--hairline); padding: 4px 9px; color: var(--text); }
        .skill-chip.match { border-color: var(--teal-green); color: var(--teal-green); }
        .skill-chip.gap { border-color: var(--flare); color: var(--flare); }

        .missing-item { font-size: 12.5px; margin-bottom: 10px; padding-left: 10px; border-left: 2px solid var(--hairline); }
        .missing-item.genuine { border-left-color: var(--flare); }
        .missing-item.notmentioned { border-left-color: var(--teal); }
        .missing-item strong { display: block; margin-bottom: 3px; color: var(--text); }
        .missing-item span.status { font-size: 11px; color: var(--text-muted); }

        .note-item { margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--hairline); }
        .note-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
        .note-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
        .tag { font-size: 10px; letter-spacing: 0.04em; border: 1px solid; padding: 2px 6px; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
        .note-section { font-size: 11.5px; color: var(--text-muted); }
        .note-detail { font-size: 13px; margin: 4px 0 3px; color: var(--text); }
        .note-reason { font-size: 11.5px; color: var(--text-muted); }
        .format-note { font-size: 12px; color: var(--text-muted); margin: 0 0 6px; padding-left: 10px; border-left: 2px solid var(--hairline); }

        .history-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .history-item { display: flex; justify-content: space-between; align-items: center; padding: 14px 0; border-bottom: 1px solid var(--hairline); gap: 12px; flex-wrap: wrap; }
        .history-item:last-child { border-bottom: none; }
        .history-title { font-family: 'Bebas Neue', sans-serif; font-size: 18px; letter-spacing: 0.01em; }
        .history-meta { font-size: 12px; color: var(--text-muted); margin-top: 3px; }
        .history-actions { display: flex; gap: 8px; flex-shrink: 0; }

        .print-resume { display: none; }
        @media print {
          .shell, .letterbox, .grain { display: none !important; }
          .margin-app { background: #fff !important; color: #111 !important; }
          .print-resume { display: block !important; padding: 30px; font-family: Georgia, 'Times New Roman', serif; font-size: 12pt; line-height: 1.6; color: #111; white-space: pre-wrap; max-width: 700px; margin: 0 auto; }
          .print-resume .print-lead { font-style: italic; margin-bottom: 16px; }
        }
      `}</style>

      <div className="grain" />
      <div className="letterbox letterbox-top">
        <span className="wordmark">MARGIN</span>
        <span className="scene-count">SCENE {sceneNumber}</span>
      </div>
      <div className="letterbox letterbox-bottom">
        <span>resume review, made for one specific job</span>
        <span>{STEPS[stepIndex].label}</span>
      </div>

      <div className="shell">
        <div className="masthead">
          <div className="title-block">
            <h1>MARGIN</h1>
            <p>A resume review written in the margin, specific to the job you want next.</p>
          </div>
          <div className="masthead-actions">
            <button className="reset-link" onClick={openHistory}>
              <Clock size={13} /> History
            </button>
            {step !== "resume" && (
              <button className="reset-link" onClick={resetAll}>
                <RotateCcw size={13} /> Start over
              </button>
            )}
          </div>
        </div>

        <div className="config-warning" style={{ background: "rgba(79, 168, 201, 0.12)", borderColor: "var(--teal)", color: "var(--text)" }}>
          <Sparkles size={15} style={{ flexShrink: 0, marginTop: 1, color: "var(--teal)" }} />
          <span>
            <strong>Browser Prototype Active:</strong> All resume parsing, ATS scoring, and review history are processed and saved directly in your browser's local storage. Your data stays 100% private on your machine with zero configuration.
          </span>
        </div>

        {mockMode && pipelineDone && (
          <div className="mock-banner">
            <AlertCircle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
            <span>
              Running in local heuristic mode — classical TF-IDF, section extraction, and keyword similarity estimates are active.
            </span>
          </div>
        )}

        {showHistory ? (
          <div className="panel">
            <div className="history-head">
              <div>
                <h2>Past reviews</h2>
                <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "2px" }}>
                  Saved in your browser's local storage
                </div>
              </div>
              <div style={{ display: "flex", gap: "8px" }}>
                {historyList.length > 0 && (
                  <button
                    className="ghost-btn"
                    onClick={handleClearAllHistory}
                    style={{ color: "var(--flare)", borderColor: "var(--flare)" }}
                  >
                    Clear history
                  </button>
                )}
                <button className="ghost-btn" onClick={() => setShowHistory(false)}>
                  Back
                </button>
              </div>
            </div>
            {historyLoading && <p className="hint">Loading saved reviews...</p>}
            {historyError && (
              <div className="analysis-error">
                <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
                <span>{historyError}</span>
              </div>
            )}
            {!historyLoading && !historyError && historyList.length === 0 && (
              <p className="hint">No saved reviews yet. Every completed review is stored in your browser automatically.</p>
            )}
            {historyList.map((h) => (
              <div key={h.id} className="history-item">
                <div>
                  <div className="history-title">{h.jobTitle}</div>
                  <div className="history-meta">
                    {h.roleLabel} · match {h.matchScore} · ATS {h.atsScore} · {formatDate(h.timestamp)}
                  </div>
                </div>
                <div className="history-actions">
                  <button className="ghost-btn" onClick={() => reopenReview(h.id)}>
                    Reopen
                  </button>
                  <button className="ghost-btn" onClick={() => deleteReview(h.id)}>
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <>
            <div className="steps">
              {STEPS.map((s, i) => (
                <div key={s.key} className={"step-item " + (i < stepIndex ? "done" : i === stepIndex ? "active" : "")}>
                  <span className="step-num">{String(i + 1).padStart(2, "0")}</span>
                  <span className="step-label">{s.label}</span>
                </div>
              ))}
            </div>

            {step === "resume" && (
              <div className="panel">
                <h2>Start with your resume</h2>
                <p className="hint">Upload a .pdf, .docx, or .txt file, or paste your resume text directly below.</p>
                <div className="upload-row">
                  <button className="upload-btn" onClick={() => fileInputRef.current && fileInputRef.current.click()}>
                    <Upload size={14} /> Upload resume (.pdf, .docx, .txt)
                  </button>
                  <button className="ghost-btn" type="button" onClick={handleLoadSampleResume}>
                    <Sparkles size={14} style={{ color: "var(--amber)" }} /> Try sample resume
                  </button>
                  <input ref={fileInputRef} type="file" accept=".docx,.txt,.md,.pdf,application/pdf" style={{ display: "none" }} onChange={handleFile} />
                  {fileName && (
                    <span className="file-name">
                      <FileText size={13} style={{ verticalAlign: "text-bottom", marginRight: 4 }} />
                      {fileName}
                    </span>
                  )}
                </div>
                {fileError && (
                  <div className="file-error">
                    <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
                    <span>{fileError}</span>
                  </div>
                )}
                <div className="divider-or">or paste it below</div>
                <textarea
                  value={resumeText}
                  onChange={(e) => setResumeText(e.target.value)}
                  placeholder="Paste your resume text here..."
                />
                <div className="actions-row">
                  <button className="primary-btn" disabled={!canGoToJob} onClick={() => setStep("job")}>
                    Next: the role <ArrowRight size={15} />
                  </button>
                </div>
              </div>
            )}

            {step === "job" && (
              <div className="panel">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "10px" }}>
                  <div>
                    <h2>What job are you applying for?</h2>
                    <p className="hint">Paste the full job description. The more detail it has, the sharper the review will be.</p>
                  </div>
                  <button className="ghost-btn" type="button" onClick={handleLoadSampleJob}>
                    <Sparkles size={14} style={{ color: "var(--amber)" }} /> Fill sample role
                  </button>
                </div>
                <textarea
                  value={jobText}
                  onChange={(e) => setJobText(e.target.value)}
                  placeholder="Paste the job description here..."
                />
                <div className="actions-row" style={{ justifyContent: "space-between" }}>
                  <button className="ghost-btn" onClick={() => setStep("resume")}>
                    Back
                  </button>
                  <button className="primary-btn" disabled={!canAnalyze} onClick={runPipeline}>
                    Roll the review <ArrowRight size={15} />
                  </button>
                </div>
              </div>
            )}

            {step === "review" && !pipelineDone && !pipelineError && (
              <div className="panel">
                <div className="analyzing-wrap">
                  <Iris size={72} />
                  <h2>Running the pipeline</h2>
                  <p>Five passes over both documents — nothing here is one shot.</p>
                </div>
                <div className="acts-wrap">
                  {ACTS.map((a, i) => {
                    const data = actData[a.key];
                    const isDone = !!data;
                    const isRunning = !isDone && i === actIndex && running;
                    return (
                      <div key={a.key} className={"act-row " + (isDone ? "done" : isRunning ? "running active" : "")}>
                        <span className="act-roman">{a.roman}</span>
                        <div className="act-body">
                          <div className="act-title">{a.title}</div>
                          {isDone && <div className="act-result">{actSummary(a.key, data)}</div>}
                        </div>
                        <div className="act-icon">
                          {isDone && <Check size={16} color="var(--teal-green)" />}
                          {isRunning && <Iris size={22} />}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {step === "review" && pipelineError && (
              <div className="panel">
                <div className="analysis-error">
                  <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
                  <span>{pipelineError}</span>
                </div>
                <div className="actions-row" style={{ justifyContent: "flex-start", marginTop: 20 }}>
                  <button className="ghost-btn" onClick={runPipeline}>
                    Try again
                  </button>
                </div>
              </div>
            )}

            {step === "review" && pipelineDone && (
              <div>
                <div className="result-head">
                  <span className="role-badge">{jobParsed.roleLabel}</span>
                  <h2>{jobParsed.jobTitle}</h2>
                </div>

                <div className="meters">
                  <LightMeter
                    label="Job match"
                    value={hybrid.blendedMatch}
                    color="var(--amber)"
                    glow="var(--amber-glow)"
                  >
                    <MiniStat label="Model estimate" value={gapAnalysis.matchScore} />
                    <MiniStat label="TF-IDF / cosine (computed)" value={hybrid.classicalScore} />
                  </LightMeter>
                  <LightMeter
                    label="ATS score"
                    value={atsAnalysis.atsScore}
                    color="var(--teal)"
                    glow="var(--teal-glow)"
                    caption={atsAnalysis.explanation}
                  >
                    <MiniStat label="Keyword coverage (model)" value={atsAnalysis.keywordCoverage} />
                    <MiniStat label="Keyword coverage (computed)" value={hybrid.classicalKeywordCoverage} />
                    <MiniStat label="Title alignment" value={atsAnalysis.titleAlignment} />
                    <MiniStat label="Formatting" value={atsAnalysis.formattingScore} />
                  </LightMeter>
                </div>
                <p className="method-note">
                  Job match blends the language model's read with an independent TF-IDF and cosine-similarity score computed
                  locally, with no AI involved. ATS keyword coverage is shown both as modeled by the language model and as
                  directly counted against the parsed job keywords, so the two can be checked against each other.
                </p>

                <div className="grid">
                  <div>
                    <div className="tabs">
                      <button className={"tab-btn " + (activeTab === "original" ? "active" : "")} onClick={() => setActiveTab("original")}>
                        Original
                      </button>
                      <button className={"tab-btn " + (activeTab === "optimized" ? "active" : "")} onClick={() => setActiveTab("optimized")}>
                        Optimized
                      </button>
                      <button className={"tab-btn " + (activeTab === "extracted" ? "active" : "")} onClick={() => setActiveTab("extracted")}>
                        Extracted
                      </button>
                    </div>

                    {activeTab === "original" && (
                      <div className="manuscript">
                        {highlightText(resumeText, gapAnalysis.emphasizeSkills, gapAnalysis.unnecessarySkills)}
                      </div>
                    )}

                    {activeTab === "optimized" && (
                      <>
                        {optimization && optimization.error && (
                          <div className="analysis-error">
                            <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
                            <span>{optimization.message}</span>
                          </div>
                        )}
                        {optimization && !optimization.error && (
                          <>
                            <div className="manuscript">
                              {optimization.optimizedSummary && (
                                <p className="summary-lead">{optimization.optimizedSummary}</p>
                              )}
                              {optimization.optimizedResume}
                            </div>
                            <div className="action-row">
                              <button className="ghost-btn" onClick={handleCopy}>
                                {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? "Copied" : "Copy text"}
                              </button>
                              <button className="ghost-btn" onClick={handleDownload}>
                                <Download size={13} /> Download .txt
                              </button>
                              <button className="ghost-btn" onClick={handlePrint}>
                                <Printer size={13} /> Save as PDF
                              </button>
                              <button className="ghost-btn" onClick={handleDownloadWord}>
                                <FileText size={13} /> Download Word
                              </button>
                            </div>
                            {downloadError && <p className="dl-error">{downloadError}</p>}
                          </>
                        )}
                      </>
                    )}

                    {activeTab === "extracted" && (
                      <div className="extract-grid">
                        <div className="extract-col">
                          <h4>From your resume</h4>
                          {resumeParsed.professionalSummary && (
                            <p className="extract-summary">{resumeParsed.professionalSummary}</p>
                          )}
                          {(resumeParsed.education || []).length > 0 && (
                            <div className="extract-block">
                              <span className="extract-label">Education</span>
                              {resumeParsed.education.map((e, i) => (
                                <div key={i} className="extract-entry">{e}</div>
                              ))}
                            </div>
                          )}
                          <div className="extract-block">
                            <span className="extract-label">Skills</span>
                            <div className="skill-chip-row">
                              {(resumeParsed.skills || []).map((s, i) => (
                                <span key={i} className="skill-chip">{s}</span>
                              ))}
                            </div>
                          </div>
                          <div className="extract-block">
                            <span className="extract-label">Technical skills</span>
                            <div className="skill-chip-row">
                              {(resumeParsed.technicalSkills || []).map((s, i) => (
                                <span key={i} className="skill-chip">{s}</span>
                              ))}
                            </div>
                          </div>
                          {(resumeParsed.experience || []).length > 0 && (
                            <div className="extract-block">
                              <span className="extract-label">Experience</span>
                              {resumeParsed.experience.map((e, i) => (
                                <div key={i} className="extract-entry">
                                  <strong>{e.title}</strong>{e.org ? " — " + e.org : ""} <span className="muted">{e.duration}</span>
                                  {(e.highlights || []).map((h, j) => (
                                    <p key={j}>{h}</p>
                                  ))}
                                </div>
                              ))}
                            </div>
                          )}
                          {(resumeParsed.projects || []).length > 0 && (
                            <div className="extract-block">
                              <span className="extract-label">Projects</span>
                              {resumeParsed.projects.map((p, i) => (
                                <div key={i} className="extract-entry">
                                  <strong>{p.name}</strong> <span className="muted">{p.tech}</span>
                                  {p.highlight && <p>{p.highlight}</p>}
                                </div>
                              ))}
                            </div>
                          )}
                          {(resumeParsed.certifications || []).length > 0 && (
                            <div className="extract-block">
                              <span className="extract-label">Certifications</span>
                              <div className="skill-chip-row">
                                {resumeParsed.certifications.map((c, i) => (
                                  <span key={i} className="skill-chip">{c}</span>
                                ))}
                              </div>
                            </div>
                          )}
                          {(resumeParsed.achievements || []).length > 0 && (
                            <div className="extract-block">
                              <span className="extract-label">Achievements</span>
                              {resumeParsed.achievements.map((a, i) => (
                                <div key={i} className="extract-entry">{a}</div>
                              ))}
                            </div>
                          )}
                        </div>

                        <div className="extract-col">
                          <h4>From the job</h4>
                          <div className="extract-block">
                            <span className="extract-label">Domain / seniority</span>
                            <div className="extract-entry">
                              {jobParsed.domain}
                              {jobParsed.domain && jobParsed.seniority ? " — " : ""}
                              {jobParsed.seniority}
                            </div>
                          </div>
                          <div className="extract-block">
                            <span className="extract-label">Required skills</span>
                            <div className="skill-chip-row">
                              {(jobParsed.requiredSkills || []).map((s, i) => (
                                <span key={i} className="skill-chip">{s}</span>
                              ))}
                            </div>
                          </div>
                          <div className="extract-block">
                            <span className="extract-label">Preferred skills</span>
                            <div className="skill-chip-row">
                              {(jobParsed.preferredSkills || []).map((s, i) => (
                                <span key={i} className="skill-chip">{s}</span>
                              ))}
                            </div>
                          </div>
                          <div className="extract-block">
                            <span className="extract-label">Technologies</span>
                            <div className="skill-chip-row">
                              {(jobParsed.technologies || []).map((s, i) => (
                                <span key={i} className="skill-chip">{s}</span>
                              ))}
                            </div>
                          </div>
                          {(jobParsed.responsibilities || []).length > 0 && (
                            <div className="extract-block">
                              <span className="extract-label">Responsibilities</span>
                              {jobParsed.responsibilities.map((r, i) => (
                                <div key={i} className="extract-entry">{r}</div>
                              ))}
                            </div>
                          )}
                          {(jobParsed.qualifications || []).length > 0 && (
                            <div className="extract-block">
                              <span className="extract-label">Qualifications</span>
                              {jobParsed.qualifications.map((q, i) => (
                                <div key={i} className="extract-entry">{q}</div>
                              ))}
                            </div>
                          )}
                          <div className="extract-block">
                            <span className="extract-label">ATS keywords</span>
                            <div className="skill-chip-row">
                              {(jobParsed.keywords || []).map((k, i) => (
                                <span key={i} className="skill-chip">{k}</span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  <div>
                    <div className="side-section">
                      <h3>Matching skills</h3>
                      <div className="skill-chip-row">
                        {(gapAnalysis.matchingSkills || []).map((s, i) => (
                          <span key={i} className="skill-chip match">{s}</span>
                        ))}
                      </div>
                    </div>

                    <div className="side-section">
                      <h3>Missing skills</h3>
                      {(gapAnalysis.missingSkills || []).map((m, i) => (
                        <div key={i} className={"missing-item " + (m.status === "genuinely_missing" ? "genuine" : "notmentioned")}>
                          <strong>{m.skill}</strong>
                          <span className="status">
                            {m.status === "genuinely_missing" ? "Genuinely missing — " : "You may have this, unstated — "}
                            {m.note}
                          </span>
                        </div>
                      ))}
                    </div>

                    <div className="side-section">
                      <h3>ATS gaps</h3>
                      {(atsAnalysis.missingKeywords || []).length > 0 && (
                        <div className="skill-chip-row" style={{ marginBottom: 12 }}>
                          {atsAnalysis.missingKeywords.map((k, i) => (
                            <span key={i} className="skill-chip gap">{k}</span>
                          ))}
                        </div>
                      )}
                      {(atsAnalysis.formattingNotes || []).map((f, i) => (
                        <p key={i} className="format-note">{f}</p>
                      ))}
                    </div>

                    <div className="side-section">
                      <h3>Editor's notes</h3>
                      {(gapAnalysis.recommendations || []).map((r, i) => (
                        <div key={i} className="note-item">
                          <div className="note-head">
                            <Tag type={r.type} />
                            <span className="note-section">{r.section}</span>
                          </div>
                          <p className="note-detail">{r.detail}</p>
                          <p className="note-reason">{r.reason}</p>
                        </div>
                      ))}
                    </div>

                    {optimization && !optimization.error && optimization.changes && (
                      <div className="side-section">
                        <h3>What changed</h3>
                        {optimization.changes.map((c, i) => (
                          <div key={i} className="note-item">
                            <div className="note-head">
                              <Tag type={c.type} />
                            </div>
                            <p className="note-detail">{c.note}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {optimization && !optimization.error && (
        <div className="print-resume">
          {optimization.optimizedSummary && <p className="print-lead">{optimization.optimizedSummary}</p>}
          <pre>{optimization.optimizedResume}</pre>
        </div>
      )}
    </div>
  );
}
