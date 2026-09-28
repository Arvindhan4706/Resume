// src/browserStorage.js
// Client-side browser storage (localStorage) for the Margin prototype.
// Keeps all user reviews, ATS scores, and history completely private and stored locally.

const STORAGE_KEY = "margin_reviews_v1";

/**
 * Get all reviews stored in the browser's localStorage.
 * @returns {Array} List of stored review objects (newest first)
 */
export function getStoredReviews() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list : [];
  } catch (err) {
    console.error("Failed to read reviews from localStorage:", err);
    return [];
  }
}

/**
 * Save a new review to the browser's localStorage.
 * @param {Object} reviewData
 * @returns {Object} The saved review with ID and timestamp
 */
export function saveReviewToStorage(reviewData) {
  try {
    const reviews = getStoredReviews();
    const newEntry = {
      id: "rev_" + Date.now() + "_" + Math.random().toString(36).substring(2, 7),
      created_at: new Date().toISOString(),
      job_title: reviewData.job_title || "Untitled Role",
      role_label: reviewData.role_label || "Candidate Review",
      match_score: reviewData.match_score ?? 0,
      ats_score: reviewData.ats_score ?? 0,
      resume_text: reviewData.resume_text || "",
      job_text: reviewData.job_text || "",
      resume_parsed: reviewData.resume_parsed || null,
      job_parsed: reviewData.job_parsed || null,
      gap_analysis: reviewData.gap_analysis || null,
      ats_analysis: reviewData.ats_analysis || null,
      optimization: reviewData.optimization || null,
      hybrid: reviewData.hybrid || null,
    };

    // Store up to 40 reviews locally, newest first
    const updated = [newEntry, ...reviews.slice(0, 39)];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return newEntry;
  } catch (err) {
    console.error("Failed to save review to localStorage:", err);
    return null;
  }
}

/**
 * Get a specific review by ID.
 * @param {string} id
 * @returns {Object|null}
 */
export function getStoredReviewById(id) {
  const reviews = getStoredReviews();
  return reviews.find((r) => r.id === id) || null;
}

/**
 * Delete a review by ID from localStorage.
 * @param {string} id
 * @returns {boolean}
 */
export function deleteStoredReview(id) {
  try {
    const reviews = getStoredReviews();
    const filtered = reviews.filter((r) => r.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    return true;
  } catch (err) {
    console.error("Failed to delete review from localStorage:", err);
    return false;
  }
}

/**
 * Clear all stored reviews from localStorage.
 */
export function clearAllStoredReviews() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    return true;
  } catch (err) {
    return false;
  }
}
