// server.js
// B.Com / CA Student Tutor Chatbot Backend
// Deploy target: Render.com
// AI Provider: Groq (OpenAI-compatible API)

const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const PORT = process.env.PORT || 5000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const MODEL = process.env.GROQ_MODEL || "openai/gpt-oss-120b";

// Render provides this automatically once deployed (e.g. https://your-app.onrender.com)
// You can also set it manually in Render's Environment tab as SELF_URL.
const SELF_URL = process.env.RENDER_EXTERNAL_URL || process.env.SELF_URL || null;

// ---- Usage control settings (to protect a free-tier API quota) ----
const MAX_INPUT_CHARS = 2000;       // reject overly long pastes
const MAX_HISTORY_TURNS = 6;        // only keep the last N turns of context
const MAX_OUTPUT_TOKENS = 512;      // cap how much the model can generate
const RATE_LIMIT_WINDOW_MS = 60 * 60 * 1000; // 1 hour
const RATE_LIMIT_MAX_REQUESTS = 30; // max requests per IP per window

if (!GROQ_API_KEY) {
  console.warn("WARNING: GROQ_API_KEY is not set. Set it in Render's Environment settings.");
}

// ---- Keep-alive: self-ping every 5 minutes so Render's free tier doesn't cold-sleep ----
if (SELF_URL) {
  setInterval(() => {
    fetch(SELF_URL)
      .then(() => console.log("Self-ping OK:", new Date().toISOString()))
      .catch((err) => console.error("Self-ping failed:", err.message));
  }, 5 * 60 * 1000); // every 5 minutes
} else {
  console.warn(
    "SELF_URL / RENDER_EXTERNAL_URL not found. Self-ping keep-alive is disabled. " +
    "On Render this should be set automatically once deployed."
  );
}

// ---- Simple in-memory per-IP rate limiter (resets on server restart) ----
const requestLog = new Map(); // ip -> [timestamps]

function isRateLimited(ip) {
  const now = Date.now();
  const timestamps = (requestLog.get(ip) || []).filter(
    (t) => now - t < RATE_LIMIT_WINDOW_MS
  );
  timestamps.push(now);
  requestLog.set(ip, timestamps);
  return timestamps.length > RATE_LIMIT_MAX_REQUESTS;
}

// System prompt: defines the assistant's personality and behavior
const SYSTEM_PROMPT = `
You are a friendly, knowledgeable tutor built specifically for B.Com and Chartered Accountancy (CA) students.

Your job:
- If the student pastes a paragraph and asks to "cut short" / "shorten" / "summarize" it, give a concise summary that keeps all key points, figures, and terms.
- If the student asks to "explain" a paragraph or concept, explain it clearly and simply, like a teacher would to a student preparing for exams.
- Help with topics relevant to B.Com and CA students: Accountancy, Costing, Taxation (Income Tax, GST), Auditing, Corporate Law, Financial Management, Economics, Business Studies, journal entries, ledger concepts, numericals, formulas, and exam-oriented explanations.
- Give definitions, examples, and step-by-step working for numerical problems when asked.
- Keep answers exam-relevant and easy to revise from.



If the student asks for a 2-mark answer:

Give a short and direct answer containing the key definition
or important point.

If the student asks for a 5-mark answer:

Give:

Definition or introduction
Important points
Simple explanation
Example when useful

If the student asks for a 10-mark answer:

Give:

Introduction
Detailed explanation
Important points
Examples when useful
Conclusion when appropriate

Make exam answers easy for B.Com students to understand,
learn, and remember.


Use previous conversation messages when they are relevant.

For example:

Student:
What is inflation?

Assistant:
Inflation is...


STRICT FORMATTING RULE:
- Never use markdown symbols in your response. Do NOT use asterisks (*), double asterisks (**), hash symbols (#), underscores (_), or backticks.
- Do not try to simulate bold, italic, or heading formatting using symbols.
- Write in plain, clean sentences and plain numbered or lettered lists (e.g., "1.", "2.", "a)", "b)") instead of markdown bullets.
- Just write plain text, the way it would appear in a plain notebook.
`.trim();

// Removes any stray markdown symbols the model might still produce
function cleanText(text) {
  if (!text) return text;
  return text
    .replace(/\*\*(.*?)\*\*/g, "$1")   // **bold**
    .replace(/\*(.*?)\*/g, "$1")       // *italic*
    .replace(/__(.*?)__/g, "$1")       // __bold__
    .replace(/_(.*?)_/g, "$1")         // _italic_
    .replace(/`{1,3}([^`]*)`{1,3}/g, "$1") // `code` / ```code```
    .replace(/^\s{0,3}#{1,6}\s*/gm, "") // # Headings
    .replace(/^\s*[-*+]\s+/gm, "")     // markdown bullet points -> plain
    .replace(/[*#_`]/g, "")            // any remaining stray symbols
    .trim();
}

app.get("/", (req, res) => {
  res.send("B.Com/CA Tutor Chatbot backend is running.");
});

app.post("/api/chat", async (req, res) => {
  try {
    const ip = req.headers["x-forwarded-for"]?.split(",")[0]?.trim() || req.socket.remoteAddress;
    if (isRateLimited(ip)) {
      return res.status(429).json({
        error: `Too many requests. Limit is ${RATE_LIMIT_MAX_REQUESTS} messages per hour to protect the free API quota. Please try again later.`,
      });
    }

    const { message, history } = req.body;

    if (!message || typeof message !== "string" || !message.trim()) {
      return res.status(400).json({ error: "Field 'message' is required and must be a non-empty string." });
    }

    if (message.length > MAX_INPUT_CHARS) {
      return res.status(400).json({
        error: `Message is too long (${message.length} characters). Please keep it under ${MAX_INPUT_CHARS} characters to conserve API usage.`,
      });
    }

    if (!GROQ_API_KEY) {
      return res.status(500).json({ error: "Server misconfiguration: GROQ_API_KEY is missing." });
    }

    // Build conversation: system prompt + optional prior history + new user message
    const messages = [{ role: "system", content: SYSTEM_PROMPT }];

    if (Array.isArray(history)) {
      // Only keep the most recent turns to limit tokens sent per request
      const trimmedHistory = history.slice(-MAX_HISTORY_TURNS);
      for (const turn of trimmedHistory) {
        if (
          turn &&
          (turn.role === "user" || turn.role === "assistant") &&
          typeof turn.content === "string"
        ) {
          messages.push({ role: turn.role, content: turn.content });
        }
      }
    }

    messages.push({ role: "user", content: message });

    const groqResponse = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: MODEL,
        messages,
        temperature: 0.4,
        max_tokens: MAX_OUTPUT_TOKENS,
      }),
    });

    if (!groqResponse.ok) {
      const errText = await groqResponse.text();
      console.error("Groq API error:", groqResponse.status, errText);
      return res.status(502).json({ error: "Failed to get a response from the AI provider." });
    }

    const data = await groqResponse.json();
    const rawReply = data?.choices?.[0]?.message?.content || "";
    const reply = cleanText(rawReply);

    return res.json({ reply });
  } catch (err) {
    console.error("Server error:", err);
    return res.status(500).json({ error: "Something went wrong on the server." });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
