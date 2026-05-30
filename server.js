import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { crawlWebsite } from "./crawler.js";
import { askRag } from "./rag.js";
import { getGroqDebugConfig } from "./groq.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const CRAWL_SECRET = process.env.CRAWL_SECRET;
const CHAT_DEBUG = process.env.CHAT_DEBUG === "true";

app.use(cors({
  origin: [
    "https://eurohardware.id.vn",
    "https://www.eurohardware.id.vn",
    "http://localhost:3000",
    "http://localhost:5173"
  ],
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type"]
}));

app.use(express.json());

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    message: "Euro Hardware RAG backend is running"
  });
});

app.post("/crawl", async (req, res) => {
  try {
    const { secret } = req.body;

    if (!CRAWL_SECRET) {
      return res.status(500).json({
        success: false,
        message: "CRAWL_SECRET is not configured"
      });
    }

    if (secret !== CRAWL_SECRET) {
      return res.status(401).json({
        success: false,
        message: "Invalid crawl secret"
      });
    }

    const stats = await crawlWebsite();

    res.json({
      success: true,
      message: "Crawl completed",
      ...stats
    });
  } catch (error) {
    console.error("Crawl failed:", error);

    res.status(500).json({
      success: false,
      message: "Crawl failed",
      error: error.message
    });
  }
});

app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body;

    if (!message || !message.trim()) {
      return res.status(400).json({
        success: false,
        message: "Message is required"
      });
    }

    const startedAt = Date.now();
    const result = await askRag(message);

    if (CHAT_DEBUG) {
      console.log("Chat debug:", JSON.stringify({
        message,
        elapsed_ms: Date.now() - startedAt,
        provider: getGroqDebugConfig(),
        sources: result.sources,
        retrieval: result.debug
      }, null, 2));
    }

    res.json({
      success: true,
      reply: result.reply,
      sources: result.sources
    });
  } catch (error) {
    console.error("Chat failed:", error);

    res.status(500).json({
      success: false,
      message: "Chat failed",
      error: error.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
