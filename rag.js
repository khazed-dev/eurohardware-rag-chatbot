import { supabase } from "./supabase.js";
import { createEmbedding } from "./ollama.js";
import { generateAnswer } from "./gemini.js";
import { sanitizeSnippet, normalizeWhitespace } from "./utils.js";

const MATCH_COUNT = Number(process.env.RAG_MATCH_COUNT || 8);
const MIN_SIMILARITY = Number(process.env.RAG_MIN_SIMILARITY || 0.45);
const CHAT_DEBUG = process.env.CHAT_DEBUG === "true";

const STOP_WORDS = new Set([
  "la",
  "co",
  "cho",
  "xin",
  "voi",
  "gi",
  "nao",
  "bao",
  "muc",
  "ve",
  "toi",
  "cua",
  "va",
  "nhu",
  "the",
  "tren",
  "duoi",
  "hay",
  "tai",
  "mot",
  "nhung",
  "anh",
  "chi",
  "em"
]);

function normalizeForSearch(text = "") {
  return String(text)
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function extractKeywords(question = "") {
  return normalizeForSearch(question)
    .split(" ")
    .filter((word) => word.length >= 2)
    .filter((word) => !STOP_WORDS.has(word));
}

function detectIntent(question = "") {
  const normalized = normalizeForSearch(question);

  return {
    isPriceIntent:
      /bao gia|gia bao nhieu|gia bao nhieu|gia sao|bang gia|xin gia|bao nhieu tien|quotation/.test(
        normalized
      ),
    isCategoryIntent:
      /danh muc|nhom san pham|loai nao|dong nao|phan loai|giai phap/.test(normalized),
    isContactIntent:
      /hotline|zalo|so dien thoai|lien he|tu van/.test(normalized),
    isTechnicalIntent:
      /thong so|kich thuoc|chat lieu|mau sac|do day|cau tao|phu hop|ung dung|dac diem/.test(
        normalized
      )
  };
}

function getSourceTypeBonus(intent, sourceType = "") {
  if (intent.isPriceIntent && sourceType === "product") {
    return 8;
  }

  if (intent.isCategoryIntent && (sourceType === "category" || sourceType === "page")) {
    return 6;
  }

  if (intent.isContactIntent && sourceType === "page") {
    return 5;
  }

  if (intent.isTechnicalIntent && sourceType === "product") {
    return 5;
  }

  return 0;
}

function computeKeywordScore(questionKeywords, match) {
  const titleText = normalizeForSearch(match.title || "");
  const contentText = normalizeForSearch(match.content || "");
  const urlText = normalizeForSearch(match.url || "");

  return questionKeywords.reduce((score, keyword) => {
    let nextScore = score;

    if (titleText.includes(keyword)) {
      nextScore += 3;
    }

    if (contentText.includes(keyword)) {
      nextScore += 1;
    }

    if (urlText.includes(keyword)) {
      nextScore += 1;
    }

    return nextScore;
  }, 0);
}

function rerankMatches(question, matches = []) {
  const keywords = extractKeywords(question);
  const intent = detectIntent(question);

  return matches
    .map((match) => {
      const similarity = Number(match.similarity || 0);
      const keywordScore = computeKeywordScore(keywords, match);
      const sourceTypeBonus = getSourceTypeBonus(intent, match.source_type || "");
      const rerankScore = similarity * 100 + keywordScore * 4 + sourceTypeBonus;

      return {
        ...match,
        similarity,
        keywordScore,
        sourceTypeBonus,
        rerankScore
      };
    })
    .sort((a, b) => b.rerankScore - a.rerankScore)
    .filter((match, index) => index < 5)
    .filter((match, index, all) => {
      if (index === 0) {
        return true;
      }

      return (
        match.similarity >= MIN_SIMILARITY ||
        match.keywordScore >= 2 ||
        match.rerankScore >= all[0].rerankScore * 0.72
      );
    });
}

function dedupeSources(matches = []) {
  const seen = new Set();

  return matches.filter((match) => {
    const key = match.url || `${match.title}-${match.source_type}`;

    if (seen.has(key)) {
      return false;
    }

    seen.add(key);
    return true;
  });
}

function buildContext(matches = []) {
  if (!matches.length) {
    return "";
  }

  return matches
    .map((item, index) => {
      const relevance = item.similarity ? item.similarity.toFixed(3) : "0.000";

      return [
        `[Nguon ${index + 1}]`,
        `Tieu de: ${item.title || "Khong ro tieu de"}`,
        `Loai: ${item.source_type || "unknown"}`,
        item.url ? `Link: ${item.url}` : "",
        `Do lien quan vector: ${relevance}`,
        `Diem tu khoa: ${item.keywordScore || 0}`,
        "Noi dung trich doan:",
        sanitizeSnippet(item.content || "", 900)
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n");
}

function buildRetrievalDebug(question, rawMatches, finalMatches) {
  return {
    question,
    raw_match_count: rawMatches.length,
    final_match_count: finalMatches.length,
    final_matches: finalMatches.map((item) => ({
      title: item.title,
      source_type: item.source_type,
      similarity: Number(item.similarity || 0).toFixed(3),
      keywordScore: item.keywordScore,
      sourceTypeBonus: item.sourceTypeBonus,
      rerankScore: Number(item.rerankScore || 0).toFixed(3),
      url: item.url
    }))
  };
}

export async function askRag(question) {
  if (!question || !question.trim()) {
    throw new Error("Question is required");
  }

  const cleanQuestion = normalizeWhitespace(question);
  const queryEmbedding = await createEmbedding(cleanQuestion);

  const { data: rawMatches, error } = await supabase.rpc("match_website_documents", {
    query_embedding: queryEmbedding,
    match_count: MATCH_COUNT
  });

  if (error) {
    throw error;
  }

  const matches = rerankMatches(cleanQuestion, rawMatches || []);
  const retrievalDebug = buildRetrievalDebug(cleanQuestion, rawMatches || [], matches);

  if (CHAT_DEBUG) {
    console.log("RAG retrieval debug:", JSON.stringify(retrievalDebug, null, 2));
  }

  if (!matches.length) {
    return {
      reply:
        "Da, hien em chua tim thay thong tin du sat tren website. Anh/chi co the de lai nhu cau cu the hoac lien he Hotline/Zalo 079 619 2091 de doi ngu Euro Hardware ho tro nhanh hon a.",
      sources: [],
      debug: retrievalDebug
    };
  }

  const context = buildContext(matches);
  const reply = await generateAnswer({
    question: cleanQuestion,
    context
  });

  const sources = dedupeSources(matches).map((item) => ({
    title: item.title,
    url: item.url,
    source_type: item.source_type,
    similarity: item.similarity
  }));

  return {
    reply,
    sources,
    debug: retrievalDebug
  };
}
