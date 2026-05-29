import { supabase } from "./supabase.js";
import { createEmbedding } from "./ollama.js";
import { generateAnswer } from "./gemini.js";
import { sanitizeSnippet, normalizeWhitespace } from "./utils.js";

const MATCH_COUNT = Number(process.env.RAG_MATCH_COUNT || 8);
const MIN_SIMILARITY = Number(process.env.RAG_MIN_SIMILARITY || 0.35);
const CHAT_DEBUG = process.env.CHAT_DEBUG === "true";
const KEYWORD_SEARCH_LIMIT = Number(process.env.RAG_KEYWORD_SEARCH_LIMIT || 20);
const SOURCE_CONTEXT_LIMIT = Number(process.env.RAG_SOURCE_CONTEXT_LIMIT || 3);

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
  "em",
  "toi",
  "muon",
  "can",
  "duoc"
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

function escapeLikeValue(value = "") {
  return String(value).replace(/[,%]/g, " ");
}

function extractKeywords(question = "") {
  return normalizeForSearch(question)
    .split(" ")
    .filter((word) => word.length >= 2)
    .filter((word) => !STOP_WORDS.has(word));
}

function extractStrongTokens(question = "") {
  return Array.from(
    new Set(
      normalizeForSearch(question)
        .split(" ")
        .filter((word) => word.length >= 3)
        .filter((word) => /[a-z]/.test(word) && /\d/.test(word))
    )
  );
}

function detectIntent(question = "") {
  const normalized = normalizeForSearch(question);

  return {
    isPriceIntent:
      /bao gia|gia bao nhieu|gia sao|bang gia|xin gia|bao nhieu tien|quotation/.test(
        normalized
      ),
    isCategoryIntent:
      /danh muc|nhom san pham|loai nao|dong nao|phan loai|giai phap/.test(normalized),
    isContactIntent: /hotline|zalo|so dien thoai|lien he|tu van/.test(normalized),
    isTechnicalIntent:
      /thong so|kich thuoc|chat lieu|mau sac|do day|cau tao|phu hop|ung dung|dac diem/.test(
        normalized
      )
  };
}

function getSourceRoot(sourceId = "") {
  return sourceId.includes("::chunk::") ? sourceId.split("::chunk::")[0] : sourceId;
}

function getSourceTypeBonus(intent, sourceType = "") {
  if (intent.isPriceIntent && sourceType === "product") {
    return 12;
  }

  if (intent.isCategoryIntent && (sourceType === "category" || sourceType === "page")) {
    return 8;
  }

  if (intent.isContactIntent && sourceType === "page") {
    return 6;
  }

  if (intent.isTechnicalIntent && sourceType === "product") {
    return 7;
  }

  if (sourceType === "product") {
    return 2;
  }

  return 0;
}

function computeKeywordScore(questionKeywords, strongTokens, match) {
  const titleText = normalizeForSearch(match.title || "");
  const contentText = normalizeForSearch(match.content || "");
  const urlText = normalizeForSearch(match.url || "");

  let score = questionKeywords.reduce((currentScore, keyword) => {
    let nextScore = currentScore;

    if (titleText.includes(keyword)) {
      nextScore += 4;
    }

    if (contentText.includes(keyword)) {
      nextScore += 1;
    }

    if (urlText.includes(keyword)) {
      nextScore += 1;
    }

    return nextScore;
  }, 0);

  strongTokens.forEach((token) => {
    if (titleText.includes(token)) {
      score += 15;
    }

    if (contentText.includes(token)) {
      score += 6;
    }

    if (urlText.includes(token)) {
      score += 8;
    }
  });

  return score;
}

async function searchKeywordMatches(questionKeywords, strongTokens) {
  const searchTerms = Array.from(new Set([...strongTokens, ...questionKeywords])).slice(0, 6);

  if (!searchTerms.length) {
    return [];
  }

  const orFilters = searchTerms.flatMap((term) => {
    const safeTerm = escapeLikeValue(term);
    return [`title.ilike.%${safeTerm}%`, `content.ilike.%${safeTerm}%`, `url.ilike.%${safeTerm}%`];
  });

  const { data, error } = await supabase
    .from("website_documents")
    .select("source_id, source_type, title, url, content")
    .or(orFilters.join(","))
    .limit(KEYWORD_SEARCH_LIMIT);

  if (error) {
    throw error;
  }

  return (data || []).map((item) => ({
    ...item,
    similarity: 0
  }));
}

function mergeMatches(vectorMatches = [], keywordMatches = []) {
  const merged = new Map();

  for (const match of [...vectorMatches, ...keywordMatches]) {
    const key = match.source_id;
    const existing = merged.get(key);

    if (!existing) {
      merged.set(key, { ...match });
      continue;
    }

    merged.set(key, {
      ...existing,
      similarity: Math.max(Number(existing.similarity || 0), Number(match.similarity || 0))
    });
  }

  return Array.from(merged.values());
}

function scoreMatches(question, matches = []) {
  const keywords = extractKeywords(question);
  const strongTokens = extractStrongTokens(question);
  const intent = detectIntent(question);

  return matches.map((match) => {
    const similarity = Number(match.similarity || 0);
    const keywordScore = computeKeywordScore(keywords, strongTokens, match);
    const sourceTypeBonus = getSourceTypeBonus(intent, match.source_type || "");
    const exactTokenBonus =
      strongTokens.some((token) => normalizeForSearch(match.title || "").includes(token)) ? 20 : 0;
    const rerankScore = similarity * 100 + keywordScore * 4 + sourceTypeBonus + exactTokenBonus;

    return {
      ...match,
      similarity,
      keywordScore,
      sourceTypeBonus,
      exactTokenBonus,
      rerankScore,
      source_root: getSourceRoot(match.source_id || "")
    };
  });
}

function groupMatchesBySource(scoredMatches = []) {
  const grouped = new Map();

  for (const match of scoredMatches) {
    const sourceRoot = match.source_root;
    const existing = grouped.get(sourceRoot);

    if (!existing) {
      grouped.set(sourceRoot, {
        source_root: sourceRoot,
        title: match.title,
        url: match.url,
        source_type: match.source_type,
        similarity: match.similarity,
        keywordScore: match.keywordScore,
        sourceTypeBonus: match.sourceTypeBonus,
        exactTokenBonus: match.exactTokenBonus,
        rerankScore: match.rerankScore,
        chunks: [match]
      });
      continue;
    }

    existing.similarity = Math.max(existing.similarity, match.similarity);
    existing.keywordScore = Math.max(existing.keywordScore, match.keywordScore);
    existing.sourceTypeBonus = Math.max(existing.sourceTypeBonus, match.sourceTypeBonus);
    existing.exactTokenBonus = Math.max(existing.exactTokenBonus, match.exactTokenBonus);
    existing.rerankScore = Math.max(existing.rerankScore, match.rerankScore) + 1;
    existing.chunks.push(match);
  }

  return Array.from(grouped.values())
    .sort((a, b) => b.rerankScore - a.rerankScore)
    .slice(0, SOURCE_CONTEXT_LIMIT)
    .filter((group, index, all) => {
      if (index === 0) {
        return true;
      }

      return (
        group.similarity >= MIN_SIMILARITY ||
        group.keywordScore >= 4 ||
        group.rerankScore >= all[0].rerankScore * 0.65
      );
    });
}

async function fetchSourceContext(sourceRoots = []) {
  if (!sourceRoots.length) {
    return [];
  }

  const orFilter = sourceRoots
    .flatMap((sourceRoot) => [`source_id.eq.${sourceRoot}`, `source_id.like.${sourceRoot}::chunk::%`])
    .join(",");

  const { data, error } = await supabase
    .from("website_documents")
    .select("source_id, source_type, title, url, content")
    .or(orFilter)
    .limit(100);

  if (error) {
    throw error;
  }

  return data || [];
}

function buildContext(sourceGroups = [], sourceDocuments = []) {
  if (!sourceGroups.length) {
    return "";
  }

  const documentsBySourceRoot = sourceDocuments.reduce((map, document) => {
    const sourceRoot = getSourceRoot(document.source_id || "");

    if (!map.has(sourceRoot)) {
      map.set(sourceRoot, []);
    }

    map.get(sourceRoot).push(document);
    return map;
  }, new Map());

  return sourceGroups
    .map((group, index) => {
      const documents = (documentsBySourceRoot.get(group.source_root) || []).sort((a, b) =>
        a.source_id.localeCompare(b.source_id)
      );

      const combinedContent = documents.map((document) => document.content || "").join("\n\n");

      return [
        `[Nguon ${index + 1}]`,
        `Tieu de: ${group.title || "Khong ro tieu de"}`,
        `Loai: ${group.source_type || "unknown"}`,
        group.url ? `Link: ${group.url}` : "",
        `Do lien quan vector: ${group.similarity.toFixed(3)}`,
        `Diem tu khoa: ${group.keywordScore}`,
        `So chunk lien quan: ${documents.length}`,
        "Noi dung tong hop:",
        sanitizeSnippet(combinedContent, 2400)
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n");
}

function buildRetrievalDebug(question, vectorMatches, keywordMatches, sourceGroups) {
  return {
    question,
    vector_match_count: vectorMatches.length,
    keyword_match_count: keywordMatches.length,
    selected_source_count: sourceGroups.length,
    selected_sources: sourceGroups.map((item) => ({
      source_root: item.source_root,
      title: item.title,
      source_type: item.source_type,
      similarity: Number(item.similarity || 0).toFixed(3),
      keywordScore: item.keywordScore,
      exactTokenBonus: item.exactTokenBonus,
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

  const { data: vectorMatches, error } = await supabase.rpc("match_website_documents", {
    query_embedding: queryEmbedding,
    match_count: MATCH_COUNT
  });

  if (error) {
    throw error;
  }

  const keywords = extractKeywords(cleanQuestion);
  const strongTokens = extractStrongTokens(cleanQuestion);
  const keywordMatches = await searchKeywordMatches(keywords, strongTokens);
  const mergedMatches = mergeMatches(vectorMatches || [], keywordMatches);
  const scoredMatches = scoreMatches(cleanQuestion, mergedMatches);
  const sourceGroups = groupMatchesBySource(scoredMatches);
  const retrievalDebug = buildRetrievalDebug(
    cleanQuestion,
    vectorMatches || [],
    keywordMatches,
    sourceGroups
  );

  if (CHAT_DEBUG) {
    console.log("RAG retrieval debug:", JSON.stringify(retrievalDebug, null, 2));
  }

  if (!sourceGroups.length) {
    return {
      reply:
        "Da, hien em chua tim thay thong tin du sat tren website. Anh/chi co the de lai nhu cau cu the hoac lien he Hotline/Zalo 079 619 2091 de doi ngu Euro Hardware ho tro nhanh hon a.",
      sources: [],
      debug: retrievalDebug
    };
  }

  const sourceDocuments = await fetchSourceContext(sourceGroups.map((group) => group.source_root));
  const context = buildContext(sourceGroups, sourceDocuments);
  const reply = await generateAnswer({
    question: cleanQuestion,
    context
  });

  const sources = sourceGroups.map((item) => ({
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
