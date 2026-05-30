import { supabase } from "./supabase.js";
import { createEmbedding } from "./ollama.js";
import { generateAnswer } from "./groq.js";
import { sanitizeSnippet, normalizeWhitespace } from "./utils.js";

const MATCH_COUNT = Number(process.env.RAG_MATCH_COUNT || 8);
const MIN_SIMILARITY = Number(process.env.RAG_MIN_SIMILARITY || 0.35);
const CHAT_DEBUG = process.env.CHAT_DEBUG === "true";
const KEYWORD_SEARCH_LIMIT = Number(process.env.RAG_KEYWORD_SEARCH_LIMIT || 50);
const SOURCE_CONTEXT_LIMIT = Number(process.env.RAG_SOURCE_CONTEXT_LIMIT || 2);
const SOURCE_SNIPPET_CHAR_LIMIT = Number(process.env.RAG_SOURCE_SNIPPET_CHAR_LIMIT || 1200);
const DEFAULT_HOTLINE = process.env.CONTACT_HOTLINE || "079 619 2091";

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

function buildFallbackSourceKey(match = {}) {
  return match.url || `${match.source_type || "unknown"}::${match.title || "untitled"}`;
}

function getSourceTypeBonus(intent, sourceType = "") {
  if (intent.isPriceIntent && sourceType === "product") {
    return 24;
  }

  if (intent.isPriceIntent && sourceType === "post") {
    return -6;
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

function buildFallbackReply(question, sourceGroups) {
  const intent = detectIntent(question);
  const primarySource = sourceGroups[0];

  if (!primarySource) {
    return `Chao anh/chi, hien thong tin em co chua du de tu van that sat nhu cau nay. Anh/chi co the de lai yeu cau cu the hon hoac lien he Hotline/Zalo ${DEFAULT_HOTLINE} de ben em ho tro nhanh va dung hon nha.`;
  }

  const productLink = primarySource.url ? `\nAnh/chi xem them tai day: ${primarySource.url}` : "";

  if (intent.isPriceIntent) {
    return `Chao anh/chi, ben em co san pham "${primarySource.title}" nha. Hien muc gia tren website co the dang de theo hinh thuc tu van truc tiep, nen anh/chi vui long dang ky bao gia tai https://eurohardware.id.vn/bao-gia hoac lien he Hotline/Zalo ${DEFAULT_HOTLINE} de ben em gui bao gia chinh xac va nhanh hon.${productLink}`;
  }

  if (primarySource.source_type === "product") {
    return `Chao anh/chi, san pham phu hop voi nhu cau anh/chi la "${primarySource.title}" nha.${productLink}`;
  }

  return `Chao anh/chi, ben em gui anh/chi thong tin tham khao phu hop nhat o day nha.${productLink}`;
}

function isProductFocusedQuestion(question, sourceGroups = []) {
  const normalizedQuestion = normalizeForSearch(question);
  const primarySource = sourceGroups[0];

  if (!primarySource || primarySource.source_type !== "product") {
    return false;
  }

  if ((primarySource.exactTokenBonus || 0) > 0) {
    return true;
  }

  return /ma|model|khoa|san pham|thong tin/.test(normalizedQuestion);
}

function formatProductFocusedReply(sourceGroup, generatedReply = "") {
  if (!sourceGroup || sourceGroup.source_type !== "product") {
    return String(generatedReply || "").trim();
  }

  const cleanedReply = normalizeWhitespace(
    String(generatedReply || "")
      .replace(/\n+/g, " ")
      .replace(/\s{2,}/g, " ")
  );
  const sentences = cleanedReply
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)
    .filter((sentence) => !/^xem them|^tham khao|^lien he/i.test(sentence));
  const summary = sentences[0] || `${sourceGroup.title} la san pham phu hop voi nhu cau anh/chi.`;
  const productLink = sourceGroup.url ? ` Xem them: ${sourceGroup.url}` : "";

  return normalizeWhitespace(`${summary}${productLink}`).trim();
}

function isLikelyTruncatedAnswer(answer = "") {
  const text = String(answer).trim();

  if (!text) {
    return true;
  }

  if (/[.!?)]$/.test(text)) {
    return false;
  }

  if (/[,:;*\-]$/.test(text)) {
    return true;
  }

  if (/\*\*[^*]{0,80}$/.test(text)) {
    return true;
  }

  const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
  const lastLine = lines.at(-1) || "";

  if (lastLine.length > 0 && lastLine.length <= 32) {
    return true;
  }

  const sentenceEndCount = (text.match(/[.!?]/g) || []).length;
  return sentenceEndCount < 2 && text.length >= 40;
}

function dedupeSources(sources = []) {
  const seen = new Set();

  return sources.filter((item) => {
    const key = item.url || `${item.title || ""}::${item.source_type || ""}`;

    if (seen.has(key)) {
      return false;
    }

    seen.add(key);
    return true;
  });
}

function compareSourceGroups(a, b, intent) {
  const aHasStrongProductSignal = a.source_type === "product" && (a.exactTokenBonus || 0) > 0;
  const bHasStrongProductSignal = b.source_type === "product" && (b.exactTokenBonus || 0) > 0;

  if (aHasStrongProductSignal !== bHasStrongProductSignal) {
    return bHasStrongProductSignal - aHasStrongProductSignal;
  }

  if (intent.isPriceIntent) {
    const aIsProduct = a.source_type === "product" ? 1 : 0;
    const bIsProduct = b.source_type === "product" ? 1 : 0;

    if (aIsProduct !== bIsProduct) {
      return bIsProduct - aIsProduct;
    }
  }

  if (b.rerankScore !== a.rerankScore) {
    return b.rerankScore - a.rerankScore;
  }

  if ((b.exactTokenBonus || 0) !== (a.exactTokenBonus || 0)) {
    return (b.exactTokenBonus || 0) - (a.exactTokenBonus || 0);
  }

  if ((b.keywordScore || 0) !== (a.keywordScore || 0)) {
    return (b.keywordScore || 0) - (a.keywordScore || 0);
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
  const searchTerms = strongTokens.length
    ? Array.from(new Set(strongTokens))
    : Array.from(new Set(questionKeywords)).slice(0, 6);

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
    const key = match.source_id || buildFallbackSourceKey(match);
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
    const contentTokenBonus =
      strongTokens.some((token) => normalizeForSearch(match.content || "").includes(token)) ? 10 : 0;
    const titleTokenPenalty = strongTokens.length && exactTokenBonus === 0 && contentTokenBonus === 0 ? -18 : 0;
    const rerankScore =
      similarity * 100 +
      keywordScore * 4 +
      sourceTypeBonus +
      exactTokenBonus +
      contentTokenBonus +
      titleTokenPenalty;

    return {
      ...match,
      similarity,
      keywordScore,
      sourceTypeBonus,
      exactTokenBonus,
      contentTokenBonus,
      titleTokenPenalty,
      rerankScore,
      source_root: getSourceRoot(match.source_id || "") || buildFallbackSourceKey(match)
    };
  });
}

function groupMatchesBySource(question, scoredMatches = []) {
  const strongTokens = extractStrongTokens(question);
  const intent = detectIntent(question);
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

  const rankedGroups = Array.from(grouped.values())
    .sort((a, b) => compareSourceGroups(a, b, intent))
    .filter((group) => {
      if (strongTokens.length && group.source_type === "product" && group.exactTokenBonus > 0) {
        return true;
      }

      if (intent.isPriceIntent && strongTokens.length && group.source_type === "post") {
        const hasExactToken = group.exactTokenBonus > 0;
        const hasStrongKeyword = group.keywordScore >= 20;
        return hasExactToken && hasStrongKeyword;
      }

      return true;
    })
    .slice(0, SOURCE_CONTEXT_LIMIT)
    .filter((group, index, all) => {
      if (strongTokens.length && group.exactTokenBonus <= 0 && group.keywordScore < 8) {
        return false;
      }

      if (index === 0) {
        return true;
      }

      return (
        group.similarity >= MIN_SIMILARITY ||
        group.keywordScore >= 4 ||
        group.rerankScore >= all[0].rerankScore * 0.65
      );
    });

  const primaryGroup = rankedGroups[0];

  if (
    primaryGroup &&
    primaryGroup.source_type === "product" &&
    (primaryGroup.exactTokenBonus || 0) > 0
  ) {
    return rankedGroups.filter((group) => group.source_root === primaryGroup.source_root);
  }

  return rankedGroups;
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
        sanitizeSnippet(combinedContent, SOURCE_SNIPPET_CHAR_LIMIT)
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n");
}

function buildRetrievalDebug(question, vectorMatches, keywordMatches, sourceGroups, timings = {}) {
  return {
    question,
    vector_match_count: vectorMatches.length,
    keyword_match_count: keywordMatches.length,
    selected_source_count: sourceGroups.length,
    timings_ms: timings,
    selected_sources: sourceGroups.map((item) => ({
      source_root: item.source_root,
      title: item.title,
      source_type: item.source_type,
      similarity: Number(item.similarity || 0).toFixed(3),
      keywordScore: item.keywordScore,
      exactTokenBonus: item.exactTokenBonus,
      contentTokenBonus: item.contentTokenBonus,
      titleTokenPenalty: item.titleTokenPenalty,
      rerankScore: Number(item.rerankScore || 0).toFixed(3),
      url: item.url
    }))
  };
}

export async function askRag(question) {
  if (!question || !question.trim()) {
    throw new Error("Question is required");
  }

  const requestStartedAt = Date.now();
  const cleanQuestion = normalizeWhitespace(question);

  const embeddingStartedAt = Date.now();
  const queryEmbedding = await createEmbedding(cleanQuestion);
  const embeddingMs = Date.now() - embeddingStartedAt;

  const vectorSearchStartedAt = Date.now();
  const { data: vectorMatches, error } = await supabase.rpc("match_website_documents", {
    query_embedding: queryEmbedding,
    match_count: MATCH_COUNT
  });
  const vectorSearchMs = Date.now() - vectorSearchStartedAt;

  if (error) {
    throw error;
  }

  const keywords = extractKeywords(cleanQuestion);
  const strongTokens = extractStrongTokens(cleanQuestion);
  const keywordSearchStartedAt = Date.now();
  const keywordMatches = await searchKeywordMatches(keywords, strongTokens);
  const keywordSearchMs = Date.now() - keywordSearchStartedAt;
  const mergedMatches = mergeMatches(vectorMatches || [], keywordMatches);
  const scoredMatches = scoreMatches(cleanQuestion, mergedMatches);
  const sourceGroups = groupMatchesBySource(cleanQuestion, scoredMatches);
  const timings = {
    embedding: embeddingMs,
    vector_search: vectorSearchMs,
    keyword_search: keywordSearchMs,
    retrieval_processing:
      Math.max(Date.now() - requestStartedAt - embeddingMs - vectorSearchMs - keywordSearchMs, 0)
  };
  const retrievalDebug = buildRetrievalDebug(
    cleanQuestion,
    vectorMatches || [],
    keywordMatches,
    sourceGroups,
    timings
  );

  if (CHAT_DEBUG) {
    console.log("RAG retrieval debug:", JSON.stringify(retrievalDebug, null, 2));
  }

  if (!sourceGroups.length) {
    return {
      reply:
        "Da, hien em chua tim thay thong tin du sat tren website. Anh/chi co the de lai nhu cau cu the hoac lien he Hotline/Zalo 079 619 2091 de doi ngu Euro Hardware ho tro nhanh hon a.",
      sources: [],
      debug: {
        ...retrievalDebug,
        timings_ms: {
          ...timings,
          total: Date.now() - requestStartedAt
        }
      }
    };
  }

  const sourceFetchStartedAt = Date.now();
  const sourceDocuments = await fetchSourceContext(sourceGroups.map((group) => group.source_root));
  timings.source_context_fetch = Date.now() - sourceFetchStartedAt;
  const context = buildContext(sourceGroups, sourceDocuments);
  const productFocused = isProductFocusedQuestion(cleanQuestion, sourceGroups);
  let reply;

  try {
    const generationStartedAt = Date.now();
    reply = await generateAnswer({
      question: cleanQuestion,
      context,
      concise: productFocused,
      productFocused
    });
    timings.answer_generation = Date.now() - generationStartedAt;

    if (productFocused) {
      reply = formatProductFocusedReply(sourceGroups[0], reply);
    }

    if (isLikelyTruncatedAnswer(reply)) {
      try {
        const regenerationStartedAt = Date.now();
        const retriedReply = await generateAnswer({
          question: cleanQuestion,
          context,
          concise: true,
          productFocused
        });

        if (!isLikelyTruncatedAnswer(retriedReply)) {
          reply = productFocused
            ? formatProductFocusedReply(sourceGroups[0], retriedReply)
            : retriedReply;
        }

        timings.answer_regeneration = Date.now() - regenerationStartedAt;
      } catch (regenerationError) {
        timings.answer_regeneration = timings.answer_regeneration || 0;

        if (CHAT_DEBUG) {
          console.warn("Answer regeneration skipped after error:", regenerationError.message);
        }
      }
    }
  } catch (error) {
    const message = error.message || "";
    const normalizedMessage = message.toLowerCase();
    const isTemporaryAnswerError =
      message.includes("429") ||
      message.includes("Too Many Requests") ||
      message.includes("quota") ||
      message.includes("503") ||
      message.includes("Service Unavailable") ||
      normalizedMessage.includes("groq answer generation failed") ||
      normalizedMessage.includes("ollama answer generation failed") ||
      normalizedMessage.includes("timeout") ||
      normalizedMessage.includes("econnrefused") ||
      normalizedMessage.includes("unavailable");

    if (!isTemporaryAnswerError) {
      throw error;
    }

    console.warn("Answer fallback reply activated:", message);
    reply = buildFallbackReply(cleanQuestion, sourceGroups);
    timings.answer_generation = timings.answer_generation || 0;
  }

  timings.total = Date.now() - requestStartedAt;

  const sources = dedupeSources(
    sourceGroups.map((item) => ({
      title: item.title,
      url: item.url,
      source_type: item.source_type,
      similarity: item.similarity
    }))
  );

  return {
    reply,
    sources,
    debug: {
      ...retrievalDebug,
      timings_ms: timings
    }
  };
}
