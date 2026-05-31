import { supabase } from "./supabase.js";
import { createEmbedding } from "./ollama.js";
import { generateAnswer, getGroqDebugConfig } from "./groq.js";
import { sanitizeSnippet, normalizeWhitespace } from "./utils.js";

const MATCH_COUNT = Number(process.env.RAG_MATCH_COUNT || 8);
const MIN_SIMILARITY = Number(process.env.RAG_MIN_SIMILARITY || 0.35);
const CHAT_DEBUG = process.env.CHAT_DEBUG === "true";
const KEYWORD_SEARCH_LIMIT = Number(process.env.RAG_KEYWORD_SEARCH_LIMIT || 50);
const SOURCE_CONTEXT_LIMIT = Number(
  process.env.RAG_SOURCE_CONTEXT_LIMIT || process.env.MAX_CONTEXT_CHUNKS || 4
);
const SOURCE_SNIPPET_CHAR_LIMIT = Number(
  process.env.RAG_SOURCE_SNIPPET_CHAR_LIMIT || process.env.MAX_CHARS_PER_CHUNK || 1200
);
const DEFAULT_HOTLINE = process.env.CONTACT_HOTLINE || "082 820 8218";

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
    .replace(/đ/g, "d")
    .replace(/Đ/g, "d")
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

function extractPriorityTokens(question = "") {
  return Array.from(
    new Set(
      normalizeForSearch(question)
        .split(" ")
        .filter((word) => word.length >= 4)
        .filter((word) => !STOP_WORDS.has(word))
    )
  );
}

function extractEntityTokens(question = "") {
  return Array.from(
    new Set(
      extractPriorityTokens(question).filter(
        (word) =>
          word.length >= 5 &&
          ![
            "thong",
            "thongtin",
            "huong",
            "dan",
            "dung",
            "loai",
            "nhung",
            "co",
            "khong",
            "bao",
            "gia",
            "cua",
            "nhom",
            "keo",
            "khoa",
            "gioang",
            "phukien"
          ].includes(word)
      )
    )
  );
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

function levenshteinDistance(a = "", b = "") {
  if (a === b) {
    return 0;
  }

  const rows = a.length + 1;
  const cols = b.length + 1;
  const matrix = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let row = 0; row < rows; row += 1) {
    matrix[row][0] = row;
  }

  for (let col = 0; col < cols; col += 1) {
    matrix[0][col] = col;
  }

  for (let row = 1; row < rows; row += 1) {
    for (let col = 1; col < cols; col += 1) {
      const cost = a[row - 1] === b[col - 1] ? 0 : 1;
      matrix[row][col] = Math.min(
        matrix[row - 1][col] + 1,
        matrix[row][col - 1] + 1,
        matrix[row - 1][col - 1] + cost
      );
    }
  }

  return matrix[a.length][b.length];
}

function hasApproxTokenMatch(textTokens = [], queryToken = "") {
  if (!queryToken) {
    return false;
  }

  return textTokens.some((token) => {
    if (token === queryToken) {
      return true;
    }

    if (Math.abs(token.length - queryToken.length) > 2) {
      return false;
    }

    const maxDistance = queryToken.length >= 6 ? 2 : 1;
    return levenshteinDistance(token, queryToken) <= maxDistance;
  });
}

function matchTokenAcrossFields(match, token = "") {
  if (!token) {
    return false;
  }

  const titleTokens = extractNormalizedTokens(match.title || "");
  const contentTokens = extractNormalizedTokens(match.content || "");
  const urlTokens = extractNormalizedTokens(match.url || "");

  return (
    hasApproxTokenMatch(titleTokens, token) ||
    hasApproxTokenMatch(contentTokens, token) ||
    hasApproxTokenMatch(urlTokens, token)
  );
}

function extractNormalizedTokens(text = "") {
  return normalizeForSearch(text)
    .split(" ")
    .filter(Boolean);
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
    isContactIntent: /hotline|zalo|so dien thoai|dien thoai|so lien he|sdt|lien he/.test(normalized),
    isUsageIntent: /huong dan|cach dung|su dung|lap dat|bao quan|van hanh/.test(normalized),
    isAdviceIntent: /tu van|goi y|nen dung|phu hop|chon loai nao|chon mau nao/.test(normalized),
    isRetailIntent: /ban le|mua le|chi ban si|ban si/.test(normalized),
    isShippingIntent: /giao hang tinh|giao tinh|gui tinh|khu vuc nao|toan quoc|o xa/.test(normalized),
    isQuoteInfoIntent:
      /cung cap gi de bao gia|thong tin gi de bao gia|bao gia chinh xac|can cung cap gi/.test(
        normalized
      ),
    isPolicyIntent:
      /ban le|mua le|chi ban si|ban si|giao tinh|toan quoc|o xa|bao gia chinh xac|can cung cap gi/.test(
        normalized
      ),
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

function extractUsefulContentLines(text = "") {
  return String(text)
    .split(/\n+/)
    .map((line) => normalizeWhitespace(line))
    .filter(Boolean)
    .filter(
      (line) =>
        !/^(loai du lieu|tieu de|danh muc|gia \/ trang thai gia|link|sku|thuong hieu|the|tinh trang kho|thuoc tinh|noi dung):/i.test(
          line
        )
    );
}

function buildSourceSummary(question, group, sourceDocuments = []) {
  const combinedContent = sourceDocuments.map((document) => document.content || "").join("\n");
  const usefulLines = extractUsefulContentLines(combinedContent);
  const relevantLines = selectRelevantLines(
    question,
    usefulLines,
    group?.source_type === "product" ? 6 : 8
  );
  const usefulText = normalizeWhitespace((relevantLines.length ? relevantLines : usefulLines).join(" "));
  const descriptiveSentences = usefulText
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)
    .filter((sentence) => sentence.length >= 24)
    .filter((sentence) => !/^(gia hien tai|gia niem yet|gia khuyen mai|gia lien he)/i.test(sentence));

  return {
    title: group.title || "Không rõ tiêu đề",
    url: group.url || "",
    sourceType: group.source_type || "unknown",
    summary: descriptiveSentences.slice(0, 3).join(" "),
    highlights: relevantLines.length ? relevantLines : usefulLines.slice(0, 6)
  };
}

function selectRelevantLines(question = "", usefulLines = [], limit = 6) {
  if (!usefulLines.length) {
    return [];
  }

  const questionKeywords = extractKeywords(question);
  const strongTokens = extractStrongTokens(question);
  const priorityTokens = extractPriorityTokens(question);

  const rankedLines = usefulLines
    .map((line, index) => {
      const normalizedLine = normalizeForSearch(line);
      const faqScore = scoreFaqLine(questionKeywords, strongTokens, line);
      const priorityScore = priorityTokens.reduce((total, token) => {
        if (!token || !normalizedLine.includes(token)) {
          return total;
        }

        return total + 5;
      }, 0);
      const qaBonus = /^hoi:|^tra loi:/i.test(line) ? 4 : 0;

      return {
        line,
        index,
        score: faqScore + priorityScore + qaBonus
      };
    })
    .sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }

      return a.index - b.index;
    });

  const positiveMatches = rankedLines.filter((item) => item.score > 0);
  const selected = (positiveMatches.length ? positiveMatches : rankedLines)
    .slice(0, limit)
    .sort((a, b) => a.index - b.index)
    .map((item) => item.line);

  return Array.from(new Set(selected));
}

function getSourceTypeBonus(intent, sourceType = "") {
  if (intent.isPolicyIntent && sourceType === "page") {
    return 20;
  }

  if (intent.isPriceIntent && sourceType === "product") {
    return 24;
  }

  if (intent.isPriceIntent && sourceType === "post") {
    return -6;
  }

  if (intent.isCategoryIntent && (sourceType === "category" || sourceType === "page")) {
    return 24;
  }

  if (intent.isCategoryIntent && sourceType === "product") {
    return -10;
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
    return `Chào anh/chị, hiện thông tin bên em có chưa đủ để tư vấn thật sát nhu cầu này. Anh/chị có thể để lại yêu cầu cụ thể hơn hoặc liên hệ Hotline/Zalo ${DEFAULT_HOTLINE} để bên em hỗ trợ nhanh và đúng hơn nhé.`;
  }

  const productLink = primarySource.url ? `\nAnh/chị xem thêm tại đây: ${primarySource.url}` : "";

  if (intent.isPriceIntent) {
    return `Chào anh/chị, bên em có sản phẩm "${primarySource.title}" nhé. Hiện mức giá trên website có thể đang để theo hình thức tư vấn trực tiếp, nên anh/chị vui lòng đăng ký báo giá tại https://eurohardware.id.vn/bao-gia hoặc liên hệ Hotline/Zalo ${DEFAULT_HOTLINE} để bên em gửi báo giá chính xác và nhanh hơn.${productLink}`;
  }

  if (primarySource.source_type === "product") {
    return `Chào anh/chị, sản phẩm phù hợp với nhu cầu anh/chị là "${primarySource.title}" nhé.${productLink}`;
  }

  return `Chào anh/chị, bên em gửi anh/chị thông tin tham khảo phù hợp nhất ở đây nhé.${productLink}`;
}

function isProductFocusedQuestion(question, sourceGroups = []) {
  const primarySource = sourceGroups[0];
  const intent = detectIntent(question);

  if (!primarySource || primarySource.source_type !== "product") {
    return false;
  }

  if (intent.isCategoryIntent || intent.isAdviceIntent || intent.isPolicyIntent) {
    return false;
  }

  const strongTokens = extractStrongTokens(question);
  const entityTokens = extractEntityTokens(question);
  const normalizedQuestion = normalizeForSearch(question);
  const normalizedPrimaryTitle = normalizeForSearch(primarySource.title || "");
  const primaryCoverage = computeTitleCoverage(question, primarySource.title || "");
  const secondCoverage = computeTitleCoverage(question, sourceGroups[1]?.title || "");

  if (strongTokens.length) {
    return true;
  }

  if (normalizedQuestion && normalizedQuestion.length >= 12 && normalizedPrimaryTitle.includes(normalizedQuestion)) {
    return true;
  }

  if (primaryCoverage >= 0.72 && primaryCoverage > secondCoverage) {
    return true;
  }

  if (entityTokens.length && sourceGroups.length === 1 && (primarySource.keywordScore || 0) >= 20) {
    return true;
  }

  return sourceGroups.length === 1 && (primarySource.keywordScore || 0) >= 28;
}

function computeTitleCoverage(question = "", title = "") {
  const questionTokens = extractPriorityTokens(question);

  if (!questionTokens.length || !title) {
    return 0;
  }

  const titleTokens = extractNormalizedTokens(title);
  const matchedCount = questionTokens.filter((token) => hasApproxTokenMatch(titleTokens, token)).length;
  return matchedCount / questionTokens.length;
}

function formatProductFocusedReply(sourceGroup, generatedReply = "") {
  if (!sourceGroup || sourceGroup.source_type !== "product") {
    return String(generatedReply || "").trim();
  }

  const cleanedReply = normalizeWhitespace(
    String(generatedReply || "")
      .replace(/\n+/g, " ")
      .replace(/\s{2,}/g, " ")
      .replace(/^cam on ban da gui cau hoi[^\n.:]*[:.]?\s*/i, "")
      .replace(/^duoi day la cau tra loi cua toi[:.]?\s*/i, "")
      .replace(/^cau tra loi[:.]?\s*/i, "")
  );
  const sentences = cleanedReply
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)
    .filter((sentence) => !/^xem them|^tham khao|^lien he|^neu ban can/i.test(sentence))
    .filter((sentence) => !sentence.includes("?"));
  const trimmedReply = sentences.slice(0, 3).join(" ");
  const productLink = sourceGroup.url ? ` Xem chi tiết tại: ${sourceGroup.url}` : "";

  if (!trimmedReply) {
    return normalizeWhitespace(
      `${sourceGroup.title} là sản phẩm anh/chị có thể tham khảo cho nhu cầu này nhé.${productLink}`
    ).trim();
  }

  if (sourceGroup.url && !trimmedReply.includes(sourceGroup.url)) {
    return normalizeWhitespace(`${trimmedReply}${productLink}`).trim();
  }

  return trimmedReply.trim();
}

function buildDeterministicProductReply(sourceGroup, sourceDocuments = []) {
  if (!sourceGroup || sourceGroup.source_type !== "product") {
    return "";
  }

  const combinedContent = sourceDocuments
    .map((document) => document.content || "")
    .join("\n");

  const usefulLines = extractUsefulContentLines(combinedContent);
  const relevantLines = usefulLines
    .filter((line) => line.length >= 20)
    .filter((line) => !/^(sku|thuong hieu|the|tinh trang kho|thuoc tinh):/i.test(line))
    .filter((line) => !/^xem chi tiet tai:/i.test(line));

  const primaryLine =
    relevantLines.find((line) => line.length >= 40 && !/[):]$/.test(line)) ||
    relevantLines.find((line) => line.length >= 24) ||
    "";
  const usefulText = normalizeWhitespace(relevantLines.join(" "));
  const descriptiveSentences = usefulText
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)
    .filter((sentence) => sentence.length >= 24)
    .filter((sentence) => !/^(gia hien tai|gia niem yet|gia khuyen mai|gia lien he)/i.test(sentence));

  const primarySentence = descriptiveSentences[0] || primaryLine;
  const shortDescription =
    primarySentence && primarySentence.length <= 180
      ? primarySentence
      : `${sourceGroup.title} là sản phẩm anh/chị có thể tham khảo cho nhu cầu này nhé.`;
  const productLink = sourceGroup.url ? ` Xem chi tiết tại: ${sourceGroup.url}` : "";

  return normalizeWhitespace(`${shortDescription}${productLink}`).trim();
}

function buildStructuredProductReply(sourceGroup, sourceDocuments = []) {
  if (!sourceGroup || sourceGroup.source_type !== "product") {
    return "";
  }

  const combinedContent = sourceDocuments
    .map((document) => document.content || "")
    .join("\n");
  const usefulLines = extractUsefulContentLines(combinedContent)
    .filter((line) => line.length >= 20)
    .filter((line) => !/^(sku|thuong hieu|the|tinh trang kho|thuoc tinh):/i.test(line))
    .filter((line) => !/^xem chi tiet tai:/i.test(line));
  const usefulText = normalizeWhitespace(usefulLines.join(" "));
  const descriptiveSentences = usefulText
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => normalizeWhitespace(sentence))
    .filter(Boolean)
    .filter((sentence) => sentence.length >= 24)
    .filter((sentence) => !/^(gia hien tai|gia niem yet|gia khuyen mai|gia lien he)/i.test(sentence));
  const pickedSentences = [];

  for (const sentence of descriptiveSentences) {
    if (pickedSentences.includes(sentence)) {
      continue;
    }

    pickedSentences.push(sentence);

    if (pickedSentences.length >= 2) {
      break;
    }
  }

  const productLink = sourceGroup.url ? ` Xem chi tiết tại: ${sourceGroup.url}` : "";

  if (pickedSentences.length) {
    return normalizeWhitespace(`${pickedSentences.join(" ")}${productLink}`).trim();
  }

  const primaryLine =
    usefulLines.find((line) => line.length >= 40 && !/[):]$/.test(line)) ||
    usefulLines.find((line) => line.length >= 24) ||
    "";

  if (primaryLine) {
    return normalizeWhitespace(`${primaryLine}${productLink}`).trim();
  }

  return normalizeWhitespace(
    `${stripProductVariantSuffix(sourceGroup.title || "") || sourceGroup.title} là sản phẩm anh/chị có thể tham khảo cho nhu cầu này nhé.${productLink}`
  ).trim();
}

function stripProductVariantSuffix(title = "") {
  return normalizeWhitespace(String(title || "").replace(/\s*[–-]\s*[A-Z0-9./()]+$/i, ""));
}

function buildContactReply() {
  return `Anh/chị có thể liên hệ Hotline/Zalo ${DEFAULT_HOTLINE} để bên em hỗ trợ nhanh nhé.`;
}

function buildRetailReply() {
  return "Bên em có phục vụ cả khách mua lẻ và khách mua số lượng lớn nhé. Nếu anh/chị mua thường xuyên hoặc cần số lượng nhiều, bên em có thể tư vấn mức giá phù hợp hơn qua Hotline/Zalo 082 820 8218.";
}

function buildShippingReply() {
  return "Bên em phục vụ khách hàng tại Đà Nẵng và nhiều tỉnh thành khác trên toàn quốc nhé. Nếu anh/chị ở xa, bên em vẫn hỗ trợ tư vấn, báo giá và gửi hàng theo hình thức vận chuyển phù hợp.";
}

function buildPriceReply(sourceGroup) {
  const productLink = sourceGroup?.url ? ` Xem sản phẩm tại: ${sourceGroup.url}` : "";
  const productName = sourceGroup?.title ? `${sourceGroup.title}` : "san pham nay";
  return `Anh/chị vui lòng đăng ký báo giá tại https://eurohardware.id.vn/bao-gia hoặc liên hệ Hotline/Zalo ${DEFAULT_HOTLINE} để bên em gửi báo giá cho ${productName} nhé.${productLink}`;
}

function buildQuoteInfoReply() {
  return "Để bên em báo giá chính xác, anh/chị giúp gửi tên sản phẩm, mã sản phẩm nếu có, số lượng cần mua, khu vực giao hàng và thông tin công trình hoặc hệ cửa nếu liên quan nhé. Nếu là phụ kiện kỹ thuật, anh/chị gửi thêm hình ảnh, bản vẽ hoặc mẫu cũ thì bên em tư vấn sát hơn.";
}

function buildUsageReply(sourceGroup, sourceDocuments = []) {
  const productReply = buildStructuredProductReply(sourceGroup, sourceDocuments);

  if (productReply) {
    return `${productReply} Nếu anh/chị cần hướng dẫn sử dụng hoặc lắp đặt chi tiết hơn, bên em hỗ trợ qua Hotline/Zalo ${DEFAULT_HOTLINE} nhé.`;
  }

  return `Hiện website chưa có hướng dẫn sử dụng chi tiết cho sản phẩm này. Anh/chị liên hệ Hotline/Zalo ${DEFAULT_HOTLINE} để bên em hỗ trợ nhanh nhé.`;
}

function buildAdviceReply(sourceGroups = []) {
  const primarySource = sourceGroups[0];

  if (primarySource?.source_type === "product") {
    const productLink = primarySource.url ? ` Xem chi tiết tại: ${primarySource.url}` : "";
    return `${primarySource.title} là phương án anh/chị có thể tham khảo cho nhu cầu này nhé.${productLink}`;
  }

  return `Anh/chị cho bên em biết thêm loại cửa, chất liệu và nhu cầu sử dụng để bên em tư vấn sát hơn nhé. Hoặc anh/chị liên hệ Hotline/Zalo ${DEFAULT_HOTLINE} để bên em hỗ trợ nhanh.`;
}

function buildClarifyingAdviceReply(sourceGroups = []) {
  const primarySource = sourceGroups[0];
  const productHint = primarySource?.url ? ` Anh/chị cũng có thể tham khảo trước tại: ${primarySource.url}` : "";
  return `Để bên em tư vấn sát hơn, anh/chị giúp bên em 1 thông tin quan trọng nhất: loại cửa hoặc hệ cửa anh/chị đang dùng là gì nhé?${productHint}`;
}

function scoreFaqLine(questionKeywords = [], strongTokens = [], line = "") {
  const normalizedLine = normalizeForSearch(line);

  let score = questionKeywords.reduce((total, keyword) => {
    if (!keyword || !normalizedLine.includes(keyword)) {
      return total;
    }

    return total + (keyword.length >= 4 ? 3 : 1);
  }, 0);

  strongTokens.forEach((token) => {
    if (normalizedLine.includes(token)) {
      score += 6;
    }
  });

  return score;
}

function buildFaqStyleReply(question, sourceGroups = [], sourceDocuments = []) {
  const primarySource = sourceGroups[0];

  if (!primarySource || primarySource.source_type === "product") {
    return "";
  }

  const questionKeywords = extractKeywords(question);
  const strongTokens = extractStrongTokens(question);
  const sourceRoot = primarySource.source_root;
  const relevantDocuments = sourceDocuments.filter(
    (document) => getSourceRoot(document.source_id || "") === sourceRoot
  );
  const usefulLines = extractUsefulContentLines(
    relevantDocuments.map((document) => document.content || "").join("\n")
  ).filter((line) => line.length >= 24);

  const rankedLines = usefulLines
    .map((line) => ({
      line,
      score: scoreFaqLine(questionKeywords, strongTokens, line)
    }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map((item) => item.line);

  if (!rankedLines.length) {
    return "";
  }

  const uniqueLines = Array.from(new Set(rankedLines));
  const answer = uniqueLines.slice(0, 2).join(" ");
  const sourceLink = primarySource.url ? ` Xem them tai: ${primarySource.url}` : "";

  return normalizeWhitespace(`${answer}${sourceLink}`).trim();
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

  if (intent.isCategoryIntent) {
    const aCategoryPreferred = a.source_type === "category" || a.source_type === "page" ? 1 : 0;
    const bCategoryPreferred = b.source_type === "category" || b.source_type === "page" ? 1 : 0;

    if (aCategoryPreferred !== bCategoryPreferred) {
      return bCategoryPreferred - aCategoryPreferred;
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

function computeKeywordScore(questionKeywords, priorityTokens, strongTokens, match) {
  const titleText = normalizeForSearch(match.title || "");
  const contentText = normalizeForSearch(match.content || "");
  const urlText = normalizeForSearch(match.url || "");
  const titleTokens = extractNormalizedTokens(match.title || "");
  const contentTokens = extractNormalizedTokens(match.content || "");
  const urlTokens = extractNormalizedTokens(match.url || "");

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

  let matchedPriorityCount = 0;

  priorityTokens.forEach((token) => {
    const inTitle = hasApproxTokenMatch(titleTokens, token);
    const inContent = hasApproxTokenMatch(contentTokens, token);
    const inUrl = hasApproxTokenMatch(urlTokens, token);

    if (inTitle || inContent || inUrl) {
      matchedPriorityCount += 1;
    }

    if (inTitle) {
      score += 10;
    }

    if (inContent) {
      score += 4;
    }

    if (inUrl) {
      score += 6;
    }
  });

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

  if (priorityTokens.length) {
    score += matchedPriorityCount * 8;

    if (matchedPriorityCount === 0) {
      score -= 20;
    } else if (matchedPriorityCount < Math.ceil(priorityTokens.length / 2)) {
      score -= 8;
    }
  }

  return score;
}

function buildEntitySearchTerms(question = "", questionKeywords = [], strongTokens = []) {
  const entityTokens = extractEntityTokens(question);

  if (entityTokens.length) {
    return entityTokens.flatMap((token) => {
      const terms = [token];

      if (token.length >= 5) {
        terms.push(token.slice(0, 4));
      }

      return terms;
    });
  }

  if (strongTokens.length) {
    return Array.from(new Set(strongTokens));
  }

  return Array.from(new Set(questionKeywords)).slice(0, 6);
}

async function searchKeywordMatches(question, questionKeywords, strongTokens) {
  const searchTerms = buildEntitySearchTerms(question, questionKeywords, strongTokens);

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

function applyEntityFirstRetrieval(question, matches = []) {
  const entityTokens = extractEntityTokens(question);

  if (!entityTokens.length) {
    return matches;
  }

  const rankedMatches = matches.map((match) => {
    const matchedEntityTokens = entityTokens.filter((token) => matchTokenAcrossFields(match, token));

    return {
      ...match,
      matchedEntityTokens
    };
  });

  const entityMatched = rankedMatches.filter((match) => match.matchedEntityTokens.length > 0);

  if (!entityMatched.length) {
    return rankedMatches;
  }

  const strongEntityMatched = entityMatched.filter(
    (match) => match.matchedEntityTokens.length >= Math.max(1, Math.ceil(entityTokens.length / 2))
  );

  return strongEntityMatched.length ? strongEntityMatched : entityMatched;
}

function scoreMatches(question, matches = []) {
  const keywords = extractKeywords(question);
  const priorityTokens = extractPriorityTokens(question);
  const strongTokens = extractStrongTokens(question);
  const intent = detectIntent(question);

  return matches.map((match) => {
    const similarity = Number(match.similarity || 0);
    const keywordScore = computeKeywordScore(keywords, priorityTokens, strongTokens, match);
    const sourceTypeBonus = getSourceTypeBonus(intent, match.source_type || "");
    const exactTokenBonus =
      strongTokens.some((token) => normalizeForSearch(match.title || "").includes(token)) ? 20 : 0;
    const contentTokenBonus =
      strongTokens.some((token) => normalizeForSearch(match.content || "").includes(token)) ? 10 : 0;
    const entityTokenBonus = Array.isArray(match.matchedEntityTokens)
      ? match.matchedEntityTokens.length * 18
      : 0;
    const titleTokenPenalty = strongTokens.length && exactTokenBonus === 0 && contentTokenBonus === 0 ? -18 : 0;
    const rerankScore =
      similarity * 100 +
      keywordScore * 4 +
      sourceTypeBonus +
      exactTokenBonus +
      entityTokenBonus +
      contentTokenBonus +
      titleTokenPenalty;

    return {
      ...match,
      similarity,
      keywordScore,
      sourceTypeBonus,
      exactTokenBonus,
      entityTokenBonus,
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

function buildContext(question = "", sourceGroups = [], sourceDocuments = []) {
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
      const sourceSummary = buildSourceSummary(question, group, documents);

      return [
        `[Nguon ${index + 1}]`,
        `Tieu de: ${sourceSummary.title}`,
        `Loai: ${sourceSummary.sourceType}`,
        sourceSummary.url ? `Link: ${sourceSummary.url}` : "",
        `Do lien quan vector: ${group.similarity.toFixed(3)}`,
        `Diem tu khoa: ${group.keywordScore}`,
        "Tom tat nhanh:",
        sanitizeSnippet(sourceSummary.summary, 420),
        sourceSummary.highlights.length ? "Chi tiet uu tien:" : "",
        sourceSummary.highlights.length
          ? sourceSummary.highlights
              .slice(0, group.source_type === "product" ? 4 : 5)
              .map((line) => `- ${sanitizeSnippet(line, 220)}`)
              .join("\n")
          : ""
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n");
}

function buildQuestionGuidance(question = "", sourceGroups = []) {
  const intent = detectIntent(question);
  const primarySource = sourceGroups[0];

  return [
    "[Huong dan cho model]",
    intent.isPriceIntent
      ? "- Day la cau hoi ve bao gia. Neu website khong co gia cu the, huong khach ve trang bao gia va hotline."
      : "- Day la cau hoi can tra loi dua tren noi dung website duoc chon.",
    primarySource?.source_type === "product"
      ? "- Neu nguon chinh la product, uu tien mo ta ngan gon san pham va chen 1 link san pham."
      : "- Neu nguon chinh la page/post/category, uu tien tom tat y chinh gan nhat voi cau hoi.",
    intent.isAdviceIntent
      ? "- Neu cau hoi dang mang tinh tu van, neu thong tin chua du thi hoi lai 1 thong tin quan trong nhat."
      : "- Khong mo rong sang noi dung khong lien quan.",
    `- Cau hoi goc: ${question}`
  ].join("\n");
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
      entityTokenBonus: item.entityTokenBonus,
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
  const intent = detectIntent(cleanQuestion);

  if (intent.isContactIntent) {
    return {
      reply: buildContactReply(),
      sources: [],
      debug: {
        question: cleanQuestion,
        vector_match_count: 0,
        keyword_match_count: 0,
        selected_source_count: 0,
        selected_sources: [],
        answer_mode: "deterministic_contact_early",
        provider: getGroqDebugConfig(),
        timings_ms: {
          total: Date.now() - requestStartedAt
        }
      }
    };
  }

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
  const keywordMatches = await searchKeywordMatches(cleanQuestion, keywords, strongTokens);
  const keywordSearchMs = Date.now() - keywordSearchStartedAt;
  const mergedMatches = mergeMatches(vectorMatches || [], keywordMatches);
  const entityFirstMatches = applyEntityFirstRetrieval(cleanQuestion, mergedMatches);
  const scoredMatches = scoreMatches(cleanQuestion, entityFirstMatches);
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
        `Dạ, hiện bên em chưa tìm thấy thông tin đủ sát trên website. Anh/chị có thể để lại nhu cầu cụ thể hoặc liên hệ Hotline/Zalo ${DEFAULT_HOTLINE} để đội ngũ Euro Hardware hỗ trợ nhanh hơn ạ.`,
      sources: [],
      debug: {
        ...retrievalDebug,
        answer_mode: "no_source_match",
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
  const context = [buildQuestionGuidance(cleanQuestion, sourceGroups), buildContext(cleanQuestion, sourceGroups, sourceDocuments)]
    .filter(Boolean)
    .join("\n\n");
  const productFocused = isProductFocusedQuestion(cleanQuestion, sourceGroups);
  const primarySourceDocuments = sourceDocuments.filter(
    (document) => getSourceRoot(document.source_id || "") === sourceGroups[0]?.source_root
  );
  let reply;

  if (intent.isRetailIntent) {
    timings.answer_generation = 0;
    timings.total = Date.now() - requestStartedAt;

    return {
      reply: buildRetailReply(),
      sources: sourceGroups[0]?.url
        ? [
            {
              title: sourceGroups[0].title,
              url: sourceGroups[0].url,
              source_type: sourceGroups[0].source_type,
              similarity: sourceGroups[0].similarity
            }
          ]
        : [],
      debug: {
        ...retrievalDebug,
        answer_mode: "deterministic_retail",
        provider: getGroqDebugConfig(),
        timings_ms: {
          ...timings
        }
      }
    };
  }

  if (intent.isShippingIntent) {
    timings.answer_generation = 0;
    timings.total = Date.now() - requestStartedAt;

    return {
      reply: buildShippingReply(),
      sources: sourceGroups[0]?.url
        ? [
            {
              title: sourceGroups[0].title,
              url: sourceGroups[0].url,
              source_type: sourceGroups[0].source_type,
              similarity: sourceGroups[0].similarity
            }
          ]
        : [],
      debug: {
        ...retrievalDebug,
        answer_mode: "deterministic_shipping",
        provider: getGroqDebugConfig(),
        timings_ms: {
          ...timings
        }
      }
    };
  }

  if (intent.isQuoteInfoIntent) {
    timings.answer_generation = 0;
    timings.total = Date.now() - requestStartedAt;

    return {
      reply: buildQuoteInfoReply(),
      sources: sourceGroups[0]?.url
        ? [
            {
              title: sourceGroups[0].title,
              url: sourceGroups[0].url,
              source_type: sourceGroups[0].source_type,
              similarity: sourceGroups[0].similarity
            }
          ]
        : [],
      debug: {
        ...retrievalDebug,
        answer_mode: "deterministic_quote_info",
        provider: getGroqDebugConfig(),
        timings_ms: {
          ...timings
        }
      }
    };
  }

  if (intent.isContactIntent && !productFocused && !intent.isPriceIntent) {
    timings.answer_generation = 0;
    timings.total = Date.now() - requestStartedAt;

    return {
      reply: buildContactReply(),
      sources: [],
      debug: {
        ...retrievalDebug,
        answer_mode: "deterministic_contact",
        provider: getGroqDebugConfig(),
        timings_ms: {
          ...timings
        }
      }
    };
  }

  if (intent.isPriceIntent) {
    timings.answer_generation = 0;
    timings.total = Date.now() - requestStartedAt;

    return {
      reply: buildPriceReply(sourceGroups[0]),
      sources: dedupeSources(
        sourceGroups.map((item) => ({
          title: item.title,
          url: item.url,
          source_type: item.source_type,
          similarity: item.similarity
        }))
      ),
      debug: {
        ...retrievalDebug,
        answer_mode: "deterministic_price",
        provider: getGroqDebugConfig(),
        timings_ms: {
          ...timings
        }
      }
    };
  }

  if (productFocused) {
    reply = intent.isUsageIntent
      ? buildUsageReply(sourceGroups[0], primarySourceDocuments)
      : buildStructuredProductReply(sourceGroups[0], primarySourceDocuments);
    timings.answer_generation = 0;
    timings.total = Date.now() - requestStartedAt;

    return {
      reply,
      sources: dedupeSources(
        sourceGroups.map((item) => ({
          title: item.title,
          url: item.url,
          source_type: item.source_type,
          similarity: item.similarity
        }))
      ),
      debug: {
        ...retrievalDebug,
        answer_mode: intent.isUsageIntent ? "deterministic_usage" : "deterministic_product",
        provider: getGroqDebugConfig(),
        timings_ms: {
          ...timings
        }
      }
    };
  }

  if (intent.isAdviceIntent) {
    timings.answer_generation = 0;
    timings.total = Date.now() - requestStartedAt;

    return {
      reply: buildClarifyingAdviceReply(sourceGroups),
      sources: dedupeSources(
        sourceGroups.map((item) => ({
          title: item.title,
          url: item.url,
          source_type: item.source_type,
          similarity: item.similarity
        }))
      ),
      debug: {
        ...retrievalDebug,
        answer_mode: "clarifying_advice",
        provider: getGroqDebugConfig(),
        timings_ms: {
          ...timings
        }
      }
    };
  }

  try {
    const generationStartedAt = Date.now();
    reply = await generateAnswer({
      question: cleanQuestion,
      context,
      concise: productFocused,
      productFocused
    });
    timings.answer_generation = Date.now() - generationStartedAt;

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
          reply = retriedReply;
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
      answer_mode: productFocused ? "product_llm" : "general_llm",
      provider: getGroqDebugConfig(),
      timings_ms: timings
    }
  };
}
