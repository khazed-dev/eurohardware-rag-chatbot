import crypto from "crypto";

const HTML_ENTITY_MAP = {
  nbsp: " ",
  amp: "&",
  quot: "\"",
  apos: "'",
  lt: "<",
  gt: ">",
  hellip: "...",
  ndash: "-",
  mdash: "-",
  rsquo: "'",
  lsquo: "'",
  rdquo: "\"",
  ldquo: "\"",
  trade: "TM",
  reg: "(R)",
  copy: "(C)",
  deg: " do"
};

function decodeHtmlEntities(text = "") {
  return String(text).replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (match, entity) => {
    const normalizedEntity = entity.toLowerCase();

    if (normalizedEntity.startsWith("#x")) {
      return String.fromCodePoint(Number.parseInt(normalizedEntity.slice(2), 16));
    }

    if (normalizedEntity.startsWith("#")) {
      return String.fromCodePoint(Number.parseInt(normalizedEntity.slice(1), 10));
    }

    return HTML_ENTITY_MAP[normalizedEntity] ?? match;
  });
}

export function sanitizeText(input = "") {
  return String(input)
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, " ")
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/g, " ")
    .replace(/(^|[^\uD800-\uDBFF])[\uDC00-\uDFFF]/g, "$1 ");
}

export function normalizeWhitespace(input = "") {
  return sanitizeText(input)
    .replace(/\r/g, "\n")
    .replace(/\t/g, " ")
    .replace(/\u00a0/g, " ")
    .replace(/[ ]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \n]+$/g, "")
    .replace(/^[ \n]+/g, "")
    .trim();
}

export function stripHtml(input = "") {
  return normalizeWhitespace(
    decodeHtmlEntities(
      String(input)
        .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ")
        .replace(/<br\s*\/?>/gi, "\n")
        .replace(/<\/p>/gi, "\n\n")
        .replace(/<\/div>/gi, "\n")
        .replace(/<\/li>/gi, "\n")
        .replace(/<\/tr>/gi, "\n")
        .replace(/<\/h[1-6]>/gi, "\n\n")
        .replace(/<\/a>/gi, "\n")
        .replace(/<\/table>/gi, "\n\n")
        .replace(/<\/ul>/gi, "\n\n")
        .replace(/<\/ol>/gi, "\n\n")
        .replace(/<\/?[^>]+(>|$)/g, " ")
        .replace(/\[.*?\]/g, " ")
    )
  );
}

function removeCssAndTemplateNoise(text = "") {
  return String(text)
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(
      /\b(color|background(?:-color)?|font-size|font-weight|padding(?:-[a-z]+)?|margin(?:-[a-z]+)?|border(?:-[a-z]+)?|line-height|text-align|display|justify-content|align-items|width|height|max-width|min-width|max-height|min-height|position|top|left|right|bottom|z-index|overflow(?:-[a-z]+)?|white-space|word-break)\s*:\s*[^;}{]+;?/gi,
      " "
    )
    .replace(/[{}]/g, " ")
    .replace(/\s*;\s*/g, "; ")
    .replace(/\s{2,}/g, " ");
}

function isLikelyNoiseSegment(segment = "") {
  const text = normalizeWhitespace(segment);

  if (!text) {
    return true;
  }

  if (text.length <= 2) {
    return true;
  }

  if (/^(color|padding|margin|border|display|justify-content|align-items|font-size|line-height)\s*:/i.test(text)) {
    return true;
  }

  if (/^(javascript|function|\$\(document\)|var\s|let\s|const\s)/i.test(text)) {
    return true;
  }

  if (/^(toggle|chuyen doi|aria|icon-angle-down)$/i.test(normalizeForComparison(text))) {
    return true;
  }

  if (/^(tong don|chiet khau|duoi \d+|tu \d+ den \d+|tren \d+)/i.test(normalizeForComparison(text))) {
    return true;
  }

  if (/^\d+%$/.test(text)) {
    return true;
  }

  if (/^[#.;:{}()[\]0-9a-z-]+$/i.test(text) && !/[a-z]/i.test(text.replace(/[a-z]{3,}/gi, ""))) {
    return true;
  }

  const alphaCount = (text.match(/[a-zA-ZÀ-ỹ]/g) || []).length;
  const symbolCount = (text.match(/[^a-zA-ZÀ-ỹ0-9\s]/g) || []).length;

  if (alphaCount > 0 && symbolCount > alphaCount) {
    return true;
  }

  return false;
}

function normalizeForComparison(text = "") {
  return String(text)
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function cleanFaqQuestion(question = "") {
  return normalizeWhitespace(stripHtml(question))
    .replace(/^[-*]\s*/, "")
    .replace(/\?{2,}/g, "?")
    .trim();
}

function cleanFaqAnswer(answer = "") {
  return cleanExtractedContent(stripHtml(answer))
    .replace(/^[-*]\s*/, "")
    .trim();
}

export function extractStructuredHtmlContent(input = "") {
  const html = String(input || "");

  if (!html.trim()) {
    return "";
  }

  const accordionPattern =
    /<div[^>]*class="accordion-item"[^>]*>[\s\S]*?<span>([\s\S]*?)<\/span>[\s\S]*?<div[^>]*class="accordion-inner"[^>]*>([\s\S]*?)<\/div>\s*<\/div>/gi;
  const faqBlocks = [];
  let match;

  while ((match = accordionPattern.exec(html))) {
    const question = cleanFaqQuestion(match[1]);
    const answer = cleanFaqAnswer(match[2]);

    if (!question || !answer) {
      continue;
    }

    faqBlocks.push(`Hoi: ${question}\nTra loi: ${answer}`);
  }

  if (faqBlocks.length >= 2) {
    const titleMatches = Array.from(
      html.matchAll(/<h[1-6][^>]*>([\s\S]*?)<\/h[1-6]>/gi),
      (headingMatch) => normalizeWhitespace(stripHtml(headingMatch[1]))
    ).filter(Boolean);
    const intro = cleanExtractedContent(
      stripHtml(
        html
          .replace(/<div[^>]*class="accordion"[\s\S]*$/i, " ")
          .replace(/<style[\s\S]*?<\/style>/gi, " ")
      )
    );
    const introLines = intro
      .split(/\n+/)
      .map((line) => normalizeWhitespace(line))
      .filter(Boolean)
      .filter((line) => !/^cau hoi thuong gap$/i.test(normalizeForComparison(line)))
      .filter((line) => !/^nhan bao gia ngay$/i.test(normalizeForComparison(line)))
      .slice(0, 4);

    return normalizeWhitespace(
      [
        titleMatches[0] ? `Tieu de trang: ${titleMatches[0]}` : "",
        ...introLines,
        ...faqBlocks
      ]
        .filter(Boolean)
        .join("\n\n")
    );
  }

  return cleanExtractedContent(stripHtml(html));
}

export function cleanExtractedContent(input = "") {
  const cleaned = normalizeWhitespace(removeCssAndTemplateNoise(input));

  if (!cleaned) {
    return "";
  }

  const segments = cleaned
    .split(/\n+|(?<=[.!?;:])\s+/)
    .map((segment) => normalizeWhitespace(segment))
    .filter(Boolean)
    .filter((segment) => !isLikelyNoiseSegment(segment));

  const dedupedSegments = [];
  const seen = new Set();

  for (const segment of segments) {
    const key = segment.toLowerCase();

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    dedupedSegments.push(segment);
  }

  return normalizeWhitespace(dedupedSegments.join("\n"));
}

export function createHash(text = "") {
  return crypto.createHash("sha256").update(text).digest("hex");
}

export function truncateText(text = "", maxLength = 6000) {
  if (!text) return "";
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength).trim()}...`;
}

export function buildDocumentContent({
  title,
  url,
  sourceType,
  category,
  price,
  content,
  extraSections = []
}) {
  const sections = [
    `Loai du lieu: ${sourceType || ""}`,
    `Tieu de: ${title || ""}`,
    category ? `Danh muc: ${category}` : "",
    price ? `Gia / trang thai gia: ${price}` : "",
    url ? `Link: ${url}` : "",
    ...extraSections.filter(Boolean),
    "",
    "Noi dung:",
    content || ""
  ].filter(Boolean);

  return normalizeWhitespace(sections.join("\n"));
}

export function normalizeUrl(baseUrl, path) {
  const base = baseUrl.replace(/\/$/, "");
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${base}${cleanPath}`;
}

export function chunkText(text = "", maxLength = 1200, overlap = 200) {
  const normalizedText = normalizeWhitespace(text);

  if (!normalizedText) {
    return [];
  }

  if (normalizedText.length <= maxLength) {
    return [normalizedText];
  }

  const blocks = normalizedText
    .split(/\n+/)
    .map((block) => normalizeWhitespace(block))
    .filter(Boolean)
    .flatMap((block) => splitBlockForChunking(block, maxLength));

  if (!blocks.length) {
    return [normalizedText];
  }

  const chunks = [];
  let currentBlocks = [];
  let currentLength = 0;

  for (const block of blocks) {
    const separatorLength = currentBlocks.length ? 1 : 0;
    const projectedLength = currentLength + separatorLength + block.length;

    if (projectedLength <= maxLength || !currentBlocks.length) {
      currentBlocks.push(block);
      currentLength = projectedLength;
      continue;
    }

    chunks.push(currentBlocks.join("\n").trim());
    currentBlocks = buildOverlapBlocks(currentBlocks, overlap);
    currentLength = currentBlocks.join("\n").length;

    if (currentLength) {
      currentBlocks.push(block);
      currentLength = currentBlocks.join("\n").length;
    } else {
      currentBlocks = [block];
      currentLength = block.length;
    }
  }

  if (currentBlocks.length) {
    chunks.push(currentBlocks.join("\n").trim());
  }

  return chunks.filter(Boolean);
}

function splitBlockForChunking(block = "", maxLength = 1200) {
  const normalizedBlock = normalizeWhitespace(block);

  if (!normalizedBlock) {
    return [];
  }

  if (normalizedBlock.length <= maxLength) {
    return [normalizedBlock];
  }

  const sentences = normalizedBlock
    .split(/(?<=[.!?])\s+|(?<=:)\s+/)
    .map((sentence) => normalizeWhitespace(sentence))
    .filter(Boolean);

  if (sentences.length <= 1) {
    return splitLongSentence(normalizedBlock, maxLength);
  }

  const result = [];
  let current = "";

  for (const sentence of sentences) {
    const candidate = current ? `${current} ${sentence}` : sentence;

    if (candidate.length <= maxLength) {
      current = candidate;
      continue;
    }

    if (current) {
      result.push(current);
    }

    if (sentence.length <= maxLength) {
      current = sentence;
    } else {
      result.push(...splitLongSentence(sentence, maxLength));
      current = "";
    }
  }

  if (current) {
    result.push(current);
  }

  return result.filter(Boolean);
}

function splitLongSentence(text = "", maxLength = 1200) {
  const words = normalizeWhitespace(text).split(" ").filter(Boolean);
  const chunks = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;

    if (candidate.length <= maxLength) {
      current = candidate;
      continue;
    }

    if (current) {
      chunks.push(current);
    }

    current = word;
  }

  if (current) {
    chunks.push(current);
  }

  return chunks;
}

function buildOverlapBlocks(blocks = [], overlap = 200) {
  if (!blocks.length || overlap <= 0) {
    return [];
  }

  const selected = [];
  let length = 0;

  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index];
    const separatorLength = selected.length ? 1 : 0;
    const projectedLength = length + separatorLength + block.length;

    if (projectedLength > overlap && selected.length) {
      break;
    }

    selected.unshift(block);
    length = projectedLength;
  }

  return selected;
}

export function buildChunkedDocuments(baseDocument, options = {}) {
  const { maxChunkLength = 1200, overlap = 200 } = options;
  const chunks = chunkText(baseDocument.content, maxChunkLength, overlap);

  return chunks.map((chunk, index) => ({
    ...baseDocument,
    source_id: `${baseDocument.source_id}::chunk::${index + 1}`,
    content: chunk,
    content_hash: createHash(chunk),
    chunk_index: index + 1,
    chunk_count: chunks.length
  }));
}

export function sanitizeSnippet(text = "", maxLength = 320) {
  return truncateText(normalizeWhitespace(text), maxLength);
}
