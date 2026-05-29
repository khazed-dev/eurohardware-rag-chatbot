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
        .replace(/<\/?[^>]+(>|$)/g, " ")
        .replace(/\[.*?\]/g, " ")
    )
  );
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

  const chunks = [];
  let start = 0;

  while (start < normalizedText.length) {
    let end = Math.min(start + maxLength, normalizedText.length);

    if (end < normalizedText.length) {
      const lineBreak = normalizedText.lastIndexOf("\n", end);
      const sentenceBreak = Math.max(
        normalizedText.lastIndexOf(". ", end),
        normalizedText.lastIndexOf("; ", end),
        normalizedText.lastIndexOf(": ", end)
      );
      const safeBreak = Math.max(lineBreak, sentenceBreak);

      if (safeBreak > start + Math.floor(maxLength * 0.6)) {
        end = safeBreak + 1;
      }
    }

    chunks.push(normalizedText.slice(start, end).trim());

    if (end >= normalizedText.length) {
      break;
    }

    start = Math.max(end - overlap, start + 1);
  }

  return chunks.filter(Boolean);
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
