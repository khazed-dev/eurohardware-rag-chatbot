import crypto from "crypto";

export function stripHtml(input = "") {
  return String(input)
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<\/?[^>]+(>|$)/g, " ")
    .replace(/\[.*?\]/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, "\"")
    .replace(/&#039;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim();
}

export function createHash(text = "") {
  return crypto.createHash("sha256").update(text).digest("hex");
}

export function truncateText(text = "", maxLength = 6000) {
  if (!text) return "";
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength);
}

export function buildDocumentContent({
  title,
  url,
  sourceType,
  category,
  price,
  content
}) {
  return `
Loại dữ liệu: ${sourceType || ""}
Tiêu đề: ${title || ""}
Danh mục: ${category || ""}
Giá / trạng thái giá: ${price || ""}
Link: ${url || ""}

Nội dung:
${content || ""}
  `
    .replace(/\s+/g, " ")
    .trim();
}

export function normalizeUrl(baseUrl, path) {
  const base = baseUrl.replace(/\/$/, "");
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${base}${cleanPath}`;
}