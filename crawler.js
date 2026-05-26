import dotenv from "dotenv";
import { supabase } from "./supabase.js";
import { createEmbedding } from "./ollama.js";
import {
  stripHtml,
  createHash,
  truncateText,
  buildDocumentContent
} from "./utils.js";

dotenv.config();

const WEBSITE_URL = process.env.WEBSITE_URL || "https://eurohardware.id.vn";

async function fetchJson(url) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Fetch failed: ${url} - ${response.status}`);
  }

  return response.json();
}

async function fetchPaginated(endpoint, perPage = 100, maxPages = 20) {
  const allItems = [];

  for (let page = 1; page <= maxPages; page++) {
    const separator = endpoint.includes("?") ? "&" : "?";
    const url = `${endpoint}${separator}per_page=${perPage}&page=${page}`;

    try {
      const items = await fetchJson(url);

      if (!Array.isArray(items) || items.length === 0) {
        break;
      }

      allItems.push(...items);

      if (items.length < perPage) {
        break;
      }
    } catch (error) {
      console.warn(`Cannot fetch page ${page}: ${url}`);
      console.warn(error.message);
      break;
    }
  }

  return allItems;
}

function mapProduct(product) {
  const title = stripHtml(product.name || "");
  const url = product.permalink || product.link || "";
  const categories = Array.isArray(product.categories)
    ? product.categories.map((c) => c.name).join(", ")
    : "";

  const shortDescription = stripHtml(product.short_description || "");
  const description = stripHtml(product.description || "");

  let price = "Giá liên hệ";
  if (product.prices?.price) {
    price = product.prices.price;
  }

  const content = buildDocumentContent({
    title,
    url,
    sourceType: "product",
    category: categories,
    price,
    content: `${shortDescription}\n${description}`
  });

  return {
    source_id: `product_${product.id}`,
    source_type: "product",
    title,
    url,
    content: truncateText(content),
    content_hash: createHash(content)
  };
}

function mapPost(post) {
  const title = stripHtml(post.title?.rendered || "");
  const url = post.link || "";
  const contentText = stripHtml(post.content?.rendered || post.excerpt?.rendered || "");

  const content = buildDocumentContent({
    title,
    url,
    sourceType: "post",
    category: "",
    price: "",
    content: contentText
  });

  return {
    source_id: `post_${post.id}`,
    source_type: "post",
    title,
    url,
    content: truncateText(content),
    content_hash: createHash(content)
  };
}

function mapPage(page) {
  const title = stripHtml(page.title?.rendered || "");
  const url = page.link || "";
  const contentText = stripHtml(page.content?.rendered || page.excerpt?.rendered || "");

  const content = buildDocumentContent({
    title,
    url,
    sourceType: "page",
    category: "",
    price: "",
    content: contentText
  });

  return {
    source_id: `page_${page.id}`,
    source_type: "page",
    title,
    url,
    content: truncateText(content),
    content_hash: createHash(content)
  };
}

function mapCategory(category) {
  const title = stripHtml(category.name || "");
  const url = category.permalink || category.link || "";
  const description = stripHtml(category.description || "");

  const content = buildDocumentContent({
    title,
    url,
    sourceType: "category",
    category: title,
    price: "",
    content: description || `Danh mục sản phẩm: ${title}`
  });

  return {
    source_id: `category_${category.id}`,
    source_type: "category",
    title,
    url,
    content: truncateText(content),
    content_hash: createHash(content)
  };
}

async function getExistingDocument(sourceId) {
  const { data, error } = await supabase
    .from("website_documents")
    .select("id, content_hash")
    .eq("source_id", sourceId)
    .maybeSingle();

  if (error) {
    throw error;
  }

  return data;
}

async function saveDocument(document) {
  const existing = await getExistingDocument(document.source_id);

  if (existing && existing.content_hash === document.content_hash) {
    return "skipped";
  }

  const embedding = await createEmbedding(document.content);

  const payload = {
    ...document,
    embedding,
    updated_at: new Date().toISOString()
  };

  const { error } = await supabase
    .from("website_documents")
    .upsert(payload, {
      onConflict: "source_id"
    });

  if (error) {
    throw error;
  }

  return existing ? "updated" : "inserted";
}

export async function crawlWebsite() {
  const stats = {
    inserted: 0,
    updated: 0,
    skipped: 0,
    failed: 0
  };

  const productEndpoint = `${WEBSITE_URL}/wp-json/wc/store/v1/products`;
  const postEndpoint = `${WEBSITE_URL}/wp-json/wp/v2/posts`;
  const pageEndpoint = `${WEBSITE_URL}/wp-json/wp/v2/pages`;
  const categoryEndpoint = `${WEBSITE_URL}/wp-json/wc/store/v1/products/categories`;

  console.log("Crawling products...");
  const products = await fetchPaginated(productEndpoint, 100, 20);

  console.log("Crawling posts...");
  const posts = await fetchPaginated(postEndpoint, 100, 20);

  console.log("Crawling pages...");
  const pages = await fetchPaginated(pageEndpoint, 100, 20);

  console.log("Crawling product categories...");
  const categories = await fetchPaginated(categoryEndpoint, 100, 10);

  const documents = [
    ...products.map(mapProduct),
    ...posts.map(mapPost),
    ...pages.map(mapPage),
    ...categories.map(mapCategory)
  ].filter((doc) => doc.title && doc.content);

  console.log(`Total documents: ${documents.length}`);

  for (const document of documents) {
    try {
      const result = await saveDocument(document);
      stats[result] += 1;
      console.log(`${result.toUpperCase()}: ${document.title}`);
    } catch (error) {
      stats.failed += 1;
      console.error(`FAILED: ${document.title}`);
      console.error(error.message);
    }
  }

  return stats;
}