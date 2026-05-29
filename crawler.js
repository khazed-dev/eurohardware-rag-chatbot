import dotenv from "dotenv";
import { supabase } from "./supabase.js";
import { createEmbedding } from "./ollama.js";
import {
  stripHtml,
  createHash,
  truncateText,
  buildDocumentContent,
  buildChunkedDocuments,
  normalizeWhitespace,
  sanitizeText,
  cleanExtractedContent
} from "./utils.js";

dotenv.config();

const WEBSITE_URL = process.env.WEBSITE_URL || "https://eurohardware.id.vn";
const CHUNK_MAX_LENGTH = Number(process.env.RAG_CHUNK_MAX_LENGTH || 1200);
const CHUNK_OVERLAP = Number(process.env.RAG_CHUNK_OVERLAP || 200);
const FETCH_CONCURRENCY = Number(process.env.CRAWL_FETCH_CONCURRENCY || 4);
const EMBEDDING_CONCURRENCY = Number(process.env.CRAWL_EMBEDDING_CONCURRENCY || 3);
const SUPABASE_PAGE_SIZE = Number(process.env.CRAWL_SUPABASE_PAGE_SIZE || 1000);
const PROGRESS_LOG_EVERY = Number(process.env.CRAWL_PROGRESS_EVERY || 10);

function buildPagedUrl(endpoint, page, perPage) {
  const separator = endpoint.includes("?") ? "&" : "?";
  return `${endpoint}${separator}per_page=${perPage}&page=${page}`;
}

async function fetchPage(endpoint, page, perPage) {
  const url = buildPagedUrl(endpoint, page, perPage);
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Fetch failed: ${url} - ${response.status}`);
  }

  const items = await response.json();
  const totalPages = Number(response.headers.get("x-wp-totalpages") || 1);
  return {
    items: Array.isArray(items) ? items : [],
    totalPages
  };
}

async function runInBatches(items, concurrency, worker) {
  const normalizedConcurrency = Math.max(1, concurrency);

  for (let index = 0; index < items.length; index += normalizedConcurrency) {
    const batch = items.slice(index, index + normalizedConcurrency);
    await Promise.all(batch.map(worker));
  }
}

function createProgressTracker(label, total, every = PROGRESS_LOG_EVERY) {
  const startedAt = Date.now();
  let completed = 0;

  if (total === 0) {
    console.log(`${label}: nothing to process`);
  } else {
    console.log(`${label}: 0/${total}`);
  }

  return {
    tick(extra = "") {
      completed += 1;

      if (
        completed === total ||
        completed === 1 ||
        completed % Math.max(1, every) === 0
      ) {
        const elapsedSeconds = ((Date.now() - startedAt) / 1000).toFixed(1);
        const suffix = extra ? ` | ${extra}` : "";
        console.log(`${label}: ${completed}/${total} | elapsed=${elapsedSeconds}s${suffix}`);
      }
    }
  };
}

async function fetchPaginated(endpoint, perPage = 100, fallbackMaxPages = 20) {
  const firstPage = await fetchPage(endpoint, 1, perPage);
  const allItems = [...firstPage.items];
  const totalPages = Math.min(firstPage.totalPages || 1, fallbackMaxPages);

  if (totalPages <= 1) {
    return allItems;
  }

  const remainingPages = Array.from({ length: totalPages - 1 }, (_, index) => index + 2);

  await runInBatches(remainingPages, FETCH_CONCURRENCY, async (page) => {
    try {
      const result = await fetchPage(endpoint, page, perPage);
      allItems.push(...result.items);
    } catch (error) {
      console.warn(`Cannot fetch page ${page}: ${buildPagedUrl(endpoint, page, perPage)}`);
      console.warn(error.message);
    }
  });

  return allItems;
}

function joinValues(values = []) {
  return values
    .map((value) => stripHtml(value))
    .filter(Boolean)
    .join(", ");
}

function formatPrice(product) {
  const price = product.prices?.price;
  const regularPrice = product.prices?.regular_price;
  const salePrice = product.prices?.sale_price;
  const currencyCode = product.prices?.currency_code || "VND";

  if (price) {
    const parts = [`Gia hien tai: ${price} ${currencyCode}`];

    if (regularPrice && regularPrice !== price) {
      parts.push(`Gia niem yet: ${regularPrice} ${currencyCode}`);
    }

    if (salePrice && salePrice !== price) {
      parts.push(`Gia khuyen mai: ${salePrice} ${currencyCode}`);
    }

    return parts.join(" | ");
  }

  return "Gia lien he";
}

function buildProductContent(product, title, url, categories) {
  const shortDescription = cleanExtractedContent(stripHtml(product.short_description || ""));
  const description = cleanExtractedContent(stripHtml(product.description || ""));
  const brand = joinValues(product.brands?.map((brandItem) => brandItem.name) || []);
  const tags = joinValues(product.tags?.map((tag) => tag.name) || []);
  const attributes = Array.isArray(product.attributes)
    ? product.attributes
        .map((attribute) => {
          const attributeName = stripHtml(attribute.name || "");
          const terms = joinValues(attribute.terms?.map((term) => term.name) || []);
          return attributeName && terms ? `${attributeName}: ${terms}` : "";
        })
        .filter(Boolean)
    : [];
  const sku = stripHtml(product.sku || "");
  const stockStatus = stripHtml(product.stock_status || "");
  const price = formatPrice(product);

  const content = buildDocumentContent({
    title,
    url,
    sourceType: "product",
    category: categories,
    price,
    content: [shortDescription, description].filter(Boolean).join("\n\n"),
    extraSections: [
      sku ? `SKU: ${sku}` : "",
      brand ? `Thuong hieu: ${brand}` : "",
      tags ? `The: ${tags}` : "",
      stockStatus ? `Tinh trang kho: ${stockStatus}` : "",
      attributes.length ? `Thuoc tinh: ${attributes.join(" | ")}` : ""
    ]
  });

  return normalizeWhitespace(content);
}

function mapProduct(product) {
  const title = stripHtml(product.name || "");
  const url = product.permalink || product.link || "";
  const categories = joinValues(product.categories?.map((category) => category.name) || []);
  const content = buildProductContent(product, title, url, categories);

  return {
    source_id: `product_${product.id}`,
    source_type: "product",
    title,
    url,
    content: truncateText(content, 12000)
  };
}

function mapPost(post) {
  const title = stripHtml(post.title?.rendered || "");
  const url = post.link || "";
  const excerpt = cleanExtractedContent(stripHtml(post.excerpt?.rendered || ""));
  const contentText = cleanExtractedContent(stripHtml(post.content?.rendered || excerpt));

  const content = buildDocumentContent({
    title,
    url,
    sourceType: "post",
    category: "",
    price: "",
    content: contentText,
    extraSections: [excerpt ? `Tom tat: ${excerpt}` : ""]
  });

  return {
    source_id: `post_${post.id}`,
    source_type: "post",
    title,
    url,
    content: truncateText(content, 12000)
  };
}

function mapPage(page) {
  const title = stripHtml(page.title?.rendered || "");
  const url = page.link || "";
  const excerpt = cleanExtractedContent(stripHtml(page.excerpt?.rendered || ""));
  const contentText = cleanExtractedContent(stripHtml(page.content?.rendered || excerpt));

  const content = buildDocumentContent({
    title,
    url,
    sourceType: "page",
    category: "",
    price: "",
    content: contentText,
    extraSections: [excerpt ? `Tom tat: ${excerpt}` : ""]
  });

  return {
    source_id: `page_${page.id}`,
    source_type: "page",
    title,
    url,
    content: truncateText(content, 12000)
  };
}

function mapCategory(category) {
  const title = stripHtml(category.name || "");
  const url = category.permalink || category.link || "";
  const description = cleanExtractedContent(stripHtml(category.description || ""));

  const content = buildDocumentContent({
    title,
    url,
    sourceType: "category",
    category: title,
    price: "",
    content: description || `Danh muc san pham: ${title}`
  });

  return {
    source_id: `category_${category.id}`,
    source_type: "category",
    title,
    url,
    content: truncateText(content, 8000)
  };
}

function buildDocumentsToSave(baseDocument) {
  const chunkedDocuments = buildChunkedDocuments(baseDocument, {
    maxChunkLength: CHUNK_MAX_LENGTH,
    overlap: CHUNK_OVERLAP
  });

  if (chunkedDocuments.length) {
    return chunkedDocuments;
  }

  return [
    {
      ...baseDocument,
      content_hash: createHash(baseDocument.content)
    }
  ];
}

function groupBySourceRoot(documents) {
  return documents.reduce((map, document) => {
    const sourceId = document.source_id.includes("::chunk::")
      ? document.source_id.split("::chunk::")[0]
      : document.source_id;

    if (!map.has(sourceId)) {
      map.set(sourceId, []);
    }

    map.get(sourceId).push(document);
    return map;
  }, new Map());
}

async function loadAllExistingDocuments() {
  const allDocuments = [];
  let from = 0;

  while (true) {
    const to = from + SUPABASE_PAGE_SIZE - 1;
    const { data, error } = await supabase
      .from("website_documents")
      .select("source_id, content_hash")
      .order("source_id", { ascending: true })
      .range(from, to);

    if (error) {
      throw error;
    }

    if (!data || data.length === 0) {
      break;
    }

    allDocuments.push(...data);

    if (data.length < SUPABASE_PAGE_SIZE) {
      break;
    }

    from += SUPABASE_PAGE_SIZE;
  }

  return groupBySourceRoot(allDocuments);
}

function planDocumentChanges(sourceDocuments, existingDocumentsBySourceRoot) {
  const upserts = [];
  const deletes = [];
  const nextSourceRoots = new Set();
  const stats = {
    skipped: 0,
    upsertSources: 0,
    unchangedSources: 0,
    changedChunks: 0,
    totalChunks: 0,
    insertedChunks: 0,
    updatedChunks: 0
  };

  for (const baseDocument of sourceDocuments) {
    const nextDocuments = buildDocumentsToSave(baseDocument);
    const existingDocuments = existingDocumentsBySourceRoot.get(baseDocument.source_id) || [];
    const existingBySourceId = new Map(
      existingDocuments.map((document) => [document.source_id, document.content_hash])
    );
    const nextSourceIds = new Set(nextDocuments.map((document) => document.source_id));

    nextSourceRoots.add(baseDocument.source_id);
    stats.totalChunks += nextDocuments.length;

    const changedDocuments = nextDocuments.filter((document) => {
      return existingBySourceId.get(document.source_id) !== document.content_hash;
    });

    const staleSourceIds = existingDocuments
      .map((document) => document.source_id)
      .filter((sourceId) => !nextSourceIds.has(sourceId));

    if (!changedDocuments.length && !staleSourceIds.length) {
      stats.skipped += nextDocuments.length;
      stats.unchangedSources += 1;
      continue;
    }

    upserts.push(...changedDocuments);
    deletes.push(...staleSourceIds);
    stats.upsertSources += 1;
    stats.changedChunks += changedDocuments.length;
    stats.skipped += nextDocuments.length - changedDocuments.length;
    stats.insertedChunks += changedDocuments.filter((document) => !existingBySourceId.has(document.source_id)).length;
    stats.updatedChunks += changedDocuments.filter((document) => existingBySourceId.has(document.source_id)).length;
  }

  for (const [sourceRoot, existingDocuments] of existingDocumentsBySourceRoot.entries()) {
    if (!nextSourceRoots.has(sourceRoot)) {
      deletes.push(...existingDocuments.map((document) => document.source_id));
    }
  }

  return {
    upserts,
    deletes,
    stats
  };
}

async function upsertDocuments(documents, stats) {
  const now = new Date().toISOString();
  const progress = createProgressTracker("Embedding and upserting chunks", documents.length);

  await runInBatches(documents, EMBEDDING_CONCURRENCY, async (document) => {
    const safeDocument = {
      ...document,
      title: sanitizeText(document.title || ""),
      url: sanitizeText(document.url || ""),
      content: sanitizeText(document.content || "")
    };

    try {
      const embedding = await createEmbedding(safeDocument.content);
      const payload = {
        source_id: safeDocument.source_id,
        source_type: safeDocument.source_type,
        title: safeDocument.title,
        url: safeDocument.url,
        content: safeDocument.content,
        content_hash: safeDocument.content_hash,
        embedding,
        updated_at: now
      };

      const { error } = await supabase
        .from("website_documents")
        .upsert(payload, { onConflict: "source_id" });

      if (error) {
        throw error;
      }

      progress.tick(safeDocument.title || safeDocument.source_id);
    } catch (error) {
      console.error("Upsert chunk failed:", {
        source_id: safeDocument.source_id,
        title: safeDocument.title,
        content_length: safeDocument.content.length,
        error: error.message || error
      });
      throw error;
    }
  });
}

async function deleteDocumentsInChunks(sourceIds, stats) {
  if (!sourceIds.length) {
    console.log("Deleting stale chunks: nothing to delete");
    return;
  }

  const chunkSize = 200;
  const progress = createProgressTracker("Deleting stale chunks", sourceIds.length, 25);

  for (let index = 0; index < sourceIds.length; index += chunkSize) {
    const batch = sourceIds.slice(index, index + chunkSize);
    const { error } = await supabase
      .from("website_documents")
      .delete()
      .in("source_id", batch);

    if (error) {
      throw error;
    }

    stats.deleted += batch.length;
    batch.forEach(() => progress.tick());
  }
}

export async function crawlWebsite() {
  const stats = {
    inserted: 0,
    updated: 0,
    skipped: 0,
    deleted: 0,
    failed: 0,
    sources: 0,
    chunks: 0,
    unchangedSources: 0,
    changedChunks: 0
  };

  const productEndpoint = `${WEBSITE_URL}/wp-json/wc/store/v1/products`;
  const postEndpoint = `${WEBSITE_URL}/wp-json/wp/v2/posts`;
  const pageEndpoint = `${WEBSITE_URL}/wp-json/wp/v2/pages`;
  const categoryEndpoint = `${WEBSITE_URL}/wp-json/wc/store/v1/products/categories`;

  console.log("Loading existing document metadata from Supabase...");
  const existingDocumentsBySourceRoot = await loadAllExistingDocuments();

  console.log("Crawling products...");
  const products = await fetchPaginated(productEndpoint, 100, 20);

  console.log("Crawling posts...");
  const posts = await fetchPaginated(postEndpoint, 100, 20);

  console.log("Crawling pages...");
  const pages = await fetchPaginated(pageEndpoint, 100, 20);

  console.log("Crawling product categories...");
  const categories = await fetchPaginated(categoryEndpoint, 100, 10);

  const sourceDocuments = [
    ...products.map(mapProduct),
    ...posts.map(mapPost),
    ...pages.map(mapPage),
    ...categories.map(mapCategory)
  ].filter((document) => document.title && document.content);

  stats.sources = sourceDocuments.length;

  console.log(`Total source documents: ${sourceDocuments.length}`);

  const plan = planDocumentChanges(sourceDocuments, existingDocumentsBySourceRoot);
  stats.skipped = plan.stats.skipped;
  stats.chunks = plan.stats.totalChunks;
  stats.unchangedSources = plan.stats.unchangedSources;
  stats.changedChunks = plan.stats.changedChunks;
  stats.inserted = plan.stats.insertedChunks;
  stats.updated = plan.stats.updatedChunks;

  console.log(
    `Plan: changed_sources=${plan.stats.upsertSources}, unchanged_sources=${plan.stats.unchangedSources}, changed_chunks=${plan.upserts.length}, delete_chunks=${plan.deletes.length}`
  );

  try {
    await upsertDocuments(plan.upserts, stats);
    await deleteDocumentsInChunks(plan.deletes, stats);
  } catch (error) {
    stats.failed += 1;
    throw error;
  }

  return stats;
}
