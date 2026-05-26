import { supabase } from "./supabase.js";
import { createEmbedding } from "./ollama.js";
import { generateAnswer } from "./gemini.js";

function buildContext(matches = []) {
  if (!matches.length) {
    return "";
  }

  return matches
    .map((item, index) => {
      return `
[Nguồn ${index + 1}]
Tiêu đề: ${item.title}
Loại: ${item.source_type}
Link: ${item.url}
Độ liên quan: ${item.similarity}

Nội dung:
${item.content}
`;
    })
    .join("\n\n");
}

export async function askRag(question) {
  if (!question || !question.trim()) {
    throw new Error("Question is required");
  }

  const queryEmbedding = await createEmbedding(question);

  const { data: matches, error } = await supabase.rpc("match_website_documents", {
    query_embedding: queryEmbedding,
    match_count: 5
  });

  if (error) {
    throw error;
  }

  if (!matches || matches.length === 0) {
    return {
      reply:
        "Dạ, hiện em chưa tìm thấy thông tin phù hợp trên website. Anh/chị vui lòng liên hệ Hotline/Zalo: 079 619 2091 để nhân viên hỗ trợ chính xác hơn ạ.",
      sources: []
    };
  }

  const context = buildContext(matches);
  const reply = await generateAnswer({
    question,
    context
  });

  const sources = matches.map((item) => ({
    title: item.title,
    url: item.url,
    source_type: item.source_type,
    similarity: item.similarity
  }));

  return {
    reply,
    sources
  };
}