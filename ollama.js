const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
const OLLAMA_EMBEDDING_MODEL = process.env.OLLAMA_EMBEDDING_MODEL || "nomic-embed-text";

export async function createEmbedding(text) {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: OLLAMA_EMBEDDING_MODEL,
      prompt: text
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Ollama embedding failed: ${errorText}`);
  }

  const data = await response.json();

  if (!data.embedding) {
    throw new Error("Ollama did not return embedding");
  }

  return data.embedding;
}