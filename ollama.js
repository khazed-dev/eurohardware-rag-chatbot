const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
const OLLAMA_EMBEDDING_MODEL = process.env.OLLAMA_EMBEDDING_MODEL || "nomic-embed-text";
const OLLAMA_TIMEOUT_MS = Number(process.env.OLLAMA_TIMEOUT_MS || 45000);
const OLLAMA_MAX_RETRIES = Number(process.env.OLLAMA_MAX_RETRIES || 2);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function createEmbedding(text) {
  let lastError;

  for (let attempt = 1; attempt <= OLLAMA_MAX_RETRIES; attempt += 1) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), OLLAMA_TIMEOUT_MS);

    try {
      const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: OLLAMA_EMBEDDING_MODEL,
          prompt: text
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ollama embedding failed: ${errorText}`);
      }

      const data = await response.json();

      if (!data.embedding) {
        throw new Error("Ollama did not return embedding");
      }

      return data.embedding;
    } catch (error) {
      clearTimeout(timeoutId);
      lastError = error;

      if (attempt < OLLAMA_MAX_RETRIES) {
        await sleep(1000 * attempt);
      }
    }
  }

  const reason =
    lastError?.name === "AbortError"
      ? `timeout after ${OLLAMA_TIMEOUT_MS}ms`
      : lastError?.message || "unknown error";

  throw new Error(
    `Ollama embedding request failed for model "${OLLAMA_EMBEDDING_MODEL}" at ${OLLAMA_BASE_URL}: ${reason}`
  );
}
