import dotenv from "dotenv";

dotenv.config();

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
const OLLAMA_CHAT_MODEL =
  process.env.OLLAMA_CHAT_MODEL || process.env.OLLAMA_MODEL || "qwen2.5:7b-instruct";
const OLLAMA_TIMEOUT_MS = Number(process.env.OLLAMA_TIMEOUT_MS || 45000);
const OLLAMA_MAX_RETRIES = Number(process.env.OLLAMA_MAX_RETRIES || 2);
const OLLAMA_TEMPERATURE = Number(process.env.OLLAMA_TEMPERATURE || 0.4);
const OLLAMA_NUM_PREDICT = Number(process.env.OLLAMA_NUM_PREDICT || 550);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function cleanAnswer(text = "") {
  return String(text)
    .replace(/\r/g, "")
    .replace(/\n\s*Tham khảo thêm:\s*[\s\S]*$/i, "")
    .replace(/\n\s*Tham khao them:\s*[\s\S]*$/i, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function buildPrompt({ question, context, concise = false }) {
  const brevityRules = concise
    ? [
        "- Tra loi that ngan gon, toi da 3 cau ngan.",
        "- Khong tao muc 'Tham khao them' hoac danh sach link o cuoi cau tra loi.",
        "- Neu can nhac link, chi nhac toi da 1 link va chen tu nhien trong 1 cau.",
        "- Khong mo dau dai dong, vao thang cau tra loi chinh."
      ]
    : [
        "- Tra loi ngan gon, uu tien toi da 4 cau hoac 3 muc ngan.",
        "- Khong tao muc 'Tham khao them', khong danh so 1. 2. 3., khong liet ke danh sach link o cuoi.",
        "- Neu can chen link, chi nhac toi da 1 link lien quan nhat va chen tu nhien trong cau.",
        "- Uu tien cau tra loi hoan chinh, tranh mo rong khong can thiet."
      ];

  return `
Ban la tro ly tu van cua Euro Door Hardware JSC, chuyen ho tro khach hang ve phu kien cua, khoa thong minh, gioang EPDM, keo silicone va vat tu nganh cua.

Ban chi duoc phep tra loi dua tren DU LIEU WEBSITE duoc cung cap. Khong tu bo sung thong tin ben ngoai.

Nguyen tac tra loi:
- Luon tra loi bang tieng Viet tu nhien, lich su, de doc.
- Tra loi dung trong tam cau hoi, uu tien phong cach tu van ban hang chuyen nghiep.
- Giong van phai tu nhien nhu nhan vien cham soc khach hang hoac tu van vien dang nhan tin voi khach.
- Neu tim thay thong tin phu hop, hay tra loi truc tiep truoc, sau do co the goi y them 1 buoc tiep theo neu can.
- Neu co nhieu lua chon phu hop, hay tom tat ngan gon tung lua chon thay vi liet ke qua dai.
- Neu du lieu khong du de ket luan, noi ro "hien website chua co thong tin day du" va moi khach lien he Hotline/Zalo 079 619 2091.
- Neu khach hoi bao gia, luon gui link dang ky nhan bao gia: https://eurohardware.id.vn/bao-gia
- Neu trong du lieu co link san pham, danh muc hoac bai viet lien quan, chi chen khi that su can thiet va khong tao danh sach link rieng.
- Khong tu dua ra ton kho, chiet khau, thong so ky thuat, chinh sach bao hanh hay thoi gian giao hang neu du lieu khong neu.
- Khong nhac den "context", "nguon", "embedding" hay cac thuong thuat ky thuat noi bo.
- Khong dung cac cau may moc nhu "de em tim tren website", "em da tim thay thong tin", "dua tren du lieu website", "he thong dang kiem tra".
- Uu tien thong tin tu san pham va trang lien quan nhat. Neu cac nguon mau thuan, uu tien nguon co do lien quan cao hon va noi dung cu the hon.
- Neu cau hoi ve san pham, hay tra loi theo cau truc: thong tin chinh, diem noi bat neu co, roi ket thuc gon.
- Neu khach hoi rat ngan hoac mo ho, hay hoi lai 1 cau duy nhat de lam ro nhu cau thay vi tra loi dai.
${brevityRules.join("\n")}

DU LIEU WEBSITE:
${context}

CAU HOI KHACH HANG:
${question}

Hay viet 1 cau tra loi cu the, mem mai, huu ich va bam sat du lieu.
  `.trim();
}

function isTemporaryOllamaError(message = "") {
  return [
    "timeout",
    "timed out",
    "econnrefused",
    "socket hang up",
    "connection reset",
    "model is loading",
    "try again",
    "unavailable",
    "overloaded"
  ].some((keyword) => message.toLowerCase().includes(keyword));
}

async function requestOllama(prompt) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), OLLAMA_TIMEOUT_MS);

  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: OLLAMA_CHAT_MODEL,
        prompt,
        stream: false,
        options: {
          temperature: OLLAMA_TEMPERATURE,
          num_predict: OLLAMA_NUM_PREDICT
        }
      }),
      signal: controller.signal
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama chat failed: ${errorText}`);
    }

    const data = await response.json();

    if (!data.response) {
      throw new Error("Ollama did not return generated text");
    }

    return cleanAnswer(data.response);
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function generateAnswer({ question, context, concise = false }) {
  const prompt = buildPrompt({ question, context, concise });
  let lastError;

  for (let attempt = 1; attempt <= OLLAMA_MAX_RETRIES; attempt += 1) {
    try {
      return await requestOllama(prompt);
    } catch (error) {
      lastError = error;
      const message = error?.message || "";
      const isAbort = error?.name === "AbortError";

      if (attempt >= OLLAMA_MAX_RETRIES || (!isAbort && !isTemporaryOllamaError(message))) {
        break;
      }

      console.warn(
        `Ollama temporary error. Retry ${attempt}/${OLLAMA_MAX_RETRIES} after ${1000 * attempt}ms`
      );
      await sleep(1000 * attempt);
    }
  }

  const reason =
    lastError?.name === "AbortError"
      ? `timeout after ${OLLAMA_TIMEOUT_MS}ms`
      : lastError?.message || "unknown error";

  throw new Error(
    `Ollama answer generation failed for model "${OLLAMA_CHAT_MODEL}" at ${OLLAMA_BASE_URL}: ${reason}`
  );
}
