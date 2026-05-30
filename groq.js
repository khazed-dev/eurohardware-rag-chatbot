import dotenv from "dotenv";

dotenv.config();

const GROQ_API_BASE_URL =
  process.env.GROQ_API_BASE_URL || process.env.GROQ_BASE_URL || "https://api.groq.com/openai/v1";
const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const GROQ_CHAT_MODEL =
  process.env.GROQ_CHAT_MODEL || process.env.GROQ_MODEL || "qwen/qwen3-32b";
const GROQ_TIMEOUT_MS = Number(process.env.GROQ_TIMEOUT_MS || 20000);
const GROQ_MAX_RETRIES = Number(process.env.GROQ_MAX_RETRIES || 1);
const GROQ_TEMPERATURE = Number(process.env.GROQ_TEMPERATURE || process.env.TEMPERATURE || 0.2);
const GROQ_MAX_TOKENS = Number(process.env.GROQ_MAX_TOKENS || process.env.MAX_TOKENS || 550);
const CHAT_DEBUG = process.env.CHAT_DEBUG === "true";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function cleanAnswer(text = "") {
  return String(text)
    .replace(/\r/g, "")
    .replace(/<think>[\s\S]*?<\/think>\s*/gi, "")
    .replace(/^(chao|xin chao|cam on ban da lien he)[^\n]*\n+/i, "")
    .replace(/^cam on ban da gui cau hoi[^\n.:]*[:.]?\s*/i, "")
    .replace(/^duoi day la cau tra loi cua toi[:.]?\s*/i, "")
    .replace(/^cam on ban da cung cap thong tin[^\n]*\n+/i, "")
    .replace(/^toi se tra loi cau hoi cua ban[^\n]*\n+/i, "")
    .replace(/^\*\*cau tra loi:\*\*\s*/i, "")
    .replace(/^cau tra loi[:.]?\s*/i, "")
    .replace(/\n\s*tham\s*khao\s*them\s*:?\s*[\s\S]*$/i, "")
    .replace(/\n\s*\*\*thong tin lien quan:\*\*[\s\S]*$/i, "")
    .replace(/\n\s*\*\*thong tin chi tiet:\*\*\s*/i, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function buildPrompt({ question, context, concise = false }) {
  const brevityRules = concise
    ? [
        "- Tra loi that ngan gon, toi da 3 cau ngan.",
        "- Khong tao muc 'Tham khao them' hoac danh sach link o cuoi cau tra loi.",
        "- Neu can nhac link, chi nhac toi da 1 link va chen tu nhien trong 1 cau.",
        "- Khong mo dau dai dong, vao thang cau tra loi chinh.",
        "- Khong chao hoi, khong cam on, khong mo dau kieu tong dai cham soc khach hang.",
        "- Khong lap lai ten san pham nhieu lan, khong tao cac tieu de nhu 'Cau tra loi', 'Thong tin chi tiet', 'Thong tin lien quan'."
      ]
    : [
        "- Tra loi ngan gon, uu tien toi da 4 cau hoac 3 muc ngan.",
        "- Khong tao muc 'Tham khao them', khong danh so 1. 2. 3., khong liet ke danh sach link o cuoi.",
        "- Neu can chen link, chi nhac toi da 1 link lien quan nhat va chen tu nhien trong cau.",
        "- Uu tien cau tra loi hoan chinh, tranh mo rong khong can thiet.",
        "- Khong chao hoi, khong cam on, khong tu gioi thieu, khong noi 'toi se tra loi' hay 'dua tren thong tin da cung cap'.",
        "- Neu cau hoi dang nham den 1 san pham cu the, chi tra loi ve dung san pham do; khong mo rong sang gioi thieu chung ve cong ty hay cac san pham khac.",
        "- Khong tao cac tieu de nhu 'Cau tra loi', 'Thong tin chi tiet', 'Thong tin lien quan'."
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
- Neu du lieu khong du de ket luan, noi ro "hien website chua co thong tin day du" va moi khach lien he Hotline/Zalo 082 820 8218.
- Neu khach hoi bao gia, luon gui link dang ky nhan bao gia: https://eurohardware.id.vn/bao-gia
- Neu trong du lieu co link san pham, danh muc hoac bai viet lien quan, chi chen khi that su can thiet va khong tao danh sach link rieng.
- Khong tu dua ra ton kho, chiet khau, thong so ky thuat, chinh sach bao hanh hay thoi gian giao hang neu du lieu khong neu.
- Khong nhac den "context", "nguon", "embedding" hay cac thuat ngu ky thuat noi bo.
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

function isTemporaryGroqError(message = "") {
  return [
    "timeout",
    "timed out",
    "429",
    "rate limit",
    "too many requests",
    "connection reset",
    "socket hang up",
    "try again",
    "unavailable",
    "overloaded",
    "503"
  ].some((keyword) => message.toLowerCase().includes(keyword));
}

async function requestGroq(prompt) {
  if (!GROQ_API_KEY) {
    throw new Error("Groq API key is missing. Please set GROQ_API_KEY in the environment.");
  }

  if (CHAT_DEBUG) {
    console.log("Groq request config:", JSON.stringify({
      model: GROQ_CHAT_MODEL,
      base_url: GROQ_API_BASE_URL,
      temperature: GROQ_TEMPERATURE,
      max_tokens: GROQ_MAX_TOKENS
    }));
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), GROQ_TIMEOUT_MS);

  try {
    const response = await fetch(`${GROQ_API_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${GROQ_API_KEY}`
      },
      body: JSON.stringify({
        model: GROQ_CHAT_MODEL,
        messages: [
          {
            role: "system",
            content:
              "Ban la nhan vien tu van san pham cua Euro Hardware. Tra loi ngan gon, dung trong tam, khong van mau CSKH, khong cam on, khong liet ke link hay muc tham khao o cuoi."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: GROQ_TEMPERATURE,
        max_tokens: GROQ_MAX_TOKENS,
        stream: false
      }),
      signal: controller.signal
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Groq chat failed: ${errorText}`);
    }

    const data = await response.json();
    const answer = data?.choices?.[0]?.message?.content;

    if (!answer) {
      throw new Error("Groq did not return generated text");
    }

    return cleanAnswer(answer);
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function generateAnswer({ question, context, concise = false, productFocused = false }) {
  const prompt = buildPrompt({ question, context, concise });
  let lastError;

  for (let attempt = 1; attempt <= GROQ_MAX_RETRIES; attempt += 1) {
    try {
      return await requestGroq(
        productFocused
          ? `${prompt}\n\nYEU CAU BO SUNG:\n- Vi day la cau hoi ve 1 san pham cu the, tra loi ngan gon nhu nhan vien cham soc khach hang dang nhan tin.\n- Uu tien 2 cau, toi da 3 cau ngan.\n- Neu co link san pham trong du lieu, hay chen 1 link san pham o cuoi cau tra loi de khach xem them.\n- Khong liet ke thong so theo bullet.\n- Khong them hotline neu khach chua hoi bao gia hay lien he.`
          : prompt
      );
    } catch (error) {
      lastError = error;
      const message = error?.message || "";
      const isAbort = error?.name === "AbortError";

      if (attempt >= GROQ_MAX_RETRIES || (!isAbort && !isTemporaryGroqError(message))) {
        break;
      }

      console.warn(
        `Groq temporary error. Retry ${attempt}/${GROQ_MAX_RETRIES} after ${1000 * attempt}ms`
      );
      await sleep(1000 * attempt);
    }
  }

  const reason =
    lastError?.name === "AbortError"
      ? `timeout after ${GROQ_TIMEOUT_MS}ms`
      : lastError?.message || "unknown error";

  throw new Error(
    `Groq answer generation failed for model "${GROQ_CHAT_MODEL}" at ${GROQ_API_BASE_URL}: ${reason}`
  );
}

export function getGroqDebugConfig() {
  return {
    provider: "groq",
    model: GROQ_CHAT_MODEL,
    base_url: GROQ_API_BASE_URL,
    temperature: GROQ_TEMPERATURE,
    max_tokens: GROQ_MAX_TOKENS
  };
}
