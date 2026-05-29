import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const apiKey = process.env.GEMINI_API_KEY;
const modelName = process.env.GEMINI_MODEL || "gemini-2.5-flash";

if (!apiKey) {
  console.warn("Missing GEMINI_API_KEY in .env. Chat answer generation will fail.");
}

const genAI = apiKey ? new GoogleGenerativeAI(apiKey) : null;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function cleanAnswer(text = "") {
  return String(text)
    .replace(/\r/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function buildPrompt({ question, context }) {
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
- Neu trong du lieu co link san pham, danh muc hoac bai viet lien quan, hay chen link do vao cau tra loi.
- Khong tu dua ra ton kho, chiet khau, thong so ky thuat, chinh sach bao hanh hay thoi gian giao hang neu du lieu khong neu.
- Khong nhac den "context", "nguon", "embedding" hay cac thuong thuat ky thuat noi bo.
- Khong dung cac cau may moc nhu "de em tim tren website", "em da tim thay thong tin", "dua tren du lieu website", "he thong dang kiem tra".
- Uu tien thong tin tu san pham va trang lien quan nhat. Neu cac nguon mau thuan, uu tien nguon co do lien quan cao hon va noi dung cu the hon.
- Neu cau hoi ve san pham, hay tra loi theo cau truc: thong tin chinh, diem noi bat neu co, link tham khao.
- Neu khach hoi rat ngan hoac mo ho, hay hoi lai 1 cau duy nhat de lam ro nhu cau thay vi tra loi dai.

DU LIEU WEBSITE:
${context}

CAU HOI KHACH HANG:
${question}

Hay viet 1 cau tra loi cu the, mem mai, huu ich va bam sat du lieu.
  `.trim();
}

export async function generateAnswer({ question, context }) {
  if (!genAI) {
    throw new Error("Gemini API key is missing");
  }

  const model = genAI.getGenerativeModel({
    model: modelName,
    generationConfig: {
      temperature: 0.4,
      topP: 0.9,
      topK: 32,
      maxOutputTokens: 700
    }
  });

  const prompt = buildPrompt({ question, context });
  const maxRetries = 3;
  const delays = [2000, 5000, 10000];

  for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
    try {
      const result = await model.generateContent(prompt);
      const response = await result.response;
      return cleanAnswer(response.text());
    } catch (error) {
      const message = error.message || "";
      const isTemporaryError =
        message.includes("503") ||
        message.includes("Service Unavailable") ||
        message.includes("high demand") ||
        message.includes("429") ||
        message.includes("Too Many Requests");

      if (!isTemporaryError || attempt === maxRetries) {
        throw error;
      }

      console.warn(
        `Gemini temporary error. Retry ${attempt}/${maxRetries} after ${delays[attempt - 1]}ms`
      );
      await sleep(delays[attempt - 1]);
    }
  }

  throw new Error("Gemini failed after retries");
}
