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

function buildPrompt({ question, context }) {
  return `
Bạn là trợ lý tư vấn của Euro Door Hardware JSC, chuyên hỗ trợ khách hàng về phụ kiện cửa, khóa thông minh, gioăng EPDM, keo silicone và vật tư ngành cửa.

Hãy trả lời khách hàng dựa trên DỮ LIỆU WEBSITE được cung cấp bên dưới.

Nguyên tắc trả lời:
- Luôn trả lời bằng tiếng Việt.
- Trả lời ngắn gọn, rõ ràng, lịch sự, phù hợp tư vấn bán hàng.
- Chỉ dùng thông tin có trong dữ liệu website.
- Không tự bịa giá, tồn kho, chiết khấu, thông số kỹ thuật hoặc chính sách nếu dữ liệu không có.
- Nếu dữ liệu không đủ, hãy nói chưa có thông tin đầy đủ và đề nghị khách để lại số điện thoại/Zalo để nhân viên tư vấn.
- Khi khách hỏi báo giá, hãy Gửi link đăng kí nhận báo giá trên website: https://eurohardware.id.vn/bao-gia và đề nghị khách điền thông tin để được tư vấn nhanh chóng.
- Khi phù hợp, gợi ý khách liên hệ Hotline/Zalo: 079 619 2091.
- Nếu có link sản phẩm hoặc bài viết trong dữ liệu, hãy gửi link cho khách.

DỮ LIỆU WEBSITE:
${context}

CÂU HỎI KHÁCH HÀNG:
${question}
`;
}

export async function generateAnswer({ question, context }) {
  if (!genAI) {
    throw new Error("Gemini API key is missing");
  }

  const model = genAI.getGenerativeModel({
    model: modelName
  });

  const prompt = buildPrompt({ question, context });

  const maxRetries = 3;
  const delays = [2000, 5000, 10000];

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const result = await model.generateContent(prompt);
      const response = await result.response;
      return response.text();
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

      console.warn(`Gemini temporary error. Retry ${attempt}/${maxRetries} after ${delays[attempt - 1]}ms`);
      await sleep(delays[attempt - 1]);
    }
  }

  throw new Error("Gemini failed after retries");
}