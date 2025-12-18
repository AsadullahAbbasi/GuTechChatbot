import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { QdrantClient } from "@qdrant/js-client-rest";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3001;

// Qdrant Client Setup
const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

// Google Gemini Setup
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// USE LATEST ALIASES TO AVOID 404
const generativeModel = genAI.getGenerativeModel({
  model: "gemini-2.5-flash", 
  generationConfig: {
    temperature: 0.7,
    maxOutputTokens: 2048,
  },
});

const embeddingModel = genAI.getGenerativeModel({ 
  model: "text-embedding-004" 
});

const QDRANT_COLLECTION_NAME = "gutech_knowledge_base";

async function getEmbedding(text) {
  try {
    const result = await embeddingModel.embedContent({
      content: { parts: [{ text }] },
      taskType: "RETRIEVAL_QUERY",
      outputDimensionality: 768, 
    });
    
    if (!result?.embedding?.values) {
      throw new Error("Invalid embedding response.");
    }
    return result.embedding.values;
  } catch (error) {
    console.error("Embedding Error:", error.message);
    throw error;
  }
}

function extractUrls(searchResult) {
  const urls = new Set();
  searchResult.forEach((item) => {
    if (item.payload?.url) urls.add(item.payload.url);
  });
  return Array.from(urls);
}

function createStructuredPrompt(context, userQuery, urls) {
  const urlSection = urls.length > 0 
    ? `<p><strong>Source Links:</strong><br>${urls.map(url => `<a href="${url}" target="_blank">${url}</a>`).join("<br>")}</p>` 
    : "";

  return `You are an expert assistant for GU TECH Karachi. Respond ONLY in HTML format.
Use <h2> for titles, <ul>/<li> for lists, and <p> for paragraphs. No Markdown.

Context Data:
${context || "No specific data found in the knowledge base. Please provide a general helpful response about GUTech based on your general knowledge."}

User Query: ${userQuery}

${urlSection}`;
}

app.post("/query", async (req, res) => {
  const { userQuery } = req.body;

  if (!userQuery) return res.status(400).json({ error: "Query required" });

  try {
    const startTime = Date.now();

    // 1. Generate search vector
    const queryEmbedding = await getEmbedding(userQuery);

    // 2. Search Qdrant
    const searchResult = await qdrantClient.search(QDRANT_COLLECTION_NAME, {
      vector: queryEmbedding,
      limit: 5,
      with_payload: true,
      // REMOVED score_threshold temporarily to debug "0 chunks" issue
    });

    console.log(`Matched ${searchResult.length} data chunks.`);

    const relevantUrls = extractUrls(searchResult);
    const context = searchResult
      .map(item => item.payload?.content)
      .filter(Boolean)
      .join("\n\n---\n\n");

    // 3. Generate final HTML answer
    const prompt = createStructuredPrompt(context, userQuery, relevantUrls);
    const result = await generativeModel.generateContent(prompt);
    const answer = result.response.text();

    const responseTime = Date.now() - startTime;

    res.json({
      answer,
      responseTime: `${responseTime}ms`,
      relevantUrls,
      chunksFound: searchResult.length,
    });
  } catch (err) {
    console.error("Backend Error Log:", err);
    res.status(500).json({ error: "Server error occurred. Check backend console." });
  }
});

app.get("/health", (req, res) => {
  res.json({ status: "OK", serverTime: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`🚀 RAG Chatbot live on port ${PORT}`);
});