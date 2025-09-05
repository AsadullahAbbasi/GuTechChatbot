import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { QdrantClient } from "@qdrant/js-client-rest";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3001;

// Qdrant Client
const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

// OpenAI Client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Gemini Client (for embeddings only)
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });

// Collection
const QDRANT_COLLECTION_NAME = "gutech_knowledge_base";

/**
 * Get embedding from Gemini (768-dim, matches your Qdrant collection)
 */
async function getEmbedding(text) {
  try {
    const embeddingRes = await embeddingModel.embedContent(text);
    if (!embeddingRes?.embedding?.values) {
      throw new Error("Invalid embedding response from Gemini API.");
    }
    return embeddingRes.embedding.values;
  } catch (error) {
    console.error("Error in getEmbedding:", error);
    throw new Error(`Failed to generate embedding: ${error.message}`);
  }
}

// API Endpoint
app.post("/query", async (req, res) => {
  const { userQuery } = req.body;
  if (!userQuery) {
    return res.status(400).json({ error: "User query is required." });
  }

  try {
    // 1. Embed query using Gemini
    const queryEmbedding = await getEmbedding(userQuery);

    // 2. Search Qdrant
    const searchResult = await qdrantClient.search(QDRANT_COLLECTION_NAME, {
      vector: queryEmbedding,
      limit: 10,
      with_payload: true,
    });

    // 3. Build context
    const context = searchResult
      .map((item) => {
        let chunkContent = item.payload?.content || "";
        const chunkUrl = item.payload?.url;
        const chunkTitle = item.payload?.title;
        const chunkSourceFile = item.payload?.source_file;

        if (chunkTitle) chunkContent += `\nTitle: ${chunkTitle}`;
        if (chunkUrl) chunkContent += `\nSource URL: ${chunkUrl}`;
        if (chunkSourceFile) chunkContent += `\nSource File: ${chunkSourceFile}`;
        return chunkContent;
      })
      .filter(Boolean)
      .join("\n\n---\n\n");

    // 4. Build prompt
    let chatInput;
    if (context.trim() === "") {
      chatInput = `
You are a professional, knowledgeable university assistant.
You need to respond for GUTech/Al Ghazali University School of Technology in Karachi.
No specific data is available from the knowledge base, but you must still respond confidently and helpfully.

User Query: ${userQuery}`;
    } else {
      chatInput = `You are a helpful and expert assistant for GU TECH (Greenwich University Technology Campus).
Use only the following relevant data to answer the user’s question.
If multiple entities are possible, list them concisely.
Do not say information is missing.
Keep answers concise, professional, and user-friendly.

Relevant Data:
${context}

User Query: ${userQuery}`;
    }

    // 5. Generate response using ChatGPT
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are a helpful and expert university assistant." },
        { role: "user", content: chatInput },
      ],
    });

    const answer = completion.choices[0].message.content;

    res.json({ answer });
  } catch (err) {
    console.error("Error in /query endpoint:", err);
    res.status(500).json({
      error:
        "Failed to fetch answer from AI model or Qdrant. Please check backend logs.",
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
