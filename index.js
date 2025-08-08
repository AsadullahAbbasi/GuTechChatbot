import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { QdrantClient } from '@qdrant/js-client-rest';
import { GoogleGenerativeAI } from '@google/generative-ai';

dotenv.config(); // Load environment variables from .env file

const app = express();
app.use(cors()); // Enable CORS for all origins (for development)
app.use(express.json()); // Enable JSON body parsing

const PORT = process.env.PORT || 3001; // Use port from .env or default to 3001
console.log("Environment Variables Loaded:");
console.log(`QDRANT_URL: ${process.env.QDRANT_URL}`);
console.log(`QDRANT_API_KEY: ${process.env.QDRANT_API_KEY ? '******' : 'NOT SET'}`); // Mask API key for logs
console.log(`GEMINI_API_KEY: ${process.env.GEMINI_API_KEY ? '******' : 'NOT SET'}`); // Mask API key for logs

// Qdrant Client Setup
const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
});

// Google Gemini Generative AI Client Setup
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const generativeModel = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

// Google Gemini Embedding Model Setup
const embeddingModel = genAI.getGenerativeModel({ model: 'embedding-001' });

// Qdrant Collection Name (must match what you used for embedding)
const QDRANT_COLLECTION_NAME = 'gutech_knowledge_base'; 

/**
 * Generates an embedding for a given text using the Gemini Embedding API.
 * @param {string} text The text to embed.
 * @returns {Promise<number[]>} A promise that resolves to the embedding vector.
 * @throws {Error} If embedding generation fails.
 */
async function getEmbedding(text) {
    try {
        const embeddingRes = await embeddingModel.embedContent(text);
        
        if (!embeddingRes || !embeddingRes.embedding || !embeddingRes.embedding.values) {
            throw new Error("Invalid embedding response from Gemini API.");
        }
        return embeddingRes.embedding.values;
    } catch (error) {
        console.error("Error in getEmbedding:", error);
        throw new Error(`Failed to generate embedding: ${error.message}`);
    }
}

// API Endpoint for Chat Queries
app.post('/query', async (req, res) => {
    const { userQuery } = req.body;

    if (!userQuery) {
        return res.status(400).json({ error: 'User query is required.' });
    }

    try {
        // 1. Generate embedding for the user's query
        const queryEmbedding = await getEmbedding(userQuery);

        // 2. Search Qdrant for relevant context
        const searchResult = await qdrantClient.search(QDRANT_COLLECTION_NAME, {
            vector: queryEmbedding,
            limit: 10, // Increased limit for debugging purposes
            with_payload: true, // Ensure the original content and metadata are returned
        });

        // --- NEW LOGGING FOR QDRANT SEARCH RESULT ---
        console.log("\n--- Qdrant Search Result (Full) ---");
        console.log(JSON.stringify(searchResult, null, 2)); // Pretty print JSON
        console.log("-----------------------------------\n");
        // --- END NEW LOGGING ---

        // Extract content and relevant metadata from relevant chunks to form context
        const context = searchResult
            .map(item => {
                let chunkContent = item.payload?.content || '';
                const chunkUrl = item.payload?.url;
                const chunkTitle = item.payload?.title;
                const chunkSourceFile = item.payload?.source_file;

                // Append URL and other useful metadata to the content
                if (chunkTitle) {
                    chunkContent += `\nTitle: ${chunkTitle}`;
                }
                if (chunkUrl) {
                    chunkContent += `\nSource URL: ${chunkUrl}`;
                }
                if (chunkSourceFile) {
                    chunkContent += `\nSource File: ${chunkSourceFile}`;
                }
                return chunkContent;
            })
            .filter(Boolean) // Remove empty strings
            .join('\n\n---\n\n'); // Join with a clear separator

        // --- NEW LOGGING FOR CONTEXT SENT TO GEMINI ---
        console.log("\n--- Context sent to Gemini ---");
        console.log(context);
        console.log("------------------------------\n");
        // --- END NEW LOGGING ---

        // 3. Prepare input for the generative model
     let chatInput;

if (context.trim() === "") {
    // If no context is found, give the model a clear persona and mission.
    chatInput = `You are a helpful and professional university assistant. A user has asked a question, but no specific information from the university's knowledge base was found to answer it directly.

    Your task is to provide a helpful and polite response. Do NOT state that you lack information. Instead, assume a role as a knowledgeable assistant and offer general, useful advice or next steps.

    Example Responses:
    - "For the most up-to-date information on campus locations, I would recommend visiting our official website or contacting our admissions office directly. They'll be able to provide the most accurate details!"
    - "To find out more about our campus facilities and location, the best place to start is our main website's 'About Us' section. Our team is also available to help if you contact them by phone or email."

    User Query: ${userQuery}`;
} else {
    // If context is found, your existing prompt is great.
    chatInput = `You are a helpful and informed assistant for GU TECH (Greenwich University Technology Campus).

    Use only the **following relevant data** to answer the user's question. If something is not directly mentioned, don't deny the user or say unspecified or this info is not availabel currently — instead, provide a helpful, logical response based on related details, common practices in universities, or suggest practical next steps.

    Keep answers concise, professional, and user-friendly. Add emojis only when it adds clarity or friendliness.

    Relevant Data:
    ${context}

    User Query: ${userQuery}`;
}

        // 4. Generate response using Gemini-1.5-Flash
        const result = await generativeModel.generateContent(chatInput);
        const response = await result.response;
        const answer = response.text();

        res.json({ answer }); // Send the generated answer back to the frontend
    } catch (err) {
        console.error("Error in /query endpoint:", err);
        res.status(500).json({ error: 'Failed to fetch answer from AI model or Qdrant. Please check backend logs.' });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
