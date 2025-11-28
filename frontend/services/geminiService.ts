import type { CropData, FertilizerData } from '../types';
import { GoogleGenerativeAI } from '@google/generative-ai';

const baseUrl = ''; // same-origin Flask server

export const getCropRecommendation = async (data: CropData): Promise<string> => {
    const payload = {
        N: Number(data.nitrogen),
        P: Number(data.phosphorus),
        K: Number(data.potassium),
        temperature: Number(data.temperature),
        humidity: Number(data.humidity),
        ph: Number(data.ph),
        rainfall: Number(data.rainfall),
        top_n: 3
    };

    const res = await fetch(`${baseUrl}/api/crop/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`Crop API error ${res.status}: ${text}`);
    }
    const result = await res.json();
    const topList = (result.top_recommendations || [])
        .map((r: any) => `- ${r.crop}: ${r.confidence_percentage}`)
        .join('\n');
    return `# Recommended Crop: **${result.recommended_crop}**\n\n` +
           `**Confidence:** ${(result.confidence_score * 100).toFixed(2)}%\n\n` +
           `## Top Alternatives\n${topList || '- N/A'}\n\n` +
           `## Insights\n` +
           `${(result.explanation?.insights || []).map((s: string) => `- ${s}`).join('\n')}`;
};

export const getFertilizerRecommendation = async (data: FertilizerData): Promise<string> => {
    const payload = {
        N: Number(data.nitrogen),
        P: Number(data.phosphorus),
        K: Number(data.potassium),
        temperature: Number(data.temperature),
        humidity: Number(data.humidity),
        moisture: data.soilMoisture ? Number(data.soilMoisture) : undefined,
        soil_type: data.soilType,
        crop_type: data.cropType,
        ph: data.ph ? Number(data.ph) : undefined
    };

    const res = await fetch(`${baseUrl}/api/fertilizer/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`Fertilizer API error ${res.status}: ${text}`);
    }
    const result = await res.json();
    const topList = (result.top_recommendations || [])
        .map((r: any) => `- ${r.fertilizer}: ${r.confidence_percentage}`)
        .join('\n');
    const rec = result.recommendations || {};
    const guidelines = (rec.application_guidelines || []).map((g: string) => `- ${g}`).join('\n');
    return `# Recommended Fertilizer: **${result.recommended_fertilizer}**\n\n` +
           `**Confidence:** ${(result.confidence_score * 100).toFixed(2)}%\n\n` +
           `## Top Alternatives\n${topList || '- N/A'}\n\n` +
           `## Application Guidelines\n${guidelines || '- N/A'}`;
};

export const detectPestOrDisease = async (imageB64: string, mimeType: string): Promise<string> => {
    const apiKey = (import.meta as any).env?.VITE_GEMINI_API_KEY || (window as any).GEMINI_API_KEY;
    if (!apiKey) {
        throw new Error('Gemini API key not configured. Set VITE_GEMINI_API_KEY in .env.local');
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

    const prompt = `You are an agricultural pest and plant disease expert. Analyze the image and identify likely pest(s) or disease(s), confidence, key visual signs, affected crops, and actionable treatment steps (chemical and organic). Provide a concise, farmer-friendly report.`;

    const result = await model.generateContent({
        contents: [{
            role: 'user',
            parts: [
                { text: prompt },
                { inlineData: { data: imageB64, mimeType } }
            ]
        }]
    });

    const text = result.response?.text?.() || result.response?.candidates?.[0]?.content?.parts?.map((p: any) => p.text).join('\n') || '';
    if (!text) {
        throw new Error('No response from Gemini');
    }
    return text;
};
