"""
BrandCraft — Generative AI–Powered Branding Automation System
Backend: FastAPI + Groq LLaMA-3.3-70B + IBM Granite 3.3 + Stable Diffusion XL
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os, base64, json, httpx
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BrandCraft API",
    description="Generative AI–Powered Branding Automation System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
HF_API_KEY     = os.getenv("HF_API_KEY",   "your-hf-api-key-here")

GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.3-70b-versatile"

HF_IBM_URL     = "https://api-inference.huggingface.co/models/ibm-granite/granite-3.3-2b-instruct"
HF_SDXL_URL    = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class BrandNameRequest(BaseModel):
    industry: str
    keywords: str
    style: Optional[str] = "modern"
    count: Optional[int] = 10

class TaglineRequest(BaseModel):
    brand_name: str
    industry: str
    tone: Optional[str] = "professional"

class LogoRequest(BaseModel):
    brand_name: str
    industry: Optional[str] = ""
    style: Optional[str] = "minimalist"
    colors: Optional[str] = "blue and white"

class BrandStoryRequest(BaseModel):
    brand_name: str
    industry: str
    mission: str
    audience: Optional[str] = "general public"

class ProductDescRequest(BaseModel):
    product_name: str
    features: str
    tone: Optional[str] = "engaging"

class SocialPostRequest(BaseModel):
    brand_name: str
    topic: str
    platform: Optional[str] = "Instagram"
    tone: Optional[str] = "casual"

class EmailRequest(BaseModel):
    brand_name: str
    purpose: str
    recipient: Optional[str] = "customer"
    tone: Optional[str] = "professional"

class SentimentRequest(BaseModel):
    text: str

class SummaryRequest(BaseModel):
    text: str
    length: Optional[str] = "medium"

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

# ─────────────────────────────────────────────────────────────────────────────
# AI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
async def call_groq(system: str, user: str, temperature: float = 0.8) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        "temperature": temperature,
        "max_tokens": 1500
    }
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(GROQ_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Groq API error: {r.text}")
        return r.json()["choices"][0]["message"]["content"]


async def call_ibm_granite(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 600, "temperature": 0.7, "return_full_text": False}
    }
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(HF_IBM_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"IBM Granite error: {r.text}")
        result = r.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "").strip()
        return result.get("generated_text", "").strip()


async def call_sdxl(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"num_inference_steps": 30, "guidance_scale": 7.5, "width": 512, "height": 512}
    }
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(HF_SDXL_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"SDXL error: {r.text}")
        return base64.b64encode(r.content).decode("utf-8")


def safe_json(text: str, fallback):
    try:
        if fallback == []:
            s, e = text.find("["), text.rfind("]") + 1
            return json.loads(text[s:e])
        s, e = text.find("{"), text.rfind("}") + 1
        return json.loads(text[s:e])
    except Exception:
        return fallback

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "project": "BrandCraft",
        "tagline": "Generative AI–Powered Branding Automation System",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "models": ["groq/llama-3.3-70b", "ibm-granite/3.3-2b", "sdxl"]}


@app.post("/api/brand-names")
async def generate_brand_names(req: BrandNameRequest):
    system = "You are a world-class brand naming strategist. Return ONLY valid JSON, no markdown."
    user = f"""Generate {req.count} creative brand names for a {req.industry} company.
Keywords/themes: {req.keywords}
Naming style: {req.style}

Return ONLY a JSON array:
[{{"name":"BrandName","meaning":"why this name works","domain_friendly":true,"category":"descriptive|invented|metaphor|acronym"}}]"""
    result = await call_groq(system, user)
    names  = safe_json(result, [])
    return {"brand_names": names, "model": "groq/llama-3.3-70b-versatile", "count": len(names)}


@app.post("/api/taglines")
async def generate_taglines(req: TaglineRequest):
    system = "You are an award-winning copywriter. Return ONLY valid JSON, no markdown."
    user = f"""Create 5 memorable taglines for "{req.brand_name}", a {req.industry} brand.
Tone: {req.tone}

Return ONLY a JSON array:
[{{"tagline":"...", "emotion":"...", "type":"inspirational|witty|direct|emotional|bold", "hook":"why it works"}}]"""
    result   = await call_groq(system, user)
    taglines = safe_json(result, [])
    return {"taglines": taglines, "brand_name": req.brand_name}


@app.post("/api/logo")
async def generate_logo(req: LogoRequest):
    prompt = (
        f"Professional commercial logo design for brand called '{req.brand_name}', "
        f"{req.industry} company. Style: {req.style}. Color palette: {req.colors}. "
        "White background, clean vector art, minimalist icon with wordmark."
    )

    try:
        img_b64 = await call_sdxl(prompt)
        return {
            "image_base64": img_b64,
            "prompt_used": prompt,
            "brand_name": req.brand_name,
            "mode": "sdxl"
        }

    except Exception:
        # Fallback SVG so demo never breaks
        svg = f"""
        <svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="white"/>
            <circle cx="256" cy="200" r="80" fill="#1a1612"/>
            <text x="256" y="350" font-size="42" text-anchor="middle"
                  font-family="Arial" fill="#c8522a">
                {req.brand_name}
            </text>
        </svg>
        """
        encoded = base64.b64encode(svg.encode()).decode()
        return {
            "image_base64": encoded,
            "prompt_used": "Fallback SVG logo (SDXL unavailable)",
            "brand_name": req.brand_name,
            "mode": "fallback"
        }

@app.post("/api/brand-story")
async def generate_brand_story(req: BrandStoryRequest):
    system = "You are a brand storytelling consultant. Return ONLY valid JSON."
    user = f"""Create a complete brand identity document for "{req.brand_name}".
Industry: {req.industry} | Mission: {req.mission} | Audience: {req.audience}

Return ONLY this JSON:
{{
  "origin_story": "compelling origin narrative",
  "mission_statement": "formal mission statement",
  "vision": "long-term vision",
  "values": ["value1","value2","value3","value4"],
  "brand_promise": "what you guarantee to customers",
  "elevator_pitch": "30-second investor pitch",
  "positioning_statement": "formal positioning"
}}"""
    result = await call_groq(system, user)
    story  = safe_json(result, {})
    return {**story, "brand_name": req.brand_name}


@app.post("/api/product-description")
async def generate_product_description(req: ProductDescRequest):
    system = "You are an elite eCommerce copywriter. Return ONLY valid JSON."
    user = f"""Write product copy for "{req.product_name}".
Features: {req.features} | Tone: {req.tone}

Return ONLY this JSON:
{{
  "short": "1-2 sentence hero description",
  "long": "full paragraph product description",
  "bullets": ["benefit bullet 1","bullet 2","bullet 3","bullet 4","bullet 5"],
  "tagline": "catchy product tagline",
  "seo_description": "SEO meta description under 160 chars"
}}"""
    result = await call_groq(system, user)
    desc   = safe_json(result, {})
    return {**desc, "product_name": req.product_name}


@app.post("/api/social-post")
async def generate_social_post(req: SocialPostRequest):
    system = "You are a viral social media strategist. Return ONLY valid JSON."
    user = f"""Create 3 {req.platform} posts for "{req.brand_name}" about: {req.topic}
Tone: {req.tone}

Return ONLY a JSON array:
[{{"post":"full post content","hashtags":["tag1","tag2","tag3"],"best_time":"optimal posting time","engagement_tip":"one tip"}}]"""
    result = await call_groq(system, user)
    posts  = safe_json(result, [])
    return {"posts": posts, "platform": req.platform, "brand_name": req.brand_name}


@app.post("/api/email")
async def generate_email(req: EmailRequest):
    system = "You are an email marketing expert. Return ONLY valid JSON."
    user = f"""Write a complete marketing email for "{req.brand_name}".
Purpose: {req.purpose} | Recipient: {req.recipient} | Tone: {req.tone}

Return ONLY this JSON:
{{
  "subject": "compelling subject line",
  "preview_text": "email preview snippet under 90 chars",
  "greeting": "personalized opening",
  "body": "full email body with paragraphs",
  "cta_text": "call to action button text",
  "ps_line": "P.S. line for additional hook"
}}"""
    result = await call_groq(system, user)
    email  = safe_json(result, {})
    return {**email, "brand_name": req.brand_name}


@app.post("/api/sentiment")
async def analyze_sentiment(req: SentimentRequest):
    system = "You are a brand psychologist and linguist. Return ONLY valid JSON."
    user = f"""Analyze sentiment and brand impact of this text:
"{req.text}"

Return ONLY this JSON:
{{
  "sentiment": "positive|negative|neutral",
  "score": 0.85,
  "confidence": 0.9,
  "tone": "professional|aggressive|friendly|inspiring|urgent",
  "emotions": {{"joy":0.7,"trust":0.8,"anticipation":0.5,"fear":0.1,"surprise":0.2}},
  "brand_impact": "detailed analysis",
  "target_audience_reaction": "how audience will respond",
  "recommendation": "improvement suggestion",
  "revised_version": "improved version of the text"
}}"""
    result   = await call_groq(system, user, temperature=0.3)
    analysis = safe_json(result, {})
    return {**analysis, "original_text": req.text}


@app.post("/api/summarize")
async def summarize_text(req: SummaryRequest):
    system = "You are a content strategist. Return ONLY valid JSON."
    user = f"""Summarize this content in {req.length} format:
{req.text}

Return ONLY this JSON:
{{
  "summary": "the summary",
  "key_points": ["point1","point2","point3"],
  "action_items": ["action1","action2"],
  "word_count_original": 0,
  "word_count_summary": 0,
  "readability_improvement": "brief note"
}}"""
    result  = await call_groq(system, user, temperature=0.3)
    summary = safe_json(result, {})
    summary["word_count_original"] = len(req.text.split())
    return summary


@app.post("/api/chat")
async def branding_chat(req: ChatRequest):
    history_str = ""
    for h in req.history[-8:]:
        role    = h.get("role", "user").capitalize()
        content = h.get("content", "")
        history_str += f"{role}: {content}\n"

    prompt = f"""<|system|>
You are BrandCraft AI, an expert branding consultant powered by IBM Granite.
You specialize in brand identity, naming, positioning, messaging, and go-to-market strategy.
Give specific, actionable, expert-level branding advice.
<|end|>
<|user|>
{history_str}User: {req.message}
<|end|>
<|assistant|>"""

    try:
        response = await call_ibm_granite(prompt)
        for token in ["<|end|>", "<|assistant|>", "<|user|>", "<|system|>"]:
            response = response.replace(token, "").strip()
        model_used = "ibm-granite/granite-3.3-2b-instruct"
    except Exception:
        response = await call_groq(
            "You are BrandCraft AI, an expert branding consultant. Give specific, actionable advice.",
            req.message
        )
        model_used = "groq/llama-3.3-70b-versatile (fallback)"

    return {"response": response, "model": model_used}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)