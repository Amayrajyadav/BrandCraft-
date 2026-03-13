"""
BrandCraft — Generative AI–Powered Branding Automation System
Backend: FastAPI + Groq LLaMA-3.3-70B + IBM Granite 3.3 + Stable Diffusion XL
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os, base64, json, httpx, asyncio
from urllib.parse import quote
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

HF_IBM_URL          = "https://api-inference.huggingface.co/models/ibm-granite/granite-3.3-2b-instruct"

# ── Image generation ──────────────────────────────────────────────────────────
# HF SDXL (old) returned HTTP 410 Gone — endpoint is permanently dead.
# Pollinations.ai is 100% free, no API key, returns real JPG images in 3-8s.
# We use it as the primary engine. HF FLUX is the backup if Pollinations fails.
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true&model=flux&seed=42"
HF_FLUX_URL      = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

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
    brand_name:    str
    industry:      Optional[str] = ""
    style:         Optional[str] = "minimalist"
    colors:        Optional[str] = ""
    primary_color: Optional[str] = ""
    accent_color:  Optional[str] = ""
    bg_color:      Optional[str] = ""
    icon_shape:    Optional[str] = ""

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


async def call_image_model(prompt: str) -> tuple:
    """
    Tier 1 → Pollinations.ai  (free, no API key, real AI image, ~5-15s)
    Tier 2 → HF FLUX.1-schnell (uses HF_API_KEY, fallback)
    Returns (base64_string, model_name, mime_type)
    """

    # ── Tier 1: Pollinations.ai ───────────────────────────────────────────────
    # Uses GET request with prompt in URL. Returns a real JPEG. No key needed.
    try:
        url = POLLINATIONS_URL.format(prompt=quote(prompt))
        async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "BrandCraft/1.0"})
        if r.status_code == 200 and len(r.content) > 3000:
            ct = r.headers.get("content-type", "")
            if "image" in ct or r.content[:2] == b"\xff\xd8":
                return base64.b64encode(r.content).decode(), "pollinations", "image/jpeg"
    except Exception:
        pass

    # ── Tier 2: HF FLUX.1-schnell ────────────────────────────────────────────
    hf_headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    hf_payload  = {"inputs": prompt, "parameters": {"num_inference_steps": 4, "guidance_scale": 0.0}}
    last_err    = "no attempts"
    for _ in range(3):
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(HF_FLUX_URL, headers=hf_headers, json=hf_payload)
            if r.status_code == 503:
                wait = 20
                try: wait = float(r.json().get("estimated_time", 20))
                except: pass
                await asyncio.sleep(min(wait, 25))
                continue
            if r.status_code == 200 and len(r.content) > 3000:
                ct = r.headers.get("content-type", "")
                if "image" in ct or r.content[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0"):
                    return base64.b64encode(r.content).decode(), "flux", "image/png"
            last_err = f"flux HTTP {r.status_code}"
            break
        except Exception as ex:
            last_err = str(ex); break

    raise RuntimeError(f"All image APIs failed — {last_err}")


def resolve_palette(req_primary, req_accent, req_bg, colors_text, style_str):
    COLOR_MAP = {
        "navy":"#0a1628","deep navy":"#0d1f3c","midnight blue":"#191970",
        "blue":"#1a5276","royal blue":"#2255cc","sky blue":"#87ceeb","cobalt":"#0047ab",
        "green":"#1e6b3a","emerald":"#046f43","sage":"#4a7c59","forest":"#2d5a27",
        "mint":"#3eb489","olive":"#6b7c45","teal":"#0e7c7b","cyan":"#0097a7","turquoise":"#00897b",
        "red":"#c0392b","crimson":"#990000","scarlet":"#cc2200","coral":"#e05a4e",
        "rose":"#c2185b","pink":"#e91e8c","maroon":"#800000",
        "orange":"#c75000","amber":"#cc7700","terracotta":"#c4622d","rust":"#b84a2a","copper":"#b87333",
        "gold":"#b8860b","golden":"#d4a017","yellow":"#c9a800","champagne":"#c9a96e","bronze":"#8b6914",
        "purple":"#6c3483","violet":"#7b2d8b","indigo":"#3c1874","lavender":"#7b68ee","plum":"#5b2c6f",
        "black":"#111111","charcoal":"#2c2c2c","dark":"#1a1a2e","slate":"#445566",
        "gray":"#555555","grey":"#555555","silver":"#7f8c8d",
        "white":"#ffffff","cream":"#fffdf7","ivory":"#fffff0","beige":"#f5f0e0","sand":"#e8d5b0",
    }
    BASE_NAMES = {"navy","blue","green","red","orange","gold","purple","black","white",
                  "teal","pink","gray","grey","silver","mint","coral","rose","maroon"}

    def is_hex(v): return bool(v and v.startswith("#") and len(v) in (4,7))
    def is_light(h):
        h = h.lstrip("#")
        if len(h)==3: h = h[0]*2+h[1]*2+h[2]*2
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return (r*.299+g*.587+b*.114) > 155
    def lighten(h, amt=80):
        h = h.lstrip("#")
        if len(h)==3: h = h[0]*2+h[1]*2+h[2]*2
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f"#{min(r+amt,255):02x}{min(g+amt,255):02x}{min(b+amt,255):02x}"
    def darken(h, f=0.55):
        h = h.lstrip("#")
        if len(h)==3: h = h[0]*2+h[1]*2+h[2]*2
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f"#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}"

    text_colors = []
    if colors_text and not (is_hex(req_primary) and is_hex(req_accent)):
        c = colors_text.lower()
        used_bases = []
        for name in sorted(COLOR_MAP.keys(), key=len, reverse=True):
            if name not in c: continue
            base = next((b for b in BASE_NAMES if b in name), name)
            if base in used_bases: continue
            text_colors.append(COLOR_MAP[name])
            used_bases.append(base)
            if len(text_colors) >= 3: break

    primary = req_primary if is_hex(req_primary) else (text_colors[0] if text_colors else "#1a1628")
    accent  = req_accent  if is_hex(req_accent)  else (text_colors[1] if len(text_colors)>1 else (lighten(primary,90) if not is_light(primary) else darken(primary)))
    bg      = req_bg      if is_hex(req_bg)       else ("#ffffff" if any(x in (colors_text or "").lower() for x in ("white","cream","ivory","light")) else "#fafafa")
    on_pri  = "#ffffff" if not is_light(primary) else "#111111"
    return {"primary":primary,"accent":accent,"bg":bg,"on_primary":on_pri}


def build_svg_logo(brand_name, industry, style, colors,
                   primary_hex, accent_hex, bg_hex, icon_shape):
    import math
    pal     = resolve_palette(primary_hex, accent_hex, bg_hex, colors, style)
    primary = pal["primary"]
    accent  = pal["accent"]
    bg      = pal["bg"]
    on_pri  = pal["on_primary"]

    brand_safe = brand_name.replace("&","and").replace("<","").replace(">","").strip()[:22]
    initials   = "".join(w[0].upper() for w in brand_safe.split()[:2]) or brand_safe[:2].upper()
    industry_l = (industry or "")[:28].upper()
    s          = (style or "").lower()
    sk         = (icon_shape or "").lower()

    if not sk:
        if   "geometric"  in s: sk = "hexagon"
        elif "vintage"    in s: sk = "shield"
        elif "classic"    in s: sk = "square"
        elif "luxury"     in s: sk = "diamond"
        elif "futuristic" in s: sk = "hexagon"
        elif "tech"       in s: sk = "square"
        elif "bold"       in s: sk = "triangle"
        else:                   sk = "circle"

    CX, CY, R = 256, 188, 118

    if sk == "hexagon":
        pts  = " ".join(f"{CX+R*math.cos(math.radians(60*i-30)):.1f},{CY+R*math.sin(math.radians(60*i-30)):.1f}" for i in range(6))
        pts2 = " ".join(f"{CX+(R-18)*math.cos(math.radians(60*i-30)):.1f},{CY+(R-18)*math.sin(math.radians(60*i-30)):.1f}" for i in range(6))
        icon_svg = f'<polygon points="{pts}" fill="{primary}"/><polygon points="{pts2}" fill="none" stroke="{accent}" stroke-width="3.5"/>'

    elif sk == "diamond":
        icon_svg = (
            f'<polygon points="{CX},{CY-R} {CX+R},{CY} {CX},{CY+R} {CX-R},{CY}" fill="{primary}"/>'
            f'<polygon points="{CX},{CY-R+18} {CX+R-18},{CY} {CX},{CY+R-18} {CX-R+18},{CY}" fill="none" stroke="{accent}" stroke-width="3"/>'
        )

    elif sk == "triangle":
        h_tri = int(R * 1.1)
        icon_svg = (
            f'<polygon points="{CX},{CY-h_tri} {CX+R},{CY+R//2} {CX-R},{CY+R//2}" fill="{primary}"/>'
            f'<polygon points="{CX},{CY-h_tri+20} {CX+R-22},{CY+R//2-12} {CX-R+22},{CY+R//2-12}" fill="none" stroke="{accent}" stroke-width="3"/>'
        )

    elif sk == "square":
        hr = R - 10
        icon_svg = (
            f'<rect x="{CX-hr}" y="{CY-hr}" width="{hr*2}" height="{hr*2}" rx="14" fill="{primary}"/>'
            f'<rect x="{CX-hr+14}" y="{CY-hr+14}" width="{hr*2-28}" height="{hr*2-28}" rx="7" fill="none" stroke="{accent}" stroke-width="3"/>'
        )

    elif sk == "shield":
        icon_svg = (
            f'<path d="M{CX},{CY-R} L{CX+R},{CY-R//2} L{CX+R},{CY+R//3} Q{CX},{CY+R} {CX},{CY+R} Q{CX-R},{CY+R} {CX-R},{CY+R//3} L{CX-R},{CY-R//2} Z" fill="{primary}"/>'
            f'<path d="M{CX},{CY-R+18} L{CX+R-18},{CY-R//2+9} L{CX+R-18},{CY+R//3} Q{CX},{CY+R-18} {CX},{CY+R-18} Q{CX-R+18},{CY+R-18} {CX-R+18},{CY+R//3} L{CX-R+18},{CY-R//2+9} Z" fill="none" stroke="{accent}" stroke-width="2.5"/>'
        )

    elif sk == "star":
        outer, inner = R, int(R*0.42)
        pts_list = []
        for i in range(5):
            ax = CX + outer*math.cos(math.radians(72*i-90))
            ay = CY + outer*math.sin(math.radians(72*i-90))
            bx = CX + inner*math.cos(math.radians(72*i-54))
            by = CY + inner*math.sin(math.radians(72*i-54))
            pts_list.append(f"{ax:.1f},{ay:.1f} {bx:.1f},{by:.1f}")
        icon_svg = f'<polygon points="{" ".join(pts_list)}" fill="{primary}" stroke="{accent}" stroke-width="3"/>'

    elif sk == "pill":
        pw, ph = int(R*1.6), int(R*0.7)
        icon_svg = (
            f'<rect x="{CX-pw}" y="{CY-ph}" width="{pw*2}" height="{ph*2}" rx="{ph}" fill="{primary}"/>'
            f'<rect x="{CX-pw+14}" y="{CY-ph+14}" width="{pw*2-28}" height="{ph*2-28}" rx="{max(ph-14,4)}" fill="none" stroke="{accent}" stroke-width="3"/>'
        )

    elif sk == "arch":
        icon_svg = (
            f'<path d="M{CX-R},{CY+R} L{CX-R},{CY} Q{CX-R},{CY-R} {CX},{CY-R} Q{CX+R},{CY-R} {CX+R},{CY} L{CX+R},{CY+R} Z" fill="{primary}"/>'
            f'<path d="M{CX-R+14},{CY+R} L{CX-R+14},{CY} Q{CX-R+14},{CY-R+14} {CX},{CY-R+14} Q{CX+R-14},{CY-R+14} {CX+R-14},{CY} L{CX+R-14},{CY+R} Z" fill="none" stroke="{accent}" stroke-width="3"/>'
        )

    elif sk == "cross":
        arm, length = int(R*0.35), R
        icon_svg = (
            f'<rect x="{CX-arm}" y="{CY-length}" width="{arm*2}" height="{length*2}" rx="{int(arm*0.4)}" fill="{primary}"/>'
            f'<rect x="{CX-length}" y="{CY-arm}" width="{length*2}" height="{arm*2}" rx="{int(arm*0.4)}" fill="{primary}"/>'
        )

    else:  # circle (explicit default)
        icon_svg = (
            f'<circle cx="{CX}" cy="{CY}" r="{R}" fill="{primary}"/>'
            f'<circle cx="{CX}" cy="{CY}" r="{R-17}" fill="none" stroke="{accent}" stroke-width="3.5"/>'
        )

    # Style decorations
    deco = ""
    if "luxury" in s:
        for ang in (0,90,180,270):
            dx = CX + (R+14)*math.cos(math.radians(ang))
            dy = CY + (R+14)*math.sin(math.radians(ang))
            deco += f'<circle cx="{dx:.0f}" cy="{dy:.0f}" r="4" fill="{accent}"/>'
    elif "futuristic" in s or "tech" in s:
        deco = (f'<line x1="{CX-R-22}" y1="{CY}" x2="{CX-R+4}" y2="{CY}" stroke="{accent}" stroke-width="2"/>'
                f'<line x1="{CX+R-4}" y1="{CY}" x2="{CX+R+22}" y2="{CY}" stroke="{accent}" stroke-width="2"/>'
                f'<circle cx="{CX-R-24}" cy="{CY}" r="3.5" fill="{accent}"/>'
                f'<circle cx="{CX+R+24}" cy="{CY}" r="3.5" fill="{accent}"/>')
    elif "playful" in s:
        deco = (f'<circle cx="{CX-R-12}" cy="{CY-18}" r="7" fill="{accent}" opacity="0.65"/>'
                f'<circle cx="{CX+R+12}" cy="{CY+16}" r="5" fill="{accent}" opacity="0.5"/>'
                f'<circle cx="{CX+22}" cy="{CY-R-16}" r="6" fill="{accent}" opacity="0.6"/>')
    elif "vintage" in s or "classic" in s:
        deco = (f'<line x1="126" y1="326" x2="386" y2="326" stroke="{accent}" stroke-width="1.5" opacity="0.5"/>'
                f'<line x1="146" y1="332" x2="366" y2="332" stroke="{accent}" stroke-width="0.8" opacity="0.35"/>')
    elif "art-deco" in s or "art_deco" in s:
        for i in range(1, 5):
            deco += f'<circle cx="{CX}" cy="{CY}" r="{R + i*7}" fill="none" stroke="{accent}" stroke-width="0.5" opacity="{max(0.06, 0.28-i*0.05)}"/>'
    elif "neon" in s:
        deco = f'<circle cx="{CX}" cy="{CY}" r="{R+6}" fill="none" stroke="{accent}" stroke-width="3" opacity="0.35"/>'
    elif "geometric" in s:
        for x, y in ((128,128),(384,128),(128,350),(384,350)):
            dx2 = 1 if x < 256 else -1
            dy2 = 1 if y < 256 else -1
            deco += (f'<line x1="{x}" y1="{y}" x2="{x+16*dx2}" y2="{y}" stroke="{accent}" stroke-width="2"/>'
                     f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+16*dy2}" stroke="{accent}" stroke-width="2"/>')
    elif "bold" in s:
        deco = f'<circle cx="{CX}" cy="{CY}" r="{R+8}" fill="none" stroke="{accent}" stroke-width="6"/>'
    elif "organic" in s:
        deco = f'<ellipse cx="{int(CX-R*.3)}" cy="{int(CY-R*.3)}" rx="{int(R*.25)}" ry="{int(R*.18)}" fill="{accent}" opacity="0.15"/>'

    # For pill/arch shapes the icon center is different — adjust text Y accordingly
    text_y   = CY + 10
    name_y   = 362
    div_y    = 380
    sub_y    = 404
    if sk == "pill":
        # pill is shorter vertically, text fits below more naturally
        text_y = CY + 2
    elif sk == "arch":
        text_y = CY + R//2 + 10   # push text down inside the arch

    text_color = primary if bg != primary else accent
    return f"""<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="pri_grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="{primary}"/>
      <stop offset="60%"  stop-color="{primary}" stop-opacity="0.92"/>
      <stop offset="100%" stop-color="{accent}"/>
    </linearGradient>
    <radialGradient id="gloss" cx="35%" cy="28%" r="60%">
      <stop offset="0%"   stop-color="#ffffff" stop-opacity="0.28"/>
      <stop offset="55%"  stop-color="#ffffff" stop-opacity="0.06"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
    <filter id="soft-shadow" x="-10%" y="-10%" width="130%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000000" flood-opacity="0.22"/>
    </filter>
  </defs>
  <rect width="512" height="512" fill="{bg}"/>
  <g filter="url(#soft-shadow)">
  {icon_svg}
  </g>
  {icon_svg.replace(f'fill="{primary}"', 'fill="url(#gloss)"').replace(f'stroke="{accent}"','stroke="none"').replace('stroke-width="3"','').replace('stroke-width="3.5"','').replace('stroke-width="2.5"','')}
  {deco}
  <text x="{CX}" y="{text_y}" font-size="88" font-family="Arial Black,Arial"
        font-weight="900" text-anchor="middle" dominant-baseline="middle"
        fill="{on_pri}" letter-spacing="-3">{initials}</text>
  <text x="256" y="{name_y}" font-size="26" font-family="Arial" font-weight="700"
        text-anchor="middle" fill="{text_color}" letter-spacing="5">{brand_safe.upper()}</text>
  <rect x="216" y="{div_y}" width="80" height="2.5" fill="{accent}" rx="1.25"/>
  <text x="256" y="{sub_y}" font-size="11" font-family="Arial"
        text-anchor="middle" fill="{accent}" letter-spacing="3" opacity="0.9">{industry_l}</text>
</svg>"""



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
async def serve_frontend():
    return FileResponse("index.html")
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
    """
    Logo generation with 3-tier fallback so the demo NEVER shows a broken image:
      Tier 1 → Pollinations.ai   (free, no key, real AI image, 3-8s)
      Tier 2 → HF FLUX.1-schnell (free HF key, real AI image, 10-40s)
      Tier 3 → Rich SVG fallback  (instant, reflects brand colors + style)
    """
    prompt = (
        f"{req.style} logo design for brand '{req.brand_name}', {req.industry}. "
        f"Colors: {req.colors}. Clean minimal icon, white background, no text overlay, "
        "professional brand identity, flat vector style."
    )

    try:
        img_b64, model_used, mime = await call_image_model(prompt)
        return {
            "image_base64": img_b64,
            "mime_type":    mime,
            "prompt_used":  prompt,
            "brand_name":   req.brand_name,
            "mode":         model_used
        }

    except Exception as e:
        svg = build_svg_logo(
            req.brand_name,
            req.industry      or "",
            req.style         or "minimalist",
            req.colors        or "",
            req.primary_color or "",
            req.accent_color  or "",
            req.bg_color      or "",
            req.icon_shape    or "",
        )
        encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        return {
            "image_base64": encoded,
            "mime_type":    "image/svg+xml",
            "prompt_used":  f"SVG · style:{req.style} · {req.primary_color}/{req.accent_color}",
            "brand_name":   req.brand_name,
            "mode":         "svg_fallback"
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


# ─────────────────────────────────────────────────────────────────────────────
# PITCH DECK
# ─────────────────────────────────────────────────────────────────────────────
class PitchDeckRequest(BaseModel):
    brand_name:  str
    brand_story: str
    industry:    Optional[str] = ""
    audience:    Optional[str] = ""
    tone:        Optional[str] = "modern"
    problem:     Optional[str] = ""
    solution:    Optional[str] = ""
    product:     Optional[str] = ""
    marketing:   Optional[str] = ""
    vision:      Optional[str] = ""
    tagline:     Optional[str] = ""

class PitchPreviewRequest(BaseModel):
    brand_name:  str
    brand_story: str
    industry:    Optional[str] = ""
    audience:    Optional[str] = ""
    tone:        Optional[str] = "modern"
    tagline:     Optional[str] = ""

class PitchBuildRequest(BaseModel):
    brand_name:  str
    brand_story: str
    industry:    Optional[str] = ""
    audience:    Optional[str] = ""
    tone:        Optional[str] = "modern"
    problem:     Optional[str] = ""
    solution:    Optional[str] = ""
    product:     Optional[str] = ""
    marketing:   Optional[str] = ""
    vision:      Optional[str] = ""
    tagline:     Optional[str] = ""
    primary:     Optional[str] = ""
    secondary:   Optional[str] = ""
    accent:      Optional[str] = ""
    bg_dark:     Optional[str] = ""
    bg_light:    Optional[str] = ""
    font_heading:Optional[str] = ""
    font_body:   Optional[str] = ""
    ai_primary:  Optional[str] = ""
    ai_accent:   Optional[str] = ""


def _industry_palette_map(industry: str) -> dict:
    k = (industry or "").lower()
    if any(x in k for x in ["fashion","luxury","beauty","jewel"]):
        return {"mood":"neutral dark with gold or rose accent","good_accents":["B5924C","C9A96E","E8D5B7","2D2D2D","1C1C1C"],"avoid":"neon or overly bright colors"}
    if any(x in k for x in ["food","restaurant","bakery","cafe"]):
        return {"mood":"warm earthy — terracotta, cream, olive, deep red","good_accents":["C8522A","8B3A2A","D4A853","4A5C3E","F5E6D3"],"avoid":"cold blues or grays"}
    if any(x in k for x in ["tech","saas","software","app","fintech","ai","data"]):
        return {"mood":"dark backgrounds with electric accent — navy, midnight + electric blue","good_accents":["4F8EF7","00D4AA","7B5EA7","1A2E4A","0A0F1C"],"avoid":"warm oranges or traditional serifs"}
    if any(x in k for x in ["health","medical","wellness","care","clinic"]):
        return {"mood":"clean whites and light blues — mint, sage, sky blue","good_accents":["4A9B8F","5B8DB8","7EC8C8","FFFFFF","F0F8FF"],"avoid":"dark heavy backgrounds"}
    if any(x in k for x in ["sustain","eco","green","organic","nature","environment"]):
        return {"mood":"earthy naturals — forest green, sand, terracotta","good_accents":["3D6B4F","7A9E5C","C4955A","F5F0E0","2C4A3E"],"avoid":"neon or synthetic-looking colors"}
    if any(x in k for x in ["education","school","learn","academy"]):
        return {"mood":"bright accessible — navy, yellow, white with energetic accent","good_accents":["1A3A6B","F5C842","4CAF82","FFFFFF","2D6DB5"],"avoid":"overly dark or muted tones"}
    return {"mood":"professional and brand-appropriate","good_accents":["1C1C1C","B5924C","F5EFE6","2D4A8A","C8522A"],"avoid":"clashing color combinations"}


def _check_palette_fit(user_color: str, industry: str, role: str) -> str:
    import colorsys
    try:
        r = int(user_color[0:2], 16) / 255
        g = int(user_color[2:4], 16) / 255
        b = int(user_color[4:6], 16) / 255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hue_deg = h * 360
    except Exception:
        return ""
    k = (industry or "").lower()
    warns = []
    if any(x in k for x in ["food","restaurant","bakery","cafe"]) and role == "accent":
        if 180 <= hue_deg <= 270 and s > 0.3:
            warns.append(f"Cold blue/purple accents (#{user_color}) can suppress appetite. Consider warm terracotta or amber instead.")
    if any(x in k for x in ["tech","saas","software","fintech","ai"]) and role == "primary":
        if 20 <= hue_deg <= 60 and s > 0.4:
            warns.append(f"Warm yellow/orange primaries (#{user_color}) can feel less credible for tech. Deep navy or slate works better.")
    if any(x in k for x in ["fashion","luxury","beauty"]) and s > 0.85 and v > 0.8:
        warns.append(f"Highly saturated neons (#{user_color}) undermine luxury brand perception. Consider muted, sophisticated alternatives.")
    if any(x in k for x in ["health","medical","wellness"]) and role in ["primary","bg_dark"] and v < 0.25:
        warns.append(f"Very dark colors (#{user_color}) in healthcare feel heavy. Light, airy palettes build more trust.")
    return " ".join(warns)


@app.post("/api/pitch-preview")
async def pitch_preview(req: PitchPreviewRequest):
    """Step 1: Generate AI palette for approval — no PPTX yet."""
    system = "You are a brand strategist. Return ONLY valid JSON."
    user = f"""Generate brand colors and typography for a pitch deck.

Brand: {req.brand_name}  |  Industry: {req.industry}
Story: {req.brand_story}  |  Audience: {req.audience}  |  Tone: {req.tone}

Return ONLY this JSON:
{{
  "tagline": "short memorable tagline",
  "primary":   "hex without #",
  "secondary": "hex without #",
  "accent":    "hex without #",
  "bg_dark":   "hex without #",
  "bg_light":  "hex without #",
  "font_heading": "Georgia or Palatino or Arial Black",
  "font_body":    "Calibri or Arial or Garamond",
  "palette_reasoning": "2 sentences explaining why these colors suit this brand"
}}"""
    ai_raw = await call_groq(system, user, temperature=0.5)
    theme  = safe_json(ai_raw, {})
    return {"theme": theme, "industry_guidance": _industry_palette_map(req.industry), "brand_name": req.brand_name}


@app.post("/api/pitch-check-palette")
async def pitch_check_palette(body: dict):
    """Check user palette against industry norms and return AI recommendations."""
    industry = body.get("industry", "")
    checks = {}
    for role in ["primary", "accent", "secondary"]:
        val = (body.get(role) or "").replace("#","").strip()
        if len(val) == 6:
            msg = _check_palette_fit(val, industry, role)
            if msg:
                checks[role] = msg
    palette_guide = _industry_palette_map(industry)
    colors_desc = ", ".join([f"{r}: #{body.get(r,'')}" for r in ["primary","secondary","accent","bg_dark","bg_light"] if body.get(r)])
    system = "You are a brand color consultant. Be direct and helpful. Under 120 words."
    user_msg = f"""A {industry} brand wants: {colors_desc}.
Issues: {checks if checks else "none obvious"}.
Industry best practice: {palette_guide["mood"]}. Avoid: {palette_guide["avoid"]}.
Give a short friendly assessment: what works, what to reconsider, suggest 1-2 alternative hex codes if needed."""
    ai_advice = await call_groq(system, user_msg, temperature=0.4)
    return {"warnings": checks, "advice": ai_advice, "industry_guidance": palette_guide, "palette_ok": len(checks) == 0}


@app.post("/api/pitch-deck")
async def generate_pitch_deck(req: PitchBuildRequest):
    import subprocess, json as json_lib, os

    system = "You are a brand strategist. Return ONLY valid JSON."
    user = f"""Fill missing content for a pitch deck.
Brand: {req.brand_name}  |  Industry: {req.industry}  |  Tone: {req.tone}
Story: {req.brand_story}  |  Audience: {req.audience}
Problem: {req.problem or "(derive)"}  |  Solution: {req.solution or "(derive)"}
Product: {req.product or "(derive)"}  |  Marketing: {req.marketing or "(derive)"}
Vision: {req.vision or "(derive)"}  |  Tagline: {req.tagline or "(generate)"}

Return ONLY this JSON:
{{
  "tagline":"tagline","primary":"hex","secondary":"hex","accent":"hex","bg_dark":"hex","bg_light":"hex",
  "font_heading":"Georgia or Palatino","font_body":"Calibri or Arial",
  "story":"2-3 sentences","problem":"3 problems newline-separated",
  "solution":"2-3 sentences","product":"2-3 sentences","marketing":"2-3 sentences","vision":"2-3 sentences"
}}"""
    ai_raw = await call_groq(system, user, temperature=0.55)
    theme  = safe_json(ai_raw, {})

    def pick(uval, akey, fallback):
        v = (uval or "").replace("#","").strip()
        return v if len(v) == 6 else theme.get(akey, fallback)

    deck_data = {
        "brand_name":   req.brand_name,
        "industry":     req.industry,
        "audience":     req.audience,
        "tone":         req.tone,
        "tagline":      req.tagline or theme.get("tagline",""),
        "primary":      pick(req.primary,   "primary",   "1a1628"),
        "secondary":    pick(req.secondary, "secondary", "f5f0e8"),
        "accent":       pick(req.accent,    "accent",    "c8522a"),
        "bg_dark":      pick(req.bg_dark,   "bg_dark",   "0f0f1a"),
        "bg_light":     pick(req.bg_light,  "bg_light",  "faf8f4"),
        "font_heading": req.font_heading or theme.get("font_heading","Georgia"),
        "font_body":    req.font_body    or theme.get("font_body","Calibri"),
        "story":        req.brand_story  or theme.get("story",""),
        "problem":      req.problem      or theme.get("problem",""),
        "solution":     req.solution     or theme.get("solution",""),
        "product":      req.product      or theme.get("product",""),
        "marketing":    req.marketing    or theme.get("marketing",""),
        "vision":       req.vision       or theme.get("vision",""),
    }

    script_path = os.path.join(os.path.dirname(__file__), "generate_deck.js")
    result = subprocess.run(["node", script_path, json_lib.dumps(deck_data)], capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"PPTX generation failed: {result.stderr[:400]}")

    return {
        "pptx_base64": result.stdout.strip(),
        "theme": {k: deck_data[k] for k in ["primary","secondary","accent","bg_dark","bg_light","font_heading","font_body","tagline","tone"]},
        "brand_name": req.brand_name
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)