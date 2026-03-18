/**
 * BrandCraft — Pitch Deck Generator v5 "Behance Edition"
 * ──────────────────────────────────────────────────────
 * Implements the full 8-step visual spec:
 *   Step 1  Structured slide schema (cover / split / cards / minimal / stats / timeline)
 *   Step 2  Visual rules — max 5 bullets, hero headings, whitespace
 *   Step 3  Image generation — NVIDIA FLUX → Pollinations → Unsplash
 *   Step 4  Layout engine — COVER / SPLIT / CARDS / MINIMAL
 *   Step 5  Styling — transparent overlays, gradient panels, soft shadows
 *   Step 6  Typography — 54pt hero / 28pt sub / 11pt body
 *   Step 7  Theme engine — luxury / minimal / startup / dark / editorial
 *   Step 8  Output — looks like a design composition, not a document
 */

const pptxgen = require("pptxgenjs");
const https   = require("https");
const http    = require("http");

/* ═══════════════════════════════════════════════════════════════════════════
   INPUT & COLOUR SETUP
═══════════════════════════════════════════════════════════════════════════ */
const raw = JSON.parse(process.argv[2]);
const {
  brand_name   = "Brand",
  tagline      = "",
  story        = "",
  problem      = "",
  solution     = "",
  product      = "",
  audience     = "",
  marketing    = "",
  vision       = "",
  primary      = "1C1C1C",
  secondary    = "F5EFE6",
  accent       = "B5924C",
  bg_dark      = "111111",
  bg_light     = "FAF7F2",
  font_heading = "Georgia",
  font_body    = "Calibri",
  industry     = "general",
  tone         = "premium",
} = raw;

const fx  = h => (String(h||"").replace(/^#/,"").trim().toUpperCase() || "222222").substring(0,6);
const P   = fx(primary);
const S   = fx(secondary);
const AC  = fx(accent);
const BD  = fx(bg_dark);
const BL  = fx(bg_light);
const W   = 10, H = 5.625;
const FH  = font_heading || "Georgia";
const FB  = font_body    || "Calibri";

/* ── Luminance helper for auto text colour ── */
const lum = hex => {
  try {
    const h = hex.replace(/^#/,"");
    return parseInt(h.slice(0,2),16)*0.299 + parseInt(h.slice(2,4),16)*0.587 + parseInt(h.slice(4,6),16)*0.114;
  } catch { return 128; }
};
const onDark  = "#FFFFFF";
const onLight = "#" + P;
const textOn  = bg => lum(bg) > 140 ? P : "FFFFFF";

/* ═══════════════════════════════════════════════════════════════════════════
   STEP 7 — THEME ENGINE
   Controls density, palette, font weight, layout choice
═══════════════════════════════════════════════════════════════════════════ */
const THEMES = {
  luxury:    { headSize:54, subSize:18, bodySize:11, spacing:1.6, cardBg:P,   cardText:"FFFFFF", lineW:0.5, pillBg:AC, accentLine:true,  coverDark:true  },
  minimal:   { headSize:52, subSize:16, bodySize:11, spacing:1.8, cardBg:BL,  cardText:P,        lineW:0.4, pillBg:AC, accentLine:false, coverDark:false },
  startup:   { headSize:56, subSize:20, bodySize:12, spacing:1.5, cardBg:P,   cardText:"FFFFFF", lineW:0.6, pillBg:AC, accentLine:true,  coverDark:true  },
  dark:      { headSize:58, subSize:19, bodySize:12, spacing:1.5, cardBg:BD,  cardText:"FFFFFF", lineW:0.5, pillBg:AC, accentLine:true,  coverDark:true  },
  editorial: { headSize:52, subSize:17, bodySize:11, spacing:1.7, cardBg:P,   cardText:S,        lineW:0.5, pillBg:AC, accentLine:true,  coverDark:true  },
};

function getTheme() {
  const t = (tone||"").toLowerCase();
  if (t.includes("luxur"))   return THEMES.luxury;
  if (t.includes("minim"))   return THEMES.minimal;
  if (t.includes("startup")) return THEMES.startup;
  if (t.includes("dark"))    return THEMES.dark;
  if (t.includes("edit"))    return THEMES.editorial;
  // Auto-pick from industry
  const ind = (industry||"").toLowerCase();
  if (["fashion","beauty","jewel","luxury"].some(x=>ind.includes(x)))   return THEMES.luxury;
  if (["tech","saas","fintech","software","app","ai"].some(x=>ind.includes(x))) return THEMES.dark;
  if (["health","wellness","edu","school"].some(x=>ind.includes(x)))     return THEMES.minimal;
  if (["food","restaurant","cafe","bakery"].some(x=>ind.includes(x)))    return THEMES.editorial;
  if (["sustain","eco","organic","nature"].some(x=>ind.includes(x)))     return THEMES.minimal;
  return THEMES.startup;
}
const TH = getTheme();

/* ═══════════════════════════════════════════════════════════════════════════
   STEP 3 — IMAGE GENERATION
   NVIDIA FLUX.1-dev → Pollinations FLUX → Unsplash
═══════════════════════════════════════════════════════════════════════════ */
const NVIDIA_API_KEY  = process.env.NVIDIA_API_KEY || "";
const NVIDIA_FLUX_URL = "https://integrate.api.nvidia.com/v1/images/generations";

function httpFetch(url, opts={}) {
  return new Promise(resolve => {
    function go(u, depth=0) {
      if (depth > 5) return resolve(null);
      const mod = u.startsWith("https") ? https : http;
      const headers = { "User-Agent": "Mozilla/5.0", ...(opts.headers||{}) };
      const req = mod.get(u, { headers, timeout: opts.timeout||25000 }, res => {
        if ([301,302,307,308].includes(res.statusCode) && res.headers.location)
          return go(res.headers.location, depth+1);
        if (res.statusCode !== 200) return resolve(null);
        const chunks = [];
        res.on("data", c => chunks.push(c));
        res.on("end",  ()  => resolve(Buffer.concat(chunks)));
        res.on("error", () => resolve(null));
      });
      req.on("error", () => resolve(null));
      req.on("timeout", () => { req.destroy(); resolve(null); });
    }
    go(url);
  });
}

function httpPost(url, body, headers={}) {
  return new Promise(resolve => {
    const data = JSON.stringify(body);
    const u = new URL(url);
    const opts = {
      hostname: u.hostname, port: u.port||443, path: u.pathname,
      method: "POST", timeout: 60000,
      headers: { "Content-Type":"application/json", "Content-Length":Buffer.byteLength(data), ...headers },
    };
    const mod = url.startsWith("https") ? https : http;
    const req = mod.request(opts, res => {
      const chunks = [];
      res.on("data", c => chunks.push(c));
      res.on("end",  () => {
        try { resolve({ status: res.statusCode, body: Buffer.concat(chunks), ct: res.headers["content-type"]||"" }); }
        catch { resolve(null); }
      });
      res.on("error", () => resolve(null));
    });
    req.on("error",   () => resolve(null));
    req.on("timeout", () => { req.destroy(); resolve(null); });
    req.write(data);
    req.end();
  });
}

async function fetchImgNvidia(prompt) {
  if (!NVIDIA_API_KEY) return null;
  try {
    const r = await httpPost(NVIDIA_FLUX_URL,
      { model:"black-forest-labs/flux.1-dev", prompt, n:1, size:"1280x720" },
      { Authorization:`Bearer ${NVIDIA_API_KEY}`, Accept:"image/png,application/json" }
    );
    if (!r || r.status!==200) return null;
    if (r.ct.includes("image") && r.body.length>3000)
      return `image/jpeg;base64,${r.body.toString("base64")}`;
    const j = JSON.parse(r.body.toString());
    const item = j?.data?.[0];
    if (item?.b64_json) return `image/png;base64,${item.b64_json}`;
    if (item?.url) {
      const buf = await httpFetch(item.url, {timeout:30000});
      if (buf && buf.length>3000) return `image/jpeg;base64,${buf.toString("base64")}`;
    }
  } catch {}
  return null;
}

async function fetchImgPollinations(prompt) {
  try {
    const enc = encodeURIComponent(prompt + ", cinematic, high resolution, editorial photography");
    const url = `https://image.pollinations.ai/prompt/${enc}?width=1280&height=720&nologo=true&model=flux&seed=${Math.floor(Math.random()*99999)}`;
    const buf = await httpFetch(url, {timeout:35000});
    if (buf && buf.length>3000) return `image/jpeg;base64,${buf.toString("base64")}`;
  } catch {}
  return null;
}

async function fetchImgUnsplash(query) {
  try {
    const url = `https://source.unsplash.com/1280x720/?${encodeURIComponent(query)}&t=${Date.now()}`;
    const buf = await httpFetch(url, {timeout:15000});
    if (buf && buf.length>3000) return `image/jpeg;base64,${buf.toString("base64")}`;
  } catch {}
  return null;
}

async function fetchImg(prompt, fallbackQuery) {
  return (
    await fetchImgNvidia(prompt)    ||
    await fetchImgPollinations(prompt) ||
    await fetchImgUnsplash(fallbackQuery || prompt.split(",")[0])
  );
}

/* ── Industry-tuned image prompts ── */
const ind = (industry||"").toLowerCase();
const vibe = (
  ["fashion","beauty","luxur","jewel"].some(x=>ind.includes(x)) ? "luxury editorial fashion aesthetic, warm golden lighting" :
  ["food","restaurant","cafe","bakery"].some(x=>ind.includes(x)) ? "gourmet food photography, warm moody lighting, editorial" :
  ["tech","saas","software","ai","data"].some(x=>ind.includes(x)) ? "futuristic technology abstract, dark cinematic, neon accents" :
  ["health","wellness","medic"].some(x=>ind.includes(x)) ? "clean minimal healthcare, soft natural light, calm" :
  ["sustain","eco","organic","nature"].some(x=>ind.includes(x)) ? "lush nature photography, golden hour, earthy organic" :
  ["edu","school","learn","academy"].some(x=>ind.includes(x)) ? "bright modern learning space, inspiring, clean" :
  ["realestate","property","interior"].some(x=>ind.includes(x)) ? "luxury interior design, architectural photography, premium" :
  "professional brand identity, cinematic, premium aesthetic"
);

async function fetchAllImgs() {
  const [img1, img2, img3] = await Promise.all([
    fetchImg(`${vibe}, wide establishing shot, brand hero image`, vibe),
    fetchImg(`${brand_name} ${ind||"brand"} product close up detail, ${vibe}`, ind+" detail"),
    fetchImg(`${vibe}, abstract texture background, brand atmosphere`, vibe+" texture"),
  ]);
  return { img1, img2, img3 };
}

/* ═══════════════════════════════════════════════════════════════════════════
   STEP 4 & 5 — LAYOUT PRIMITIVES
═══════════════════════════════════════════════════════════════════════════ */
let pres;

/* Core draw functions */
const R  = (sl,x,y,w,h,col,tr)   => { const c=fx(col); const o={x,y,w,h,fill:{color:c},line:{color:c,width:0}}; if(tr!==undefined)o.fill.transparency=tr; sl.addShape(pres.shapes.RECTANGLE,o); };
const Ov = (sl,x,y,w,h,col,tr,lc,lw) => { const c=fx(col); const o={x,y,w,h,fill:{color:c},line:{color:fx(lc||c),width:lw||0}}; if(tr!==undefined)o.fill.transparency=tr; sl.addShape(pres.shapes.OVAL,o); };
const Ln = (sl,x,y,w,col,h2=0.014)  => { const c=fx(col||AC); sl.addShape(pres.shapes.RECTANGLE,{x,y,w,h:h2,fill:{color:c},line:{color:c,width:0}}); };
const VL = (sl,x,y,h2,col,w2=0.014) => { const c=fx(col||AC); sl.addShape(pres.shapes.RECTANGLE,{x,y,w:w2,h:h2,fill:{color:c},line:{color:c,width:0}}); };
const Im = (sl,data,x,y,w,h) => { if(!data){R(sl,x,y,w,h,BD,0);return;} try{sl.addImage({data,x,y,w,h,sizing:{type:"cover",w,h}});}catch{R(sl,x,y,w,h,BD,0);} };

/* Gradient overlay — stepped rects to simulate gradient */
const Grad = (sl,x,y,w,h,col,from,to,dir="v") => {
  const N=10;
  for(let i=0;i<N;i++){
    const t=Math.round(from+(to-from)*(i/(N-1)));
    if(dir==="v") R(sl,x,y+h*(i/N),w,h/N+0.05,col,t);
    else          R(sl,x+w*(i/N),y,w/N+0.05,h,col,t);
  }
};

/* Text helpers */
const T = (sl,text,x,y,w,h,opts={}) => sl.addText(String(text||""), {
  x,y,w,h,
  fontSize:   opts.size||12,
  fontFace:   opts.font||(opts.serif?FH:FB),
  bold:       opts.bold||false,
  italic:     opts.italic||false,
  color:      fx(opts.color||P),
  align:      opts.align||"left",
  valign:     opts.valign||"top",
  charSpacing:opts.spacing||0,
  paraSpaceAfter: opts.gap||0,
  margin:     0,
});

/* Eyebrow label — small mono uppercase */
const Ey = (sl,text,x,y,col) => T(sl,String(text||"").toUpperCase(),x,y,W-1,0.22,{size:7,font:FB,color:col||AC,spacing:4.5});
/* Hero headline */
const Hd = (sl,text,x,y,w,h,col,size,align) => T(sl,text,x,y,w,h,{size:size||TH.headSize,serif:true,color:col||P,align:align||"left",bold:false});
/* Body copy */
const Bd = (sl,text,x,y,w,h,col,size,align) => T(sl,text,x,y,w,h,{size:size||TH.bodySize,font:FB,color:col||P,align:align||"left",gap:3});
/* Slide number */
const Num= (sl,n,col) => T(sl,String(n).padStart(2,"0"),9.1,5.15,0.7,0.25,{size:7.5,font:FB,color:col||"777777",align:"right",spacing:2});

/* Bullet list — max 5, with accent square bullet */
function Bullets(sl, items, x, y, w, maxH, col, size) {
  const list = (Array.isArray(items)?items:(String(items||"").split(/[\n·•]+/)))
    .map(s=>s.trim()).filter(Boolean).slice(0,5);
  if (!list.length) return;
  const lineH = (size||TH.bodySize) * 0.016;
  list.forEach((item,i) => {
    R(sl, x, y+i*lineH+0.035, 0.06, 0.06, AC);
    T(sl, item, x+0.14, y+i*lineH, w-0.14, lineH+0.02, {size:size||TH.bodySize, font:FB, color:fx(col||P)});
  });
}

/* Pill / tag chip */
function Pill(sl,text,x,y,col,textCol) {
  const c = fx(col||AC), tc = fx(textCol||P);
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, {x,y,w:1.3,h:0.26,rounding:0.5,fill:{color:c},line:{color:c,width:0}});
  T(sl,text.toUpperCase(),x,y,1.3,0.26,{size:7.5,font:FB,color:tc,align:"center",valign:"middle",spacing:1.5});
}

/* ═══════════════════════════════════════════════════════════════════════════
   STEP 4 — COVER LAYOUTS (industry-specific full-bleed designs)
═══════════════════════════════════════════════════════════════════════════ */

function coverLuxury(sl, img) {
  sl.background = {color:P};
  Im(sl, img, 0, 0, W, H);
  R(sl, 0, 0, W, H, P, 52);                      // dark tint
  Grad(sl, 0, H*0.35, W, H*0.65, BD, 85, 20);    // bottom fade
  VL(sl, 0, 0, H, AC);                            // left gold strip
  // Decorative oversized initial
  T(sl, brand_name[0]||"B", -0.2, -0.5, 5, 5, {size:220, serif:true, color:"FFFFFF", bold:false, spacing:-5});
  R(sl, -0.2, -0.5, 5, 5, "000000", 75);         // dim the big letter
  // Brand name
  T(sl, brand_name.toUpperCase(), 0.55, 1.0, 6, 2.2, {size:TH.headSize, serif:true, color:"FFFFFF", spacing:8});
  Ln(sl, 0.55, 3.35, 5.5, AC);
  T(sl, tagline||`The Art of ${industry||"Excellence"}`, 0.55, 3.52, 5.5, 0.5, {size:13, serif:true, italic:true, color:AC});
  Ey(sl, `${industry||"Brand"} · Reference Deck`, 0.55, 5.18, "555555");
  Num(sl, 1, "888888");
}

function coverModern(sl, img) {
  sl.background = {color:BL};
  Im(sl, img, 4.5, 0, 5.5, H);
  Grad(sl, 3.8, 0, 3.5, H, BL, 0, 100, "h");    // horizontal fade from right
  R(sl, 0, 0, 4.8, H, BL);
  R(sl, 4.8, 0, 0.014, H, AC);                   // divider
  T(sl, brand_name.toUpperCase(), 0.55, 0.9, 4.0, 2.0, {size:TH.headSize, serif:true, color:P, spacing:5});
  Ln(sl, 0.55, 3.1, 3.8, AC);
  T(sl, tagline||`${industry||"Brand"} Identity`, 0.55, 3.28, 4.0, 0.5, {size:13, serif:true, italic:true, color:AC});
  Pill(sl, tone||"Modern", 0.55, 4.2, AC, "FFFFFF");
  Ey(sl, `${industry||"Brand"} · Reference Deck`, 0.55, 5.18, "AAAAAA");
  Num(sl, 1, "AAAAAA");
}

function coverDark(sl, img) {
  sl.background = {color:"060A12"};
  Im(sl, img, 3.5, 0, 6.5, H);
  R(sl, 3.5, 0, 6.5, H, "060A12", 55);
  R(sl, 0, 0, 4.2, H, "060A12");
  VL(sl, 0, 0, H, AC);
  // Dot grid decoration top-right
  for(let r=0;r<3;r++) for(let c=0;c<6;c++)
    Ov(sl, 5.8+c*0.32, 0.35+r*0.32, 0.09, 0.09, AC, r+c>5?88:62);
  T(sl, brand_name.toUpperCase(), 0.55, 1.2, 3.5, 2.0, {size:TH.headSize-4, serif:true, color:S, spacing:4});
  Ln(sl, 0.55, 3.38, 3.2, AC);
  T(sl, tagline||"Build the Future", 0.55, 3.55, 3.5, 0.45, {size:11, font:FB, color:AC, spacing:2});
  Ey(sl, industry||"Technology", 0.55, 5.18, "444444");
  Num(sl, 1, "555555");
}

function coverWarm(sl, img) {
  sl.background = {color:BD};
  Im(sl, img, 0, 0, W, H);
  R(sl, 0, 0, W, H, BD, 48);
  Grad(sl, 0, H*0.4, W, H*0.6, BD, 80, 10);
  VL(sl, 0, 0, H, AC);
  T(sl, brand_name.toUpperCase(), 0.5, 0.7, W-0.7, 2.5, {size:TH.headSize, serif:true, color:"FFFFFF", align:"center", spacing:8});
  Ln(sl, 2.5, 3.5, 5.0, AC);
  T(sl, tagline||`Taste the Difference`, 1, 3.68, W-1.5, 0.55, {size:14, serif:true, italic:true, color:AC, align:"center"});
  Ey(sl, industry||"Food & Beverage", 0.5, 5.18, "AAAAAA");
  Num(sl, 1, "BBBBBB");
}

/* ═══════════════════════════════════════════════════════════════════════════
   SLIDE BUILDERS — Structured layouts per spec Step 4
═══════════════════════════════════════════════════════════════════════════ */

/* SPLIT layout — image fills half, text half */
function slideStory(sl, img) {
  sl.background = {color:BL};
  Im(sl, img, 0, 0, 4.5, H);
  R(sl, 0, 0, 4.5, H, BD, 45);
  R(sl, 4.5, 0, 5.5, H, BL);
  VL(sl, 4.5, 0, H, AC);
  Ey(sl, "02  ·  Brand Story", 0.3, 0.32, "888888");
  T(sl, "\u201C", 0.25, 0.5, 0.8, 1.0, {size:80, serif:true, color:AC});
  Hd(sl, "Our\nStory", 0.35, 0.75, 3.8, 1.5, "FFFFFF", 28);
  const txt = story||`${brand_name} was built on a belief — that ${industry||"the market"} deserves something genuinely different. We blend craft with intention.`;
  T(sl, txt, 4.75, 0.55, 4.9, 2.5, {size:13, serif:true, italic:true, color:P, gap:6});
  Ln(sl, 4.75, 3.25, 4.9, AC);
  Ey(sl, `${industry||"Brand"}  ·  ${tone||"Premium"}`, 4.75, 3.42, P);
  Num(sl, 2, "999999");
}

/* CARDS layout — 3 equal-width cards with accent tops */
function slideCards(sl, title, eyebrow, items, n, bgCol) {
  sl.background = {color:bgCol||BL};
  Ey(sl, eyebrow, 0.5, 0.28, AC);
  Ln(sl, 0.5, 0.58, W-0.7, AC);
  Hd(sl, title, 0.5, 0.72, 7, 0.9, bgCol===P?"FFFFFF":P, 28);
  items.slice(0,3).forEach((item,i) => {
    const x = 0.5+i*3.15;
    const dk = bgCol===P || i===1;
    R(sl, x, 1.82, 2.95, 3.5, dk?BD:P);
    R(sl, x, 1.82, 2.95, 0.08, AC);             // accent top strip
    T(sl, item.icon||item.num||`0${i+1}`, x+0.22, 2.05, 2.5, 0.7, {size:28, serif:true, color:AC});
    Ln(sl, x+0.22, 2.85, 2.5, AC);
    T(sl, (item.title||"").toUpperCase(), x, 3.05, 2.95, 0.35, {size:9.5, font:FB, color:dk?"EEEEEE":S, align:"center", spacing:1.5});
    Bd(sl, item.desc||item.sub||"", x+0.12, 3.5, 2.71, 1.5, dk?"AAAAAA":"888888", 9.5, "center");
  });
  Num(sl, n, "999999");
}

/* MINIMAL layout — centred, strong typographic hierarchy */
function slideMinimal(sl, eyebrow, title, subtitle, n, onImg, img) {
  sl.background = {color:onImg?BD:BL};
  if(img && onImg){ Im(sl, img, 0, 0, W, H); R(sl, 0, 0, W, H, BD, 65); }
  Ey(sl, eyebrow, W/2-3.5, 0.38, AC);
  T(sl, eyebrow.toUpperCase(), W/2-3.5, 0.38, 7, 0.22, {size:7, font:FB, color:AC, spacing:4.5, align:"center"});
  Hd(sl, title, 0.7, 0.85, W-1.4, 1.8, onImg?"FFFFFF":P, TH.headSize, "center");
  Ln(sl, W/2-2.5, 2.8, 5.0, AC);
  T(sl, subtitle, 0.7, 2.96, W-1.4, 0.65, {size:14, serif:true, italic:true, color:onImg?AC:AC, align:"center"});
  Num(sl, n, onImg?"666666":"AAAAAA");
}

/* SPLIT-RIGHT-IMAGE — text left, image right (or reverse) */
function slideSplitRight(sl, eyebrow, title, bullets, img, n, flip) {
  sl.background = {color:P};
  const tx = flip?5.1:0.55, ty = 0, tw = flip?4.45:4.35;
  const ix = flip?0:4.55, iw = flip?4.55:5.45;
  Im(sl, img, ix, 0, iw, H);
  R(sl, ix, 0, iw, H, BD, 42);
  R(sl, tx, 0, tw, H, P);
  if(flip) VL(sl, 4.55, 0, H, AC);
  else     VL(sl, 4.55, 0, H, AC);
  Ey(sl, eyebrow, tx+0.22, 0.34, AC);
  Ln(sl, tx+0.22, 0.62, tw-0.22, AC);
  Hd(sl, title, tx+0.22, 0.78, tw-0.3, 1.55, S, 26);
  Bullets(sl, bullets, tx+0.22, 2.55, tw-0.3, 2.5, "BBBBBB", TH.bodySize+0.5);
  Num(sl, n, "555555");
}

/* STATS layout — large number statements */
function slideStats(sl, eyebrow, title, stats, n) {
  sl.background = {color:BD};
  Ey(sl, eyebrow, 0.5, 0.28, AC);
  Ln(sl, 0.5, 0.58, W-0.7, AC);
  Hd(sl, title, 0.5, 0.72, 7, 0.9, S, 28);
  stats.slice(0,4).forEach((s,i) => {
    const x = 0.5+i*2.38, y = 1.88;
    R(sl, x, y, 2.2, 3.1, "0D1117");
    R(sl, x, y, 2.2, 0.065, AC);
    T(sl, s.value||"—", x, y+0.18, 2.2, 1.0, {size:36, serif:true, color:AC, align:"center"});
    Ln(sl, x+0.3, y+1.28, 1.6, AC);
    T(sl, (s.label||"").toUpperCase(), x, y+1.44, 2.2, 0.3, {size:8, font:FB, color:"CCCCCC", align:"center", spacing:1.5});
    Bd(sl, s.sub||"", x+0.1, y+1.82, 2.0, 1.0, "777777", 8.5, "center");
  });
  Num(sl, n, "555555");
}

/* TIMELINE layout — horizontal dots */
function slideTimeline(sl, eyebrow, title, img, phases, n) {
  sl.background = {color:BD};
  Im(sl, img, 0, 0, W, H);
  R(sl, 0, 0, W, H, BD, 68);
  VL(sl, W-0.1, 0, H, AC);
  Ey(sl, eyebrow, 0.55, 0.3, AC);
  Ln(sl, 0.55, 0.6, W-0.8, AC);
  T(sl, title, 0.55, 0.76, W-0.9, 1.45, {size:15, serif:true, italic:true, color:"FFFFFF", gap:5});
  Ln(sl, 0.55, 2.42, W-0.8, AC);
  phases.slice(0,3).forEach((ph,i) => {
    const x = 0.55+i*3.12;
    if(i<2) R(sl, x+1.58, 2.85, 1.54, 0.012, "444444");
    Ov(sl, x+1.14, 2.62, 0.48, 0.48, i===0?AC:"333333", 0, AC, 0.5);
    Ey(sl, ph.label||`Year ${i+1}`, x, 3.32, "888888");
    T(sl, (ph.title||"").toUpperCase(), x, 3.58, 2.95, 0.32, {size:11, font:FB, color:"FFFFFF", spacing:1});
    Bd(sl, ph.desc||"", x, 3.96, 2.95, 0.88, "888888", 9.5);
  });
  Num(sl, n, "555555");
}

/* COLOUR PALETTE slide */
function slidePalette(sl, n) {
  sl.background = {color:P};
  Ey(sl, `0${n}  ·  Brand Identity`, 0.5, 0.28, AC);
  Ln(sl, 0.5, 0.58, W-0.7, AC);
  Hd(sl, "Colours  \u00B7  Type  \u00B7  Style", 0.5, 0.74, 7.5, 0.85, S, 26);
  const swatches = [{col:P,hex:primary,role:"Primary"},{col:S,hex:secondary,role:"Secondary"},{col:AC,hex:accent,role:"Accent"},{col:BD,hex:bg_dark,role:"Dark BG"},{col:BL,hex:bg_light,role:"Light BG"}];
  swatches.forEach((sw,i) => {
    const x = 0.5+i*1.85;
    sl.addShape(pres.shapes.RECTANGLE, {x,y:1.72,w:1.75,h:2.2,fill:{color:fx(sw.col)},line:{color:"333333",width:0.3}});
    Bd(sl, `#${String(sw.hex||"").toUpperCase()}`, x, 3.98, 1.75, 0.28, "888888", 7, "center");
    T(sl, sw.role.toUpperCase(), x, 4.3, 1.75, 0.28, {size:7.5, font:FB, color:AC, align:"center", spacing:1});
  });
  Ln(sl, 0.5, 4.72, W-0.7, AC);
  Bd(sl, `${FH}  \u00B7  ${FB}`, 0.5, 4.88, 5, 0.28, "888888", 8.5);
  [tone, (ind.split(",")[0]||"").trim()||"Crafted", "Premium"].forEach((m,i) => {
    const x = 6.1+i*1.32;
    R(sl, x, 4.85, 1.22, 0.34, i===0?AC:"333333");
    Bd(sl, m.charAt(0).toUpperCase()+m.slice(1), x, 4.85, 1.22, 0.34, i===0?P:"AAAAAA", 8.5, "center");
  });
  Num(sl, n, "555555");
}

/* CLOSING slide */
function slideClosing(sl, n) {
  sl.background = {color:BL};
  R(sl, 0, 0, W, 0.6, P);
  R(sl, 0, H-0.6, W, 0.6, P);
  T(sl, brand_name.toUpperCase(), 0.5, 0.88, W-0.7, 1.95, {size:68, serif:true, color:P, align:"center", spacing:10});
  Ln(sl, 2.0, 3.0, 6.0, AC);
  T(sl, tagline||`The Art of ${industry||"Excellence"}`, 1.0, 3.16, W-1.5, 0.58, {size:15, serif:true, italic:true, color:AC, align:"center"});
  Ln(sl, 2.0, 3.85, 6.0, AC);
  Bd(sl, "Let\u2019s build something remarkable together.", 1.0, 4.02, W-1.5, 0.38, "666666", 10.5, "center");
  T(sl, "\u26A0  REFERENCE DECK ONLY \u00B7 Brand visualization aid — share with a designer for the final polished version.",
    0.6, H-0.52, 8.8, 0.38, {size:6.5, font:FB, color:"AAAAAA", align:"center"});
  T(sl, String(n).padStart(2,"0"), 9.1, H-0.55, 0.7, 0.22, {size:8, font:FB, color:P, align:"right", spacing:2});
}

/* ═══════════════════════════════════════════════════════════════════════════
   STEP 1 — STRUCTURED SLIDE SCHEMA
   Build the 10-slide content plan using brand data
═══════════════════════════════════════════════════════════════════════════ */
function buildSlideSchema() {
  const probItems = (problem||"")
    .split(/[\n.]+/).map(s=>s.trim()).filter(Boolean).slice(0,5);
  if(probItems.length<3) {
    probItems.push(...[
      `The ${ind||"market"} lacks authenticity — customers can't find brands they truly trust`,
      `Existing options force compromise on quality, values, or both`,
      `${brand_name} exists to close this gap with intention and craft`,
    ].slice(0, 3-probItems.length));
  }
  const solBullets = (solution||`${brand_name} delivers a premium, purpose-built experience for ${audience||"discerning customers"}`)
    .split(/[\n.]+/).map(s=>s.trim()).filter(s=>s.length>8).slice(0,4);
  const marketingBullets = (marketing||"Community-first storytelling. Strategic brand partnerships. Targeted digital presence.")
    .split(/[.\n]+/).map(s=>s.trim()).filter(Boolean).slice(0,4);
  return {
    cover:    { title:brand_name, subtitle:tagline||`${ind} · Premium`, mood:tone },
    story:    { text: story },
    problem:  { title:"What Needs\nto Change", bullets:probItems },
    solution: { title:"How We\nSolve It",    bullets:solBullets },
    product:  { title:"What We\nOffer",       text: product||`${brand_name} delivers a carefully considered experience — premium quality, authentic purpose.` },
    audience: {
      title:"Our Audience",
      segments:[
        { num:"60%", title:"Primary",   desc:audience||"Early adopters who prioritise quality and authentic brand storytelling" },
        { num:"30%", title:"Secondary", desc:`Established voices in ${ind||"the space"} seeking credible fresh perspectives`},
        { num:"10%", title:"Emerging",  desc:"New customers discovering the category for the first time" },
      ],
    },
    marketing:{ title:"How We\nReach Them", bullets:marketingBullets },
    palette:  { },
    roadmap:  {
      title: vision||`${brand_name} will become the defining name in ${ind||"its category"}.`,
      phases:[
        { label:"Year 1", title:"Launch",  desc:"MVP · first customers · product-market fit" },
        { label:"Year 2", title:"Scale",   desc:"Expand team · channels · partnerships" },
        { label:"Year 3", title:"Lead",    desc:"Category leadership · international growth" },
      ],
    },
    closing: { },
  };
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN BUILD
═══════════════════════════════════════════════════════════════════════════ */
async function main() {
  pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.title  = `${brand_name} — Reference Pitch Deck`;
  pres.author = "BrandCraft AI";

  const imgs   = await fetchAllImgs();
  const schema = buildSlideSchema();

  /* ── Pick cover variant by industry/theme ── */
  const indKey = (industry||"").toLowerCase();
  const useDark     = ["tech","saas","software","fintech","ai","data","gaming"].some(x=>indKey.includes(x));
  const useWarm     = ["food","restaurant","cafe","bakery","bar"].some(x=>indKey.includes(x));
  const useModern   = ["health","wellness","edu","school","sustain","eco"].some(x=>indKey.includes(x));
  const useLuxury   = !useDark && !useWarm && !useModern;

  /* 1. COVER */
  const s1 = pres.addSlide();
  if      (useDark)   coverDark  (s1, imgs.img1);
  else if (useWarm)   coverWarm  (s1, imgs.img1);
  else if (useModern) coverModern(s1, imgs.img1);
  else                coverLuxury(s1, imgs.img1);

  /* 2. STORY — SPLIT */
  const s2 = pres.addSlide();
  slideStory(s2, imgs.img2);

  /* 3. PROBLEM — SPLIT RIGHT IMAGE with bullets */
  const s3 = pres.addSlide();
  slideSplitRight(s3,
    "03  ·  The Problem We Solve",
    schema.problem.title,
    schema.problem.bullets,
    imgs.img3, 3, false
  );

  /* 4. SOLUTION — CARDS */
  const s4 = pres.addSlide();
  slideCards(s4,
    schema.solution.title,
    "04  ·  Our Solution",
    [
      { icon:"◎", title:"Authentic",    desc:"Honest at every touchpoint — no compromise on values" },
      { icon:"◈", title:"Intentional",  desc:"Every detail serves a deliberate purpose" },
      { icon:"◉", title:"Enduring",     desc:"Built to last, not follow trends" },
    ], 4, P
  );

  /* 5. PRODUCT — SPLIT LEFT */
  const s5 = pres.addSlide();
  slideSplitRight(s5,
    "05  ·  Product & Service",
    schema.product.title,
    schema.product.text.split(/[.\n]+/).map(s=>s.trim()).filter(s=>s.length>8).slice(0,4),
    imgs.img2, 5, true
  );

  /* 6. AUDIENCE — CARDS with % stats */
  const s6 = pres.addSlide();
  slideStats(s6,
    "06  ·  Who We Serve",
    "Our Audience",
    [
      { value:schema.audience.segments[0].num, label:"Primary Target",    sub:schema.audience.segments[0].desc },
      { value:schema.audience.segments[1].num, label:"Secondary Segment", sub:schema.audience.segments[1].desc },
      { value:schema.audience.segments[2].num, label:"Emerging",          sub:schema.audience.segments[2].desc },
      { value: "1×",                           label:"Market Position",   sub:`Defining the ${ind||"category"} narrative` },
    ], 6
  );

  /* 7. MARKETING — CARDS */
  const s7 = pres.addSlide();
  slideCards(s7,
    schema.marketing.title,
    "07  ·  Go-to-Market",
    [
      { num:"01", title:"Digital",    desc:"Social, SEO, content & paid media campaigns" },
      { num:"02", title:"Community",  desc:"Events, collaborations & word-of-mouth" },
      { num:"03", title:"Direct",     desc:"Retail, pop-ups & direct outreach" },
    ], 7, BL
  );

  /* 8. BRAND IDENTITY — PALETTE SLIDE */
  const s8 = pres.addSlide();
  slidePalette(s8, 8);

  /* 9. VISION / ROADMAP — TIMELINE on full-bleed image */
  const s9 = pres.addSlide();
  slideTimeline(s9,
    "09  ·  The Road Ahead",
    schema.roadmap.title,
    imgs.img1, schema.roadmap.phases, 9
  );

  /* 10. CLOSING — MINIMAL centred */
  const s10 = pres.addSlide();
  slideClosing(s10, 10);

  const b64 = await pres.write({ outputType:"base64" });
  process.stdout.write(b64);
}

main().catch(e => { process.stderr.write(e.message||String(e)); process.exit(1); });