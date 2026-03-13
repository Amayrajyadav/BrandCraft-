/**
 * BrandCraft — Deck Generator v4
 * Gamma + Behance style: full-bleed images, card overlays, industry-specific layouts
 * Input: JSON via process.argv[2]    Output: base64 PPTX to stdout
 */
const pptxgen = require("pptxgenjs");
const https   = require("https");
const http    = require("http");

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

const fix = h => String(h || "").replace(/^#/, "").trim() || "222222";
const P  = fix(primary);
const S  = fix(secondary);
const AC = fix(accent);
const BD = fix(bg_dark);
const BL = fix(bg_light);
const W = 10, H = 5.625;

/* ─── INDUSTRY MAP ────────────────────────────────────────────────────────── */
const IND = {
  fashion:        { imgs: ["fashion editorial luxury","runway model minimal","fabric texture close"],     layout:"editorial",  vibe:"luxury editorial"   },
  food:           { imgs: ["gourmet food photography","fresh ingredients overhead","restaurant interior"], layout:"warm",       vibe:"warm artisan"       },
  technology:     { imgs: ["modern tech office","code dark screen","startup product minimal"],            layout:"dark_tech",  vibe:"futuristic minimal"  },
  fintech:        { imgs: ["fintech data visualization","financial technology","mobile payment"],         layout:"dark_tech",  vibe:"trustworthy modern"  },
  healthcare:     { imgs: ["modern healthcare clean","wellness nature calm","medical technology white"],  layout:"clean_light",vibe:"calm trustworthy"    },
  sustainability: { imgs: ["green nature sunlight","eco sustainable farm","recycled materials texture"],  layout:"nature",     vibe:"earthy purposeful"   },
  beauty:         { imgs: ["luxury beauty cosmetics","skincare minimal white","floral macro close"],      layout:"editorial",  vibe:"elegant refined"     },
  education:      { imgs: ["modern classroom bright","student studying focused","open book light"],       layout:"clean_light",vibe:"inspiring accessible" },
  realestate:     { imgs: ["luxury real estate interior","modern architecture exterior","city skyline"],  layout:"editorial",  vibe:"premium confident"   },
  saas:           { imgs: ["saas dashboard screen","developer laptop dark","product ui interface"],       layout:"dark_tech",  vibe:"efficient modern"    },
  restaurant:     { imgs: ["restaurant interior warm","chef plating food","ingredients table overhead"],  layout:"warm",       vibe:"warm inviting"       },
  wellness:       { imgs: ["yoga wellness calm","nature meditation","clean organic lifestyle"],           layout:"nature",     vibe:"serene holistic"     },
};

function getInd() {
  const k = (industry||"").toLowerCase();
  for (const [key,val] of Object.entries(IND)) if (k.includes(key)) return val;
  return { imgs: ["brand identity creative","modern business workspace","team collaboration office"], layout:"editorial", vibe:"professional modern" };
}
const IND_CFG = getInd();

/* ─── IMAGE FETCH ─────────────────────────────────────────────────────────── */
function fetchImg(query, w=1280, h=720) {
  const url = `https://source.unsplash.com/featured/${w}x${h}/?${encodeURIComponent(query)}&t=${Date.now()}`;
  return new Promise(resolve => {
    function get(u, depth=0) {
      if (depth > 5) return resolve(null);
      const mod = u.startsWith("https") ? https : http;
      mod.get(u, { headers:{"User-Agent":"Mozilla/5.0"}, timeout:15000 }, res => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          return get(res.headers.location, depth+1);
        }
        if (res.statusCode !== 200) return resolve(null);
        const chunks = [];
        res.on("data", c => chunks.push(c));
        res.on("end", () => {
          const ct = res.headers["content-type"] || "image/jpeg";
          resolve(`${ct};base64,${Buffer.concat(chunks).toString("base64")}`);
        });
        res.on("error", () => resolve(null));
      }).on("error", () => resolve(null)).on("timeout", () => resolve(null));
    }
    get(url);
  });
}

async function fetchAll() {
  const [img1, img2, img3] = await Promise.all(IND_CFG.imgs.map(q => fetchImg(q)));
  return { img1: img1||null, img2: img2||null, img3: img3||null };
}

/* ─── HELPERS ─────────────────────────────────────────────────────────────── */
let pres;

function mkRect(sl, x, y, w, h, color, transp) {
  const col = fix(color);
  const o = { x, y, w, h, fill:{color:col}, line:{color:col,width:0} };
  if (transp !== undefined) o.fill.transparency = transp;
  sl.addShape(pres.shapes.RECTANGLE, o);
}
function mkLine(sl, x, y, w, color) {
  const col = fix(color||AC);
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w, h:0.014, fill:{color:col}, line:{color:col,width:0} });
}
function mkVLine(sl, x, y, h, color) {
  const col = fix(color||AC);
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w:0.014, h, fill:{color:col}, line:{color:col,width:0} });
}
function mkImg(sl, data, x, y, w, h) {
  if (!data) { mkRect(sl,x,y,w,h,P,0); return; }
  try { sl.addImage({ data, x, y, w, h, sizing:{type:"cover",w,h} }); } catch(e) { mkRect(sl,x,y,w,h,P,0); }
}
function mkEyebrow(sl, text, x, y, w, color) {
  sl.addText(text.toUpperCase(), { x, y, w:w||8, h:0.22, fontSize:7, fontFace:font_body, color:fix(color||AC), charSpacing:5, align:"left", margin:0 });
}
function mkHead(sl, text, x, y, w, h, color, size, align) {
  sl.addText(text, { x, y, w, h, fontSize:size||32, fontFace:font_heading, bold:false, color:fix(color||P), align:align||"left", valign:"top", margin:0 });
}
function mkBody(sl, text, x, y, w, h, color, size, align) {
  sl.addText(text, { x, y, w, h, fontSize:size||11, fontFace:font_body, color:fix(color||P), align:align||"left", valign:"top", paraSpaceAfter:3, margin:0 });
}
function mkNum(sl, n, color) {
  sl.addText(String(n).padStart(2,"0"), { x:9.1, y:5.15, w:0.7, h:0.25, fontSize:7.5, fontFace:font_body, color:fix(color||"777777"), align:"right", charSpacing:2, margin:0 });
}
// frosted card (semi-transparent overlay)
function mkCard(sl, x, y, w, h, dark) {
  mkRect(sl,x,y,w,h, dark?"000000":"FFFFFF", dark?55:20);
}

/* ─── LAYOUT STRATEGIES ───────────────────────────────────────────────────── */
// Each layout produces different slide structures per industry

function buildEditorial(imgs) {
  // FASHION / BEAUTY / REAL-ESTATE: dramatic editorial splits, thin rules, serif heavy
  const sl1 = pres.addSlide();
  // SLIDE 1 — COVER
  sl1.background = { color: BD };
  mkImg(sl1, imgs.img1, 0, 0, W, H);
  mkRect(sl1, 0, 0, W, H, BD, 55);
  mkRect(sl1, 0, 0, 0.1, H, AC); // left gold strip
  mkRect(sl1, 0, 0, 4.8, H, BD, 40); // left dark overlay
  sl1.addText(brand_name.toUpperCase(), { x:0.35, y:0.7, w:4.4, h:2.8, fontSize:58, fontFace:font_heading, bold:false, color:fix(S), align:"left", charSpacing:8, margin:0 });
  mkLine(sl1, 0.35, 3.6, 3.8, AC);
  sl1.addText(tagline||`The Art of ${industry}`, { x:0.35, y:3.78, w:4.4, h:0.55, fontSize:13, fontFace:font_heading, italic:true, color:fix(AC), margin:0 });
  mkEyebrow(sl1, industry||"Brand Identity", 0.35, 5.2, 4.0, "555555");
  mkNum(sl1, 1, S);

  // SLIDE 2 — STORY (split: image left, text card right)
  const sl2 = pres.addSlide();
  sl2.background = { color: BL };
  mkImg(sl2, imgs.img1, 0, 0, 4.6, H);
  mkRect(sl2, 0, 0, 4.6, H, BD, 45);
  mkRect(sl2, 4.6, 0, 5.4, H, BL);
  mkEyebrow(sl2, "Brand Story", 0.3, 0.32, 4.0, AC);
  sl2.addText("\u201C", { x:0.25, y:0.55, w:0.9, h:1.1, fontSize:80, fontFace:font_heading, color:fix(AC), margin:0 });
  mkHead(sl2, "Our Story", 0.35, 0.82, 4.0, 0.9, "FFFFFF", 28);
  mkEyebrow(sl2, "02  ·  Brand Story", 4.9, 0.32, 4.8, AC);
  mkLine(sl2, 4.9, 0.62, 4.8, AC);
  const storyTxt = story || `${brand_name} was born from a passion for ${IND_CFG.vibe} — where every detail is intentional and every customer feels it.`;
  sl2.addText(storyTxt, { x:4.9, y:0.82, w:4.8, h:2.6, fontSize:13, fontFace:font_heading, italic:true, color:fix(P), align:"left", valign:"top", paraSpaceAfter:7, margin:0 });
  mkLine(sl2, 4.9, 3.55, 4.8, AC);
  mkBody(sl2, `Industry: ${industry||"Brand"}`, 4.9, 3.72, 2.2, 0.28, AC, 8.5);
  mkBody(sl2, `Tone: ${tone}`, 4.9, 4.02, 2.2, 0.28, "777777", 8.5);
  mkBody(sl2, `For: ${audience||"Discerning consumers"}`, 4.9, 4.32, 4.6, 0.45, P, 10);
  mkNum(sl2, 2, "999999");

  return [sl1, sl2];
}

function buildWarm(imgs) {
  // FOOD / RESTAURANT: warm tones, image-dominant, text on dark card overlays
  const sl1 = pres.addSlide();
  sl1.background = { color: BD };
  mkImg(sl1, imgs.img1, 0, 0, W, H);
  mkRect(sl1, 0, 0, W, H, BD, 45);
  mkRect(sl1, 0, H*0.6, W, H*0.4, BD, 70); // bottom gradient
  sl1.addText(brand_name.toUpperCase(), { x:0.6, y:0.5, w:8.8, h:2.6, fontSize:60, fontFace:font_heading, bold:false, color:"FFFFFF", align:"center", charSpacing:6, margin:0 });
  mkLine(sl1, 2.5, 3.2, 5.0, AC);
  sl1.addText(tagline||`Taste the Difference`, { x:1.0, y:3.38, w:8.0, h:0.55, fontSize:14, fontFace:font_heading, italic:true, color:fix(AC), align:"center", margin:0 });
  mkRect(sl1, 0, 0, 0.1, H, AC);
  mkEyebrow(sl1, industry||"Food & Beverage", 0.35, 5.22, 9.0, "888888");
  mkNum(sl1, 1, S);

  const sl2 = pres.addSlide();
  sl2.background = { color: BL };
  mkImg(sl2, imgs.img2||imgs.img1, 5.2, 0, 4.8, H);
  mkRect(sl2, 5.2, 0, 4.8, H, BD, 40);
  mkRect(sl2, 0, 0, 5.2, H, P);
  mkEyebrow(sl2, "02  ·  Our Story", 0.5, 0.35, 4.4, AC);
  mkLine(sl2, 0.5, 0.65, 4.4, AC);
  mkHead(sl2, "Our\nStory", 0.5, 0.85, 4.4, 1.6, S, 34);
  const storyTxt = story || `${brand_name} started with a simple idea — that great ${industry||"food"} should be accessible, honest, and made with love. Every recipe tells a story.`;
  sl2.addText(storyTxt, { x:0.5, y:2.65, w:4.4, h:1.8, fontSize:12.5, fontFace:font_heading, italic:true, color:"CCCCCC", align:"left", valign:"top", paraSpaceAfter:6, margin:0 });
  mkLine(sl2, 0.5, 4.6, 4.4, AC);
  mkBody(sl2, `Est. ${new Date().getFullYear()}  ·  ${industry||"Food"}  ·  ${tone} tone`, 0.5, 4.78, 4.4, 0.3, "888888", 8.5);
  mkNum(sl2, 2, "555555");

  return [sl1, sl2];
}

function buildDarkTech(imgs) {
  // TECH / FINTECH / SAAS: dark, minimal, data-forward
  const sl1 = pres.addSlide();
  sl1.background = { color: "080C14" };
  mkImg(sl1, imgs.img1, 4.0, 0, 6.0, H);
  mkRect(sl1, 4.0, 0, 6.0, H, "080C14", 55);
  mkRect(sl1, 0, 0, 4.0, H, "080C14");
  mkRect(sl1, 0, 0, 0.1, H, AC);
  sl1.addText(brand_name.toUpperCase(), { x:0.3, y:0.9, w:3.5, h:2.4, fontSize:48, fontFace:font_heading, bold:false, color:fix(S), align:"left", charSpacing:4, margin:0 });
  mkLine(sl1, 0.3, 3.45, 3.4, AC);
  sl1.addText(tagline||`Build the Future`, { x:0.3, y:3.62, w:3.6, h:0.52, fontSize:12, fontFace:font_body, color:fix(AC), margin:0 });
  mkBody(sl1, industry||"Technology", 0.3, 4.6, 3.4, 0.3, "444444", 8.5);
  mkNum(sl1, 1, "555555");

  const sl2 = pres.addSlide();
  sl2.background = { color: BD };
  mkImg(sl2, imgs.img2||imgs.img1, 0, 0, W, 3.0);
  mkRect(sl2, 0, 0, W, 3.0, BD, 60);
  mkEyebrow(sl2, "02  ·  Brand Story", 0.55, 0.28, 8, AC);
  mkHead(sl2, "Our Story", 0.55, 0.58, 7, 1.0, "FFFFFF", 30);
  const storyTxt = story || `${brand_name} was built to solve what others ignored — creating technology that is elegant, powerful, and built around the real needs of ${audience||"modern teams"}.`;
  mkRect(sl2, 0.4, 3.1, 9.2, 2.3, "0D1117");
  mkLine(sl2, 0.55, 3.22, 0.4, AC);
  sl2.addText(storyTxt, { x:0.65, y:3.15, w:8.7, h:1.8, fontSize:12.5, fontFace:font_heading, italic:true, color:"CCCCCC", align:"left", valign:"top", paraSpaceAfter:6, margin:0 });
  mkBody(sl2, `${industry||"Tech"}  ·  ${tone}  ·  For: ${audience||"innovators"}`, 0.55, 5.18, 8.7, 0.25, "555555", 7.5);
  mkNum(sl2, 2, "555555");

  return [sl1, sl2];
}

function buildCleanLight(imgs) {
  // HEALTHCARE / EDUCATION: clean, light, airy
  const sl1 = pres.addSlide();
  sl1.background = { color: "FFFFFF" };
  mkImg(sl1, imgs.img1, 5.5, 0, 4.5, H);
  mkRect(sl1, 5.5, 0, 4.5, H, S, 60);
  mkRect(sl1, 0, 0, 5.5, H, "FFFFFF");
  mkRect(sl1, 0, 0, 0.1, H, AC);
  sl1.addText(brand_name.toUpperCase(), { x:0.35, y:0.9, w:5.0, h:2.2, fontSize:50, fontFace:font_heading, bold:false, color:fix(P), align:"left", charSpacing:5, margin:0 });
  mkLine(sl1, 0.35, 3.22, 4.6, AC);
  sl1.addText(tagline||`Care That Matters`, { x:0.35, y:3.4, w:5.0, h:0.5, fontSize:13, fontFace:font_heading, italic:true, color:fix(AC), margin:0 });
  mkBody(sl1, industry||"Healthcare & Wellness", 0.35, 5.22, 4.5, 0.28, "AAAAAA", 8);
  mkNum(sl1, 1, "AAAAAA");

  const sl2 = pres.addSlide();
  sl2.background = { color: BL };
  mkImg(sl2, imgs.img2||imgs.img1, 6.2, 0.6, 3.5, 4.5);
  mkRect(sl2, 6.2, 0.6, 3.5, 4.5, S, 50);
  mkEyebrow(sl2, "02  ·  Our Story", 0.5, 0.35, 5.4, AC);
  mkLine(sl2, 0.5, 0.65, 5.4, AC);
  mkHead(sl2, "Our Story", 0.5, 0.85, 5.4, 0.95, P, 30);
  const storyTxt = story || `${brand_name} was founded on a commitment to making ${industry||"healthcare"} more human — where every interaction is built around empathy, clarity, and genuine care.`;
  sl2.addText(storyTxt, { x:0.5, y:2.0, w:5.4, h:2.2, fontSize:13, fontFace:font_heading, italic:true, color:fix(P), align:"left", valign:"top", paraSpaceAfter:7, margin:0 });
  mkLine(sl2, 0.5, 4.35, 5.4, AC);
  mkBody(sl2, `${industry||"Healthcare"}  ·  ${tone}  ·  For: ${audience||"those who care"}`, 0.5, 4.52, 5.2, 0.3, "888888", 8.5);
  mkNum(sl2, 2, "AAAAAA");

  return [sl1, sl2];
}

function buildNature(imgs) {
  // SUSTAINABILITY / WELLNESS: organic, earthy, textural
  const sl1 = pres.addSlide();
  sl1.background = { color: BD };
  mkImg(sl1, imgs.img1, 0, 0, W, H);
  mkRect(sl1, 0, 0, W, H, BD, 50);
  mkRect(sl1, 0, H*0.55, W, H*0.45, BD, 75);
  mkRect(sl1, 0, 0, 0.1, H, AC);
  sl1.addText(brand_name.toUpperCase(), { x:0.5, y:1.2, w:9.0, h:2.0, fontSize:62, fontFace:font_heading, bold:false, color:"FFFFFF", align:"center", charSpacing:10, margin:0 });
  mkLine(sl1, 2.5, 3.35, 5.0, AC);
  sl1.addText(tagline||`For the Planet & People`, { x:1.0, y:3.55, w:8.0, h:0.55, fontSize:13, fontFace:font_heading, italic:true, color:fix(AC), align:"center", margin:0 });
  mkEyebrow(sl1, industry||"Sustainability", 0.5, 5.22, 9.0, "777777");
  mkNum(sl1, 1, "888888");

  const sl2 = pres.addSlide();
  sl2.background = { color: BL };
  mkImg(sl2, imgs.img2||imgs.img1, 0, 0, W, 3.2);
  mkRect(sl2, 0, 0, W, 3.2, BD, 55);
  mkEyebrow(sl2, "02  ·  Our Story", 0.55, 0.28, 8, AC);
  mkHead(sl2, "Our Story", 0.55, 0.58, 7, 1.0, "FFFFFF", 30);
  mkRect(sl2, 0.4, 3.35, 9.2, 2.0, "FFFFFF");
  mkLine(sl2, 0.55, 3.45, 0.35, AC);
  const storyTxt = story || `${brand_name} exists because the world needs better choices. We believe every purchase is a vote for the future — and we're here to make the right choice the beautiful one.`;
  sl2.addText(storyTxt, { x:0.65, y:3.42, w:8.7, h:1.7, fontSize:12.5, fontFace:font_heading, italic:true, color:fix(P), align:"left", valign:"top", paraSpaceAfter:6, margin:0 });
  mkNum(sl2, 2, "999999");

  return [sl1, sl2];
}

/* ─── SHARED SLIDES 3–10 (styled per layout) ─────────────────────────────── */
async function buildSharedSlides(imgs) {
  // SLIDE 3 — PROBLEM
  const s3 = pres.addSlide();
  s3.background = { color: P };
  mkRect(s3, 0, 0, W, 0.65, AC);
  mkEyebrow(s3, "03  ·  The Problem", 0.5, 0.2, 9, P);
  s3.addText("03", { x:7.4, y:0.55, w:2.4, h:2.0, fontSize:100, fontFace:font_heading, color:"1A1A1A", align:"right", margin:0 });
  mkHead(s3, "What Needs\nto Change", 0.5, 0.82, 5.5, 1.5, S, 30);
  mkLine(s3, 0.5, 2.55, W-0.8, AC);
  const defs = [`The ${industry||"market"} lacks truly authentic and differentiated options`,`Customers compromise on quality or values — no brand truly gets them`,"There's a clear gap that ${brand_name} is uniquely positioned to fill"];
  let probs = problem ? problem.split(/[.\n]/).map(x=>x.trim()).filter(x=>x.length>10) : [];
  while(probs.length<3) probs.push(defs[probs.length]||defs[2]);
  probs.slice(0,3).forEach((p,i)=>{
    const y = 2.75+i*0.92;
    s3.addText(`0${i+1}`, {x:0.5,y,w:0.62,h:0.65,fontSize:22,fontFace:font_heading,color:fix(AC),align:"left",margin:0});
    mkVLine(s3, 1.25, y+0.08, 0.48, AC);
    mkBody(s3, p.trim(), 1.42, y+0.07, 8.2, 0.65, S, 11.5);
  });
  mkNum(s3, 3, "555555");

  // SLIDE 4 — SOLUTION
  const s4 = pres.addSlide();
  s4.background = { color: BL };
  mkImg(s4, imgs.img2||imgs.img1, 0, 0, 4.4, H);
  mkRect(s4, 0, 0, 4.4, H, BD, 50);
  mkEyebrow(s4, "04  ·  Solution", 0.35, 0.35, 3.7, AC);
  mkLine(s4, 0.35, 0.65, 3.4, AC);
  mkHead(s4, "How We\nSolve It", 0.35, 0.85, 3.8, 1.4, "FFFFFF", 28);
  const solTxt = solution || `${brand_name} delivers an experience that is intentional, high-quality, and deeply aligned with what ${audience||"our audience"} actually cares about.`;
  mkBody(s4, solTxt, 4.7, 0.55, 5.0, 1.45, P, 12.5);
  mkLine(s4, 4.7, 2.15, 5.0, AC);
  const pillars = ["Authentic","Intentional","Enduring"];
  pillars.forEach((p,i)=>{
    const x = 4.7+i*1.72;
    mkRect(s4, x, 2.38, 1.58, 1.95, i===1?P:"EEEADF");
    mkRect(s4, x, 2.38, 1.58, 0.07, AC);
    mkBody(s4, `0${i+1}`, x+0.12, 2.52, 1.35, 0.38, AC, 11);
    mkLine(s4, x+0.12, 2.95, 1.2, AC);
    mkBody(s4, p.toUpperCase(), x, 3.12, 1.58, 0.32, i===1?S:P, 9.5, "center");
    mkBody(s4, IND_CFG.vibe.split(" ")[i]||p, x+0.1, 3.5, 1.38, 0.55, i===1?"AAAAAA":"666666", 9, "center");
  });
  mkLine(s4, 4.7, 4.5, 5.0, AC);
  mkEyebrow(s4, `${brand_name}  ·  ${IND_CFG.vibe}`, 4.7, 4.65, 5.0, P);
  mkNum(s4, 4, "999999");

  // SLIDE 5 — PRODUCT
  const s5 = pres.addSlide();
  s5.background = { color: P };
  mkImg(s5, imgs.img3||imgs.img1, 5.6, 0, 4.4, H);
  mkRect(s5, 5.6, 0, 4.4, H, BD, 40);
  mkEyebrow(s5, "05  ·  Product & Service", 0.5, 0.35, 5.0, AC);
  mkLine(s5, 0.5, 0.65, 5.0, AC);
  mkHead(s5, "What We\nOffer", 0.5, 0.85, 5.0, 1.4, S, 34);
  const prodTxt = product || `${brand_name} offers a carefully considered ${industry||"product"} experience — built with premium quality and designed for ${audience||"those who know what they want"}.`;
  mkBody(s5, prodTxt, 0.5, 2.45, 5.0, 1.2, "BBBBBB", 11.5);
  ["Signature quality & craft","Thoughtfully designed experience","Premium finish & materials","Built for the discerning buyer"].forEach((f,i)=>{
    mkLine(s5, 0.5, 3.78+i*0.37, 0.28, AC);
    mkBody(s5, f, 0.88, 3.73+i*0.37, 4.6, 0.32, S, 10);
  });
  // logo placeholder circle on image
  s5.addShape(pres.shapes.OVAL,{x:6.6,y:1.3,w:2.4,h:2.4,fill:{color:"000000",transparency:55},line:{color:fix(AC),width:1.5}});
  s5.addText(brand_name.slice(0,2).toUpperCase(),{x:6.6,y:1.3,w:2.4,h:2.4,fontSize:40,fontFace:font_heading,color:fix(AC),align:"center",valign:"middle",margin:0});
  mkBody(s5, "[ Add your logo here ]", 6.3, 3.88, 3.0, 0.32, "666666", 7.5, "center");
  mkNum(s5, 5, "555555");

  // SLIDE 6 — AUDIENCE
  const s6 = pres.addSlide();
  s6.background = { color: BL };
  mkEyebrow(s6, "06  ·  Who We Serve", 0.5, 0.28, 9, AC);
  mkLine(s6, 0.5, 0.58, W-0.8, AC);
  mkHead(s6, "Our Audience", 0.5, 0.74, 5, 0.82, P, 28);
  const segs = [
    {pct:"60%",title:"Primary",   desc:audience||"Early adopters who value quality and authentic storytelling above all else"},
    {pct:"30%",title:"Secondary", desc:`Established voices in ${industry||"the space"} looking for something fresh and credible`},
    {pct:"10%",title:"Emerging",  desc:"New customers discovering the category for the first time through community and word-of-mouth"},
  ];
  segs.forEach((sg,i)=>{
    const x = 0.5+i*3.15;
    mkRect(s6, x, 1.72, 2.95, 3.62, i===0?P:"EEEADF");
    mkRect(s6, x, 1.72, 2.95, 0.07, AC);
    s6.addText(sg.pct,{x,y:1.9,w:2.95,h:1.05,fontSize:44,fontFace:font_heading,color:i===0?fix(AC):fix(P),align:"center",margin:0});
    mkLine(s6, x+0.5, 3.05, 1.95, AC);
    mkBody(s6, sg.title.toUpperCase(), x, 3.22, 2.95, 0.3, i===0?fix(AC):fix(P), 7.5, "center");
    mkBody(s6, sg.desc, x+0.15, 3.6, 2.65, 1.4, i===0?"CCCCCC":"555555", 9.5, "center");
  });
  mkNum(s6, 6, "999999");

  // SLIDE 7 — BRAND IDENTITY
  const s7 = pres.addSlide();
  s7.background = { color: P };
  mkEyebrow(s7, "07  ·  Brand Identity", 0.5, 0.28, 9, AC);
  mkLine(s7, 0.5, 0.58, W-0.8, AC);
  mkHead(s7, "Colours  \u00B7  Type  \u00B7  Style", 0.5, 0.74, 7.5, 0.8, S, 26);
  [{color:P,label:"Primary",hex:primary},{color:S,label:"Secondary",hex:secondary},{color:AC,label:"Accent",hex:accent},{color:BD,label:"Dark BG",hex:bg_dark},{color:BL,label:"Light BG",hex:bg_light}].forEach((sw,i)=>{
    const x = 0.5+i*1.84;
    s7.addShape(pres.shapes.RECTANGLE,{x,y:1.72,w:1.74,h:2.2,fill:{color:fix(sw.color)},line:{color:"333333",width:0.3}});
    mkBody(s7, `#${String(sw.hex||"").toUpperCase()}`, x, 3.98, 1.74, 0.28, "888888", 7, "center");
    mkBody(s7, sw.label.toUpperCase(), x, 4.28, 1.74, 0.28, AC, 7.5, "center");
  });
  mkLine(s7, 0.5, 4.7, W-0.8, AC);
  s7.addText(`${font_heading}  \u00B7  ${font_body}`, {x:0.5,y:4.85,w:5,h:0.28,fontSize:8.5,fontFace:font_body,color:"888888",charSpacing:2,align:"left",margin:0});
  [tone, IND_CFG.vibe.split(" ")[0]||"clean", "crafted"].forEach((m,i)=>{
    mkRect(s7, 6.1+i*1.32, 4.82, 1.2, 0.34, i===0?AC:"333333");
    mkBody(s7, m.charAt(0).toUpperCase()+m.slice(1), 6.1+i*1.32, 4.82, 1.2, 0.34, i===0?P:"AAAAAA", 8.5, "center");
  });
  mkNum(s7, 7, "555555");

  // SLIDE 8 — MARKETING
  const s8 = pres.addSlide();
  s8.background = { color: BL };
  mkVLine(s8, 4.88, 0, H, AC);
  mkEyebrow(s8, "08  ·  Go-to-Market", 0.5, 0.32, 4.1, AC);
  mkLine(s8, 0.5, 0.62, 4.1, AC);
  mkHead(s8, "How We\nReach Them", 0.5, 0.82, 4.1, 1.5, P, 27);
  const mktTxt = marketing || `${brand_name} will grow through authentic storytelling, community-first thinking, and partnerships that place us in front of ${audience||"our audience"} at the right moment.`;
  mkBody(s8, mktTxt, 0.5, 2.5, 4.1, 1.55, "444444", 11);
  mkLine(s8, 0.5, 4.2, 4.1, AC);
  mkEyebrow(s8, "Strategy  \u00B7  Community  \u00B7  Presence", 0.5, 4.38, 4.1, P);
  [["01","Digital","Social, content, SEO & ads"],["02","Community","Events, collabs & word-of-mouth"],["03","Direct","Retail, pop-ups & outreach"],["04","Brand PR","Press, storytelling & thought leadership"]].forEach(([num,name,detail],i)=>{
    const x = 5.18+(i%2)*2.42, y = 0.5+Math.floor(i/2)*2.55;
    const dk = i%2===0;
    mkRect(s8, x, y, 2.24, 2.32, dk?P:"E6E0D2");
    mkRect(s8, x, y, 2.24, 0.07, AC);
    mkBody(s8, num, x+0.15, y+0.2, 1.95, 0.45, AC, 20);
    mkBody(s8, name.toUpperCase(), x, y+0.72, 2.24, 0.32, dk?S:P, 9.5, "center");
    mkBody(s8, detail, x+0.1, y+1.12, 2.04, 0.85, dk?"AAAAAA":"666666", 9, "center");
  });
  mkNum(s8, 8, "999999");

  // SLIDE 9 — VISION
  const s9 = pres.addSlide();
  s9.background = { color: BD };
  mkImg(s9, imgs.img3||imgs.img1, 0, 0, W, H);
  mkRect(s9, 0, 0, W, H, BD, 68);
  mkRect(s9, W-0.1, 0, 0.1, H, AC);
  mkEyebrow(s9, "09  ·  The Road Ahead", 0.55, 0.3, 8, AC);
  mkLine(s9, 0.55, 0.6, W-0.8, AC);
  const visTxt = vision || `${brand_name} will become the defining reference point in ${industry||"its category"} — a brand people return to not just for what it offers, but for what it stands for.`;
  s9.addText(visTxt, {x:0.55,y:0.82,w:W-0.9,h:1.5,fontSize:16,fontFace:font_heading,italic:true,color:"FFFFFF",align:"left",valign:"top",paraSpaceAfter:5,margin:0});
  mkLine(s9, 0.55, 2.5, W-0.8, AC);
  [["Year 1","Launch","MVP, first customers, product-market fit"],["Year 2","Scale","Expand team, channels & key partnerships"],["Year 3","Lead","Category leadership & international expansion"]].forEach(([yr,ttl,desc],i)=>{
    const x = 0.55+i*3.1;
    s9.addShape(pres.shapes.OVAL,{x:x+1.12,y:2.75,w:0.46,h:0.46,fill:{color:i===0?fix(AC):"333333"},line:{color:fix(AC),width:0.5}});
    if(i<2) mkRect(s9, x+1.58, 2.95, 1.52, 0.014, "444444");
    mkEyebrow(s9, yr, x, 3.42, 2.8, AC);
    mkBody(s9, ttl.toUpperCase(), x, 3.68, 2.8, 0.32, "FFFFFF", 11);
    mkBody(s9, desc, x, 4.06, 2.8, 0.9, "888888", 9.5);
  });
  mkNum(s9, 9, "555555");

  // SLIDE 10 — CLOSING
  const s10 = pres.addSlide();
  s10.background = { color: BL };
  mkRect(s10, 0, 0, W, 0.62, P);
  mkRect(s10, 0, H-0.62, W, 0.62, P);
  mkEyebrow(s10, brand_name.toUpperCase(), 0.5, 0.2, 9, S);
  s10.addText(brand_name.toUpperCase(), {x:0.5,y:0.88,w:9,h:1.95,fontSize:66,fontFace:font_heading,bold:false,color:fix(P),align:"center",charSpacing:10,margin:0});
  mkLine(s10, 2.0, 2.98, 6.0, AC);
  s10.addText(tagline||`The Art of ${industry||"Excellence"}`, {x:1.0,y:3.14,w:8.0,h:0.58,fontSize:15,fontFace:font_heading,italic:true,color:fix(AC),align:"center",margin:0});
  mkLine(s10, 2.0, 3.82, 6.0, AC);
  mkBody(s10, "Let\u2019s build something beautiful together.", 1.0, 4.0, 8.0, 0.38, "666666", 10, "center");
  s10.addText("\u26A0  REFERENCE DECK ONLY  \u00B7  This is a brand visualization aid, not a final investor pitch. Share with your designer for the polished production version.", {x:0.6,y:H-0.52,w:8.8,h:0.38,fontSize:7,fontFace:font_body,color:"AAAAAA",align:"center",margin:0});
  s10.addText("10",{x:9.1,y:H-0.55,w:0.7,h:0.22,fontSize:8,fontFace:font_body,color:fix(S),align:"right",charSpacing:2,margin:0});
}

/* ─── MAIN BUILD ──────────────────────────────────────────────────────────── */
async function main() {
  pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.title  = `${brand_name} — Reference Pitch Deck`;
  pres.author = "BrandCraft AI";

  const imgs = await fetchAll();

  // Pick layout by industry type
  const L = IND_CFG.layout;
  if      (L === "warm")        buildWarm(imgs);
  else if (L === "dark_tech")   buildDarkTech(imgs);
  else if (L === "clean_light") buildCleanLight(imgs);
  else if (L === "nature")      buildNature(imgs);
  else                          buildEditorial(imgs); // editorial = default

  await buildSharedSlides(imgs);

  const b64 = await pres.write({ outputType: "base64" });
  process.stdout.write(b64);
}

main().catch(e => { process.stderr.write(e.message||String(e)); process.exit(1); });