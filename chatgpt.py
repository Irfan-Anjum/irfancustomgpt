"""Meridian Chat - a custom ChatGPT-style app in ONE file.

Run:
    python chatgpt.py            (opens your browser automatically)
    python chatgpt.py 5050       (use another port)

The same file also works on Streamlit Community Cloud (GitHub + streamlit.io):
set "Main file path" to chatgpt.py and put a requirements.txt containing the
single line "streamlit" next to it. In that mode the page runs in the visitor's
browser and talks to OpenAI directly (their key never reaches the host).

No extra packages needed: it uses only Python's standard library (3.8+).

What it does
- On start the browser shows a popup asking for your OpenAI API key.
  The key is checked with OpenAI; a wrong key never gets in.
- Chats are saved in the sidebar (New chat, delete, rename, pin, search).
- Replies stream live. Personas, accent colors and a PNG chatbot avatar included.
- This server only serves the page and forwards requests to OpenAI.
  It never stores or logs your key.

Optional environment variables: HOST, PORT, NO_BROWSER=1, OPENAI_BASE_URL.
"""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MAX_BODY = 4 * 1024 * 1024
ALLOWED_FIELDS = {
    "model", "messages", "stream", "temperature", "top_p", "max_tokens",
    "max_completion_tokens", "presence_penalty", "frequency_penalty", "stop",
}

HTML = r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian Chat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Sora:wght@600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#fbfbfd;--panel:#fff;--side:#f3f4f8;--ink:#12141c;--mute:#656a7d;--line:#e3e5ee;--hover:#e9ebf3;--accent:#3d3df5;--accent2:#2e2ed1;--on:#fff;--bubble:#eceefe;--code:#141622;--err:#c62828;--shadow:0 1px 2px rgba(18,20,28,.06),0 8px 24px -12px rgba(18,20,28,.18)}
:root[data-theme=dark]{--bg:#0f1117;--panel:#161922;--side:#12141b;--ink:#eceef6;--mute:#8f94a8;--line:#262a37;--hover:#1d212c;--accent:#7b7bff;--accent2:#9a9aff;--on:#0f1117;--bubble:#222542;--code:#0a0b10;--err:#ff8a80;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6)}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--ink);font:15px/1.65 "Plus Jakarta Sans",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
button,input,select,textarea{font:inherit;color:inherit}
button{cursor:pointer}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.hidden{display:none!important}
svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex:none}
.ib{display:grid;place-items:center;width:34px;height:34px;border:0;background:none;border-radius:8px;color:var(--mute)}
.ib:hover{background:var(--hover);color:var(--ink)}

/* login */
#login{height:100%;display:grid;grid-template-columns:1.05fr 1fr}
.brand{background:color-mix(in srgb,var(--accent) 55%,#000);color:#fff;padding:48px 56px;display:flex;flex-direction:column;justify-content:space-between}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;font-size:18px}
.mark{width:30px;height:30px;border-radius:9px;background:var(--accent);display:grid;place-items:center;color:#fff}
.brand .mark{background:#fff;color:#12141c}
.brand h2{font-size:clamp(30px,3.6vw,46px);line-height:1.1;letter-spacing:-.03em;margin:0 0 18px;max-width:14ch}
.brand p{color:#c7c8f5;max-width:40ch;margin:0}
.brand ul{list-style:none;margin:0;padding:0;display:grid;gap:12px;color:#dcdcfa;font-size:14px}
.brand li{display:flex;gap:10px;align-items:center}
.brand li svg{width:16px;height:16px;color:#9fa0ff}
.formside{display:grid;place-items:center;padding:32px}
.form{width:100%;max-width:380px}
.form h1{font-size:26px;letter-spacing:-.02em;margin:0 0 6px}
.form>p{color:var(--mute);margin:0 0 24px}
label{font-size:13px;font-weight:600;display:block;margin-bottom:6px}
.field{display:flex;align-items:center;border:1px solid var(--line);border-radius:12px;background:var(--panel);padding-right:4px;transition:border-color .15s,box-shadow .15s}
.field:focus-within{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 20%,transparent)}
.field input{flex:1;border:0;background:transparent;padding:12px 14px;min-width:0;font-family:"JetBrains Mono",monospace;font-size:13.5px}
.field input:focus{outline:none}
.cta{width:100%;margin-top:14px;padding:12px;border:0;border-radius:12px;background:var(--accent);color:var(--on);font-weight:600;transition:background .15s}
.cta:hover{background:var(--accent2)}
.cta:disabled{opacity:.6;cursor:wait}
.msgline{min-height:22px;margin-top:12px;font-size:13.5px;color:var(--err)}
.fine{font-size:12.5px;color:var(--mute);border-top:1px solid var(--line);padding-top:16px;margin-top:12px}

/* app shell */
#app{height:100%;display:grid;grid-template-columns:288px 1fr}
aside{background:var(--side);border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
.s-head{display:flex;align-items:center;justify-content:space-between;padding:14px 14px 8px}
.newbtn{margin:6px 12px;display:flex;align-items:center;gap:8px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:var(--panel);font-weight:600;font-size:14px;box-shadow:var(--shadow)}
.search{margin:8px 12px 4px;display:flex;align-items:center;gap:8px;padding:0 10px;border-radius:10px;background:var(--hover);color:var(--mute)}
.search input{flex:1;border:0;background:none;padding:8px 0;font-size:13.5px;min-width:0}
.search input:focus{outline:none}
#list{flex:1;overflow:auto;padding:4px 8px 12px}
.grp{font-size:12px;font-weight:600;color:var(--mute);padding:14px 10px 4px}
.item{display:flex;align-items:center;border-radius:9px;padding-left:4px}
.item:hover{background:var(--hover)}
.item.on{background:var(--hover);box-shadow:inset 3px 0 0 var(--accent)}
.item .t{flex:1;text-align:left;border:0;background:none;padding:8px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14px}
.item .x{border:0;background:none;color:var(--mute);padding:6px 8px;visibility:hidden}.item .rn{border:0;background:none;color:var(--mute);padding:6px 4px;visibility:hidden}.item .x svg,.item .rn svg{width:15px;height:15px}
.item:hover .x,.item:hover .rn,.item.on .x,.item.on .rn{visibility:visible}@media (hover:none){.item .x,.item .rn{visibility:visible}}
.s-foot{border-top:1px solid var(--line);padding:12px;display:flex;align-items:center;gap:10px}
.avatar{width:32px;height:32px;border-radius:50%;background:var(--accent);color:var(--on);display:grid;place-items:center;font-weight:700;font-size:13px;flex:none}
.who{flex:1;min-width:0;font-size:13px;line-height:1.3}
.who b{display:block}.who span{color:var(--mute);font-family:"JetBrains Mono",monospace;font-size:11.5px}

main{display:flex;flex-direction:column;min-width:0;min-height:0;background:var(--bg)}
header{display:flex;align-items:center;gap:8px;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--bg)}
#menu{display:none}
.sel{position:relative;display:flex;align-items:center}
#model{appearance:none;border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:7px 30px 7px 12px;font-weight:600;font-size:14px;max-width:62vw}
.sel svg{position:absolute;right:8px;pointer-events:none;width:14px;height:14px;color:var(--mute)}
.sp{flex:1}
.status{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--mute)}
.status i{width:7px;height:7px;border-radius:50%;background:#2fb67c}
#scroll{flex:1;overflow:auto;padding:28px 16px 12px}
.col{max-width:780px;margin:0 auto}

.hero{margin-top:9vh;text-align:center}
.hero .mark{margin:0 auto 18px;width:46px;height:46px;border-radius:14px}
.hero .mark svg{width:24px;height:24px}
.hero h2{font-size:30px;letter-spacing:-.03em;margin:0 0 6px}
.hero p{color:var(--mute);margin:0 0 28px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px;text-align:left}
.sg{border:1px solid var(--line);background:var(--panel);border-radius:14px;padding:14px 16px;transition:border-color .15s,transform .15s;text-align:left}
.sg:hover{border-color:var(--accent);transform:translateY(-1px)}
.sg b{display:block;font-size:14px;margin-bottom:2px}
.sg span{font-size:13px;color:var(--mute)}

.msg{display:flex;gap:14px;margin:0 0 26px}
.msg.user{flex-direction:row-reverse}
.msg .av{width:30px;height:30px;border-radius:50%;flex:none;display:grid;place-items:center;font-size:12px;font-weight:700}
.msg.bot .av{overflow:hidden}.av img{width:100%;height:100%;object-fit:cover;display:block}
.msg.bot .av svg{width:16px;height:16px}
.msg.user .av{display:none}
.body{min-width:0;flex:1}
.msg.user .body{flex:none;max-width:82%}
.msg.user .b{background:var(--bubble);border-radius:18px 18px 4px 18px;padding:10px 16px;white-space:pre-wrap;overflow-wrap:anywhere}
.b{overflow-wrap:anywhere}
.b p{margin:0 0 12px}.b>:last-child{margin-bottom:0}
.b ul,.b ol{margin:0 0 12px;padding-left:22px}.b li{margin:3px 0}
.b h1,.b h2,.b h3{margin:18px 0 8px;line-height:1.3;letter-spacing:-.01em}
.b h1{font-size:21px}.b h2{font-size:18px}.b h3{font-size:16px}
.b a{color:var(--accent)}
.b code{background:var(--hover);padding:2px 6px;border-radius:6px;font:.88em "JetBrains Mono",monospace}
.pre{background:var(--code);border-radius:12px;margin:0 0 14px;overflow:hidden;border:1px solid #262a37}
.pre .bar{display:flex;justify-content:space-between;align-items:center;padding:6px 8px 6px 14px;font-size:12px;color:#8f94a8;background:#1b1e2b}
.pre .bar button{display:flex;align-items:center;gap:6px;border:0;background:none;color:#c4c8d8;font-size:12px;padding:4px 8px;border-radius:6px}
.pre .bar button:hover{background:#262a37}
.pre .bar svg{width:14px;height:14px}
.pre pre{margin:0;padding:14px;overflow:auto}
.pre code{background:none;padding:0;color:#e4e6ef;font-size:13px;line-height:1.6}
.acts{display:flex;gap:2px;margin-top:6px;opacity:0;transition:opacity .15s}
.msg:hover .acts,.acts:focus-within{opacity:1}
.acts .ib{width:30px;height:30px}
.acts svg{width:16px;height:16px}
.err{color:var(--err)}
.dots{display:inline-flex;gap:4px;padding:8px 0}
.dots i{width:7px;height:7px;border-radius:50%;background:var(--mute);animation:b 1.1s infinite ease-in-out}
.dots i:nth-child(2){animation-delay:.15s}.dots i:nth-child(3){animation-delay:.3s}
@keyframes b{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
@media (prefers-reduced-motion:reduce){.dots i{animation:none;opacity:.7}.sg{transition:none}}

footer{padding:8px 16px 14px;background:linear-gradient(transparent,var(--bg) 30%)}
.box{max-width:780px;margin:0 auto;display:flex;align-items:flex-end;gap:8px;border:1px solid var(--line);border-radius:18px;padding:8px 8px 8px 16px;background:var(--panel);box-shadow:var(--shadow);transition:border-color .15s,box-shadow .15s}
.box:focus-within{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent)}
#in{flex:1;border:0;resize:none;background:transparent;max-height:200px;padding:7px 0;line-height:1.5}
#in:focus{outline:none}
#send{width:38px;height:38px;border-radius:12px;border:0;background:var(--accent);color:var(--on);display:grid;place-items:center;flex:none;transition:opacity .15s}
#send:disabled{opacity:.35;cursor:default}
.hint{text-align:center;font-size:12px;color:var(--mute);margin-top:8px}
.toast{position:fixed;bottom:90px;left:50%;transform:translateX(-50%);background:var(--ink);color:var(--bg);padding:8px 14px;border-radius:10px;font-size:13px;z-index:20}

.dot{width:22px;height:22px;border:0;background:none;display:grid;place-items:center;flex:none;padding:0;margin-right:4px}
.dot i{width:10px;height:10px;border-radius:50%;border:2px solid var(--mute);display:block}
.hero-av{display:block;margin:0 auto 18px;width:56px;height:56px;border-radius:50%;object-fit:cover}
.modal{position:fixed;inset:0;background:rgba(10,12,20,.5);display:grid;place-items:center;z-index:30;padding:16px}
.sheet{background:var(--panel);border:1px solid var(--line);border-radius:18px;width:100%;max-width:520px;max-height:88vh;overflow:auto;padding:22px 24px;box-shadow:var(--shadow)}
.sheet h3{margin:0 0 14px;font-size:20px;letter-spacing:-.02em}
.sec{padding:14px 0;border-top:1px solid var(--line)}
.sec>b{display:block;margin-bottom:10px;font-size:14px}
.sws,.avrow{display:flex;flex-wrap:wrap;align-items:center;gap:10px}
.avrow.end{justify-content:flex-end;margin-top:6px}
.sw{width:30px;height:30px;border-radius:50%;border:2px solid var(--panel);box-shadow:0 0 0 1px var(--line);padding:0}
.sw.on{box-shadow:0 0 0 2px var(--ink)}
#cc{width:36px;height:30px;border:0;background:none;padding:0}
.btn{border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:8px 12px;font-size:13.5px;font-weight:600}
.btn:hover{background:var(--hover)}
.avrow img{width:48px;height:48px;border-radius:50%;object-fit:cover}
.chk{display:flex;gap:8px;align-items:center;font-weight:500;font-size:14px;margin:0 0 6px}
.chk input{accent-color:var(--accent)}
.sheet .note{font-size:12.5px;color:var(--mute);margin:0 0 10px}
.guide{margin:0 0 10px;padding-left:20px;display:grid;gap:10px;font-size:14px}
[data-chatstyle=outline] .msg.user .b{background:transparent;border:1px solid var(--accent)}
[data-chatstyle=square] .msg.user .b{border-radius:6px}
@media (max-width:820px){
 #login{grid-template-columns:1fr}.brand{display:none}
 #app{grid-template-columns:1fr}
 aside{position:fixed;inset:0 auto 0 0;width:300px;z-index:5;transform:translateX(-100%);transition:transform .2s}
 aside.open{transform:none;box-shadow:0 0 0 100vmax rgba(0,0,0,.45)}
 #menu{display:grid}.cards{grid-template-columns:1fr}.status{display:none}
}

:root,:root[data-theme=aurora]{--bg:transparent;--panel:rgba(255,255,255,.13);--side:rgba(22,18,70,.34);--ink:#fff;--mute:rgba(255,255,255,.74);--line:rgba(255,255,255,.24);--hover:rgba(255,255,255,.15);--accent:#fff;--accent2:#e8e2ff;--on:#4a3d9f;--bubble:rgba(255,255,255,.2);--code:rgba(18,14,52,.62);--err:#ffc2c2;--shadow:0 10px 36px -10px rgba(20,10,70,.45);--sheetbg:rgba(48,38,130,.92);--page:linear-gradient(135deg,#1560c8 0%,#3b55c2 28%,#7a4fb3 62%,#d63f8e 100%);--wave:rgba(255,255,255,.26)}
:root[data-theme=light]{--bg:transparent;--panel:rgba(255,255,255,.74);--side:rgba(255,255,255,.55);--ink:#1d1b3a;--mute:#5f5b86;--line:rgba(70,50,150,.18);--hover:rgba(70,50,150,.09);--accent:#5b3fd0;--accent2:#4a31b8;--on:#fff;--bubble:rgba(91,63,208,.13);--code:rgba(22,20,50,.92);--err:#c62828;--shadow:0 10px 30px -12px rgba(60,40,140,.3);--sheetbg:#fff;--page:linear-gradient(135deg,#dbe8ff 0%,#e6defa 55%,#ffdcec 100%);--wave:rgba(91,63,208,.18)}
body{background:var(--page) fixed}
#waves{position:fixed;inset:0;z-index:-1;pointer-events:none}
#waves svg{width:100%;height:100%}
#waves path{fill:none;stroke:var(--wave);stroke-width:.9}
aside,header,.box,.sg,.field,.form,#model,#persona{-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px)}
.sheet{background:var(--sheetbg);-webkit-backdrop-filter:blur(18px);backdrop-filter:blur(18px)}
select option{color:#1d1b3a;background:#fff}
::placeholder{color:var(--mute);opacity:1}
.mark{color:var(--on)}
.cta{border-radius:999px}
.b a{text-decoration:underline}
.toast{background:rgba(24,18,74,.94);color:#fff}
#login{grid-template-columns:1fr;place-items:center;padding:24px}
.form{text-align:center;padding:38px 34px 30px;background:rgba(255,255,255,.1);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow)}
:root[data-theme=light] .form{background:rgba(255,255,255,.6)}
.form h1{font-size:36px;font-weight:500;letter-spacing:-.01em;margin:0 0 6px}
.form>p{margin:0 0 22px}
.field{text-align:left;border-radius:6px}
.field>svg{margin-left:14px;color:var(--mute)}
.rem{display:flex;justify-content:space-between;align-items:center;margin:14px 2px 18px;font-size:13px}
.rem .chk{margin:0;font-size:13px}.rem a{color:var(--ink);opacity:.85}
.form .cta{margin-top:0}
.fine{text-align:left}
#persona{appearance:none;border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:7px 30px 7px 12px;font-weight:600;font-size:14px;max-width:40vw}
.item .rn.on{visibility:visible;color:var(--ink)}.item .rn.on svg{fill:currentColor}
.inp{width:100%;border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:10px 12px;font:inherit;color:var(--ink);resize:vertical}
.inp:focus{outline:none;border-color:var(--accent)}
.prow{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:6px 0}
@media (max-width:820px){#model,#persona{max-width:34vw}}
.cards.ft{margin-top:12px}.hero{margin-top:6vh}
.feats{list-style:none;display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:20px 0 0;padding:0;font-size:12.5px;text-align:left;color:var(--mute)}
.feats li{border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:8px 10px}
.feats b{display:block;font-size:13px;color:var(--ink)}

:root{--t1:#fff;--t2:#ffc4e3;--t3:#a9d4ff;--glow:rgba(255,255,255,.3)}
:root[data-theme=light]{--t1:#3b2bb5;--t2:#d63f8e;--t3:#1560c8;--glow:rgba(91,63,208,.25)}
.title{font-family:"Sora","Plus Jakarta Sans",system-ui,sans-serif;font-weight:700;letter-spacing:-.025em;line-height:1.15;filter:drop-shadow(0 6px 20px var(--glow))}
.form h1.title{font-size:44px;margin:0 0 8px}
.hero h2.title{font-size:34px}
.title span{display:inline-block;padding-bottom:.08em;background:linear-gradient(100deg,var(--t1) 0%,var(--t2) 35%,var(--t3) 65%,var(--t1) 100%);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent;animation:rise .9s cubic-bezier(.2,.8,.2,1) both,shine 6s linear infinite}
.title span:nth-child(2){animation-delay:.18s,0s}
@keyframes rise{from{opacity:0;transform:translateY(20px) scale(.96);filter:blur(8px)}to{opacity:1;transform:none;filter:blur(0)}}
@keyframes shine{from{background-position:0% 0}to{background-position:200% 0}}
@media (prefers-reduced-motion:reduce){.title span{animation:none}}

#login{position:fixed;inset:0;z-index:60;height:auto;background:rgba(14,10,48,.5);-webkit-backdrop-filter:blur(12px);backdrop-filter:blur(12px)}
#login .form{width:100%;max-width:420px;background:var(--sheetbg);box-shadow:0 30px 80px -20px rgba(0,0,0,.55);animation:pop .45s cubic-bezier(.2,.8,.2,1) both}
@keyframes pop{from{opacity:0;transform:translateY(16px) scale(.97)}to{opacity:1;transform:none}}
.item .x{visibility:visible;opacity:.5}.item:hover .x,.item.on .x{opacity:1}
@media (prefers-reduced-motion:reduce){#login .form{animation:none}}
</style>
</head>
<body>
<div id="waves" aria-hidden="true"></div>
<svg style="display:none" aria-hidden="true">
 <symbol id="i-logo" viewBox="0 0 24 24"><path d="M4 18V8l8 6 8-6v10M4 8l8-4 8 4"/></symbol>
 <symbol id="i-plus" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></symbol>
 <symbol id="i-search" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></symbol>
 <symbol id="i-x" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></symbol>
 <symbol id="i-copy" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h8"/></symbol>
 <symbol id="i-redo" viewBox="0 0 24 24"><path d="M20 12a8 8 0 1 1-2.5-5.8M20 4v5h-5"/></symbol>
 <symbol id="i-up" viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></symbol>
 <symbol id="i-stop" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/></symbol>
 <symbol id="i-menu" viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h10"/></symbol>
 <symbol id="i-moon" viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/></symbol>
 <symbol id="i-out" viewBox="0 0 24 24"><path d="M9 4H5v16h4M16 8l4 4-4 4M20 12H9"/></symbol>
 <symbol id="i-shield" viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/></symbol>
 <symbol id="i-bolt" viewBox="0 0 24 24"><path d="M13 3 5 14h6l-1 7 8-11h-6z"/></symbol>
 <symbol id="i-sliders" viewBox="0 0 24 24"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/></symbol>
 <symbol id="i-help" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7M12 17h.01"/></symbol>
 <symbol id="i-edit" viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16z"/></symbol>
 <symbol id="i-lock" viewBox="0 0 24 24"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></symbol>
 <symbol id="i-eye" viewBox="0 0 24 24"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></symbol>
 <symbol id="i-star" viewBox="0 0 24 24"><path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></symbol>
 <symbol id="i-chev" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></symbol>
</svg>

<section id="login" role="dialog" aria-modal="true" aria-labelledby="ltitle">
  <form class="form" id="loginForm" autocomplete="off">
    <h1 class="title" id="ltitle"><span>Let’s</span> <span>explore</span></h1>
    <p>Enter your OpenAI API key to start chatting.</p>
    <div class="field">
      <svg><use href="#i-lock"/></svg>
      <input id="key" type="password" placeholder="sk-..." aria-label="OpenAI API key" spellcheck="false" autofocus required>
      <button type="button" class="ib" id="peek" aria-label="Show or hide key"><svg><use href="#i-eye"/></svg></button>
    </div>
    <div class="rem"><label class="chk"><input type="checkbox" id="rem">Remember me</label><a href="#" id="forget">Forget me?</a></div>
    <button class="cta" id="go" type="submit">Start exploring</button>
    <div class="msgline" id="loginErr" role="alert"></div>
    <div class="fine">Your key is checked with OpenAI first. It travels only from your browser to this app on your computer and on to OpenAI, and the app never stores it. Without Remember me it stays in this tab only.</div>
  </form>
</section>

<section id="app" inert>
  <aside id="side">
    <div class="s-head"><div class="logo"><span class="mark"><svg><use href="#i-logo"/></svg></span>Meridian</div><button class="ib" id="close" aria-label="Close sidebar"><svg><use href="#i-x"/></svg></button></div>
    <button class="newbtn" id="new"><svg><use href="#i-plus"/></svg>New chat</button>
    <div class="search"><svg><use href="#i-search"/></svg><input id="q" placeholder="Search chats" aria-label="Search chats"></div>
    <div id="list"></div>
    <div class="s-foot">
      <div class="avatar">API</div>
      <div class="who"><b>Signed in</b><span id="masked"></span></div>
      <button class="ib" id="theme" aria-label="Toggle theme"><svg><use href="#i-moon"/></svg></button>
      <button class="ib" id="logout" aria-label="Sign out"><svg><use href="#i-out"/></svg></button>
    </div>
  </aside>
  <main>
    <header>
      <button class="ib" id="menu" aria-label="Open chats"><svg><use href="#i-menu"/></svg></button>
      <div class="sel"><select id="model" aria-label="Model"></select><svg><use href="#i-chev"/></svg></div>
      <div class="sel"><select id="persona" aria-label="Persona"></select><svg><use href="#i-chev"/></svg></div>
      <div class="sp"></div>
      <div class="status"><i></i>Key verified</div>
      <button class="ib" id="guideBtn" aria-label="User guide"><svg><use href="#i-help"/></svg></button>
      <button class="ib" id="setBtn" aria-label="Settings"><svg><use href="#i-sliders"/></svg></button>
    </header>
    <div id="scroll"><div class="col" id="thread"></div></div>
    <footer>
      <div class="box">
        <textarea id="in" rows="1" placeholder="Message Meridian" aria-label="Message"></textarea>
        <button id="send" aria-label="Send"><svg><use href="#i-up"/></svg></button>
      </div>
      <div class="hint">Enter to send, Shift+Enter for a new line. Check important answers.</div>
    </footer>
  </main>
</section>

<div class="modal hidden" id="modal" role="dialog" aria-modal="true"><div class="sheet" id="sheet"></div></div>
<script>
const API="/api", $=id=>document.getElementById(id);
const ic=n=>`<svg><use href="#i-${n}"/></svg>`;
let KEY="",chats=[],cur=null,busy=false,ctrl=null;
const mk=s=>({get:k=>{try{return s.getItem(k)}catch(e){return null}},set:(k,v)=>{try{s.setItem(k,v)}catch(e){}},del:k=>{try{s.removeItem(k)}catch(e){}}});
const ss=mk(typeof sessionStorage!=="undefined"?sessionStorage:{}),ls=mk(typeof localStorage!=="undefined"?localStorage:{});
const save=()=>{if(!KEY)return;if(P.keep)ls.set("chats",JSON.stringify(chats));else ls.del("chats")};
const esc=s=>s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
function toast(t){const d=document.createElement("div");d.className="toast";d.textContent=t;document.body.append(d);setTimeout(()=>d.remove(),1600)}

/* preferences, avatar, guide */
const COLORS=["","#e5484d","#f08c00","#2fb67c","#2d8cf0","#a855f7"],SW=["#ffffff","#ff7ab8","#7dd3fc","#fde68a","#a7f3d0","#c4b5fd"];
let P={};try{P=JSON.parse(ls.get("prefs")||"{}")||{}}catch(e){}
if(P.keep===undefined)P.keep=true;
const savePrefs=()=>ls.set("prefs",JSON.stringify(P));
function onColor(h){if(!/^#[0-9a-f]{6}$/i.test(h))return"#fff";const n=parseInt(h.slice(1),16);return(.299*(n>>16)+.587*(n>>8&255)+.114*(n&255))>150?"#12141c":"#fff"}
const avCache={};
function botSrc(){const pa=curPersona().avatar;return pa||baseSrc()}
function baseSrc(){
  if(P.avatar)return P.avatar;
  let c=getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();if(!/^#[0-9a-f]{6}$/i.test(c))c="#ffffff";
  if(avCache[c])return avCache[c];
  const cv=document.createElement("canvas");cv.width=cv.height=128;const x=cv.getContext("2d");
  if(P.accent){x.fillStyle=c}else{const g=x.createLinearGradient(0,0,128,128);g.addColorStop(0,"#1a6bd8");g.addColorStop(.55,"#7a4fb3");g.addColorStop(1,"#e0449a");x.fillStyle=g}x.beginPath();x.arc(64,64,64,0,7);x.fill();
  x.translate(32,32);x.scale(64/24,64/24);x.strokeStyle=P.accent?onColor(c):"#fff";x.lineWidth=2;x.lineCap="round";x.lineJoin="round";
  x.stroke(new Path2D("M4 18V8l8 6 8-6v10M4 8l8-4 8 4"));
  return avCache[c]=cv.toDataURL("image/png");
}
function refreshAvatars(){document.querySelectorAll("img[data-bot]").forEach(i=>i.src=botSrc())}
function applyPrefs(){
  const r=document.documentElement;
  if(P.accent){r.style.setProperty("--accent",P.accent);r.style.setProperty("--accent2","color-mix(in srgb,"+P.accent+" 80%,#000)");r.style.setProperty("--on",onColor(P.accent))}
  else["--accent","--accent2","--on"].forEach(k=>r.style.removeProperty(k));
  r.dataset.chatstyle=P.style||"bubble";refreshAvatars();
}
function openModal(h){$("sheet").innerHTML=h;$("modal").classList.remove("hidden")}
function closeModal(){$("modal").classList.add("hidden")}
$("modal").onclick=e=>{if(e.target.id==="modal")closeModal()};
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeModal()});
function loadPng(f){
  if(!f)return;if(f.type!=="image/png"){toast("Please choose a PNG file.");return}
  const fr=new FileReader();
  fr.onload=()=>{const im=new Image();
    im.onload=()=>{const cv=document.createElement("canvas");cv.width=cv.height=128;const s=Math.min(im.width,im.height);
      cv.getContext("2d").drawImage(im,(im.width-s)/2,(im.height-s)/2,s,s,0,0,128,128);
      P.avatar=cv.toDataURL("image/png");savePrefs();applyPrefs();openSettings();toast("Chatbot picture updated")};
    im.onerror=()=>toast("That PNG could not be read.");im.src=fr.result};
  fr.readAsDataURL(f);
}
function openGuide(){
  openModal(`<h3>User guide</h3><ol class="guide">
  <li><b>Sign in.</b> Paste your API key. It is verified first, and only a valid key opens the workspace. Closing the tab signs you out.</li>
  <li><b>Pick a model.</b> Use the selector at the top. It lists the models your key can use.</li>
  <li><b>Chat.</b> Enter sends, Shift+Enter adds a new line. Replies stream in live and the square button stops one; streaming can be switched off in Settings. Under the latest reply you can copy it or regenerate it.</li>
  <li><b>Saved chats.</b> Chats are saved on this device and grouped by day. In the sidebar, click the small circle to cycle a color label, the star to pin, the pencil to rename, and the cross to delete. Use the search box to find a chat.</li>
  <li><b>Personas.</b> The persona menu at the top sets who you are talking to. Choose Manage personas to create your own with a name, instructions and an optional PNG picture.</li>
  <li><b>Make it yours.</b> In Settings (sliders icon) change the accent color, the message style and the chatbot picture. Any PNG works as the avatar.</li>
  <li><b>History.</b> In Settings you can turn saving off, export chats to a file, import them on another device, or clear everything.</li>
  <li><b>Privacy.</b> Your key and chats stay in this browser. Messages pass through this app’s server to OpenAI and are not stored there.</li></ol>
  <div class="avrow end"><button class="cta" id="done" style="width:auto;margin:0;padding:10px 20px">Got it</button></div>`);
  $("done").onclick=closeModal;
}
function openSettings(){
  openModal(`<h3>Settings</h3>
  <div class="sec" style="border:0;padding-top:0"><b>Accent color</b><div class="sws">${SW.map(c=>`<button class="sw${P.accent===c?" on":""}" data-c="${c}" style="background:${c}" aria-label="Use ${c}"></button>`).join("")}<input type="color" id="cc" value="${P.accent||"#ffffff"}" aria-label="Custom color"><button class="btn" id="rc">Reset</button></div></div>
  <div class="sec"><b>Message style</b><select id="cs" class="btn"><option value="bubble">Rounded bubbles</option><option value="outline">Outlined</option><option value="square">Square</option></select></div>
  <div class="sec"><b>Chatbot picture (PNG)</b><div class="avrow"><img data-bot alt="Chatbot picture" src="${botSrc()}"><button class="btn" id="up">Upload PNG</button><button class="btn" id="ra">Use default</button><input type="file" id="uf" accept="image/png" class="hidden"></div></div>
  <div class="sec"><b>Replies</b><label class="chk"><input type="checkbox" id="st"${P.stream!==false?" checked":""}>Stream replies as they are written</label><p class="note">Turn off to show each answer only when it is complete.</p></div>
  <div class="sec"><b>Chat history</b><label class="chk"><input type="checkbox" id="kh"${P.keep?" checked":""}>Save chats on this device</label><p class="note">${chats.length} saved chat${chats.length===1?"":"s"}.</p><div class="avrow"><button class="btn" id="ex">Export</button><button class="btn" id="im">Import</button><button class="btn" id="cl">Clear all</button><input type="file" id="imf" accept="application/json,.json" class="hidden"></div></div>
  <div class="avrow end"><button class="cta" id="done" style="width:auto;margin:0;padding:10px 20px">Done</button></div>`);
  $("cs").value=P.style||"bubble";
  $("sheet").querySelectorAll(".sw").forEach(b=>b.onclick=()=>{P.accent=b.dataset.c;savePrefs();applyPrefs();openSettings()});
  $("cc").oninput=e=>{P.accent=e.target.value;savePrefs();applyPrefs()};
  $("rc").onclick=()=>{delete P.accent;savePrefs();applyPrefs();openSettings()};
  $("cs").onchange=e=>{P.style=e.target.value;savePrefs();applyPrefs()};
  $("st").onchange=e=>{P.stream=e.target.checked;savePrefs();toast(P.stream?"Streaming is on":"Streaming is off")};
  $("up").onclick=()=>$("uf").click();$("uf").onchange=e=>loadPng(e.target.files[0]);
  $("ra").onclick=()=>{delete P.avatar;savePrefs();applyPrefs();openSettings()};
  $("kh").onchange=e=>{
    if(!e.target.checked){if(confirm("Stop saving chats and delete the ones saved on this device?")){P.keep=false;savePrefs();save();toast("History saving is off")}else e.target.checked=true}
    else{P.keep=true;savePrefs();save();toast("Chats will be saved")}
  };
  $("ex").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(chats,null,2)],{type:"application/json"}));a.download="meridian-chats.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500)};
  $("im").onclick=()=>$("imf").click();
  $("imf").onchange=e=>{const f=e.target.files[0];if(!f)return;const fr=new FileReader();
    fr.onload=()=>{try{const a=JSON.parse(fr.result);if(!Array.isArray(a))throw 0;let n=0;
      a.forEach(c=>{if(c&&c.id&&typeof c.title==="string"&&Array.isArray(c.messages)&&!chats.some(k=>k.id===String(c.id))){
        chats.push({id:String(c.id),ts:+c.ts||0,title:c.title.slice(0,60),color:COLORS.includes(c.color)?c.color:"",messages:c.messages.filter(m=>m&&(m.role==="user"||m.role==="assistant")&&typeof m.content==="string")});n++}});
      chats.sort((a,b)=>(b.ts||0)-(a.ts||0));save();renderList();openSettings();toast(n+" chat"+(n===1?"":"s")+" imported")}
      catch(err){toast("That file is not a valid chat export.")}};
    fr.readAsText(f)};
  $("cl").onclick=()=>{if(busy)return;if(confirm("Delete all saved chats?")){chats=[];save();newChat();openSettings();toast("History cleared")}};
  $("done").onclick=closeModal;
}
$("guideBtn").onclick=openGuide;$("setBtn").onclick=openSettings;

/* personas */
const BUILTIN=[
{id:"default",name:"Assistant",prompt:""},
{id:"coder",name:"Code reviewer",prompt:"You are a meticulous senior software engineer doing code review. Find bugs, security risks and unclear code. Be concise, explain why each issue matters, and show corrected code."},
{id:"writer",name:"Writing coach",prompt:"You are a supportive writing coach. Improve clarity, structure and tone while keeping the author's voice. Briefly explain your main edits."},
{id:"tutor",name:"Friendly tutor",prompt:"You are a patient tutor. Explain step by step in plain language, use small examples, and end with a short question that checks understanding."}];
let CUSTOM=[];try{CUSTOM=JSON.parse(ls.get("personas")||"[]")||[]}catch(e){}
const allPers=()=>BUILTIN.concat(CUSTOM);
function curPersona(){const id=(cur&&cur.persona)||P.persona||"default";return allPers().find(p=>p.id===id)||BUILTIN[0]}
const qa=s=>esc(s).replace(/"/g,"&quot;");
function fillPersonas(){$("persona").innerHTML=allPers().map(p=>`<option value="${qa(p.id)}">${esc(p.name)}</option>`).join("")+'<option value="__m">Manage personas...</option>';syncPersonaUI()}
function syncPersonaUI(){const p=curPersona();$("persona").value=p.id;$("in").placeholder="Message "+p.name;refreshAvatars()}
function setPersona(id){P.persona=id;savePrefs();if(cur){cur.persona=id;save()}syncPersonaUI();if(cur&&!cur.messages.length)renderThread()}
$("persona").onchange=e=>{if(busy){syncPersonaUI();return}if(e.target.value==="__m"){syncPersonaUI();openPersonas()}else setPersona(e.target.value)};
function readPng(f,cb){
  if(!f)return;if(f.type!=="image/png"){toast("Please choose a PNG file.");return}
  const fr=new FileReader();fr.onload=()=>{const im=new Image();
    im.onload=()=>{const cv=document.createElement("canvas");cv.width=cv.height=128;const m=Math.min(im.width,im.height);
      cv.getContext("2d").drawImage(im,(im.width-m)/2,(im.height-m)/2,m,m,0,0,128,128);cb(cv.toDataURL("image/png"))};
    im.onerror=()=>toast("That PNG could not be read.");im.src=fr.result};fr.readAsDataURL(f);
}
function openPersonas(id){
  const p=CUSTOM.find(x=>x.id===id)||{name:"",prompt:""};let av=p.avatar||"";
  openModal(`<h3>${id?"Edit persona":"New persona"}</h3>
  <p class="note">A persona decides how the assistant behaves. Its instructions are sent as hidden context with every message in the chat.</p>
  <label for="pn">Name</label><input id="pn" class="inp" maxlength="30" value="${qa(p.name)}" placeholder="Startup advisor">
  <label for="pp" style="margin-top:12px">Instructions</label><textarea id="pp" class="inp" rows="6" placeholder="Describe who the assistant is, how it speaks and what it should focus on.">${esc(p.prompt)}</textarea>
  <div class="avrow" style="margin:12px 0"><img id="pav" alt="Persona picture" src="${av||baseSrc()}"><button class="btn" id="pu">Upload PNG</button><button class="btn" id="pr">Remove</button><input type="file" id="puf" accept="image/png" class="hidden"></div>
  <div class="avrow end"><button class="btn" id="pc">Cancel</button><button class="cta" id="ps" style="width:auto;margin:0;padding:10px 20px">Save persona</button></div>
  ${CUSTOM.length?`<div class="sec" style="margin-top:14px"><b>Your personas</b>${CUSTOM.map(c=>`<div class="prow"><span>${esc(c.name)}</span><span><button class="btn" data-pe="${qa(c.id)}">Edit</button> <button class="btn" data-pd="${qa(c.id)}">Delete</button></span></div>`).join("")}</div>`:""}`);
  $("pu").onclick=()=>$("puf").click();
  $("puf").onchange=e=>readPng(e.target.files[0],u=>{av=u;$("pav").src=u});
  $("pr").onclick=()=>{av="";$("pav").src=baseSrc()};
  $("pc").onclick=closeModal;
  $("ps").onclick=()=>{const n=$("pn").value.trim(),t=$("pp").value.trim();if(!n||!t){toast("Add a name and instructions.");return}
    const o={id:p.id||"c"+Date.now().toString(36),name:n,prompt:t};if(av)o.avatar=av;
    const i=CUSTOM.findIndex(x=>x.id===o.id);if(i>=0)CUSTOM[i]=o;else CUSTOM.push(o);
    ls.set("personas",JSON.stringify(CUSTOM));fillPersonas();setPersona(o.id);closeModal();toast("Persona saved")};
  $("sheet").querySelectorAll("[data-pe]").forEach(b=>b.onclick=()=>openPersonas(b.dataset.pe));
  $("sheet").querySelectorAll("[data-pd]").forEach(b=>b.onclick=()=>{if(!confirm("Delete this persona?"))return;
    CUSTOM=CUSTOM.filter(x=>x.id!==b.dataset.pd);ls.set("personas",JSON.stringify(CUSTOM));
    if(P.persona===b.dataset.pd){P.persona="default";savePrefs()}fillPersonas();openPersonas()});
}
/* background waves */
(function(){let a="",b="";for(let i=0;i<34;i++){a+=`<path d="M-60 ${70+i*5} C 260 ${-160+i*9}, 520 ${300+i*3}, 800 ${-30+i*6}"/>`;b+=`<path d="M1300 ${730-i*5} C 980 ${980-i*9}, 720 ${520-i*3}, 420 ${850-i*6}"/>`}
$("waves").innerHTML=`<svg viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice">${a}${b}</svg>`})();


/* theme */
function setTheme(t){t=t==="light"?"light":"aurora";document.documentElement.dataset.theme=t;ls.set("theme",t);refreshAvatars()}
setTheme(ls.get("theme"));
$("theme").onclick=()=>setTheme(document.documentElement.dataset.theme==="light"?"aurora":"light");
applyPrefs();

/* login */
$("peek").onclick=()=>{const k=$("key");k.type=k.type==="password"?"text":"password"};
$("forget").onclick=e=>{e.preventDefault();ls.del("k");ss.del("k");toast("Saved key removed from this device")};
async function verify(key){
  const r=await fetch(API+"/models",{headers:{Authorization:"Bearer "+key}});
  if(r.status===401)throw new Error("That API key was rejected. Check it and try again.");
  if(!r.ok)throw new Error("Could not verify the key (HTTP "+r.status+"). Try again shortly.");
  return(await r.json()).data.map(m=>m.id);
}
function chatModels(ids){
  const bad=/(embed|audio|tts|whisper|transcribe|realtime|image|dall|moderation|instruct|search|davinci|babbage|codex|computer)/;
  const o=ids.filter(i=>/^(gpt-|chatgpt|o\d)/.test(i)&&!bad.test(i)&&!/\d{4}-\d{2}-\d{2}$/.test(i)).sort();
  return o.length?o:ids.filter(i=>/^gpt-/.test(i)).sort();
}
$("loginForm").onsubmit=async e=>{
  e.preventDefault();const key=$("key").value.trim();$("loginErr").textContent="";
  if(!/^sk-/.test(key)){$("loginErr").textContent="API keys start with “sk-”.";return}
  $("go").disabled=true;$("go").textContent="Verifying key...";
  try{const ids=await verify(key);KEY=key;if($("rem").checked)ls.set("k",key);else ss.set("k",key);enter(ids)}
  catch(err){$("loginErr").textContent=err.name==="TypeError"?"Network error. Check your connection and that python chatgpt.py is running and you opened http://127.0.0.1:5000 (not the file itself).":err.message}
  finally{$("go").disabled=false;$("go").textContent="Start exploring"}
};
function enter(ids){
  $("login").classList.add("hidden");$("app").removeAttribute("inert");
  const ms=chatModels(ids),sel=$("model");
  sel.innerHTML=ms.map(m=>`<option>${m}</option>`).join("");
  const p=ls.get("model");sel.value=ms.includes(p)?p:(ms.find(m=>m==="gpt-4o-mini")||ms[0]);
  $("masked").textContent=KEY.slice(0,5)+"..."+KEY.slice(-4);
  chats=[];if(P.keep){try{chats=JSON.parse(ls.get("chats")||"[]")}catch(e){}}
  fillPersonas();newChat();$("in").focus();
  if(!ls.get("seen")){ls.set("seen","1");openGuide()}
}
$("model").onchange=()=>ls.set("model",$("model").value);
$("logout").onclick=()=>{if(ctrl)ctrl.abort();KEY="";ss.del("k");ls.del("k");$("key").value="";chats=[];cur={id:"x",ts:0,title:"",persona:"default",messages:[]};renderList();renderThread();$("app").setAttribute("inert","");$("login").classList.remove("hidden");$("key").focus()};

/* chats */
function newChat(){if(busy)return;cur={id:Date.now().toString(36),ts:Date.now(),title:"New chat",persona:P.persona||"default",messages:[]};renderThread();renderList();$("side").classList.remove("open")}
$("new").onclick=newChat;$("menu").onclick=()=>$("side").classList.add("open");$("close").onclick=()=>$("side").classList.remove("open");
$("q").oninput=renderList;
function renderList(){
  const L=$("list");L.innerHTML="";const q=$("q").value.toLowerCase();
  const day=864e5,now=new Date().setHours(0,0,0,0);
  const g={"Pinned":[],"Today":[],"Yesterday":[],"Earlier":[]};
  chats.filter(c=>c.title.toLowerCase().includes(q)).forEach(c=>{const t=c.ts||0;g[c.pin?"Pinned":t>=now?"Today":t>=now-day?"Yesterday":"Earlier"].push(c)});
  Object.entries(g).forEach(([name,arr])=>{
    if(!arr.length)return;
    const h=document.createElement("div");h.className="grp";h.textContent=name;L.append(h);
    arr.forEach(c=>{
      const d=document.createElement("div");d.className="item"+(cur&&c.id===cur.id?" on":"");
      const t=document.createElement("button");t.className="t";t.textContent=c.title;
      t.onclick=()=>{if(busy)return;cur=c;renderThread();renderList();$("side").classList.remove("open")};
      const dot=document.createElement("button");dot.className="dot";dot.setAttribute("aria-label","Change color label");
      const di=document.createElement("i");if(c.color){di.style.background=c.color;di.style.borderColor=c.color}dot.append(di);
      dot.onclick=()=>{c.color=COLORS[(COLORS.indexOf(c.color||"")+1)%COLORS.length];save();renderList()};
      const rn=document.createElement("button");rn.className="rn";rn.innerHTML=ic("edit");rn.setAttribute("aria-label","Rename chat");
      rn.onclick=()=>{const n=prompt("Rename chat",c.title);if(n&&n.trim()){c.title=n.trim().slice(0,60);save();renderList()}};
      const x=document.createElement("button");x.className="x";x.innerHTML=ic("x");x.setAttribute("aria-label","Delete chat");
      x.onclick=()=>{if(busy||!confirm("Delete this chat?"))return;chats=chats.filter(k=>k.id!==c.id);save();if(cur.id===c.id)newChat();renderList()};
      const pn=document.createElement("button");pn.className="rn"+(c.pin?" on":"");pn.innerHTML=ic("star");pn.setAttribute("aria-label",c.pin?"Unpin chat":"Pin chat");
      pn.onclick=()=>{c.pin=!c.pin;save();renderList()};
      d.append(dot,t,pn,rn,x);L.append(d);
    });
  });
}

/* markdown */
function inline(s){return esc(s).replace(/`([^`]+)`/g,"<code>$1</code>").replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>").replace(/(^|[^*])\*([^*\n]+)\*/g,"$1<em>$2</em>").replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')}
function md(text){
  let html="";
  text.split(/```/).forEach((p,i)=>{
    if(i%2){const nl=p.indexOf("\n"),lang=nl>=0?p.slice(0,nl).trim():"",code=nl>=0?p.slice(nl+1):p;
      html+=`<div class="pre"><div class="bar"><span>${esc(lang)||"code"}</span><button data-copy>${ic("copy")}Copy</button></div><pre><code>${esc(code.replace(/\n$/,""))}</code></pre></div>`;return}
    let list=null;const close=()=>{if(list){html+=`</${list}>`;list=null}};
    p.split("\n").forEach(line=>{let m;
      if(m=line.match(/^(#{1,3})\s+(.*)/)){close();html+=`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`}
      else if(m=line.match(/^\s*[-*]\s+(.*)/)){if(list!=="ul"){close();html+="<ul>";list="ul"}html+=`<li>${inline(m[1])}</li>`}
      else if(m=line.match(/^\s*\d+[.)]\s+(.*)/)){if(list!=="ol"){close();html+="<ol>";list="ol"}html+=`<li>${inline(m[1])}</li>`}
      else if(!line.trim())close();else{close();html+=`<p>${inline(line)}</p>`}});
    close();
  });
  return html;
}

/* render */
const FEAT=[["chats","Saved chats","Pin, color-label, rename and search your history"],["persona","Custom personas","Create an assistant with its own instructions and picture"],["stream","Live streaming","Watch replies appear, stop any time, or turn it off"],["style","Your look","Accent color, message style and a PNG chatbot avatar"]];
const SUG=[["Explain a concept","How does public-key cryptography work, in plain language?"],["Write code","Write a Python function that merges overlapping intervals, with tests."],["Draft an email","Draft a polite email asking a vendor for an updated quote."],["Plan something","Plan a 3-day study schedule for a statistics exam."]];
function renderThread(){
  syncPersonaUI();const T=$("thread");T.innerHTML="";
  if(!cur.messages.length){
    T.innerHTML=`<div class="hero"><img data-bot class="hero-av" alt="" src="${botSrc()}"><h2 class="title"><span>How can I</span> <span>help today?</span></h2><p>You are chatting with ${esc(curPersona().name)}. Pick a starting point or type your own message.</p><div class="cards">${SUG.map((s,i)=>`<button class="sg" data-s="${i}"><b>${s[0]}</b><span>${s[1]}</span></button>`).join("")}</div><div class="cards ft">${FEAT.map(f=>`<button class="sg" data-f="${f[0]}"><b>${f[1]}</b><span>${f[2]}</span></button>`).join("")}</div></div>`;return}
  cur.messages.forEach((m,i)=>addMsg(m.role,m.content,i===cur.messages.length-1));
  $("scroll").scrollTop=$("scroll").scrollHeight;
}
function addMsg(role,text,last){
  const d=document.createElement("div");d.className="msg "+(role==="user"?"user":"bot");
  const b=document.createElement("div");b.className="b";
  if(role==="user")b.textContent=text;else b.innerHTML=md(text);
  const body=document.createElement("div");body.className="body";body.append(b);
  if(role!=="user"&&text){body.insertAdjacentHTML("beforeend",actsHTML(last))}
  d.innerHTML=role==="user"?"":`<div class="av"><img data-bot alt="" src="${botSrc()}"></div>`;d.append(body);$("thread").append(d);return b;
}
const actsHTML=last=>`<div class="acts"><button class="ib" data-act="copy" aria-label="Copy reply">${ic("copy")}</button>${last?`<button class="ib" data-act="redo" aria-label="Regenerate">${ic("redo")}</button>`:""}</div>`;
const copyText=(t,ok)=>(navigator.clipboard?navigator.clipboard.writeText(t):Promise.reject()).then(ok).catch(()=>toast("Copy failed. Select the text manually."));
$("thread").addEventListener("click",e=>{
  const ft=e.target.closest("[data-f]");if(ft){const k=ft.dataset.f;if(k==="chats"){$("side").classList.add("open");$("q").focus()}else if(k==="persona")openPersonas();else openSettings();return}
  const s=e.target.closest("[data-s]");if(s){inp.value=SUG[s.dataset.s][1];inp.oninput();send();return}
  const c=e.target.closest("[data-copy]");if(c){copyText(c.closest(".pre").querySelector("code").textContent,()=>toast("Code copied"));return}
  const a=e.target.closest("[data-act]");if(!a||busy)return;
  if(a.dataset.act==="copy")copyText(cur.messages[cur.messages.length-1].content,()=>toast("Reply copied"));
  if(a.dataset.act==="redo"){cur.messages.pop();renderThread();run()}
});

/* send */
const inp=$("in");
inp.oninput=()=>{inp.style.height="auto";inp.style.height=Math.min(inp.scrollHeight,200)+"px";$("send").disabled=busy?false:!inp.value.trim()};
inp.onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey&&!e.isComposing){e.preventDefault();if(!busy)send()}};
$("send").onclick=()=>busy?ctrl.abort():send();
function setBusy(v){busy=v;$("send").innerHTML=ic(v?"stop":"up");$("send").setAttribute("aria-label",v?"Stop":"Send");$("send").disabled=v?false:!inp.value.trim();$("model").disabled=v;$("persona").disabled=v}
function send(){
  const text=inp.value.trim();if(!text)return;
  if(!cur.messages.length){$("thread").innerHTML="";cur.title=text.slice(0,42);cur.ts=Date.now();chats.unshift(cur)}
  cur.messages.push({role:"user",content:text});inp.value="";inp.style.height="auto";
  addMsg("user",text);run();
}
async function run(){
  const out=addMsg("assistant","");out.innerHTML='<span class="dots"><i></i><i></i><i></i></span>';
  const scr=()=>{const s=$("scroll");if(s.scrollHeight-s.scrollTop-s.clientHeight<180)s.scrollTop=s.scrollHeight};
  $("scroll").scrollTop=$("scroll").scrollHeight;save();renderList();setBusy(true);
  ctrl=new AbortController();let full="";const sys=curPersona().prompt;
  try{
    const r=await fetch(API+"/chat/completions",{method:"POST",signal:ctrl.signal,headers:{"Content-Type":"application/json",Authorization:"Bearer "+KEY},body:JSON.stringify({model:$("model").value,stream:P.stream!==false,messages:(sys?[{role:"system",content:sys}]:[]).concat(cur.messages)})});
    if(!r.ok){let m="Request failed (HTTP "+r.status+").";try{m=(await r.json()).error.message}catch(e){}if(r.status===401)m+=" Your key may have been revoked. Sign out and sign in again.";throw new Error(m)}
    if(P.stream===false){const j=await r.json();full=(j.choices[0].message.content)||"";out.innerHTML=md(full)}else{const rd=r.body.getReader(),dec=new TextDecoder();let buf="";
    for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});const ls_=buf.split("\n");buf=ls_.pop();
      for(const ln of ls_){if(!ln.startsWith("data:"))continue;const d=ln.slice(5).trim();if(d==="[DONE]")continue;
        try{const c=JSON.parse(d).choices[0].delta.content;if(c){full+=c;out.innerHTML=md(full);scr()}}catch(e){}}}}
    if(!full)out.innerHTML='<span class="err">The model returned an empty reply.</span>';
  }catch(err){
    if(err.name==="AbortError"){if(!full)out.innerHTML='<span class="err">Stopped.</span>'}
    else out.innerHTML=(full?md(full):"")+`<p class="err">${esc(err.message)}</p>`;
  }finally{
    if(full)cur.messages.push({role:"assistant",content:full});
    save();setBusy(false);ctrl=null;if(full)renderThread();renderList();inp.focus();
  }
}

/* resume */
(async()=>{const k=ss.get("k")||ls.get("k");if(!k)return;try{const ids=await verify(k);KEY=k;enter(ids)}catch(e){ss.del("k");ls.del("k")}})();
</script>
</body>
</html>
'''


class Handler(BaseHTTPRequestHandler):
    server_version = "MeridianChat"

    def log_message(self, fmt, *args):  # no request logging: keeps keys out of logs
        pass

    def _send(self, status, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message):
        self._send(status, json.dumps({"error": {"message": message}}))

    def _auth(self):
        header = self.headers.get("Authorization", "")
        return header if header.startswith("Bearer sk-") and len(header) > 20 else None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, HTML, "text/html; charset=utf-8")
        elif path == "/api/models":
            self._models()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._error(404, "Not found.")

    def _models(self):
        auth = self._auth()
        if not auth:
            return self._error(401, "Missing or malformed API key.")
        req = urllib.request.Request(OPENAI_BASE + "/models", headers={"Authorization": auth})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                self._send(resp.status, resp.read())
        except urllib.error.HTTPError as err:
            self._send(err.code, err.read() or b"{}")
        except Exception:
            self._error(502, "Could not reach OpenAI. Check your internet connection.")

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/chat/completions":
            return self._error(404, "Not found.")
        auth = self._auth()
        if not auth:
            return self._error(401, "Missing or malformed API key.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            return self._error(400, "Invalid request size.")
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            return self._error(400, "Request body is not valid JSON.")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body.get("model"):
            return self._error(400, "Request must include a model and a list of messages.")

        payload = {k: v for k, v in body.items() if k in ALLOWED_FIELDS}
        stream = bool(payload.get("stream"))
        req = urllib.request.Request(
            OPENAI_BASE + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": auth, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as err:
            return self._send(err.code, err.read() or b"{}")
        except Exception:
            return self._error(502, "Could not reach OpenAI. Check your internet connection.")

        with resp:
            if not stream:
                return self._send(resp.status, resp.read())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    line = resp.readline()  # one server-sent line at a time, forwarded instantly
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # browser closed the tab or pressed Stop
        self.close_connection = True


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    first = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else int(os.environ.get("PORT", "5000"))
    server = None
    for port in range(first, first + 20):  # if a port is busy (e.g. an old copy), use the next one
        try:
            server = ThreadingHTTPServer((host, port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port found. Close other copies of this app and try again.")
    server.daemon_threads = True
    url = "http://%s:%d" % ("127.0.0.1" if host == "0.0.0.0" else host, port)
    print("\nMeridian Chat is running at " + url + "\nPress Ctrl+C to stop.\n")
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def run_streamlit():
    """Used when Streamlit starts this file (for example on Streamlit Community Cloud)."""
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="Meridian Chat", page_icon="\U0001F4AC", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """<style>
        #MainMenu, header, footer {visibility: hidden; height: 0;}
        .block-container {padding: 0 !important; max-width: 100% !important;}
        iframe {height: 100vh !important; border: 0;}
        </style>""",
        unsafe_allow_html=True,
    )
    # No local server here, so the page talks to OpenAI straight from the browser.
    page = (
        HTML.replace('const API="/api"', 'const API="https://api.openai.com/v1"')
        .replace("It travels only from your browser to this app on your computer and on to OpenAI, and the app never stores it.",
                 "It stays in your browser and is sent only to OpenAI; the hosting app never sees it.")
        .replace("Messages pass through this app\u2019s server to OpenAI and are not stored there.",
                 "Messages go directly from your browser to OpenAI.")
        .replace("Check your connection and that python chatgpt.py is running and you opened http://127.0.0.1:5000 (not the file itself).",
                 "Check your internet connection and try again.")
    )
    if hasattr(st, "iframe"):  # newer Streamlit
        st.iframe(page, height=900)
    else:  # older Streamlit
        components.html(page, height=900, scrolling=False)


def running_in_streamlit():
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


if __name__ == "__main__":
    if running_in_streamlit():
        run_streamlit()
    else:
        main()
