"""
Generate a self-contained interactive HTML explorer for a single Polymarket
weather market day.
"""

import json
from typing import Optional


# Categorical palette — validated reference slots (light / dark)
_COLORS_L = ["#2a78d6","#008300","#eb6834","#eda100","#1baf7a","#e87ba4","#4a3aa7","#c0392b","#8e44ad"]
_COLORS_D = ["#3987e5","#22b422","#d95926","#c98500","#199e70","#d55181","#9085e9","#e74c3c","#a569bd"]


def _assign_colors(tickers: list[str]) -> dict[str, dict]:
    """Map ticker labels to light/dark hex colors."""
    mapping = {}
    for i, t in enumerate(tickers):
        mapping[t] = {
            "light": _COLORS_L[i % len(_COLORS_L)],
            "dark":  _COLORS_D[i % len(_COLORS_D)],
        }
    return mapping


def _slim_trades(enriched_trades: list[dict]) -> list:
    """
    Convert enriched trade dicts to a compact array-of-arrays for JS embedding.

    Columns: [timestamp, proxyWallet, walletShort, isBuy, price, usdcValue,
              ticker, isYes, txShort, name]
    """
    slim = []
    for t in enriched_trades:
        wallet = t.get("proxyWallet") or ""
        tx = t.get("transactionHash") or ""
        slim.append([
            int(t.get("timestamp", 0)),
            wallet,
            wallet[:10] if wallet else "",
            1 if t.get("side") == "BUY" else 0,
            round(float(t.get("price", 0)), 5),
            round(float(t.get("usdc_value", 0)), 4),
            t.get("ticker", "?"),
            1 if t.get("is_yes") else 0,
            (tx[:12] + "…") if len(tx) > 12 else tx,
            t.get("name") or t.get("pseudonym") or "",
        ])
    return slim


def generate_artifact(
    enriched_trades: list[dict],
    market_meta: dict,
    event: dict,
    title: Optional[str] = None,
) -> str:
    """
    Return a self-contained interactive HTML explorer string.

    Parameters
    ----------
    enriched_trades : output of data_api.enrich_trades()
    market_meta     : output of data_api.build_market_meta()
    event           : raw event dict from data_api.get_event_by_slug()
    title           : page title override (defaults to event title)
    """
    event_title = title or event.get("title", "Polymarket Weather Explorer")
    slug = event.get("slug", "")

    # Ordered list of unique tickers (by trade volume descending)
    vol_by_ticker: dict[str, float] = {}
    for t in enriched_trades:
        tk = t.get("ticker", "unknown")
        vol_by_ticker[tk] = vol_by_ticker.get(tk, 0) + float(t.get("usdc_value", 0))
    tickers = [k for k, _ in sorted(vol_by_ticker.items(), key=lambda x: -x[1]) if k != "unknown"]

    colors = _assign_colors(tickers)
    slim = _slim_trades(enriched_trades)

    data_json = json.dumps(slim, separators=(",", ":"), ensure_ascii=False)
    tickers_json = json.dumps(tickers, ensure_ascii=False)
    colors_l_json = json.dumps({t: v["light"] for t, v in colors.items()}, ensure_ascii=False)
    colors_d_json = json.dumps({t: v["dark"]  for t, v in colors.items()}, ensure_ascii=False)

    total_trades = len(enriched_trades)
    total_vol = sum(float(t.get("usdc_value", 0)) for t in enriched_trades)
    unique_wallets = len(set(t.get("proxyWallet", "") for t in enriched_trades))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{event_title}</title>
<style>
.viz-root{{
  --sf1:#faf9f6;--sf2:#f0ede8;--sf3:#e5e1da;--bd:#d9d5cd;
  --tp:#1c1a17;--ts:#52504c;--tm:#8a8680;
  --buy:#0a7c52;--sell:#b83020;--buy-lo:rgba(10,124,82,.12);--sell-lo:rgba(184,48,32,.12);
  --mono:'Courier New',monospace;
}}
@media(prefers-color-scheme:dark){{
  :root:where(:not([data-theme="light"])) .viz-root{{
    --sf1:#13120f;--sf2:#1d1b17;--sf3:#252219;--bd:#302e28;
    --tp:#e8e4dc;--ts:#b0ab9e;--tm:#7a7670;
    --buy:#2cb87a;--sell:#e05040;--buy-lo:rgba(44,184,122,.15);--sell-lo:rgba(224,80,64,.15);
  }}
}}
:root[data-theme="dark"] .viz-root{{
  --sf1:#13120f;--sf2:#1d1b17;--sf3:#252219;--bd:#302e28;
  --tp:#e8e4dc;--ts:#b0ab9e;--tm:#7a7670;
  --buy:#2cb87a;--sell:#e05040;--buy-lo:rgba(44,184,122,.15);--sell-lo:rgba(224,80,64,.15);
}}
:root[data-theme="light"] .viz-root{{
  --sf1:#faf9f6;--sf2:#f0ede8;--sf3:#e5e1da;--bd:#d9d5cd;
  --tp:#1c1a17;--ts:#52504c;--tm:#8a8680;
  --buy:#0a7c52;--sell:#b83020;--buy-lo:rgba(10,124,82,.12);--sell-lo:rgba(184,48,32,.12);
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--sf1);color:var(--tp);font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:13px;line-height:1.4}}
.ph{{padding:16px 22px 12px;border-bottom:1px solid var(--bd);background:var(--sf2);display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}}
.ph h1{{font-size:15px;font-weight:700;letter-spacing:-.02em}}
.ph .sub{{font-size:11px;color:var(--tm)}}
.fb{{padding:8px 18px;display:flex;align-items:center;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--bd);background:var(--sf1);position:sticky;top:0;z-index:10}}
.fl{{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--tm);margin-right:1px}}
.chip{{padding:3px 9px;border-radius:99px;font-size:11px;font-weight:500;border:1.5px solid var(--bd);cursor:pointer;background:var(--sf2);color:var(--ts);transition:all .1s;white-space:nowrap;font-family:var(--mono)}}
.chip.active{{border-color:currentColor;background:color-mix(in srgb,currentColor 10%,var(--sf1))}}
.chip-buy.active{{color:var(--buy);background:var(--buy-lo);border-color:var(--buy)}}
.chip-sell.active{{color:var(--sell);background:var(--sell-lo);border-color:var(--sell)}}
.chip-yes.active{{color:#6d28d9;background:rgba(109,40,217,.1);border-color:#6d28d9}}
.chip-no.active{{color:#b45309;background:rgba(180,83,9,.1);border-color:#b45309}}
.sep{{width:1px;height:16px;background:var(--bd);margin:0 2px}}
.wi{{font-family:var(--mono);font-size:11px;padding:3px 8px;border-radius:4px;border:1.5px solid var(--bd);background:var(--sf2);color:var(--tp);width:180px;outline:none}}
.wi:focus{{border-color:var(--buy)}}
.bc{{padding:3px 9px;border-radius:4px;font-size:11px;border:1.5px solid var(--bd);cursor:pointer;background:var(--sf2);color:var(--ts)}}
.bc:hover{{border-color:var(--ts)}}
.note{{padding:8px 18px;background:color-mix(in srgb,#2a78d6 5%,var(--sf1));border-bottom:1px solid var(--bd);font-size:11px;color:var(--ts);line-height:1.6}}
.note strong{{color:var(--tp)}}
.sr{{display:flex;border-bottom:1px solid var(--bd)}}
.st{{flex:1;padding:11px 16px;border-right:1px solid var(--bd)}}
.st:last-child{{border-right:none}}
.sl{{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--tm);margin-bottom:3px}}
.sv{{font-family:var(--mono);font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--tp)}}
.ss{{font-size:11px;color:var(--tm);margin-top:1px}}
.cr{{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid var(--bd)}}
.cp{{padding:12px 14px;border-right:1px solid var(--bd)}}
.cp:last-child{{border-right:none}}
.ct{{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--tm);margin-bottom:8px;font-weight:500}}
canvas{{display:block;width:100%}}
.lg{{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}}
.li{{display:flex;align-items:center;gap:4px;font-size:10px;color:var(--ts);font-family:var(--mono)}}
.ld{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.br{{display:grid;grid-template-columns:300px 1fr;min-height:260px}}
.pn{{border-right:1px solid var(--bd);overflow:hidden}}
.pn:last-child{{border-right:none}}
.ph2{{padding:8px 12px;font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--tm);border-bottom:1px solid var(--bd);font-weight:500;background:var(--sf2);position:sticky;top:0;z-index:5}}
.wt{{width:100%;border-collapse:collapse}}
.wt th{{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--tm);padding:5px 10px;text-align:right;font-weight:500;background:var(--sf2);border-bottom:1px solid var(--bd);cursor:pointer;white-space:nowrap}}
.wt th:first-child{{text-align:left}}
.wt td{{padding:6px 10px;font-variant-numeric:tabular-nums;text-align:right;border-bottom:1px solid var(--bd);font-size:11px;cursor:pointer}}
.wt td:first-child{{text-align:left}}
.wt tr:hover td{{background:var(--sf2)}}
.wt tr.sel td{{background:color-mix(in srgb,#2a78d6 8%,var(--sf1))}}
.wa{{font-family:var(--mono);font-size:10px;color:var(--ts);max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:block}}
.wr{{color:var(--tm);font-size:10px;margin-right:3px}}
.bb{{color:var(--buy);background:var(--buy-lo);padding:1px 4px;border-radius:2px;font-size:9px;font-family:var(--mono)}}
.bs{{color:var(--sell);background:var(--sell-lo);padding:1px 4px;border-radius:2px;font-size:9px;font-family:var(--mono)}}
.tt2{{width:100%;border-collapse:collapse}}
.tt2 th{{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--tm);padding:5px 8px;text-align:right;font-weight:500;background:var(--sf2);border-bottom:1px solid var(--bd);cursor:pointer;white-space:nowrap;position:sticky;top:0}}
.tt2 th:first-child{{text-align:left}}
.tt2 td{{padding:5px 8px;font-variant-numeric:tabular-nums;text-align:right;border-bottom:1px solid var(--bd);font-size:10px;font-family:var(--mono)}}
.tt2 td:first-child{{text-align:left;font-family:system-ui,-apple-system,sans-serif}}
.tt2 tr:hover td{{background:var(--sf2)}}
.tw{{overflow:auto;max-height:260px}}
.pg{{display:flex;align-items:center;gap:6px;padding:7px 12px;font-size:11px;color:var(--ts);border-top:1px solid var(--bd)}}
.pg button{{padding:2px 8px;border-radius:3px;border:1.5px solid var(--bd);cursor:pointer;background:var(--sf2);color:var(--ts);font-size:11px}}
.pg button:disabled{{opacity:.35;cursor:default}}
.pg button:not(:disabled):hover{{border-color:var(--ts)}}
#tip{{position:fixed;pointer-events:none;z-index:100;background:var(--sf2);border:1px solid var(--bd);padding:7px 11px;border-radius:4px;font-size:11px;color:var(--tp);box-shadow:0 3px 8px rgba(0,0,0,.1);max-width:220px;line-height:1.6;display:none}}
@media(max-width:640px){{.cr{{grid-template-columns:1fr}}.br{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="viz-root">
<div class="ph">
  <h1>{event_title}</h1>
  <span class="sub">via data-api.polymarket.com · {total_trades:,} trades · {unique_wallets:,} wallets · ${total_vol:,.0f} USDC</span>
</div>
<div class="fb">
  <span class="fl">Ticker</span><div id="tch"></div>
  <div class="sep"></div>
  <span class="fl">Side</span>
  <button class="chip chip-buy active" id="bb" onclick="toggleSide('BUY')">BUY</button>
  <button class="chip chip-sell active" id="bs" onclick="toggleSide('SELL')">SELL</button>
  <div class="sep"></div>
  <span class="fl">Token</span>
  <button class="chip chip-yes active" id="by" onclick="toggleOut('YES')">YES</button>
  <button class="chip chip-no active" id="bn" onclick="toggleOut('NO')">NO</button>
  <div class="sep"></div>
  <input class="wi" id="ws" placeholder="wallet or name…" oninput="onWS()">
  <button class="bc" onclick="clearAll()">Reset</button>
</div>
<div class="note">
  <strong>YES tokens</strong> pay $1 if that temperature range occurs — their price = market probability.
  <strong>NO tokens</strong> pay $1 if it does NOT occur. The price chart (YES only) shows probability per outcome over the day.
  Click a wallet in the leaderboard to spotlight their trades.
</div>
<div class="sr">
  <div class="st"><div class="sl">Trades</div><div class="sv" id="s1">—</div><div class="ss" id="s1s"></div></div>
  <div class="st"><div class="sl">Volume USDC</div><div class="sv" id="s2">—</div><div class="ss" id="s2s"></div></div>
  <div class="st"><div class="sl">Unique Wallets</div><div class="sv" id="s3">—</div><div class="ss" id="s3s"></div></div>
  <div class="st"><div class="sl">Avg Trade</div><div class="sv" id="s4">—</div><div class="ss" id="s4s"></div></div>
</div>
<div class="cr">
  <div class="cp">
    <div class="ct">Volume over Time — bubble = USDC · filled=YES · ring=NO</div>
    <canvas id="sc" height="170"></canvas>
    <div class="lg" id="sl2"></div>
  </div>
  <div class="cp">
    <div class="ct">YES Token Price — market probability per outcome</div>
    <canvas id="pc" height="170"></canvas>
    <div class="lg" id="pl"></div>
  </div>
</div>
<div class="br">
  <div class="pn">
    <div class="ph2">Top Wallets — click to spotlight</div>
    <div class="tw"><table class="wt">
      <thead><tr>
        <th>Wallet</th>
        <th onclick="sortW('usdc')">Vol↕</th>
        <th onclick="sortW('trades')">Tx↕</th>
        <th>Bias</th>
      </tr></thead>
      <tbody id="wtb"></tbody>
    </table></div>
  </div>
  <div class="pn">
    <div class="ph2">Trades</div>
    <div class="tw"><table class="tt2">
      <thead><tr>
        <th onclick="sortT('ts')">Time↕</th>
        <th onclick="sortT('ticker')">Ticker↕</th>
        <th onclick="sortT('out')">Token↕</th>
        <th onclick="sortT('side')">Side↕</th>
        <th onclick="sortT('price')">Price↕</th>
        <th onclick="sortT('usdc')">USDC↕</th>
        <th>Wallet / Name</th>
      </tr></thead>
      <tbody id="ttb"></tbody>
    </table></div>
    <div class="pg">
      <button id="pb" onclick="prevP()">‹</button>
      <span id="pl2">—</span>
      <button id="nb" onclick="nextP()">›</button>
      <span id="tl" style="margin-left:auto;color:var(--tm)"></span>
    </div>
  </div>
</div>
<div id="tip"></div>
</div>

<script>
// cols: [timestamp,wallet,walletShort,isBuy,price,usdc,ticker,isYes,tx,name]
const RAW={data_json};
const TICKERS={tickers_json};
const CL={colors_l_json};
const CD={colors_d_json};
const MIN_TS=RAW.length?Math.min(...RAW.map(r=>r[0])):0;
const MAX_TS=RAW.length?Math.max(...RAW.map(r=>r[0])):1;
const PAGE=50;

let aTick=new Set(TICKERS),showB=true,showS=true,showY=true,showN=true;
let wf='',selW=null,tSort={{c:'usdc',d:-1}},wSort={{c:'usdc',d:-1}},page=0;
let filtered=[],wData=[];

function isDark(){{const dt=document.documentElement.getAttribute('data-theme');if(dt==='dark')return true;if(dt==='light')return false;return window.matchMedia('(prefers-color-scheme:dark)').matches}}
function tc(t){{return(isDark()?CD:CL)[t]||'#888'}}
function cv(n){{return getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(n).trim()}}
function ha(hex,a){{try{{const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return`rgba(${{r}},${{g}},${{b}},${{a}})`}}catch{{return hex}}}}
function fmtT(ts){{const d=new Date(ts*1000);return d.toLocaleTimeString([],{{hour:'2-digit',minute:'2-digit'}})}}
function fmtN(v){{return v.toFixed(0).replace(/\B(?=(\d{{3}})+(?!\d))/g,',')}}

function buildChips(){{
  const el=document.getElementById('tch');el.innerHTML='';
  TICKERS.forEach(t=>{{
    const b=document.createElement('button');b.className='chip active';b.textContent=t;b.dataset.t=t;
    const c=tc(t);b.style.color=c;b.style.borderColor=c;b.style.background=ha(c,.1);
    b.onclick=()=>toggleTick(t,b);el.appendChild(b);
  }});
}}
function toggleTick(t,b){{
  if(aTick.has(t)){{aTick.delete(t);b.classList.remove('active');b.style.color='';b.style.borderColor='';b.style.background='';}}
  else{{aTick.add(t);b.classList.add('active');const c=tc(t);b.style.color=c;b.style.borderColor=c;b.style.background=ha(c,.1);}}
  update();
}}
function toggleSide(s){{if(s==='BUY'){{showB=!showB;document.getElementById('bb').classList.toggle('active',showB);}}else{{showS=!showS;document.getElementById('bs').classList.toggle('active',showS);}}update();}}
function toggleOut(o){{if(o==='YES'){{showY=!showY;document.getElementById('by').classList.toggle('active',showY);}}else{{showN=!showN;document.getElementById('bn').classList.toggle('active',showN);}}update();}}
function onWS(){{wf=document.getElementById('ws').value.toLowerCase();if(wf)selW=null;update();}}
function clearAll(){{
  aTick=new Set(TICKERS);showB=true;showS=true;showY=true;showN=true;wf='';selW=null;
  document.getElementById('ws').value='';
  ['bb','bs','by','bn'].forEach(id=>document.getElementById(id).classList.add('active'));
  buildChips();update();
}}

function computeF(){{
  filtered=RAW.filter(r=>{{
    if(!aTick.has(r[6]))return false;
    if(r[3]&&!showB)return false;if(!r[3]&&!showS)return false;
    if(r[7]&&!showY)return false;if(!r[7]&&!showN)return false;
    if(wf&&!r[1].toLowerCase().includes(wf)&&!r[9].toLowerCase().includes(wf))return false;
    if(selW&&r[1]!==selW)return false;
    return true;
  }});
}}
function updateStats(){{
  const n=filtered.length,vol=filtered.reduce((s,r)=>s+r[5],0);
  const ws=new Set(filtered.map(r=>r[1])).size,avg=n?vol/n:0;
  const buys=filtered.filter(r=>r[3]).length;
  document.getElementById('s1').textContent=n.toLocaleString();
  document.getElementById('s1s').textContent=`${{buys}} buy · ${{n-buys}} sell`;
  document.getElementById('s2').textContent='$'+fmtN(vol);
  document.getElementById('s2s').textContent=`of $${{fmtN(RAW.reduce((s,r)=>s+r[5],0))}} total`;
  document.getElementById('s3').textContent=ws.toLocaleString();
  document.getElementById('s3s').textContent=`of ${{new Set(RAW.map(r=>r[1])).size}} total`;
  document.getElementById('s4').textContent='$'+avg.toFixed(2);
}}

function drawScatter(){{
  const cv_=document.getElementById('sc');
  const dpr=window.devicePixelRatio||1,W=cv_.offsetWidth,H=170;
  cv_.width=W*dpr;cv_.height=H*dpr;
  const ctx=cv_.getContext('2d');ctx.scale(dpr,dpr);
  const pad={{l:30,r:8,t:6,b:22}};
  const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  ctx.fillStyle=cv('--sf2');ctx.fillRect(0,0,W,H);
  if(!filtered.length)return;
  const maxU=Math.max(...filtered.map(r=>r[5]),0.01);
  const logMin=Math.log10(0.01),logMax=Math.log10(Math.max(maxU*1.5,0.1));
  function yp(v){{return pad.t+ch-(Math.log10(Math.max(v,.01))-logMin)/(logMax-logMin)*ch}}
  function xp(ts){{return pad.l+(ts-MIN_TS)/(MAX_TS-MIN_TS||1)*cw}}
  const gc=cv('--bd'),tm=cv('--tm');
  ctx.strokeStyle=gc;ctx.lineWidth=.5;
  ctx.font='9px monospace';ctx.fillStyle=tm;
  [.01,.05,.1,.5,1,5,10,50,100,500,1000].forEach(v=>{{
    if(v>Math.pow(10,logMax))return;
    const y=yp(v);if(y<pad.t||y>pad.t+ch)return;
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+cw,y);ctx.stroke();
    ctx.textAlign='right';ctx.fillText(v>=1?`$${{v}}`:`.0${{v*100|0}}`,pad.l-3,y+3);
  }});
  const span=MAX_TS-MIN_TS||3600;
  const tickCount=Math.min(6,Math.floor(cw/60));
  for(let i=0;i<=tickCount;i++){{
    const ts=MIN_TS+i/tickCount*span;
    const x=pad.l+i/tickCount*cw;
    ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+ch);ctx.stroke();
    ctx.textAlign='center';ctx.fillText(fmtT(ts),x,pad.t+ch+13);
  }}
  const pts=[...filtered].sort((a,b)=>a[5]-b[5]);
  pts.forEach(r=>{{
    const x=xp(r[0]),y=yp(r[5]);
    const rad=Math.max(2,Math.min(10,1.5+r[5]*.03));
    const col=tc(r[6]);
    ctx.beginPath();ctx.arc(x,y,rad,0,Math.PI*2);
    if(r[7]){{ctx.fillStyle=ha(col,.7);ctx.fill();}}
    else{{ctx.fillStyle=cv('--sf2');ctx.strokeStyle=col;ctx.lineWidth=1;ctx.fill();ctx.stroke();}}
  }});
  buildLeg('sl2');
}}

function drawPrice(){{
  const cv_=document.getElementById('pc');
  const dpr=window.devicePixelRatio||1,W=cv_.offsetWidth,H=170;
  cv_.width=W*dpr;cv_.height=H*dpr;
  const ctx=cv_.getContext('2d');ctx.scale(dpr,dpr);
  const pad={{l:28,r:8,t:6,b:22}};
  const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  ctx.fillStyle=cv('--sf2');ctx.fillRect(0,0,W,H);
  function xp(ts){{return pad.l+(ts-MIN_TS)/(MAX_TS-MIN_TS||1)*cw}}
  function yp(p){{return pad.t+ch-p*ch}}
  const gc=cv('--bd'),tm=cv('--tm');
  ctx.strokeStyle=gc;ctx.lineWidth=.5;
  ctx.font='9px monospace';ctx.fillStyle=tm;
  [0,.25,.5,.75,1].forEach(p=>{{
    const y=yp(p);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+cw,y);ctx.stroke();
    ctx.textAlign='right';ctx.fillText(p===0?'0':p===1?'1.0':p.toFixed(2),pad.l-3,y+3);
  }});
  const span=MAX_TS-MIN_TS||3600;
  const tickCount=Math.min(6,Math.floor(cw/60));
  for(let i=0;i<=tickCount;i++){{
    const ts=MIN_TS+i/tickCount*span;
    const x=pad.l+i/tickCount*cw;
    ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+ch);ctx.stroke();
    ctx.textAlign='center';ctx.fillText(fmtT(ts),x,pad.t+ch+13);
  }}
  const yes=filtered.filter(r=>r[7]);
  TICKERS.filter(t=>aTick.has(t)).forEach(ticker=>{{
    const pts=yes.filter(r=>r[6]===ticker).sort((a,b)=>a[0]-b[0]);
    if(pts.length<2)return;
    const BINS=50;
    const buckets=Array.from({{length:BINS}},()=>[]);
    pts.forEach(r=>{{const b=Math.min(BINS-1,Math.floor((r[0]-MIN_TS)/(MAX_TS-MIN_TS||1)*BINS));buckets[b].push(r[4]);}});
    const points=[];
    buckets.forEach((b,i)=>{{if(b.length){{const avg=b.reduce((s,v)=>s+v,0)/b.length;points.push([xp(MIN_TS+(i+.5)/BINS*(MAX_TS-MIN_TS||1)),yp(avg)]);}}}});
    if(points.length<2)return;
    const col=tc(ticker);
    ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.lineJoin='round';
    ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);
    for(let i=1;i<points.length;i++)ctx.lineTo(points[i][0],points[i][1]);
    ctx.stroke();
    const last=points[points.length-1];
    ctx.beginPath();ctx.arc(last[0],last[1],3,0,Math.PI*2);
    ctx.fillStyle=col;ctx.fill();
    ctx.font='bold 9px monospace';ctx.fillStyle=col;
    ctx.textAlign=last[0]>W-70?'right':'left';
    ctx.fillText(ticker,last[0]+(last[0]>W-70?-5:5),last[1]-2);
  }});
  buildLeg('pl',true);
}}

function buildLeg(id,yesOnly){{
  const el=document.getElementById(id);el.innerHTML='';
  TICKERS.filter(t=>aTick.has(t)).forEach(t=>{{
    const item=document.createElement('div');item.className='li';
    item.innerHTML=`<span class="ld" style="background:${{tc(t)}}"></span>${{t}}${{yesOnly?' YES':''}}`;
    el.appendChild(item);
  }});
}}

function buildWallets(){{
  const wm={{}};
  RAW.forEach(r=>{{const w=r[1];if(!wm[w])wm[w]={{addr:w,short:r[2],name:r[9],usdc:0,trades:0,buy:0,sell:0}};
    wm[w].usdc+=r[5];wm[w].trades++;r[3]?wm[w].buy++:wm[w].sell++;
  }});
  wData=Object.values(wm).sort((a,b)=>b.usdc-a.usdc).slice(0,40);
  renderW();
}}
function sortW(c){{if(wSort.c===c)wSort.d*=-1;else{{wSort.c=c;wSort.d=-1;}}wData.sort((a,b)=>wSort.d*(a[wSort.c]>b[wSort.c]?1:-1));renderW();}}
function renderW(){{
  const tb=document.getElementById('wtb');tb.innerHTML='';
  wData.forEach((w,i)=>{{
    const tr=document.createElement('tr');if(selW===w.addr)tr.classList.add('sel');
    const bias=w.buy>w.sell*2?`<span class="bb">B${{w.buy}}</span>`:w.sell>w.buy*2?`<span class="bs">S${{w.sell}}</span>`:`<span style="font-size:9px;color:var(--tm)">B${{w.buy}}/S${{w.sell}}</span>`;
    const label=w.name?`<span style="color:var(--ts)">${{w.name}}</span>`:`<span class="wa">${{w.addr.slice(0,10)}}…</span>`;
    tr.innerHTML=`<td><span class="wr">#${{i+1}}</span>${{label}}</td><td>$${{w.usdc.toFixed(0)}}</td><td>${{w.trades}}</td><td>${{bias}}</td>`;
    tr.onclick=()=>{{selW=selW===w.addr?null:w.addr;update();}};
    tb.appendChild(tr);
  }});
}}

const CI={{ts:0,ticker:6,out:7,side:3,price:4,usdc:5}};
function sortT(c){{if(tSort.c===c)tSort.d*=-1;else{{tSort.c=c;tSort.d=-1;}}page=0;renderT();}}
function renderT(){{
  const ci=CI[tSort.c]??5;
  const s=[...filtered].sort((a,b)=>tSort.d*(a[ci]>b[ci]?1:a[ci]<b[ci]?-1:0));
  const tot=s.length,pd=s.slice(page*PAGE,(page+1)*PAGE);
  const tb=document.getElementById('ttb');tb.innerHTML='';
  pd.forEach(r=>{{
    const tr=document.createElement('tr');
    const col=tc(r[6]);
    const tok=r[7]?'<span style="color:#6d28d9;font-size:9px">YES</span>':'<span style="color:#b45309;font-size:9px">NO</span>';
    const label=r[9]?r[9]:`${{r[2]}}…`;
    tr.innerHTML=`<td style="color:var(--tm)">${{fmtT(r[0])}}</td><td><span style="color:${{col}};font-weight:600">${{r[6]}}</span></td><td>${{tok}}</td><td style="color:${{r[3]?'var(--buy)':'var(--sell)'}};">${{r[3]?'BUY':'SELL'}}</td><td>${{r[4].toFixed(4)}}</td><td style="font-weight:${{r[5]>50?700:400}}">$${{r[5].toFixed(2)}}</td><td style="color:var(--ts);font-size:10px">${{label}}</td>`;
    tb.appendChild(tr);
  }});
  const pages=Math.ceil(tot/PAGE)||1;
  document.getElementById('pl2').textContent=`${{page+1}}/${{pages}}`;
  document.getElementById('tl').textContent=`${{tot.toLocaleString()}} trades`;
  document.getElementById('pb').disabled=page===0;
  document.getElementById('nb').disabled=page>=pages-1;
}}
function prevP(){{if(page>0){{page--;renderT();}}}}
function nextP(){{if(page<Math.ceil(filtered.length/PAGE)-1){{page++;renderT();}}}}

const tip=document.getElementById('tip');
document.getElementById('sc').addEventListener('mousemove',e=>{{
  const canvas=e.currentTarget,rect=canvas.getBoundingClientRect();
  const pad={{l:30,r:8,t:6,b:22}};
  const cw=rect.width-pad.l-pad.r,ch=rect.height-pad.t-pad.b;
  const mx=e.clientX-rect.left-pad.l,my=e.clientY-rect.top-pad.t;
  const maxU=filtered.length?Math.max(...filtered.map(r=>r[5]),.01):.01;
  const logMin=Math.log10(.01),logMax=Math.log10(Math.max(maxU*1.5,.1));
  let best=null,bestD=18;
  filtered.forEach(r=>{{
    const x=(r[0]-MIN_TS)/(MAX_TS-MIN_TS||1)*cw;
    const y=ch-(Math.log10(Math.max(r[5],.01))-logMin)/(logMax-logMin)*ch;
    const d=Math.hypot(x-mx,y-my);
    if(d<bestD){{bestD=d;best=r;}}
  }});
  if(best){{
    const col=tc(best[6]);
    tip.innerHTML=`<div style="color:${{col}};font-weight:700;font-family:var(--mono)">${{best[6]}} ${{best[7]?'YES':'NO'}}</div><div>${{best[3]?'<b style="color:var(--buy)">BUY</b>':'<b style="color:var(--sell)">SELL</b>'}} $${{best[5].toFixed(3)}} · p=${{best[4].toFixed(4)}}</div><div style="color:var(--tm);font-size:10px">${{fmtT(best[0])}} · ${{best[8]}}</div><div style="color:var(--tm);font-size:10px">${{best[9]||best[2]}}…</div>`;
    tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-8)+'px';
  }}else tip.style.display='none';
}});
document.getElementById('sc').addEventListener('mouseleave',()=>{{tip.style.display='none';}});

function update(){{computeF();updateStats();renderW();page=0;renderT();requestAnimationFrame(()=>{{drawScatter();drawPrice();}});}}
window.addEventListener('load',()=>{{buildChips();buildWallets();update();}});
let rT;window.addEventListener('resize',()=>{{clearTimeout(rT);rT=setTimeout(()=>{{drawScatter();drawPrice();}},80);}});
const obs=new MutationObserver(()=>{{buildChips();drawScatter();drawPrice();}});
obs.observe(document.documentElement,{{attributes:true,attributeFilter:['data-theme']}});
window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change',()=>{{buildChips();drawScatter();drawPrice();}});
</script>
</body>
</html>"""
