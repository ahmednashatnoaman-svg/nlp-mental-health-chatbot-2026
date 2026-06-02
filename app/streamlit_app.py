"""
MindBot — Expert Streamlit UI  v2
RAG-Based Mental Health Support Chatbot · NLP Final Project 2026
"""
import sys, os, uuid, json, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindBot — Mental Health Support",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#F0F4F8;}
.block-container{padding:1.5rem 2rem 3rem;max-width:1200px;}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a2744 0%,#2D3748 100%);border-right:1px solid #3d4f6b;}
[data-testid="stSidebar"] *{color:#E2E8F0!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#fff!important;}
[data-testid="stSidebar"] .stMetric label{color:#94A3B8!important;}
[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"]{color:#fff!important;}

/* Header */
.mindbot-header{background:linear-gradient(135deg,#1a2744 0%,#2B5BA5 50%,#4A90D9 100%);border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem;box-shadow:0 4px 20px rgba(26,39,68,.25);}
.mindbot-header h1{color:#fff;font-size:1.8rem;font-weight:700;margin:0;}
.mindbot-header p{color:#A8C5F0;font-size:.9rem;margin:0;}

/* Chat */
.chat-container{background:#fff;border-radius:16px;padding:1.5rem;min-height:460px;max-height:520px;overflow-y:auto;box-shadow:0 2px 12px rgba(0,0,0,.07);margin-bottom:1rem;border:1px solid #E2E8F0;}
.msg-user{display:flex;justify-content:flex-end;margin:.6rem 0;}
.msg-user .bubble{background:linear-gradient(135deg,#2B5BA5,#4A90D9);color:#fff;padding:.75rem 1.1rem;border-radius:18px 18px 4px 18px;max-width:72%;font-size:.95rem;line-height:1.55;box-shadow:0 2px 8px rgba(43,91,165,.3);}
.msg-bot{display:flex;justify-content:flex-start;margin:.6rem 0;gap:.6rem;align-items:flex-start;}
.bot-avatar{width:34px;height:34px;background:linear-gradient(135deg,#1a2744,#2B5BA5);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;margin-top:2px;}
.msg-bot .bubble{background:#fff;color:#2D3748;padding:.75rem 1.1rem;border-radius:18px 18px 18px 4px;max-width:72%;font-size:.95rem;line-height:1.55;border:1px solid #E2E8F0;box-shadow:0 2px 8px rgba(0,0,0,.05);}
.msg-bot .bubble.crisis{background:#FFF5F5;border-color:#FC8181;color:#742A2A;}
.msg-bot .bubble.medium-risk{background:#FFFBEB;border-color:#F59E0B;color:#92400E;}

/* Badges */
.meta-row{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.5rem;}
.badge{display:inline-flex;align-items:center;gap:.25rem;padding:.2rem .6rem;border-radius:20px;font-size:.73rem;font-weight:500;color:#fff;}
.badge-lang{background:#2B7A0B;}.badge-intent{background:#2B5BA5;}.badge-emotion{background:#6B4B9F;}
.badge-time{background:#4A5568;}.badge-crisis{background:#C53030;}.badge-medium{background:#D97706;}
.badge-device{background:#0F766E;}

/* Crisis banners */
.crisis-banner{background:linear-gradient(135deg,#FFF5F5,#FED7D7);border:2px solid #FC8181;border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;color:#742A2A;font-weight:500;}
.medium-banner{background:linear-gradient(135deg,#FFFBEB,#FEF3C7);border:2px solid #F59E0B;border-radius:12px;padding:.9rem 1.25rem;margin-bottom:1rem;color:#92400E;font-weight:500;}

/* Status */
.status-card{background:#23304a;border:1px solid #3d4f6b;border-radius:10px;padding:.6rem .8rem;margin:.3rem 0;display:flex;align-items:center;gap:.5rem;font-size:.82rem;}
.dot-green{width:10px;height:10px;border-radius:50%;background:#48BB78;flex-shrink:0;}
.dot-red{width:10px;height:10px;border-radius:50%;background:#FC8181;flex-shrink:0;}

/* Input */
.stTextInput>div>div>input{border-radius:12px!important;border:2px solid #CBD5E0!important;padding:.65rem 1rem!important;font-size:.95rem!important;transition:border-color .2s;}
.stTextInput>div>div>input:focus{border-color:#4A90D9!important;box-shadow:0 0 0 3px rgba(74,144,217,.15)!important;}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#2B5BA5,#4A90D9);color:#fff!important;border:none!important;border-radius:10px!important;padding:.55rem 1.2rem!important;font-weight:600!important;font-size:.9rem!important;transition:all .2s;box-shadow:0 2px 8px rgba(43,91,165,.25);}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(43,91,165,.35)!important;}

/* Source */
.source-card{background:#F7FAFC;border-left:3px solid #4A90D9;border-radius:0 8px 8px 0;padding:.7rem .9rem;font-size:.82rem;color:#4A5568;margin:.3rem 0;line-height:1.6;}
.source-q{color:#1a2744;font-weight:600;font-size:.8rem;margin-bottom:.3rem;}
.source-a{color:#4A5568;font-size:.8rem;}

/* Metric card */
.metric-card{background:#fff;border-radius:12px;padding:1rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06);border:1px solid #E2E8F0;}
.metric-card .value{font-size:1.8rem;font-weight:700;color:#2B5BA5;}
.metric-card .label{font-size:.8rem;color:#718096;margin-top:.2rem;}

/* Section title */
.section-title{color:#2D3748;font-weight:700;font-size:1.05rem;padding-bottom:.5rem;border-bottom:2px solid #4A90D9;margin-bottom:.8rem;}

/* Scrollbar */
.chat-container::-webkit-scrollbar{width:5px;}
.chat-container::-webkit-scrollbar-track{background:#F0F4F8;border-radius:3px;}
.chat-container::-webkit-scrollbar-thumb{background:#CBD5E0;border-radius:3px;}

/* Welcome chips */
.chip{background:#EBF4FF;color:#2B5BA5;padding:.3rem .8rem;border-radius:20px;font-size:.82rem;display:inline-block;margin:.2rem;}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "session_id":        str(uuid.uuid4()),
        "messages":          [],
        "emotion_history":   [],
        "intent_history":    [],
        "lang_history":      [],
        "response_times":    [],
        "turn_count":        0,
        "engine_loaded":     False,
        "engine":            None,
        "last_crisis_level": "low",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Engine loading ─────────────────────────────────────────────────────────────
_ENGINE_VERSION = "v4"   # bump to force cache reset

@st.cache_resource(show_spinner=False)
def _load_engine(_v=_ENGINE_VERSION):
    from src.pipeline.chat_engine import ChatEngine
    return ChatEngine().load()


def get_engine():
    if not st.session_state.engine_loaded:
        with st.spinner("🔄 Loading AI modules (MPS accelerated)…"):
            try:
                st.session_state.engine = _load_engine()
                st.session_state.engine_loaded = True
            except Exception as e:
                st.error(f"Engine failed to load: {e}")
    return st.session_state.engine


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(engine):
    with st.sidebar:
        st.markdown("## 🧠 MindBot")
        st.caption("Mental Health Support Chatbot")
        st.divider()

        # Module status
        st.markdown("### Module Status")
        status = engine.status.summary() if engine else {}
        for label, key, tag, desc in [
            ("Language Detection",   "language", "M1", "TF-IDF + LinearSVC"),
            ("Emotion Classifier",   "emotion",  "M2", "DistilBERT · MPS"),
            ("Intent Classifier",    "intent",   "M3", "Groq few-shot"),
            ("RAG Pipeline",         "rag",      "M4", "Qdrant + BAAI/bge"),
        ]:
            ready = status.get(key, False)
            dot   = "dot-green" if ready else "dot-red"
            st.markdown(
                f'<div class="status-card"><div class="{dot}"></div>'
                f'<span><b>{tag}</b> {label}<br><small style="color:#94A3B8">{desc}</small></span></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Session stats
        st.markdown("### Session Stats")
        c1, c2 = st.columns(2)
        c1.metric("Turns", st.session_state.turn_count)
        c2.metric("Emotions", len(set(st.session_state.emotion_history)))

        if st.session_state.response_times:
            avg_ms = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.caption(f"⚡ Avg response: {avg_ms:.0f}ms")

        if st.session_state.emotion_history:
            from src.modules.emotion_classifier import EMOTION_META
            top_emo = Counter(st.session_state.emotion_history).most_common(1)[0]
            meta    = EMOTION_META.get(top_emo[0], {})
            st.markdown(f"**Dominant emotion:** {meta.get('emoji','')} {top_emo[0].title()}")

        st.divider()

        # Settings
        st.markdown("### Settings")
        strong_model = st.toggle("Strong model (120B)", value=True,
                                  help="gpt-oss-120b for answers; gpt-oss-20b for greetings")
        top_k        = st.slider("Retrieved docs", 1, 10, 5,
                                  help="Passages retrieved from Qdrant per query")
        show_sources = st.toggle("Show sources", value=True)
        show_scores  = st.toggle("Show confidence scores", value=False)
        show_device  = st.toggle("Show device info", value=False)

        st.divider()

        # Actions
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages         = []
            st.session_state.emotion_history  = []
            st.session_state.intent_history   = []
            st.session_state.lang_history     = []
            st.session_state.response_times   = []
            st.session_state.turn_count       = 0
            st.session_state.last_crisis_level = "low"
            st.session_state.session_id       = str(uuid.uuid4())
            if engine:
                engine.clear_history(st.session_state.session_id)
            st.rerun()

        if st.session_state.messages:
            export = json.dumps(
                [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                indent=2, ensure_ascii=False,
            )
            st.download_button("💾 Export chat", data=export,
                               file_name="mindbot_chat.json", mime="application/json")

        st.divider()
        st.caption("NLP Final Project 2026\nRAG-Based Mental Health Chatbot")

    return dict(strong_model=strong_model, top_k=top_k,
                show_sources=show_sources, show_scores=show_scores,
                show_device=show_device)


# ── Badge builder ──────────────────────────────────────────────────────────────
def _badges(meta: dict, show_scores: bool, show_device: bool) -> str:
    parts = []
    if meta.get("language"):
        flag = {"en":"🇬🇧","ar":"🇸🇦","fr":"🇫🇷","de":"🇩🇪","es":"🇪🇸","it":"🇮🇹",
                "pt":"🇵🇹","zh":"🇨🇳","ja":"🇯🇵","ru":"🇷🇺","hi":"🇮🇳","tr":"🇹🇷",
                "nl":"🇳🇱","pl":"🇵🇱","sv":"🇸🇪","ko":"🇰🇷"}.get(meta["language"], "🌐")
        conf = f" {meta.get('lang_conf',0)*100:.0f}%" if show_scores else ""
        parts.append(f'<span class="badge badge-lang">{flag} {meta.get("language_name","")}{conf}</span>')
    if meta.get("intent"):
        label = meta["intent"].replace("_"," ").title()
        parts.append(f'<span class="badge badge-intent">{meta.get("intent_emoji","")} {label}</span>')
    if meta.get("emotion"):
        conf = f" {meta.get('emo_conf',0)*100:.0f}%" if show_scores else ""
        parts.append(f'<span class="badge badge-emotion">{meta.get("emo_emoji","")} {meta["emotion"].title()}{conf}</span>')
    if meta.get("elapsed_ms"):
        parts.append(f'<span class="badge badge-time">⚡ {meta["elapsed_ms"]}ms</span>')
    if meta.get("crisis"):
        parts.append('<span class="badge badge-crisis">🆘 Crisis</span>')
    elif meta.get("crisis_level") == "medium":
        parts.append('<span class="badge badge-medium">💛 Sensitive</span>')
    if show_device and meta.get("device") and meta["device"] not in ("n/a",""):
        parts.append(f'<span class="badge badge-device">🔥 {meta["device"].upper()}</span>')
    return '<div class="meta-row">' + "".join(parts) + "</div>" if parts else ""


# ── Chat renderer ──────────────────────────────────────────────────────────────
def render_chat(messages, show_sources, show_scores, show_device):
    html = []
    for msg in messages:
        role    = msg["role"]
        content = msg["content"].replace("\n","<br>")
        meta    = msg.get("meta", {})

        if role == "user":
            html.append(f'<div class="msg-user"><div class="bubble">{content}</div></div>')
        else:
            crisis_cls = " crisis" if meta.get("crisis") else (" medium-risk" if meta.get("crisis_level")=="medium" else "")
            badges     = _badges(meta, show_scores, show_device)
            html.append(
                f'<div class="msg-bot">'
                f'<div class="bot-avatar">🧠</div>'
                f'<div><div class="bubble{crisis_cls}">{content}</div>{badges}</div>'
                f'</div>'
            )

    st.markdown(f'<div class="chat-container">{"".join(html)}</div>', unsafe_allow_html=True)

    # Sources
    if show_sources:
        last_bot = next((m for m in reversed(messages) if m["role"]=="assistant"), None)
        if last_bot and last_bot.get("sources"):
            with st.expander(f"📚 Knowledge base ({len(last_bot['sources'])} passages)", expanded=False):
                for i, src in enumerate(last_bot["sources"], 1):
                    ctx  = src.get("context","")
                    resp = src.get("response","")
                    if ctx and resp:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<div class="source-q">#{i} · score {src["score"]:.3f}</div>'
                            f'<div class="source-q">Q: {ctx[:200]}{"…" if len(ctx)>200 else ""}</div>'
                            f'<div class="source-a">A: {resp[:300]}{"…" if len(resp)>300 else ""}</div>'
                            f'</div>', unsafe_allow_html=True,
                        )
                    else:
                        text = src.get("text","")
                        st.markdown(
                            f'<div class="source-card">'
                            f'<div class="source-q">#{i} · score {src["score"]:.3f}</div>'
                            f'{text[:350]}{"…" if len(text)>350 else ""}'
                            f'</div>', unsafe_allow_html=True,
                        )


# ── Analytics tab ──────────────────────────────────────────────────────────────
def render_analytics():
    st.markdown('<div class="section-title">📊 Session Analytics</div>', unsafe_allow_html=True)
    emo  = st.session_state.emotion_history
    ints = st.session_state.intent_history
    rts  = st.session_state.response_times
    n    = st.session_state.turn_count

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f'<div class="metric-card"><div class="value">{n}</div><div class="label">Total Turns</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="value">{len(set(emo))}</div><div class="label">Unique Emotions</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="value">{ints.count("asking_mental_health_question")}</div><div class="label">RAG Queries</div></div>', unsafe_allow_html=True)
    avg_rt = f"{sum(rts)/len(rts):.0f}ms" if rts else "—"
    k4.markdown(f'<div class="metric-card"><div class="value">{avg_rt}</div><div class="label">Avg Response</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Emotion Distribution")
        if emo:
            from src.modules.emotion_classifier import EMOTION_META
            ec = Counter(emo)
            fig = go.Figure(go.Bar(
                x=list(ec.keys()), y=list(ec.values()),
                marker_color=[EMOTION_META.get(e,{}).get("color","#4A90D9") for e in ec],
                text=list(ec.values()), textposition="outside",
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=20,b=20,l=20,r=20), height=270,
                              xaxis=dict(showgrid=False), yaxis=dict(showgrid=True,gridcolor="#E2E8F0"),
                              font=dict(family="Inter"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emotion data yet.")

    with col_r:
        st.markdown("#### Intent Breakdown")
        if ints:
            ic = Counter(ints)
            labels = [i.replace("_"," ").title() for i in ic]
            fig2 = go.Figure(go.Pie(
                labels=labels, values=list(ic.values()),
                marker=dict(colors=["#2B5BA5","#82B366","#6B4B9F","#D79B00","#FC8181"][:len(labels)]),
                hole=0.45, textinfo="label+percent", textfont_size=11,
            ))
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=20,b=20,l=20,r=20),
                               height=270, showlegend=False, font=dict(family="Inter"))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No intent data yet.")

    if rts and len(rts) >= 2:
        st.markdown("#### Response Time Over Conversation")
        fig3 = go.Figure(go.Scatter(
            x=list(range(1, len(rts)+1)), y=rts,
            mode="lines+markers",
            marker=dict(size=8, color="#4A90D9"),
            line=dict(color="#4A90D9", width=2),
            fill="tozeroy", fillcolor="rgba(74,144,217,0.1)",
        ))
        fig3.update_layout(
            xaxis=dict(title="Turn", showgrid=False),
            yaxis=dict(title="ms", showgrid=True, gridcolor="#E2E8F0"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20,b=40,l=60,r=20), height=220, font=dict(family="Inter"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    if emo and len(emo) >= 2:
        st.markdown("#### Emotion Timeline")
        from src.modules.emotion_classifier import EMOTION_LABELS, EMOTION_META
        emo_int = {e: i for i, e in enumerate(EMOTION_LABELS)}
        ys = [emo_int.get(e, 0) for e in emo]
        colors = [EMOTION_META.get(e,{}).get("color","#4A90D9") for e in emo]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=list(range(1, len(ys)+1)), y=ys,
            mode="lines+markers",
            marker=dict(size=10, color=colors),
            line=dict(color="#4A90D9", width=2),
        ))
        fig4.update_layout(
            yaxis=dict(tickvals=list(emo_int.values()), ticktext=list(emo_int.keys()),
                       showgrid=True, gridcolor="#E2E8F0"),
            xaxis=dict(title="Turn", showgrid=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20,b=40,l=80,r=20), height=240, font=dict(family="Inter"),
        )
        st.plotly_chart(fig4, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    engine   = get_engine()
    settings = render_sidebar(engine)

    # Header
    st.markdown("""
    <div class="mindbot-header">
      <div style="font-size:2.5rem">🧠</div>
      <div>
        <h1>MindBot</h1>
        <p>RAG-Based Mental Health Support · Language Detection · Emotion Analysis · Intent Routing · MPS Accelerated</p>
      </div>
    </div>""", unsafe_allow_html=True)

    # Crisis banners
    level = st.session_state.get("last_crisis_level","low")
    if level == "high":
        st.markdown('<div class="crisis-banner">🆘 <b>Crisis signal detected.</b> Emergency resources provided. Please reach out to a helpline immediately.</div>', unsafe_allow_html=True)
    elif level == "medium":
        st.markdown('<div class="medium-banner">💛 <b>Difficult moment detected.</b> Response has been adapted with extra care. You are not alone.</div>', unsafe_allow_html=True)

    # Tabs
    tab_chat, tab_analytics, tab_about = st.tabs(["💬 Chat", "📊 Analytics", "ℹ️ About"])

    # ════════════ CHAT ════════════
    with tab_chat:
        if st.session_state.messages:
            render_chat(
                st.session_state.messages,
                show_sources=settings["show_sources"],
                show_scores=settings["show_scores"],
                show_device=settings["show_device"],
            )
        else:
            st.markdown("""
            <div class="chat-container" style="display:flex;align-items:center;justify-content:center;flex-direction:column;gap:1rem;color:#718096;">
              <div style="font-size:3.5rem">🧠</div>
              <div style="font-size:1.1rem;font-weight:600;color:#4A5568">Welcome to MindBot</div>
              <div style="text-align:center;max-width:420px;font-size:.9rem">
                I'm here to provide mental health support with empathy and understanding.
                Feel free to share what's on your mind — I'm listening.
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center;margin-top:.5rem">
                <span class="chip">💭 I've been feeling anxious</span>
                <span class="chip">😔 I can't sleep from stress</span>
                <span class="chip">🙋 How to manage depression?</span>
                <span class="chip">😤 I feel angry all the time</span>
              </div>
            </div>""", unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            col_in, col_btn = st.columns([6,1])
            with col_in:
                user_input = st.text_input("msg", placeholder="Share what's on your mind…",
                                           label_visibility="collapsed")
            with col_btn:
                submitted = st.form_submit_button("Send ↗", use_container_width=True)

        if submitted and user_input.strip():
            msg = user_input.strip()
            st.session_state.messages.append({"role":"user","content":msg,"meta":{},"sources":[]})

            if engine and engine.status.intent:
                with st.spinner("🧠 Processing…"):
                    result = engine.process(msg, session_id=st.session_state.session_id)

                emotion_meta = result.get("emotion_meta", {})
                meta = {
                    "language":      result.get("language"),
                    "language_name": result.get("language_name"),
                    "lang_conf":     result.get("intent_meta",{}).get("confidence",0),
                    "intent":        result.get("intent"),
                    "intent_emoji":  result.get("intent_meta",{}).get("emoji",""),
                    "emotion":       result.get("emotion"),
                    "emo_emoji":     emotion_meta.get("emoji",""),
                    "emo_conf":      emotion_meta.get("confidence",0),
                    "elapsed_ms":    result.get("elapsed_ms"),
                    "crisis":        result.get("crisis",False),
                    "crisis_level":  result.get("crisis_level","low"),
                    "device":        emotion_meta.get("device",""),
                }
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result["response"],
                    "meta":    meta,
                    "sources": result.get("sources",[]),
                })

                if result.get("emotion"):
                    st.session_state.emotion_history.append(result["emotion"])
                if result.get("intent"):
                    st.session_state.intent_history.append(result["intent"])
                st.session_state.lang_history.append(result.get("language","en"))
                st.session_state.response_times.append(result.get("elapsed_ms",0))
                st.session_state.turn_count         += 1
                st.session_state.last_crisis_level   = result.get("crisis_level","low")
            else:
                st.session_state.messages.append({
                    "role":"assistant",
                    "content":"⚠️ Some modules are not ready. Check the sidebar and ensure your `.env` keys are set.",
                    "meta":{},"sources":[],
                })
            st.rerun()

    # ════════════ ANALYTICS ════════════
    with tab_analytics:
        render_analytics()

    # ════════════ ABOUT ════════════
    with tab_about:
        st.markdown("""
## About MindBot

**MindBot** is a RAG-Based Mental Health Support Chatbot — NLP Final Project 2026.

### Pipeline Architecture
```
User Message
     │
     ▼
[Crisis Check]            ← 3-level scoring (HIGH/MEDIUM/LOW)
     │
     ▼
[M1: Language Detection]  ← TF-IDF char n-grams + LinearSVC (20 languages)
     │
     ▼
[Translation → English]   ← Groq gpt-oss-20b (if non-English)
     │
     ▼
[M3: Intent Classification] ← Few-shot LLM (gpt-oss-20b)
     │
     ├── greeting/goodbye/gratitude/out_of_scope → Direct LLM reply
     │
     └── asking_mental_health_question
               │
               ▼
         [M2: Emotion Classification]  ← DistilBERT · MPS accelerated
               │
               ▼
         [M4: RAG Pipeline]            ← Qdrant + BAAI/bge-base-en-v1.5 + gpt-oss-120b
               │
               ▼
         [Translate Response Back]     ← if non-English input
```

### Datasets
| Module | Dataset | Size |
|--------|---------|------|
| Language Detection | `papluca/language-identification` | 70k · 20 langs |
| Emotion Classifier | `dair-ai/emotion` | 16k · 6 classes |
| RAG Knowledge Base | `heliosbrahma/mental_health_counseling_conversations` | 10,042 vectors |

### Tech Stack
| Component | Technology |
|-----------|-----------|
| Language Detection | TF-IDF char n-grams + LinearSVC (sklearn) |
| Emotion | DistilBERT fine-tuned (HuggingFace Transformers) · **MPS** |
| Intent | Few-shot Groq `gpt-oss-20b` |
| Embeddings | `BAAI/bge-base-en-v1.5` (768-dim) · **MPS** |
| Vector DB | Qdrant Cloud (free tier) |
| LLM | Groq `gpt-oss-120b` / `gpt-oss-20b` |
| UI | Streamlit 1.58 |
| API | FastAPI + uvicorn |

### Bonus Features
- 🆘 3-level crisis detection (HIGH/MEDIUM/LOW)
- 🌍 Multi-language support with translation
- 💬 Conversation memory (last 10 turns used in RAG context)
- 📊 Real-time analytics dashboard
- 💾 Chat export
- 🔥 MPS acceleration (Apple Silicon)
        """)


if __name__ == "__main__":
    main()
