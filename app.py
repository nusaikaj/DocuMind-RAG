"""DocuMind RAG v2.3 — optional PDF vision indexing on top of v2.2 text RAG.

Visual pages are uploaded to Gemini ONLY after explicit user action.
Existing local text retrieval and citation mapping remain unchanged.
"""
from __future__ import annotations

import hashlib
import html
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from rag_core import build_prompt, encode_chunks, extract_chunks, retrieve, resolve_citations, source_label
from visual_core import (MAX_IMAGE_PAGE_ATTACHMENTS, describe_pdf_image,
                         is_visual_chunk, make_visual_chunk, parse_page_selection,
                         pdf_page_count, render_pdf_page, suggest_visual_pages)

load_dotenv()
st.set_page_config(
    page_title="DocuMind AI | Chat with your documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling is intentionally plain CSS, so no additional package is required.
st.markdown(
    """
<style>
:root {
    --dm-background: #090e1c;
    --dm-surface: #131c30;
    --dm-line: #283551;
    --dm-text: #edf3ff;
    --dm-muted: #a8b7d4;
    --dm-accent: #8b79ff;
}
.stApp {
    background:
        radial-gradient(circle at 80% 8%, rgba(101,84,225,.13), transparent 34%),
        radial-gradient(circle at 16% 81%, rgba(38,115,189,.09), transparent 38%),
        var(--dm-background);
    color: var(--dm-text);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
section.main .block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1150px;
    padding-top: 2.2rem;
    padding-bottom: 7rem;
}
section[data-testid="stSidebar"] {
    background: #11182a;
    border-right: 1px solid var(--dm-line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.dm-logo {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -.7px;
    margin: 0 0 .15rem;
    color: #f6f7ff;
}
.dm-small { color: var(--dm-muted); font-size: .86rem; margin: 0 0 .85rem; }
.dm-eyebrow {
    display: inline-block;
    background: rgba(125,111,255,.14);
    border: 1px solid rgba(153,143,255,.25);
    padding: .28rem .7rem;
    border-radius: 999px;
    color: #baafff;
    font-size: .77rem;
    font-weight: 700;
    letter-spacing: .04em;
}
.dm-hero {
    border: 1px solid #2b3958;
    border-radius: 22px;
    padding: 2rem 2.3rem 1.75rem;
    background: linear-gradient(120deg, rgba(33,47,84,.96), rgba(17,24,45,.96) 63%, rgba(34,29,68,.95));
    box-shadow: 0 14px 52px rgba(0,0,0,.15);
    margin-bottom: 1.25rem;
}
.dm-title {
    margin: .75rem 0 .4rem;
    font-size: clamp(2.1rem, 3.5vw, 3.15rem);
    font-weight: 830;
    line-height: 1.12;
    letter-spacing: -1.6px;
    color: #f8faff;
}
.dm-title span { color: #a99cff; }
.dm-subtitle { color: #bec9df; font-size: .98rem; margin: 0; line-height: 1.65; }
.dm-info-card {
    border: 1px solid var(--dm-line);
    border-radius: 15px;
    padding: .85rem 1.05rem;
    background: rgba(21,31,53,.88);
    min-height: 79px;
}
.dm-metric { color: #f1f4ff; font-size: 1.18rem; font-weight: 750; }
.dm-label { font-size: .80rem; color: #a8b7d4; margin-top: .1rem; }
.dm-section { font-size: 1rem; font-weight: 760; margin: 1.35rem 0 .7rem; color: #e7edff; }
.dm-empty {
    padding: 1.45rem 1.35rem;
    border-radius: 15px;
    border: 1px dashed #364564;
    background: rgba(23,33,54,.45);
    color: #acbdd8;
    margin: .65rem 0 1.2rem;
    text-align: center;
}
.dm-ready {
    display: inline-block;
    color: #88f4ca;
    border: 1px solid rgba(78,209,151,.27);
    background: rgba(38,139,103,.13);
    padding: .42rem .72rem;
    border-radius: 10px;
    font-size: .83rem;
    margin: .2rem 0 1rem;
}
[data-testid="stChatMessage"] {
    border: 1px solid #2a3856;
    border-radius: 17px;
    background: rgba(19,28,48,.78);
    padding: .9rem 1.05rem;
    margin-bottom: .85rem;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { line-height: 1.75; }
[data-testid="stChatInput"] { border-radius: 14px; }
[data-testid="stFileUploader"] { border-radius: 12px; }
div.stButton > button {
    border-radius: 12px;
    border-color: #354366;
    background: #1c2741;
    color: #e9edff;
    transition: background .15s ease, border-color .15s ease;
}
div.stButton > button:hover { border-color: #8e82ff; background: #2b3255; }
[data-testid="stExpander"] { border: 1px solid #34405a; border-radius: 11px; }
@media(max-width: 760px) {
    .dm-hero { padding: 1.3rem 1.1rem; }
    .dm-title { letter-spacing: -.9px; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the local embedding model...")
def get_encoder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def file_fingerprint(files) -> str:
    digest = hashlib.sha256()
    for uploaded in files:
        data = uploaded.getvalue()
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def show_sources(sources: list[dict[str, str]]) -> None:
    """Display retrieved snippets behind a clean collapsible panel."""
    if not sources:
        return
    with st.expander(f"📖 View {len(sources)} retrieved source passages", expanded=False):
        for i, source in enumerate(sources, start=1):
            st.markdown(f"**{i}. {source['label']}**")
            st.caption(source["text"])
            if i < len(sources):
                st.divider()


def friendly_error(exc: Exception) -> str:
    """Give actionable model errors without exposing raw request details or secrets."""
    error = str(exc)
    if "503" in error or "UNAVAILABLE" in error:
        return "⚠️ Gemini is busy right now (503). Try again or switch to another supported model."
    if "429" in error or "RESOURCE_EXHAUSTED" in error:
        return "⚠️ Gemini's request limit was reached (429). Check your free-tier quota and try later."
    if "404" in error or "NOT_FOUND" in error:
        return "⚠️ Gemini model not found (404). Check the model name in the sidebar."
    if "401" in error or "403" in error or "API_KEY_INVALID" in error:
        return "⚠️ Gemini authorization failed. Please check your API key and project access."
    return f"⚠️ Gemini request failed ({type(exc).__name__}). Check your API key, model and connection."


# Sidebar: configuration and files
with st.sidebar:
    st.markdown('<div class="dm-logo">📚 DocuMind <span style="color:#a99cff">AI</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="dm-small">Your personal document assistant</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown("#### 📁 Knowledge base")
    uploads = st.file_uploader(
        "Upload PDF, TXT or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Up to 15 MB per file. Scanned/image-heavy PDFs can be analyzed via the optional visual page feature.",
    )
    st.caption("Supported: PDF · TXT · Markdown")
    st.divider()
    st.markdown("#### ⚙️ AI settings")
    api_key = st.text_input(
        "Gemini API key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="If GEMINI_API_KEY is present in your local .env, this field is pre-filled. Never commit the key to GitHub.",
    )
    model_name = st.text_input(
        "Gemini model",
        value=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    )
    top_k = st.slider("Passages per answer", min_value=2, max_value=8, value=6)
    st.caption("🔒 Local PDF text extraction and embeddings. Gemini receives retrieved text when answering.")
    st.divider()
    st.markdown("#### 🖼️ PDF diagrams & tables (optional)")
    pdf_uploads = [f for f in (uploads or []) if f.name.lower().endswith(".pdf")
                   and f.size <= 15 * 1024 * 1024]
    analyze_pages_clicked = False
    selected_pdf = None
    requested_pages = ""
    attach_images_in_answers = False
    if pdf_uploads:
        selected_pdf_index = st.selectbox(
            "Choose PDF for visual analysis",
            range(len(pdf_uploads)),
            format_func=lambda i: pdf_uploads[i].name,
        )
        selected_pdf = pdf_uploads[selected_pdf_index]
        try:
            suggested_visual_pages = suggest_visual_pages(selected_pdf.getvalue())
            if suggested_visual_pages:
                st.caption("Suggested image-heavy pages: "
                           + ", ".join(map(str, suggested_visual_pages)))
        except Exception:
            st.caption("Enter the exact pages containing diagrams or tables.")
        requested_pages = st.text_input(
            "Pages to analyze (up to 6 at once)",
            placeholder="e.g. 5, 21-23",
            help="Only these selected page images are sent to Gemini when you click Analyze.",
        )
        analyze_pages_clicked = st.button(
            "🔎 Analyze selected PDF pages", use_container_width=True,
        )
        attach_images_in_answers = st.checkbox(
            "Send relevant analyzed page images when answering", value=False,
            help="Optional. Sends up to 2 selected source page images with an answer request.",
        )
    else:
        st.caption("Upload a PDF to analyze diagrams, charts and scanned pages.")
    st.caption("⚠️ Vision uses Gemini API requests and can use quota. Page images are "
               "not sent unless you click Analyze or enable the image option above.")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Index documents only when uploads change. Reuse the existing rag_core retrieval code.
if uploads:
    fingerprint = file_fingerprint(uploads)
    if fingerprint != st.session_state.get("index_fingerprint"):
        with st.spinner("Reading documents and generating embeddings..."):
            extracted = []
            processed = 0
            for uploaded in uploads:
                if uploaded.size > 15 * 1024 * 1024:
                    st.warning(f"Skipped {uploaded.name}: exceeds 15 MB.")
                    continue
                try:
                    chunks = extract_chunks(uploaded.name, uploaded.getvalue())
                    extracted.extend(chunks)
                    if chunks:
                        processed += 1
                except Exception as exc:
                    st.warning(f"Could not read {uploaded.name}: {exc}")
            st.session_state["base_chunks"] = extracted
            st.session_state["visual_cache"] = {}
            if extracted:
                try:
                    vectors = encode_chunks(extracted, get_encoder())
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
                    st.stop()
                st.session_state["chunks"] = extracted
                st.session_state["vectors"] = vectors
            else:
                st.session_state.pop("chunks", None)
                st.session_state.pop("vectors", None)
            st.session_state["index_fingerprint"] = fingerprint
            st.session_state["indexed_file_count"] = processed
            st.session_state["messages"] = []
else:
    if st.session_state.get("index_fingerprint") is not None:
        # Uploaded files were removed; do not leave old document answers on screen.
        st.session_state["messages"] = []
    for key in ("chunks", "vectors", "index_fingerprint", "indexed_file_count",
                "base_chunks", "visual_cache"):
        st.session_state.pop(key, None)

# Streamlit can retain Session State on source-code hot reload.
if uploads and "base_chunks" not in st.session_state:
    st.session_state["base_chunks"] = list(st.session_state.get("chunks", []))
if uploads and "visual_cache" not in st.session_state:
    st.session_state["visual_cache"] = {}

if analyze_pages_clicked and selected_pdf is not None:
    if not api_key.strip():
        st.sidebar.warning("Enter your Gemini API key first.")
    elif not model_name.strip():
        st.sidebar.warning("Enter a Gemini model name first.")
    else:
        try:
            pdf_bytes = selected_pdf.getvalue()
            pages = parse_page_selection(requested_pages, pdf_page_count(pdf_bytes))
        except Exception as exc:
            st.sidebar.error(str(exc))
        else:
            new_cache = dict(st.session_state.get("visual_cache", {}))
            client = genai.Client(api_key=api_key.strip())
            success_count = 0
            progress = st.sidebar.progress(0, text="Analyzing selected PDF pages...")
            for i, page in enumerate(pages, start=1):
                cache_id = (selected_pdf_index, page)
                try:
                    if cache_id not in new_cache:
                        png_bytes = render_pdf_page(pdf_bytes, page)
                        description = describe_pdf_image(
                            client, model_name.strip(), png_bytes, selected_pdf.name, page,
                        )
                        new_cache[cache_id] = make_visual_chunk(
                            selected_pdf.name, page, description,
                        )
                    success_count += 1
                except Exception as exc:
                    st.sidebar.warning(f"Page {page}: {friendly_error(exc)}")
                progress.progress(i / len(pages), text=f"Processed {i}/{len(pages)} pages")
            progress.empty()
            if success_count:
                combined = list(st.session_state.get("base_chunks", [])) + list(new_cache.values())
                try:
                    combined_vectors = encode_chunks(combined, get_encoder())
                except Exception as exc:
                    st.sidebar.error(f"Could not rebuild the search index: {type(exc).__name__}")
                else:
                    st.session_state["visual_cache"] = new_cache
                    st.session_state["chunks"] = combined
                    st.session_state["vectors"] = combined_vectors
                    st.session_state["messages"] = []
                    st.sidebar.success(f"Visually analyzed {success_count} page(s).")
                    st.rerun()

has_documents = "chunks" in st.session_state
indexed_files = st.session_state.get("indexed_file_count", 0) if has_documents else 0
passage_count = len(st.session_state["chunks"]) if has_documents else 0

# Main page
st.markdown(
    """
<div class="dm-hero">
    <div class="dm-eyebrow">✦ YOUR AI DOCUMENT WORKSPACE</div>
    <div class="dm-title">Ask your documents.<br><span>Find better answers.</span></div>
    <p class="dm-subtitle">Upload notes, reports or research papers. Ask questions and explore
    source-grounded answers powered by retrieval and Gemini.</p>
</div>
""",
    unsafe_allow_html=True,
)

metric_cols = st.columns(3, gap="small")
metrics = [
    ("📄", str(indexed_files), "Indexed documents"),
    ("🧩", str(passage_count), "Searchable passages"),
    ("🟢" if api_key.strip() else "🟡", "Gemini" if api_key.strip() else "Preview", "Answer mode"),
]
for col, (emoji, value, label) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f'<div class="dm-info-card"><div class="dm-metric">{emoji} {html.escape(value)}</div>'
            f'<div class="dm-label">{html.escape(label)}</div></div>',
            unsafe_allow_html=True,
        )

if has_documents:
    visual_count = len(st.session_state.get("visual_cache", {}))
    st.markdown(
        f'<div class="dm-ready">✓ Knowledge base ready · {indexed_files} document(s) · '
        f'{passage_count} passages indexed · {visual_count} visual page(s)</div>',
        unsafe_allow_html=True,
    )
    if visual_count:
        with st.expander(f"🖼️ View {visual_count} indexed diagram/table descriptions"):
            for visual in st.session_state["visual_cache"].values():
                st.markdown(f"**{source_label(visual)}**")
                st.caption(visual.text)
                st.divider()
elif uploads:
    st.warning("No searchable text detected. Use the optional 'PDF diagrams & tables' sidebar feature to analyze specific image-only pages with Gemini.")
else:
    st.markdown(
        '<div class="dm-empty">📂 <strong>No documents uploaded yet.</strong><br>'
        'Upload PDF, TXT or Markdown files in the sidebar to start chatting.</div>',
        unsafe_allow_html=True,
    )

suggested_prompt = None
if has_documents and not st.session_state["messages"]:
    st.markdown('<div class="dm-section">✨ Try a question</div>', unsafe_allow_html=True)
    suggestions = [
        "Summarize the main ideas in these documents.",
        "Explain the important concepts in simple words.",
        "Give examples with their page references.",
    ]
    prompt_cols = st.columns(3, gap="small")
    for i, (col, suggestion) in enumerate(zip(prompt_cols, suggestions)):
        with col:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                suggested_prompt = suggestion

st.markdown('<div class="dm-section">💬 Conversation</div>', unsafe_allow_html=True)
if not st.session_state["messages"]:
    st.caption("Your conversation will appear here. Answers are based on passages retrieved from your uploaded files.")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "✨"):
        st.markdown(message["text"])
        if message.get("sources"):
            show_sources(message["sources"])

typed_question = st.chat_input(
    "Ask anything about your uploaded documents...",
    disabled=not has_documents,
)
question = typed_question or suggested_prompt

if question and has_documents:
    st.session_state["messages"].append({"role": "user", "text": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Finding relevant passages..."):
            try:
                hits = retrieve(
                    question,
                    st.session_state["chunks"],
                    st.session_state["vectors"],
                    get_encoder(),
                    top_k,
                )
            except Exception as exc:
                st.error(f"Document search failed: {exc}")
                st.stop()

        source_data = [
            {"label": source_label(hit.chunk), "text": hit.chunk.text}
            for hit in hits
        ]
        if not hits:
            answer = "I couldn't find a relevant passage in the uploaded documents. Try a different question."
        elif not api_key.strip():
            previews = "\n\n".join(
                f"**{item['label']}**\n\n> {item['text'][:400]}"
                for item in source_data
            )
            answer = "**Retrieval preview** — Enter a Gemini API key to generate an answer.\n\n" + previews
        else:
            try:
                with st.spinner("Generating an answer with Gemini..."):
                    client = genai.Client(api_key=api_key.strip())
                    prompt = build_prompt(question, hits)
                    contents = [prompt]
                    if attach_images_in_answers and pdf_uploads:
                        # Optional: include page images for at most two retrieved
                        # *visually indexed* PDF passages, with source IDs preserved.
                        attachments = 0
                        for source_number, hit in enumerate(hits, start=1):
                            if attachments >= MAX_IMAGE_PAGE_ATTACHMENTS:
                                break
                            if not is_visual_chunk(hit.chunk) or hit.chunk.page is None:
                                continue
                            # Match the actual visual chunk instance to the original
                            # uploaded PDF. This avoids mixing two same-named PDFs.
                            matched_file = next(
                                (pdf_uploads[file_idx]
                                 for (file_idx, page_no), cached_chunk in
                                 st.session_state.get("visual_cache", {}).items()
                                 if cached_chunk is hit.chunk
                                 and page_no == hit.chunk.page
                                 and file_idx < len(pdf_uploads)),
                                None,
                            )
                            if matched_file is None:
                                continue
                            png_bytes = render_pdf_page(matched_file.getvalue(), hit.chunk.page)
                            contents.append(
                                f"The following image is source [S{source_number}] "
                                f"({source_label(hit.chunk)}). Check it against the visual summary."
                            )
                            contents.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))
                            attachments += 1
                    response = client.models.generate_content(
                        model=model_name.strip(),
                        contents=contents,
                    )
                answer = (response.text or "Gemini returned an empty answer.").strip()
                answer, citation_issues = resolve_citations(answer, hits)
                if citation_issues:
                    st.warning("Some generated citations could not be verified against retrieved sources. Check the source passages before relying on the answer.")
            except Exception as exc:
                answer = friendly_error(exc)
        st.markdown(answer)
        show_sources(source_data)

    st.session_state["messages"].append(
        {"role": "assistant", "text": answer, "sources": source_data}
    )
