import os
import requests
import streamlit as st
from datetime import datetime
from vertexai.generative_models import GenerativeModel
from vertexai import init as vertexai_init

S2_API_KEY = os.getenv("S2_API_KEY", "")
URL = "https://api.semanticscholar.org/graph/v1"

# ---------- Semantic Scholar ----------
def lit_search(topic: str, start_year: str, end_year: str, limit: int = 10):
    params = {
        "query": (topic or "").strip(),
        "fields": "title,abstract,year,authors,url",
        "limit": limit,
        "sort": "year",
    }
    if start_year and end_year:
        params["year"] = f"{start_year}-{end_year}"
    elif start_year and not end_year:
        params["year"] = f"{start_year}-"
    elif end_year and not start_year:
        params["year"] = f"-{end_year}"

    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}

    try:
        r = requests.get(f"{URL}/paper/search", params=params, headers=headers, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Semantic Scholar request failed: {e}")
        return {"papers": []}

    data = r.json() if r.content else {}
    papers = data.get("data", []) or []
    return {"papers": papers}



def init_vertex():
    project = os.getenv("GOOGLE_CLOUD_PROJECT")  # must be set separately
    location = "us-central1"  # pick a region where Gemini is available
    credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # path to JSON

    if not project or not credentials:
        st.error("Missing GOOGLE_CLOUD_PROJECT or GOOGLE_APPLICATION_CREDENTIALS env vars.")
        return False

    try:
        vertexai_init(project=project, location=location)
        return True
    except Exception as e:
        st.error(f"Vertex AI init failed: {e}")
        return False
    
def build_context_from_papers(papers, k: int = 8, max_abs_chars: int = 900):
    selected = papers[:k]
    source_lines, abstract_chunks = [], []
    for idx, p in enumerate(selected, 1):
        title = p.get("title") or "Untitled"
        year = p.get("year")
        url = p.get("url") or ""
        authors = ", ".join(a.get("name", "") for a in p.get("authors", []))
        source_lines.append(f"[{idx}] {title} ({year}) — {authors} — {url}".strip())

        abstract = (p.get("abstract") or "").strip()
        if abstract:
            short_abs = abstract[:max_abs_chars] + ("..." if len(abstract) > max_abs_chars else "")
            abstract_chunks.append(f"[{idx}] Title: {title}\nAbstract: {short_abs}")
        else:
            abstract_chunks.append(f"[{idx}] Title: {title}\nAbstract: (no abstract available)")
    return "\n".join(source_lines), "\n\n".join(abstract_chunks)

def ask_gemini_about_papers(user_prompt: str, papers):
    if not init_vertex():
        return None

    model = GenerativeModel("gemini-2.5-pro")

    source_list, abstracts_block = build_context_from_papers(papers)
    system_instructions = (
        "You are a biomedical research assistant. Use ONLY the provided sources. "
        "Answer concisely, in bullet points where helpful. Include bracketed citations "
        "like [1], [2] that refer to the numbered Source List. If uncertain, say so."
    )
    prompt = f"""{system_instructions}

User question:
{user_prompt}

Source List:
{source_list}

Abstracts:
{abstracts_block}

Instructions:
- Synthesize the main findings relevant to the question.
- Note any limitations or conflicts across studies.
- End with a short 'Further reading' section listing the most relevant 2–3 sources by their numbers.
- Cite with [#] that map to the Source List above.
"""

    try:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None) or "No response text."
    except Exception as e:
        st.error(f"Gemini request failed: {e}")
        return None




def main():
    st.title("🧬 Genomic Assistant")
    st.sidebar.header("🔍 Select a Function")

    option = st.sidebar.radio(
        "Choose a task:",
        ("💡 Literature Search", "🧪 Variant Analysis Flow", "📊 General Geneomic Query")
    )

    if option == "💡 Literature Search":
        st.subheader("📊 Literature Search")
        topic = st.text_input("Enter a set of keywords for your topic:")
        col1, col2, col3 = st.columns(3)
        with col1:
            this_year = datetime.now().year
            start_year = st.text_input("Start year (YYYY):", value=str(this_year - 5))
        with col2:
            end_year = st.text_input("End year (YYYY):", value=str(this_year))
        with col3:
            limit = st.number_input("Results", min_value=1, max_value=50, value=10, step=1)

        user_prompt = st.text_input(
            "Add any specific question/details to answer from the literature (optional):",
            placeholder="e.g., Does DRD4 activation correlate with pyschological conditions?"
        )


        # Run search op
        if st.button("Search"):
            result = lit_search(topic, start_year, end_year, limit=int(limit))
            st.session_state["papers"] = result.get("papers", [])
            st.session_state["user_prompt"] = user_prompt.strip()

        # Render results
        papers = st.session_state.get("papers", [])
        if papers:
            st.success(f"Found {len(papers)} papers (showing up to {limit}).")
            for i, p in enumerate(papers[:int(limit)], 1):
                title = p.get("title") or "Untitled"
                year = p.get("year")
                authors = ", ".join(a.get("name","") for a in p.get("authors", []))
                abstract = p.get("abstract") or "(no abstract available)"
                url = p.get("url")

                st.markdown(f"**{i}. {title} ({year})**")
                if authors:
                    st.caption(authors)
                if url:
                    st.markdown(f"[View on Semantic Scholar]({url})")
                with st.expander("Abstract"):
                    st.write(abstract)
                st.divider()

        # Ask Gemini (separate button, works across reruns)
        if papers and st.session_state.get("user_prompt"):
            if st.button("Ask Gemini about these results"):
                with st.spinner("Thinking with Gemini…"):
                    answer = ask_gemini_about_papers(st.session_state["user_prompt"], papers[:int(limit)])
                st.subheader("🧠 Gemini Synthesis")
                st.write(answer or "No response from Gemini.")
                st.session_state["gemini_answer"] = answer or ""

        # Optional: download the synthesis
        if st.session_state.get("gemini_answer"):
            st.download_button(
                "Download summary",
                data=st.session_state["gemini_answer"],
                file_name="gemini_summary.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
