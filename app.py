"""
Legal Data Protection Engine and Legal Advisory Chatbot
Main Streamlit Application
"""
import streamlit as st
import os
import json
from datetime import datetime

st.set_page_config(
    page_title="Legal Advisory Engine",
    page_icon="scales",
    layout="wide"
)

from config import EngineConfig
from src.legal_data_engine import LegalDataEngine
from src.models.legal_analysis import PermissionLevel
from src.chatbot.response_generator import ResponseGenerator, MiniMaxClient, GeminiClient
from src.chatbot.prompt_builder import PromptBuilder
from src.rag.query_engine import QueryEngine

if "legal_engine" not in st.session_state:
    st.session_state.legal_engine = LegalDataEngine()

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

def get_llm_client():
    """Get or create MiniMax LLM client."""
    config = EngineConfig()
    api_key = config.minimax_api_key or os.environ.get("MINIMAX_API_KEY")
    if api_key and (st.session_state.get("llm_client") is None):
        st.session_state.llm_client = MiniMaxClient(api_key=api_key, base_url=config.minimax_base_url)
    return st.session_state.get("llm_client")


def get_gemini_client():
    """Get or create Gemini LLM client."""
    if "gemini_client" not in st.session_state:
        st.session_state.gemini_client = None
    config = EngineConfig()
    api_key = config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    model = st.session_state.get("gemini_model", config.gemini_model)
    if api_key and (st.session_state.gemini_client is None or st.session_state.gemini_client.model != model):
        st.session_state.gemini_client = GeminiClient(api_key=api_key, model=model)
    return st.session_state.gemini_client

def get_response_generator():
    """Get or create response generator with the currently selected LLM client."""
    provider = st.session_state.get("llm_provider", "minimax")

    if provider == "gemini":
        client = get_gemini_client()
    else:
        client = get_llm_client()

    if "response_generator" not in st.session_state:
        engine = st.session_state.legal_engine
        query_engine = QueryEngine(
            engine.embedding_generator,
            engine.chroma_client
        )
        prompt_builder = PromptBuilder()
        st.session_state.response_generator = ResponseGenerator(
            query_engine=query_engine,
            prompt_builder=prompt_builder,
            llm_client=client,
            summaries_directory=st.session_state.legal_engine.config.summaries_directory
        )
    else:
        # Update client reference if provider changed
        st.session_state.response_generator.llm_client = client
    return st.session_state.response_generator

def render_llm_settings_sidebar():
    """Render LLM provider/model selector in sidebar. Returns selected provider."""
    config = EngineConfig()
    st.sidebar.subheader("🤖 LLM Settings")

    # Initialize session state defaults
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = config.llm_provider
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = config.gemini_model

    provider = st.sidebar.selectbox(
        "Provider",
        ["minimax", "gemini"],
        index=0 if st.session_state.llm_provider == "minimax" else 1,
        key="llm_provider_select"
    )
    st.session_state.llm_provider = provider

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or config.gemini_api_key
        if api_key:
            gemini_models = [
                "gemini-2.5-flash",
                "gemini-3-flash",
                "gemini-3.1-flash-lite",
                "gemini-2.5-flash-lite",
            ]
            model_idx = 0
            if st.session_state.gemini_model in gemini_models:
                model_idx = gemini_models.index(st.session_state.gemini_model)
            selected_model = st.sidebar.selectbox(
                "Gemini Model",
                gemini_models,
                index=model_idx,
                key="gemini_model_select"
            )
            st.session_state.gemini_model = selected_model
        else:
            st.sidebar.warning("GEMINI_API_KEY not set in Secrets")

    return provider


def main():
    st.title("[Legal] Data Protection Engine & Advisory Chatbot")

    # LLM settings in sidebar
    current_provider = render_llm_settings_sidebar()

    # Update config from session state (for legal_data_engine)
    config = EngineConfig()
    config.llm_provider = st.session_state.llm_provider
    config.gemini_model = st.session_state.gemini_model

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Website Analysis", "Manual Submission", "Legal Advisory Chatbot", "Summary Dashboard"]
    )
    if page == "Website Analysis":
        render_website_analysis_page()
    elif page == "Manual Submission":
        render_manual_submission_page()
    elif page == "Legal Advisory Chatbot":
        render_chatbot_page()
    elif page == "Summary Dashboard":
        render_dashboard_page()

def render_website_analysis_page():
    st.header("Website Legal Analysis")
    url_input = st.text_input(
        "Enter website URL to analyze",
        placeholder="https://example.com"
    )
    if st.button("Analyze Website"):
        if url_input:
            with st.spinner("Analyzing website legal documents..."):
                try:
                    analysis = st.session_state.legal_engine.process_website(url_input)
                    display_analysis(analysis)
                    display_scraped_urls(url_input)
                except Exception as e:
                    st.error(f"Error analyzing website: {str(e)}")
    else:
        st.info("Enter a URL and click Analyze Website to start.")

def display_analysis(analysis):
    st.success(f"Analysis complete for {analysis.website_domain}!")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Website Information")
        st.write(f"**URL:** {analysis.website_url}")
        st.write(f"**Domain:** {analysis.website_domain}")
        st.write(f"**Category:** Bucket {analysis.category.value}")
    with col2:
        st.subheader("Permission Summary")
        for param, perm in analysis.permissions.items():
            if perm.permission == PermissionLevel.ALLOWED:
                label = f"[ALLOWED] {param}"
            elif perm.permission == PermissionLevel.NOT_ALLOWED:
                label = f"[DENIED] {param}"
            else:
                label = f"[UNCERTAIN] {param}"
            with st.expander(label):
                if perm.reasoning:
                    st.text(f"Reasoning: {perm.reasoning}")
                if perm.relevant_excerpts:
                    st.markdown("**Evidence:**")
                    for i, exc in enumerate(perm.relevant_excerpts, 1):
                        if isinstance(exc, dict):
                            source = exc.get('source', '')
                            text = exc.get('text', '')[:500]
                            if source and source.startswith('http'):
                                st.markdown(f"[{source}]({source})")
                            elif source:
                                st.text(f"Source: {source}")
                            st.markdown(f"> \"{text}\"")
                        else:
                            st.markdown(f"> \"{str(exc)[:500]}\"")
    if analysis.unique_findings:
        st.subheader("Unique Findings")
        for finding in analysis.unique_findings:
            st.write(f"- {finding}")
    st.subheader("Analysis Summary")
    st.text(analysis.summary_text)


def display_analysis_from_dict(data: dict):
    """Display analysis results from a dict (used for manual submissions)."""
    st.success(f"Analysis complete for {data['website_domain']}!")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Website Information")
        st.write(f"**URL:** {data['website_url']}")
        st.write(f"**Domain:** {data['website_domain']}")
        st.write(f"**Category:** Bucket {data['category']}" if isinstance(data['category'], int) else f"**Category:** {data['category']}")
        source = data.get("source", "auto")
        st.write(f"**Source:** {'Manual Submission' if source == 'manual' else 'Auto Scraped'}")
    with col2:
        st.subheader("Permission Summary")
        for param, perm in data.get("permissions", {}).items():
            level = perm.get("level", "uncertain")
            if level == "allowed":
                label = f"[✅ ALLOWED] {param}"
            elif level == "not_allowed":
                label = f"[❌ DENIED] {param}"
            else:
                label = f"[❓ UNCERTAIN] {param}"
            with st.expander(label):
                reasoning = perm.get("reasoning", "")
                if reasoning:
                    st.text(f"Reasoning: {reasoning}")
                excerpts = perm.get("relevant_excerpts", [])
                if excerpts:
                    st.markdown("**Evidence:**")
                    for exc in excerpts:
                        if isinstance(exc, dict):
                            source = exc.get('source', '')
                            text = exc.get('text', '')[:500]
                            st.markdown(f"> \"{text}\"")
                        else:
                            st.markdown(f"> \"{str(exc)[:500]}\"")
    findings = data.get("unique_findings", [])
    if findings:
        st.subheader("Unique Findings")
        for finding in findings:
            st.write(f"- {finding}")
    st.subheader("Analysis Summary")
    st.text(data.get("summary_text", ""))

def display_scraped_urls(url):
    """Display successfully scraped document URLs in an expander."""
    website_data = st.session_state.legal_engine._scrape_website(url)
    successful_docs = [doc for doc in website_data.documents if doc.success]
    if not successful_docs:
        return
    with st.expander("Scraped Documents"):
        st.markdown(f"**Total successful: {len(successful_docs)}**")
        for doc in successful_docs:
            st.markdown(f"- **{doc.document_type}**: [{doc.url}]({doc.url})")

def render_chatbot_page():
    st.header("Legal Advisory Chatbot")

    # Read provider from session state (set by main() sidebar)
    provider = st.session_state.get("llm_provider", "minimax")
    config = EngineConfig()
    if provider == "minimax":
        api_key = os.environ.get("MINIMAX_API_KEY") or config.minimax_api_key
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or config.gemini_api_key

    if not api_key:
        st.warning(f"{provider.upper()} API key not configured. Add it in Settings → Secrets.")
        return

    websites = get_analyzed_websites()
    if not websites:
        st.warning("No websites analyzed yet. Please analyze a website first.")
        return
    selected_website = st.selectbox("Select a website", websites)
    if selected_website:
        render_chat_interface(selected_website)

def get_analyzed_websites():
    config = EngineConfig()
    summary_dir = config.summaries_directory
    websites = []
    if os.path.exists(summary_dir):
        for f in os.listdir(summary_dir):
            if f.startswith("summary_") and f.endswith(".json"):
                domain = f.replace("summary_", "").replace(".json", "").replace("_", ".")
                websites.append(domain)
    return websites

def render_chat_interface(website_domain: str):
    if website_domain not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[website_domain] = []
    for message in st.session_state.chat_sessions[website_domain]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    user_query = st.chat_input(f"Ask about {website_domain} legal terms...")
    if user_query:
        st.session_state.chat_sessions[website_domain].append({
            "role": "user",
            "content": user_query
        })
        with st.chat_message("user"):
            st.write(user_query)
        with st.spinner("Generating response..."):
            try:
                generator = get_response_generator()
                result = generator.generate_response(
                    query=user_query,
                    website_domain=website_domain
                )
                response = result["response"]
            except Exception as e:
                response = f"Error: {str(e)}"
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.chat_sessions[website_domain].append({
            "role": "assistant",
            "content": response
        })

def render_dashboard_page():
    st.header("Website Summary Dashboard")
    websites = get_analyzed_websites()
    if not websites:
        st.info("No websites have been analyzed yet.")
        return
    config = EngineConfig()
    summary_dir = config.summaries_directory
    data = []
    for domain in websites:
        summary_path = os.path.join(summary_dir, f"summary_{domain.replace('.', '_')}.json")
        if os.path.exists(summary_path):
            import json
            with open(summary_path, "r") as f:
                summary = json.load(f)
            def fmt_level(level):
                if level == "allowed": return "✅"
                if level == "not_allowed": return "❌"
                if level == "uncertain": return "❓"
                return level
            data.append({
                "Domain": domain,
                "URL": summary.get("website_url", "N/A"),
                "Category": f"Bucket {summary.get('category', '?')}",
                "Scraping": fmt_level(summary.get("permissions", {}).get("scraping", {}).get("level", "")),
                "Storing": fmt_level(summary.get("permissions", {}).get("storing", {}).get("level", "")),
                "Display": fmt_level(summary.get("permissions", {}).get("free_display", {}).get("level", "")),
                "Redistributing": fmt_level(summary.get("permissions", {}).get("free_redistribute", {}).get("level", ""))
            })
    if data:
        import pandas as pd
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No summary data found.")

def render_manual_submission_page():
    config = EngineConfig()
    st.header("📝 Manual Terms Submission")
    st.info("Use this if automatic scraping failed or you want to provide your own terms of use document.")

    url_input = st.text_input("Website URL", placeholder="https://example.com", key="manual_url")
    terms_text = st.text_area(
        "Terms of Use / Privacy Policy text",
        placeholder="Paste the full legal text here...",
        height=300,
        key="manual_terms"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        submit = st.button("Analyze & Save", type="primary")

    if submit:
        if not url_input:
            st.warning("Please enter a website URL.")
            return
        if not terms_text or len(terms_text) < 200:
            st.warning("Please enter a valid legal document (at least 200 characters).")
            return

        domain = url_input.replace("https://", "").replace("http://", "").split("/")[0]

        with st.spinner("Saving document and analyzing..."):
            try:
                # 1. Save raw document
                manual_dir = os.path.join(config.data_directory, "manual_docs")
                os.makedirs(manual_dir, exist_ok=True)
                raw_path = os.path.join(manual_dir, f"{domain}_raw.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(f"URL: {url_input}\n")
                    f.write(f"Submitted: {datetime.now().isoformat()}\n")
                    f.write(f"{'='*60}\n\n")
                    f.write(terms_text)

                # 2. Use LegalClassifier to analyze
                api_key = os.environ.get("MINIMAX_API_KEY") or config.minimax_api_key
                gemini_key = os.environ.get("GEMINI_API_KEY") or config.gemini_api_key
                provider = st.session_state.get("llm_provider", "minimax")

                if provider == "gemini":
                    classifier_api_key = gemini_key
                    classifier_base_url = None
                    classifier_provider = "gemini"
                    classifier_gemini_model = st.session_state.get("gemini_model", config.gemini_model)
                else:
                    classifier_api_key = api_key
                    classifier_base_url = config.minimax_base_url
                    classifier_provider = "minimax"
                    classifier_gemini_model = None

                if not classifier_api_key:
                    st.warning(f"{provider.upper()} API key not configured.")
                    return

                from src.classifier.legal_classifier import LegalClassifier
                classifier = LegalClassifier(
                    api_key=classifier_api_key,
                    base_url=classifier_base_url or "https://api.minimax.chat/v1",
                    provider=classifier_provider,
                    gemini_api_key=classifier_gemini_model if provider == "gemini" else None,
                    gemini_model=classifier_gemini_model if provider == "gemini" else "gemini-2.5-flash"
                )

                analysis = classifier.classify_permissions(
                    text=terms_text,
                    website_url=url_input,
                    website_domain=domain,
                    robots_txt="",
                    document_urls={"manual_submission": url_input}
                )

                # 3. Save analysis as summary JSON (same format as automatic analysis)
                summary_path = os.path.join(config.summaries_directory, f"summary_{domain.replace('.', '_')}.json")
                os.makedirs(config.summaries_directory, exist_ok=True)

                summary_data = {
                    "website_url": url_input,
                    "website_domain": domain,
                    "category": analysis.category.value if hasattr(analysis.category, 'value') else analysis.category,
                    "summary_text": analysis.summary_text,
                    "permissions": {
                        param: {
                            "level": perm.permission.value,
                            "reasoning": perm.reasoning,
                            "relevant_excerpts": perm.relevant_excerpts,
                            "source_documents": perm.source_documents,
                        } for param, perm in analysis.permissions.items()
                    },
                    "unique_findings": getattr(analysis, 'unique_findings', []),
                    "source": "manual",  # Mark as manual submission
                    "raw_doc_path": raw_path,
                    "analyzed_at": datetime.now().isoformat()
                }

                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary_data, f, indent=2, ensure_ascii=False)

                st.success(f"✅ Saved and analyzed! Domain: {domain}")
                display_analysis_from_dict(summary_data)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Show previously submitted manual documents
    st.divider()
    st.subheader("📂 Previously Submitted Documents")
    manual_dir = os.path.join(config.data_directory, "manual_docs")
    if os.path.exists(manual_dir):
        files = [f for f in os.listdir(manual_dir) if f.endswith("_raw.txt")]
        if files:
            for f in sorted(files):
                domain = f.replace("_raw.txt", "")
                with st.expander(domain):
                    path = os.path.join(manual_dir, f)
                    content = open(path, "r", encoding="utf-8").read()
                    st.text(content[:1000] + ("..." if len(content) > 1000 else ""))
                    st.caption(f"Full text: {path}")
        else:
            st.info("No manual submissions yet.")


if __name__ == "__main__":
    main()
