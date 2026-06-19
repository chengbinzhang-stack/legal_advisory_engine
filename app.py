"""
Legal Data Protection Engine and Legal Advisory Chatbot
Main Streamlit Application
"""
import streamlit as st
import os

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
    config = EngineConfig()
    st.session_state.legal_engine = LegalDataEngine(config)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "minimax"
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-2.5-flash"

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

def main():
    st.title("[Legal] Data Protection Engine & Advisory Chatbot")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Website Analysis", "Legal Advisory Chatbot", "Summary Dashboard"]
    )
    if page == "Website Analysis":
        render_website_analysis_page()
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

    # LLM Provider & Model selection
    config = EngineConfig()
    with st.sidebar:
        st.subheader("LLM Settings")
        provider = st.selectbox(
            "Provider",
            ["minimax", "gemini"],
            index=0 if config.llm_provider == "minimax" else 1,
            key="llm_provider_select"
        )
        if provider == "minimax":
            api_key = os.environ.get("MINIMAX_API_KEY") or EngineConfig().minimax_api_key
            st.session_state.llm_provider = "minimax"
        else:
            api_key = os.environ.get("GEMINI_API_KEY") or EngineConfig().gemini_api_key
            st.session_state.llm_provider = "gemini"
            if api_key:
                gemini_models = [
                    "gemini-2.5-flash",
                    "gemini-3-flash",
                    "gemini-3.1-flash-lite",
                    "gemini-2.5-flash-lite",
                ]
                selected_model = st.selectbox("Gemini Model", gemini_models, index=0)
                st.session_state.gemini_model = selected_model

        if not api_key:
            st.warning(f"{provider.upper()} API key not configured.")

    if provider == "minimax":
        api_key = os.environ.get("MINIMAX_API_KEY") or EngineConfig().minimax_api_key
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or EngineConfig().gemini_api_key

    if not api_key:
        if provider == "minimax":
            st.warning("MINIMAX_API_KEY not configured. Please set it in environment variables.")
            st.code("export MINIMAX_API_KEY=your_api_key_here", language="bash")
        else:
            st.warning("GEMINI_API_KEY not configured. Please set it in environment variables.")
            st.code("export GEMINI_API_KEY=your_api_key_here", language="bash")
        return

    config.llm_provider = provider

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
            data.append({
                "Domain": domain,
                "URL": summary.get("website_url", "N/A"),
                "Category": summary.get("category", "N/A"),
                "Scraping": summary.get("permissions", {}).get("scraping", {}).get("level", "N/A"),
                "Storing": summary.get("permissions", {}).get("storing", {}).get("level", "N/A"),
                "Display": summary.get("permissions", {}).get("free_display", {}).get("level", "N/A"),
                "Redistributing": summary.get("permissions", {}).get("free_redistribute", {}).get("level", "N/A")
            })
    if data:
        import pandas as pd
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No summary data found.")

if __name__ == "__main__":
    main()
