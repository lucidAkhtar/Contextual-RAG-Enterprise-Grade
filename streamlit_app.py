"""
Streamlit UI for Contextual RAG System
Interactive interface for querying, benchmarking, and analyzing the RAG system.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, List
import time

# Page configuration
st.set_page_config(
    page_title="Contextual RAG System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


def check_api_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    try:
        response = requests.get(f"{API_BASE_URL}/info")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def query_rag_system(
    question: str,
    method: str = "hybrid",
    top_k: int = 5
) -> Dict[str, Any]:
    """Query the RAG system."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "q": question,
                "k": top_k,
                "retrieval_method": method
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/cache/stats")
        if response.status_code == 200:
            return response.json().get("cache_stats", {})
    except:
        pass
    return {}


def get_query_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get query history."""
    try:
        response = requests.get(f"{API_BASE_URL}/cache/history?limit={limit}")
        if response.status_code == 200:
            return response.json().get("history", [])
    except:
        pass
    return []


def load_ground_truth() -> List[Dict[str, Any]]:
    """Load ground truth data."""
    try:
        with open("data/ground_truth_arxiv.json", "r") as f:
            data = json.load(f)
            # Return qa_pairs array from the new structure
            return data.get("qa_pairs", [])
    except:
        return []


def get_audit_logs(limit: int = 50, user_id: str = None) -> List[Dict[str, Any]]:
    """Get audit logs."""
    try:
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        response = requests.get(f"{API_BASE_URL}/audit/logs", params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("audit_logs", [])
    except:
        pass
    return []


def get_audit_stats() -> Dict[str, Any]:
    """Get audit statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/audit/stats", timeout=5)
        if response.status_code == 200:
            return response.json().get("audit_stats", {})
    except:
        pass
    return {}


def main():
    """Main Streamlit application."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=RAG+System", use_column_width=True)
        st.title("Navigation")
        
        # Check API health
        if check_api_health():
            st.success("API Connected")
        else:
            st.error("API Offline")
            st.warning("Please start the API server:\n```bash\npython -m src.main\n```")
            return
        
        # Get system info
        system_info = get_system_info()
        if system_info:
            st.info(f"Documents: {len(system_info.get('document_names', ['Unknown']))}")
            st.info(f"Chunks: {system_info.get('num_chunks', 'N/A')}")
            st.info(f"Cache: {'✓' if system_info.get('cache_enabled') else '✗'}")
        
        # Navigation
        page = st.radio(
            "Select Page",
            [" Query System", " Analytics & Benchmarks", " Audit Logs", " Ground Truth Data", " System Info"]
        )
    
    # Main content
    if page == " Query System":
        render_query_page()
    elif page == " Analytics & Benchmarks":
        render_analytics_page()
    elif page == " Audit Logs":
        render_audit_logs_page()
    elif page == " Ground Truth Data":
        render_ground_truth_page()
    elif page == " System Info":
        render_system_info_page()


def render_query_page():
    """Render the main query interface."""
    st.markdown('<p class="main-header"> Contextual RAG System</p>', unsafe_allow_html=True)
    st.markdown("Ask questions and get answers from your documents using advanced retrieval methods.")
    
    # Query input
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="What is the improvement in retrieval accuracy using contextual RAG?"
        )
    with col2:
        method = st.selectbox(
            "Retrieval Method",
            ["hybrid", "contextual", "bm25", "tfidf"],
            help="Select the retrieval algorithm to use"
        )
    with col3:
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of chunks to retrieve"
        )
    
    if st.button(" Get Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question!")
            return
        
        with st.spinner(" Processing your question..."):
            start_time = time.time()
            result = query_rag_system(question, method, top_k)
            elapsed_time = time.time() - start_time
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
            return
        
        # Display confidence score prominently
        st.markdown("---")
        confidence_score = result.get('confidence_score', 0.0)
        confidence_level = result.get('confidence_level', 'unknown')
        
        # Confidence badge color
        confidence_colors = {
            'high': ('🟢', '#27ae60', 'High Confidence'),
            'medium': ('🟡', '#f39c12', 'Medium Confidence'),
            'low': ('🔴', '#e74c3c', 'Low Confidence')
        }
        badge_emoji, badge_color, badge_text = confidence_colors.get(confidence_level, ('⚪', '#95a5a6', 'Unknown'))
        
        col_conf1, col_conf2 = st.columns([3, 1])
        with col_conf1:
            st.markdown(f"""
            <div style="background-color: {badge_color}20; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid {badge_color};">
                <h3 style="margin: 0; color: {badge_color};">{badge_emoji} {badge_text}</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Confidence Score: <strong>{confidence_score:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        with col_conf2:
            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': badge_color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "lightblue"},
                        {'range': [75, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Display answer
        st.markdown("---")
        st.subheader(" Answer")
        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #3498db;">
            {result.get('answer', 'No answer generated')}
        </div>
        """, unsafe_allow_html=True)
        
        # Display citations (enterprise feature)
        citations = result.get('citations', [])
        if citations:
            st.markdown("---")
            st.subheader(" Citations & Source Verification")
            st.markdown("*Banking-grade answer verification with source traceability*")
            
            for idx, citation in enumerate(citations, 1):
                col_cite1, col_cite2 = st.columns([4, 1])
                with col_cite1:
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 3px solid #667eea;">
                        <p style="margin: 0; font-weight: bold; color: #667eea;"> Citation {idx}: {citation.get('source_document', 'Unknown')}</p>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;"><strong>Page:</strong> {citation.get('page', 'N/A')} | <strong>Chunk ID:</strong> {citation.get('chunk_id', 'N/A')}</p>
                        <p style="margin: 0.5rem 0 0 0; font-style: italic; color: #555;">"{citation.get('excerpt', 'No excerpt')}"</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_cite2:
                    st.metric("Citation Confidence", f"{citation.get('confidence', 0):.1%}")
        
        # Display metrics
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        stats = result.get('retrieval_stats', {})
        with col1:
            st.metric(" Total Latency", f"{result.get('latency_ms', 0):.0f} ms")
        with col2:
            st.metric(" Retrieval Time", f"{stats.get('retrieval_time_ms', 0):.0f} ms")
        with col3:
            st.metric(" Generation Time", f"{stats.get('generation_time_ms', 0):.0f} ms")
        with col4:
            cache_hit = stats.get('cache_hit', False)
            st.metric(" Cache", "HIT ✓" if cache_hit else "MISS")
        with col5:
            st.metric(" Citations", len(citations))
        
        # Display sources
        st.markdown("---")
        st.subheader(" Retrieved Sources")
        
        sources = result.get('sources', [])
        if sources:
            for idx, source in enumerate(sources, 1):
                with st.expander(f"Source {idx} - {source.get('source_document', 'Unknown')} (Page {source.get('page', 'N/A')}) - Score: {source.get('score', 0):.3f}"):
                    st.markdown(f"**Method:** {source.get('method', 'Unknown')}")
                    st.markdown(f"**Content:**\n\n{source.get('content', 'No content')}")
                    
                    # Show method scores for hybrid
                    if source.get('method') == 'hybrid' and 'metadata' in source:
                        method_scores = source['metadata'].get('method_scores', {})
                        if method_scores:
                            st.markdown("**Individual Method Scores:**")
                            score_df = pd.DataFrame([method_scores])
                            st.dataframe(score_df, use_container_width=True)
        else:
            st.info("No sources retrieved")


def render_analytics_page():
    """Render analytics and benchmarks page."""
    st.markdown('<p class="main-header"> Analytics & Benchmarks</p>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([" Query History", " Cache Statistics", " Benchmarks"])
    
    with tab1:
        st.subheader("Recent Query History")
        history = get_query_history(100)
        
        if history:
            # Convert to DataFrame
            df = pd.DataFrame(history)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(" Total Queries", len(df))
            with col2:
                avg_latency = df['latency_ms'].mean()
                st.metric(" Avg Latency", f"{avg_latency:.1f} ms")
            with col3:
                cache_hit_rate = (df['cache_hit'].sum() / len(df)) * 100
                st.metric(" Cache Hit Rate", f"{cache_hit_rate:.1f}%")
            with col4:
                most_used_method = df['method'].mode()[0] if len(df) > 0 else "N/A"
                st.metric(" Top Method", most_used_method)
            
            # Latency over time
            st.subheader("Latency Over Time")
            fig = px.line(df, x=df.index, y='latency_ms', color='method',
                         title="Query Latency by Method",
                         labels={'latency_ms': 'Latency (ms)', 'index': 'Query Number'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Method distribution
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Method Usage")
                method_counts = df['method'].value_counts()
                fig = px.pie(values=method_counts.values, names=method_counts.index,
                            title="Retrieval Method Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Latency by Method")
                fig = px.box(df, x='method', y='latency_ms',
                            title="Latency Distribution by Method",
                            labels={'latency_ms': 'Latency (ms)'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent queries table
            st.subheader("Recent Queries")
            display_df = df[['query', 'method', 'latency_ms', 'cache_hit', 'timestamp']].head(20)
            st.dataframe(display_df, use_container_width=True)
            
        else:
            st.info("No query history available yet. Start asking questions!")
    
    with tab2:
        st.subheader("Cache Performance")
        cache_stats = get_cache_stats()
        
        if cache_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", cache_stats.get('total_queries', 0))
            with col2:
                st.metric("Cache Hits", cache_stats.get('cache_hits', 0))
            with col3:
                st.metric("Cache Misses", cache_stats.get('cache_misses', 0))
            with col4:
                st.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1f}%")
            
            # Visualize cache performance
            hit_rate = cache_stats.get('hit_rate', 0)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=hit_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Cache Hit Rate (%)"},
                delta={'reference': 30, 'suffix': "%"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 20], 'color': "#ffcccc"},
                        {'range': [20, 40], 'color': "#fff4cc"},
                        {'range': [40, 100], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f" Cached Entries: {cache_stats.get('cached_entries', 0)}")
            
        else:
            st.info("No cache statistics available")
    
    with tab3:
        st.subheader("Benchmark Results")
        
        # Check if benchmark file exists
        benchmark_path = Path("benchmarks/results.md")
        if benchmark_path.exists():
            with open(benchmark_path, 'r') as f:
                benchmark_content = f.read()
            st.markdown(benchmark_content)
        else:
            st.warning("No benchmark results found. Run benchmarks using:")
            st.code("python -m scripts.run_benchmarks", language="bash")
            
            # Show sample benchmark data
            st.subheader("Sample Benchmark Comparison")
            sample_data = {
                'Method': ['Traditional RAG', 'BM25', 'Contextual', 'Hybrid'],
                'Recall@5': [0.67, 0.71, 0.82, 0.91],
                'Latency (ms)': [145, 95, 156, 178],
                'Semantic Similarity': [0.78, 0.74, 0.89, 0.93]
            }
            df = pd.DataFrame(sample_data)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(df, x='Method', y='Recall@5',
                            title="Recall@5 by Method",
                            color='Method')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='Method', y='Semantic Similarity',
                            title="Semantic Similarity by Method",
                            color='Method')
                st.plotly_chart(fig, use_container_width=True)


def render_audit_logs_page():
    """Render enterprise audit logs page for banking compliance."""
    st.markdown('<p class="main-header"> Audit Logs & Compliance</p>', unsafe_allow_html=True)
    st.markdown("**Banking-grade audit trail for regulatory compliance and security monitoring.**")
    
    # Get audit statistics
    audit_stats = get_audit_stats()
    
    if audit_stats:
        # Display audit statistics
        st.markdown("###  Audit Statistics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(" Total Queries", audit_stats.get('total_queries', 0))
        with col2:
            st.metric(" Successful", audit_stats.get('successful_queries', 0))
        with col3:
            st.metric(" Failed", audit_stats.get('failed_queries', 0))
        with col4:
            st.metric(" Unique Users", audit_stats.get('unique_users', 0))
        with col5:
            st.metric(" Documents Accessed", audit_stats.get('unique_documents', 0))
        with col6:
            avg_conf = audit_stats.get('avg_confidence', 0.0)
            st.metric(" Avg Confidence", f"{avg_conf:.1%}")
        
        # Confidence distribution gauge
        st.markdown("###  System Confidence Overview")
        fig_conf = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_conf * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%"},
            delta={'reference': 75, 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 75], 'color': "#ffffcc"},
                    {'range': [75, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            },
            title={'text': "Average Query Confidence"}
        ))
        fig_conf.update_layout(height=300)
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Audit log viewer
    st.markdown("---")
    st.markdown("### 📋 Audit Log Entries")
    
    # Filters
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        limit = st.slider("Number of logs to display", min_value=10, max_value=200, value=50, step=10)
    with col_f2:
        success_filter = st.selectbox("Status Filter", ["All", "Successful Only", "Failed Only"])
    
    # Fetch audit logs
    audit_logs = get_audit_logs(limit=limit)
    
    if audit_logs:
        st.info(f"Showing {len(audit_logs)} audit log entries")
        
        # Apply success filter
        if success_filter == "Successful Only":
            audit_logs = [log for log in audit_logs if log.get('success', True)]
        elif success_filter == "Failed Only":
            audit_logs = [log for log in audit_logs if not log.get('success', True)]
        
        # Display logs in an expandable format
        for idx, log in enumerate(audit_logs, 1):
            success = log.get('success', True)
            badge = "CORRECT" if success else "INCORRECT"
            confidence = log.get('confidence_score', 0.0)
            timestamp = log.get('timestamp', 'N/A')
            
            with st.expander(f"{badge} Log {idx} - {timestamp} - Confidence: {confidence:.1%}"):
                col_log1, col_log2 = st.columns([2, 1])
                
                with col_log1:
                    st.markdown(f"**Log ID:** `{log.get('log_id', 'N/A')}`")
                    st.markdown(f"**Timestamp:** {timestamp}")
                    st.markdown(f"**Query:** {log.get('query', 'N/A')}")
                    st.markdown(f"**Answer:** {log.get('answer', 'N/A')[:200]}...")
                    st.markdown(f"**User ID:** {log.get('user_id', 'Anonymous')}")
                    st.markdown(f"**Session ID:** {log.get('session_id', 'N/A')}")
                    st.markdown(f"**IP Address:** {log.get('ip_address', 'N/A')}")
                
                with col_log2:
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.metric("Latency", f"{log.get('latency_ms', 0):.0f} ms")
                    st.metric("Cache Hit", "Yes" if log.get('cache_hit', False) else "No")
                    st.metric("Status", "Success" if success else "Failed")
                
                # Documents accessed
                docs_accessed = log.get('documents_accessed', [])
                if docs_accessed:
                    st.markdown(f"**Documents Accessed ({len(docs_accessed)}):**")
                    for doc in docs_accessed:
                        st.markdown(f"-  {doc}")
                
                # Retrieval method
                st.markdown(f"**Retrieval Method:** `{log.get('retrieval_method', 'N/A')}`")
                
                # Error message if failed
                if not success and log.get('error_message'):
                    st.error(f"**Error:** {log.get('error_message')}")
                
                # Metadata
                metadata = log.get('metadata', {})
                if metadata:
                    st.markdown("**Additional Metadata:**")
                    st.json(metadata)
        
        # Export functionality
        st.markdown("---")
        st.download_button(
            label=" Export Audit Logs (JSON)",
            data=json.dumps(audit_logs, indent=2),
            file_name=f"audit_logs_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.warning("No audit logs available. Start querying the system to generate audit logs.")
    
    # Compliance information
    st.markdown("---")
    st.markdown("###  Compliance Information")
    st.info("""
    **Enterprise Audit Logging Features:**
    -  Full query traceability with unique log IDs
    -  User and session tracking
    -  IP address logging for security analysis
    -  Confidence scoring for answer verification
    -  Document access tracking
    -  Performance metrics per query
    -  Error logging and investigation
    -  Export capabilities for external audits
    
    **Regulatory Compliance:** This audit system helps meet requirements for Qatar Central Bank, 
    UAE regulatory standards, and international banking compliance frameworks (Basel III, GDPR, etc.).
    """)


def render_ground_truth_page():
    """Render ground truth data page."""
    st.markdown('<p class="main-header">📚 Ground Truth Data</p>', unsafe_allow_html=True)
    st.markdown("View and explore the ground truth QA pairs used for evaluation.")
    
    ground_truth = load_ground_truth()
    
    if ground_truth:
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total QA Pairs", len(ground_truth))
        with col2:
            documents = list(set([qa['source_document'] for qa in ground_truth]))
            st.metric("Documents", len(documents))
        with col3:
            avg_answer_len = sum(len(qa['answer']) for qa in ground_truth) / len(ground_truth)
            st.metric("Avg Answer Length", f"{avg_answer_len:.0f} chars")
        
        # Filter by document
        st.subheader("Filter by Document")
        selected_doc = st.selectbox("Select Document", ["All"] + documents)
        
        # Filter data
        filtered_data = ground_truth
        if selected_doc != "All":
            filtered_data = [qa for qa in ground_truth if qa['source_document'] == selected_doc]
        
        # Display QA pairs
        st.subheader(f"Showing {len(filtered_data)} QA Pairs")
        
        for idx, qa in enumerate(filtered_data, 1):
            with st.expander(f"Q{idx}: {qa['question'][:100]}..."):
                st.markdown(f"**Document:** {qa['source_document']}")
                if 'difficulty' in qa:
                    st.markdown(f"**Difficulty:** `{qa['difficulty'].upper()}`")
                if 'category' in qa:
                    st.markdown(f"**Category:** `{qa['category'].replace('_', ' ').title()}`")
                st.markdown(f"**Question:**\n\n{qa['question']}")
                st.markdown(f"**Answer:**\n\n{qa['answer']}")
        
        # Download button
        st.download_button(
            label=" Download Ground Truth JSON",
            data=json.dumps(ground_truth, indent=2),
            file_name="ground_truth.json",
            mime="application/json"
        )
    else:
        st.warning("No ground truth data found at data/ground_truth_arxiv.json")


def render_system_info_page():
    """Render system information page."""
    st.markdown('<p class="main-header"> System Information</p>', unsafe_allow_html=True)
    
    system_info = get_system_info()
    
    if system_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration")
            st.json({
                "Version": system_info.get('version', 'Unknown'),
                "Chunking Strategy": system_info.get('chunking_strategy', 'Unknown'),
                "Total Chunks": system_info.get('num_chunks', 'Unknown'),
                "Contextual Retrieval": system_info.get('contextual_retrieval_enabled', False),
                "Cache Enabled": system_info.get('cache_enabled', False)
            })
        
        with col2:
            st.subheader("Available Methods")
            methods = system_info.get('retrieval_methods', [])
            for method in methods:
                st.success(f"✓ {method.upper()}")
            
            st.subheader("Documents Loaded")
            docs = system_info.get('document_names', [])
            for doc in docs:
                st.info(f" {doc}")
    
    st.markdown("---")
    st.subheader("API Endpoints")
    endpoints = [
        {"Method": "POST", "Endpoint": "/api/v1/query", "Description": "Query the RAG system"},
        {"Method": "GET", "Endpoint": "/api/v1/health", "Description": "Health check"},
        {"Method": "GET", "Endpoint": "/api/v1/info", "Description": "System information"},
        {"Method": "GET", "Endpoint": "/api/v1/metrics", "Description": "Performance metrics"},
        {"Method": "GET", "Endpoint": "/api/v1/cache/stats", "Description": "Cache statistics"},
        {"Method": "GET", "Endpoint": "/api/v1/cache/history", "Description": "Query history"},
    ]
    st.dataframe(pd.DataFrame(endpoints), use_container_width=True)


if __name__ == "__main__":
    main()
