import streamlit as st
import pandas as pd
import os
from PIL import Image

# Import our modules
from data_loader import PatientDataLoader
from kg_model import build_graph_from_csv, build_graph
from rag_engine import retrieve_context, format_prompt
from visualize import visualize_reasoning_chain, visualize_graph
from main import get_llm_response

# Page configuration
st.set_page_config(
    page_title="KG-RAG POC - Obesity Management",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2ca02c;
        margin-top: 1.5rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .evidence-chain {
        background-color: #fff3cd;
        color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = "mok.csv"
if 'patients' not in st.session_state:
    st.session_state.patients = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 KG-RAG POC")
    st.markdown("**Knowledge Graph-based RAG System**")
    st.markdown("Obesity Management Use Case")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 KG Visualization", "💬 LLM Chat", "📈 Batch Analysis"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("#### Data Source")
    
    # CSV upload
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        st.session_state.csv_path = uploaded_file
    else:
        if st.button("📂 Use Default (mok.csv)"):
            st.session_state.csv_path = "mok.csv"
    
    st.markdown("---")
    st.markdown("#### About")
    st.markdown("""
    This system integrates:
    - **DHT**: Digital Health Tech
    - **COA**: Clinical Outcomes
    - **PPI**: Patient Preferences
    
    Using knowledge graphs and RAG for explainable AI.
    """)

# Helper function to load data
@st.cache_data
def load_patient_data(csv_path):
    if isinstance(csv_path, str):
        loader = PatientDataLoader(csv_path)
    else:
        # Handle uploaded file
        loader = PatientDataLoader()
        loader.df = pd.read_csv(csv_path)
    
    patients = loader.process_data()
    stats = loader.get_summary_stats()
    return patients, stats, loader.df

# PAGE 1: DASHBOARD
if page == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Patient Dashboard</div>', unsafe_allow_html=True)
    
    try:
        patients, stats, df = load_patient_data(st.session_state.csv_path)
        st.session_state.patients = patients
        
        # Summary Statistics
        st.markdown('<div class="sub-header">Summary Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", stats['total_patients'])
        
        with col2:
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
        
        with col3:
            st.metric("Avg Weight Loss", f"{stats['avg_weight_change']:.2f} kg")
        
        with col4:
            st.metric("Avg Daily Steps", f"{stats['avg_daily_steps']:.0f}")
        
        # Outcome breakdown
        st.markdown('<div class="sub-header">Outcome Distribution</div>', unsafe_allow_html=True)
        st.caption("病患體重變化結果分類統計")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="✅ Successful (成功)",
                value=stats["successful_count"],
                help="體重成功減輕的病患數量"
            )
        
        with col2:
            st.metric(
                label="🐌 Slow (緩慢)",
                value=stats["slow_count"],
                help="體重減輕緩慢的病患數量"
            )
        
        with col3:
            st.metric(
                label="📈 Increase (增加)",
                value=stats["increase_count"],
                help="體重增加的病患數量"
            )
        
        # Patient Table
        st.markdown('<div class="sub-header">Patient Details</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=400)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure 'mok.csv' is in the working directory or upload a valid CSV file.")

# PAGE 2: KG VISUALIZATION
elif page == "🔍 KG Visualization":
    st.markdown('<div class="main-header">🔍 Knowledge Graph Visualization</div>', unsafe_allow_html=True)
    
    try:
        # Load data if not already loaded
        if st.session_state.patients is None:
            patients, stats, df = load_patient_data(st.session_state.csv_path)
            st.session_state.patients = patients
        
        # Build graph
        if st.session_state.graph is None:
            with st.spinner("Building knowledge graph..."):
                G, patients_data = build_graph_from_csv(st.session_state.csv_path if isinstance(st.session_state.csv_path, str) else "mok.csv")
                st.session_state.graph = G
        
        # Patient selector
        patient_ids = [p['id'] for p in st.session_state.patients]
        selected_patient = st.selectbox("Select Patient", patient_ids)
        
        if selected_patient:
            patient_data = next(p for p in st.session_state.patients if p['id'] == selected_patient)
            
            # Display patient info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Weight Change", f"{patient_data['weight_change_value']} kg")
                st.caption(patient_data['weight_change_category'])
            
            with col2:
                st.metric("Daily Steps", f"{patient_data['daily_steps']:,}")
                st.caption(patient_data['daily_steps_class'])
            
            with col3:
                st.metric("Caloric Intake", f"{patient_data['caloric_intake']:,}")
                st.caption(patient_data['caloric_intake_class'])
            
            # Retrieve KG context
            st.markdown('<div class="sub-header">Knowledge Graph Context</div>', unsafe_allow_html=True)
            
            reasoning_text, evidence = retrieve_context(st.session_state.graph, selected_patient)
            
            for text in reasoning_text:
                st.info(text)
            
            # Visualize reasoning chain
            if evidence:
                st.markdown('<div class="sub-header">Reasoning Chain Visualization</div>', unsafe_allow_html=True)
                
                with st.spinner("Generating visualization..."):
                    viz_file = f"temp_{selected_patient}_reasoning.png"
                    visualize_reasoning_chain(st.session_state.graph, selected_patient, evidence, output_file=viz_file)
                    
                    if os.path.exists(viz_file):
                        image = Image.open(viz_file)
                        st.image(image, use_container_width=True)
                        os.remove(viz_file)
                
                # Evidence chain
                st.markdown('<div class="sub-header">Evidence Chain</div>', unsafe_allow_html=True)
                for i, item in enumerate(evidence, 1):
                    st.markdown(f"""
                    <div class="evidence-chain">
                    <strong>Chain {i}:</strong><br>
                    {item['preference']} <span style="color:#d62728;">--[{item['conflict_relation']}]--></span> 
                    {item['behavior']} <span style="color:#2ca02c;">--[{item['behavior_relation']}]--></span> 
                    {item['outcome']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No evidence chains found for this patient.")
    
    except Exception as e:
        st.error(f"Error: {e}")

# PAGE 3: LLM CHAT
elif page == "💬 LLM Chat":
    st.markdown('<div class="main-header">💬 LLM Chat Interface</div>', unsafe_allow_html=True)
    
    try:
        # Load data if not already loaded
        if st.session_state.patients is None:
            patients, stats, df = load_patient_data(st.session_state.csv_path)
            st.session_state.patients = patients
        
        # Build graph
        if st.session_state.graph is None:
            G, patients_data = build_graph_from_csv(st.session_state.csv_path if isinstance(st.session_state.csv_path, str) else "mok.csv")
            st.session_state.graph = G
        
        # Patient selector
        patient_ids = [p['id'] for p in st.session_state.patients]
        selected_patient = st.selectbox("Select Patient", patient_ids, key="chat_patient")
        
        if selected_patient:
            patient_data = next(p for p in st.session_state.patients if p['id'] == selected_patient)
            
            # Display patient summary
            with st.expander("Patient Summary", expanded=False):
                st.json(patient_data)
            
            # Chat interface
            st.markdown('<div class="sub-header">Ask a Question</div>', unsafe_allow_html=True)
            
            query = st.text_input(
                "Your question:",
                placeholder=f"Why is {selected_patient} experiencing {patient_data['weight_change_category']} weight change?",
                key="chat_input"
            )
            
            if st.button("🚀 Submit Query", type="primary"):
                if query:
                    with st.spinner("Retrieving context and generating response..."):
                        # Retrieve context
                        reasoning_text, evidence = retrieve_context(st.session_state.graph, selected_patient)
                        
                        # Format prompt
                        prompt = format_prompt(query, reasoning_text)
                        
                        # Get LLM response
                        llm_response = get_llm_response(prompt)
                        
                        # Display response
                        st.markdown('<div class="sub-header">LLM Response</div>', unsafe_allow_html=True)
                        st.success(llm_response)
                        
                        # Display evidence
                        if evidence:
                            st.markdown('<div class="sub-header">Supporting Evidence</div>', unsafe_allow_html=True)
                            for i, item in enumerate(evidence, 1):
                                st.markdown(f"""
                                <div class="evidence-chain">
                                {i}. {item['preference']} --[{item['conflict_relation']}]--> 
                                {item['behavior']} --[{item['behavior_relation']}]--> 
                                {item['outcome']}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Add to history
                        st.session_state.chat_history.append({
                            'patient': selected_patient,
                            'query': query,
                            'response': llm_response,
                            'evidence': evidence
                        })
                else:
                    st.warning("Please enter a question.")
            
            # Chat history
            if st.session_state.chat_history:
                st.markdown('<div class="sub-header">Chat History</div>', unsafe_allow_html=True)
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                    with st.expander(f"{chat['patient']}: {chat['query'][:50]}..."):
                        st.markdown(f"**Response:** {chat['response']}")
    
    except Exception as e:
        st.error(f"Error: {e}")

# PAGE 4: BATCH ANALYSIS
elif page == "📈 Batch Analysis":
    st.markdown('<div class="main-header">📈 Batch Analysis</div>', unsafe_allow_html=True)
    
    st.info("""
    **批次分析功能說明**：
    - 🤖 對所有病患執行完整的 **KG-RAG + LLM 分析**
    - 📊 自動提取知識圖譜證據鏈
    - 💬 使用 Gemini AI 生成個別化臨床解釋
    - 📥 可下載包含 AI 見解的完整報告
    
    ⚠️ 注意：此功能會調用 LLM API，處理 10 位病患約需 30-60 秒
    """)
    
    try:
        # Load data if not already loaded
        if st.session_state.patients is None:
            patients, stats, df = load_patient_data(st.session_state.csv_path)
            st.session_state.patients = patients
        
        # Show patient count
        st.metric("待分析病患數", len(st.session_state.patients))
        
        if st.button("🔄 開始批次分析", type="primary", use_container_width=True):
            # Build graph
            with st.spinner("建構知識圖譜..."):
                G, patients_data = build_graph_from_csv(st.session_state.csv_path if isinstance(st.session_state.csv_path, str) else "mok.csv")
                st.session_state.graph = G
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, patient in enumerate(st.session_state.patients):
                patient_id = patient['id']
                status_text.text(f"正在分析 {patient_id}... ({i+1}/{len(st.session_state.patients)})")
                
                # Retrieve context
                reasoning_text, evidence = retrieve_context(G, patient_id)
                
                # Generate query
                query = f"What explains {patient_id}'s {patient['weight_change_category']} weight change?"
                
                # Get LLM response
                prompt = format_prompt(query, reasoning_text)
                llm_response = get_llm_response(prompt)
                
                results.append({
                    'Patient ID': patient_id,
                    'Weight Change': f"{patient['weight_change_value']} kg",
                    'Category': patient['weight_change_category'],
                    'Evidence Chains': len(evidence),
                    'AI Analysis': llm_response[:300] + "..." if len(llm_response) > 300 else llm_response
                })
                
                progress_bar.progress((i + 1) / len(st.session_state.patients))
            
            status_text.empty()
            
            # Display results table
            st.markdown('<div class="sub-header">分析結果</div>', unsafe_allow_html=True)
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, height=400)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 下載完整分析報告 (CSV)",
                data=csv,
                file_name=f"kg_rag_batch_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.success(f"✅ 成功分析 {len(results)} 位病患！")
            
            # Show summary statistics
            st.markdown('<div class="sub-header">分析摘要</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_chains = sum(r['Evidence Chains'] for r in results) / len(results)
                st.metric("平均證據鏈數", f"{avg_chains:.1f}")
            with col2:
                successful = sum(1 for r in results if r['Category'] == 'Successful')
                st.metric("成功案例", f"{successful}/{len(results)}")
            with col3:
                avg_analysis_length = sum(len(r['AI Analysis']) for r in results) / len(results)
                st.metric("平均分析長度", f"{avg_analysis_length:.0f} 字元")
    
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("**KG-RAG POC** | Powered by Streamlit & Gemini AI")
