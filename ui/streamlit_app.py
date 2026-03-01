import streamlit as st
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.routing_agent import RoutingAgent
import json
import asyncio
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

# Page Configuration
st.set_page_config(
    page_title="AI Call Center Assistant",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(76, 175, 80, 0.3);
        backdrop-filter: blur(10px);
        margin: 10px 0;
    }
    .metric-label {
        color: #b0b0b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #4CAF50;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-suffix {
        color: #666;
        font-size: 1.2rem;
    }
    h1, h2, h3 {
        color: #4CAF50 !important;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# sidebar
with st.sidebar:
    st.title("Settings")
    st.info("Ensure your .env file contains the necessary API keys.")
    st.info("To enable LangSmith tracing, add `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` to your `.env`.")
    st.divider()
    st.write("Built with LangGraph & GPT-4o")

# Main Header
st.title("📞 AI Call Center Assistant")
st.subheader("Transform Raw Call Data into Actionable Insights")

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Call Details")
    customer_name = st.text_input("Customer Name", "John Doe")
    agent_name = st.text_input("Agent Name", "Alice Agent")

with col2:
    st.markdown("### Input Mode")
    input_mode = st.radio("Choose Input Method", ["Text Transcript", "Audio File (Whisper)"])

if input_mode == "Text Transcript":
    transcript_text = st.text_area("Paste Transcript Here", height=200, 
                                placeholder="Customer: ... \nAgent: ...")
else:
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
    transcript_text = None

# Process Button
if st.button("Generate Insights"):
    if input_mode == "Text Transcript" and not transcript_text:
        st.error("Please provide a transcript.")
    elif input_mode == "Audio File (Whisper)" and not uploaded_file:
        st.error("Please upload an audio file.")
    else:
        with st.spinner("🤖 Agents are working on the analysis..."):
            # Prepare input
            logger.info("User requested insight generation.")
            raw_input = {
                "customer_name": customer_name,
                "agent_name": agent_name
            }
            
            if input_mode == "Text Transcript":
                raw_input["text"] = transcript_text
            else:
                # Save audio temporarily
                temp_path = f"data/temp_{datetime.now().timestamp()}.wav"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                raw_input["audio_path"] = temp_path
            
            # Run Routing Agent
            try:
                agent = RoutingAgent()
                result = asyncio.run(agent.run(raw_input))
                
                if result.get("error"):
                    logger.error(f"Analysis failed due to agent logic error: {result['error']}")
                    st.error("We couldn't complete the analysis. Please check your transcript/audio and try again.")
                    with st.expander("Technical details"):
                        st.write(result["error"])
                else:
                    st.success("Analysis Complete!")
                    
                    # Display Results
                    st.divider()
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown(f'''
                            <div class="metric-card">
                                <div class="metric-label">Professionalism</div>
                                <div class="metric-value">{result["quality_scores"].professionalism_score}<span class="metric-suffix">/10</span></div>
                            </div>
                        ''', unsafe_allow_html=True)
                    with m_col2:
                        st.markdown(f'''
                            <div class="metric-card">
                                <div class="metric-label">Soft Skills</div>
                                <div class="metric-value">{result["quality_scores"].soft_skills_score}<span class="metric-suffix">/10</span></div>
                            </div>
                        ''', unsafe_allow_html=True)
                    with m_col3:
                        st.markdown(f'''
                            <div class="metric-card">
                                <div class="metric-label">Technical</div>
                                <div class="metric-value">{result["quality_scores"].technical_score}<span class="metric-suffix">/10</span></div>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    st.markdown("### Sentiment Analysis")
                    s_col1, s_col2, s_col3 = st.columns(3)
                    with s_col1:
                        st.markdown(f"**Customer Sentiment:** {result['quality_scores'].customer_sentiment_overall.value}")
                        st.markdown(f"**Sentiment Shift:** {result['quality_scores'].sentiment_shift}")
                    with s_col2:
                        st.markdown(f"**Customer Emotion:** {result['quality_scores'].customer_primary_emotion}")
                        st.markdown(f"**Agent Tone:** {result['quality_scores'].agent_tone}")
                    with s_col3:
                        churn_color = "#ff4b4b" if result['quality_scores'].churn_risk_detected else "#4CAF50"
                        churn_text = "Yes ⚠️" if result['quality_scores'].churn_risk_detected else "No ✅"
                        st.markdown(f"**Churn Risk:** <span style='color:{churn_color}; font-weight:bold;'>{churn_text}</span>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    st.markdown("### Summary")
                    st.info(result["summary"].one_line_summary)
                    
                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        st.markdown("#### Key Points")
                        for point in result["summary"].key_points:
                            st.write(f"- {point}")
                    with t_col2:
                        st.markdown("#### Action Items")
                        actions = [a for a in (result["summary"].action_items or []) if a.strip()]
                        if actions:
                            for item in actions:
                                st.write(f"- {item}")
                        else:
                            st.caption("No specific action items or follow-ups identified.")
                            
                    with st.expander("View Full Transcript"):
                        st.text(result["transcription"].text)
                    
                    with st.expander("Quality Scoring Notes"):
                        st.write(result["quality_scores"].rubric_notes)
                        
            except Exception as e:
                logger.error(f"Unexpected error in Streamlit UI: {e}", exc_info=True)
                st.error("An unexpected error occurred while processing your request. Please try again later.")
                with st.expander("Technical details"):
                    st.exception(e)
            finally:
                if input_mode == "Audio File (Whisper)" and os.path.exists(temp_path):
                    os.remove(temp_path)

# Sample Data Info
st.divider()
st.markdown("""
<div style="font-size: 0.8em; color: #666;">
    <b>Note:</b> This is a prototype system. Transcription uses Whisper (local or via API), and logic is orchestrated by LangGraph.
</div>
""", unsafe_allow_html=True)
