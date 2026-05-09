import streamlit as st
import os
import sys
import queue
import time
import threading
import av

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
try:
    from RealtimeSTT import AudioToTextRecorder
except ImportError:
    AudioToTextRecorder = None

from agents.routing_agent import RoutingAgent
import json
import asyncio
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

# Sync Streamlit Secrets to Environment Variables
def sync_secrets_to_env():
    """Syncs Streamlit secrets to os.environ for compatibility with downstream agents."""
    try:
        for key, value in st.secrets.items():
            if key not in os.environ or not os.environ[key]:
                os.environ[key] = str(value)
                # logger.debug(f"Synced secret {key} to environment.")
    except Exception:
        # st.secrets is not available in local dev unless .streamlit/secrets.toml exists
        pass

sync_secrets_to_env()

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
    input_mode = st.radio("Choose Input Method", ["Text Transcript", "Audio File (Whisper)", "Live Call (WebRTC)"])

if input_mode == "Text Transcript":
    transcript_text = st.text_area("Paste Transcript Here", height=200, 
                                placeholder="Customer: ... \nAgent: ...")
elif input_mode == "Audio File (Whisper)":
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
    transcript_text = None
elif input_mode == "Live Call (WebRTC)":
    st.markdown("### Start Live Call")
    
    if "live_transcript" not in st.session_state:
        st.session_state.live_transcript = ""
    if "current_realtime" not in st.session_state:
        st.session_state.current_realtime = ""
    if "text_queue" not in st.session_state:
        st.session_state.text_queue = queue.Queue()
    if "realtime_queue" not in st.session_state:
        st.session_state.realtime_queue = queue.Queue()
        
    text_queue = st.session_state.text_queue
    realtime_queue = st.session_state.realtime_queue

    if "resampler" not in st.session_state:
        st.session_state.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)

    if "recorder" not in st.session_state:
        with st.spinner("Loading speech model (this takes a moment the first time)..."):
            if AudioToTextRecorder is None:
                st.error("RealtimeSTT not found. Please install it.")
                st.stop()
                
            def on_realtime_update(text):
                if text.strip():
                    realtime_queue.put(text)
                    
            recorder = AudioToTextRecorder(
                use_microphone=False, 
                model="tiny.en", 
                spinner=False, 
                language="en",
                enable_realtime_transcription=True,
                on_realtime_transcription_update=on_realtime_update,
                realtime_processing_pause=0.2, # Ease CPU load
                post_speech_silence_duration=2.0 # Prevent premature cutoffs
            )
            st.session_state.recorder = recorder
            
            def stt_worker():
                while True:
                    # recorder.text() blocks until silence is detected, then returns the finalized sentence
                    text = recorder.text()
                    if text and text.strip():
                        text_queue.put(text)
                        realtime_queue.put("")
            
            # Start STT processing in a background thread
            t = threading.Thread(target=stt_worker, daemon=True)
            t.start()

    class STTAudioProcessor(AudioProcessorBase):
        def __init__(self, recorder, resampler):
            self.recorder = recorder
            self.resampler = resampler

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            resampled_frames = self.resampler.resample(frame)
            for resampled_frame in resampled_frames:
                self.recorder.feed_audio(resampled_frame.to_ndarray().tobytes())
            return frame

    current_recorder = st.session_state.recorder
    current_resampler = st.session_state.resampler

    webrtc_ctx = webrtc_streamer(
        key="live_call",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=lambda: STTAudioProcessor(current_recorder, current_resampler),
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    
    if webrtc_ctx.state.playing:
        st.markdown("### Live Transcription")
        transcript_container = st.empty()
            
        while webrtc_ctx.state.playing:
            
            # 2. Check for new transcribed text in the queue
            while not text_queue.empty():
                try:
                    new_text = text_queue.get_nowait()
                    st.session_state.live_transcript += new_text + " "
                except queue.Empty:
                    break
                    
            # 3. Check for real-time word updates
            while not realtime_queue.empty():
                try:
                    st.session_state.current_realtime = realtime_queue.get_nowait()
                except queue.Empty:
                    break
                
            # Update UI
            display_text = st.session_state.live_transcript
            if st.session_state.current_realtime:
                display_text += f" *{st.session_state.current_realtime}*"
                
            transcript_container.info(display_text if display_text else "Listening...")
            # Yield to Streamlit
            time.sleep(0.1)

    # Flush any remaining finalized text from the queue after stopping
    if not webrtc_ctx.state.playing:
        if "text_queue" in st.session_state:
            while not st.session_state.text_queue.empty():
                try:
                    st.session_state.live_transcript += st.session_state.text_queue.get_nowait() + " "
                except queue.Empty:
                    break
                    
        # Append any unfinalized text that was cut off when stopping
        if "current_realtime" in st.session_state and st.session_state.current_realtime:
            st.session_state.live_transcript += st.session_state.current_realtime + " "
            st.session_state.current_realtime = ""

    if st.session_state.live_transcript:
        st.markdown("### Recorded Transcript")
        st.success(st.session_state.live_transcript)
        if st.button("Clear Transcript"):
            st.session_state.live_transcript = ""
            st.rerun()
            
    transcript_text = st.session_state.get("live_transcript", "")

# Process Button
if st.button("Generate Insights"):
    if input_mode == "Text Transcript" and not transcript_text:
        st.error("Please provide a transcript.")
    elif input_mode == "Audio File (Whisper)" and not uploaded_file:
        st.error("Please upload an audio file.")
    elif input_mode == "Live Call (WebRTC)" and not transcript_text:
        st.error("Please record a live call first.")
    else:
        with st.spinner("🤖 Agents are working on the analysis..."):
            # Prepare input
            logger.info("User requested insight generation.")
            raw_input = {
                "customer_name": customer_name,
                "agent_name": agent_name
            }
            
            if input_mode in ["Text Transcript", "Live Call (WebRTC)"]:
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
