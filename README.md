# 📞 AI Call Center Assistant

An end-to-end **Agentic AI** system that transforms raw call center data — audio recordings or text transcripts — into structured insights using a **multi-agent LangGraph pipeline** powered by **GPT-4o**.

---

## 🧠 System Overview

The AI Call Center Assistant automatically performs the following on every call:

1. **Validates & registers** call metadata (customer, agent, timestamp)
2. **Transcribes** audio to text (via OpenAI Whisper) or accepts text directly
3. **Summarizes** the conversation into key points, sentiment, and action items
4. **Quality scores** the agent's performance against a structured rubric
5. **Detects sentiment & churn risk** from the customer's behavior
6. **Presents all results** through an interactive Streamlit UI

---

## 🏗️ Architecture

```
User Input (Text / Audio)
         │
         ▼
┌─────────────────────────┐
│      Intake Agent        │   Validates & enriches metadata (UUID, timestamp)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Transcription Agent   │   Whisper API (audio) or passthrough (text)
└────────────┬────────────┘
             │
        ┌────┴────┐
        │         │   (parallel async execution via LangGraph)
        ▼         ▼
┌──────────────┐ ┌───────────────────┐
│ Summarization│ │  Quality Scoring  │
│    Agent     │ │      Agent        │
└──────┬───────┘ └────────┬──────────┘
        │                  │
        └────────┬─────────┘
                 ▼
       Streamlit UI (Results)
```

All agents are connected via a **LangGraph StateGraph** with full `async/await` support, ensuring summarization and quality scoring run **in parallel** after transcription completes.

---

## 🤖 Agents

### 1. `IntakeAgent` — `agents/intake_agent.py`
- Validates raw user input using **Pydantic** models
- Auto-generates `call_id` (UUID) and `timestamp` if not provided
- Returns a structured `CallMetadata` object

### 2. `TranscriptionAgent` — `agents/transcription_agent.py`
- **Audio mode**: Calls OpenAI `whisper-1` via `AsyncOpenAI` for speech-to-text
- **Text mode**: Directly wraps provided text into a `TranscriptionResult`
- Client wrapped with **LangSmith** `wrap_openai` for full trace visibility

### 3. `SummarizationAgent` — `agents/summarization_agent.py`
- Uses `gpt-4o` with a structured prompt via LangChain
- Extracts:
  - One-line call summary
  - Key discussion points
  - Overall sentiment (Positive / Neutral / Negative)
  - **Action items** — including callbacks, follow-ups, and future dates
- Returns a structured `CallSummary` Pydantic object

### 4. `QualityScoreAgent` — `agents/quality_score_agent.py`
- Uses `gpt-4o` with a loaded rubric from `config/rubrics.json`
- Scores agent performance across 5 dimensions (1–10 scale):
  | Dimension | Description |
  |---|---|
  | `technical_score` | Technical knowledge & issue resolution |
  | `professionalism_score` | Demeanor, respectfulness, brand representation |
  | `communication_score` | Clarity, conciseness, language appropriateness |
  | `process_adherence_score` | Policy compliance & verification steps |
  | `soft_skills_score` | Empathy, active listening, emotional de-escalation |
- Also extracts: customer sentiment, primary emotion, agent tone, sentiment shift, and churn risk

### 5. `RoutingAgent` — `agents/routing_agent.py`
- **LangGraph orchestrator** that connects all agents
- Exposes a single `async run(raw_input)` entry point
- Manages the state machine (`GraphState`) across all nodes
- Handles errors gracefully per node without crashing the full pipeline

---

## 🖥️ User Interface

**Streamlit** (`ui/streamlit_app.py`) provides an interactive web dashboard:

- Choose input mode: **Text Transcript** or **Audio File upload (WAV/MP3/M4A)**
- Enter customer and agent names
- Click **Generate Insights** to run the full pipeline
- View results:
  - Quality score metric cards (Professionalism, Soft Skills, Technical)
  - Sentiment analysis (sentiment, emotion, agent tone, churn risk, sentiment shift)
  - Call summary (one-line, key points, action items)
  - Full transcript expander
  - Quality scoring rubric notes expander

---

## 📊 Quality Rubric

Scoring rubric is externalized to `config/rubrics.json` for easy customization without code changes.

Each category maps score brackets to agent behavior descriptions:

| Score | Meaning |
|---|---|
| 1–3 | Poor / Non-compliant |
| 4–6 | Average / Partial compliance |
| 7–8 | Good / Meets expectations |
| 9–10 | Exceptional / Exceeds expectations |

---

## 🔍 Observability — LangSmith

The system integrates **LangSmith** for full trace visibility into every call processed:

- All LangChain/LangGraph calls are automatically traced (via `LANGCHAIN_TRACING_V2`)
- Direct OpenAI Whisper calls are traced via `langsmith.wrappers.wrap_openai`
- View token usage, latency, prompts, outputs, and LangGraph node transitions at [smith.langchain.com](https://smith.langchain.com)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API Key
- LangSmith API Key *(optional, for tracing)*

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd "AI Call Center Assistant"

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
source venv/bin/activate          # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY="sk-..."

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="lsv2_..."
LANGCHAIN_PROJECT="AI Call Center Assistant"
```

### Run the Application

```bash
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing

The project includes a **closed-ended LLM-as-judge evaluation** framework.

```bash
python -m tests.test_closed_ended_validation
```

This runs a second `gpt-4o` model as an **independent judge** that evaluates each pipeline output against pre-defined yes/no validation questions per sample transcript.

**Latest results: 10 / 10 tests passed (100%)**

Sample validation questions include:
- *"Does the one_line_summary mention an issue with the internet cutting out or fluctuating?"*
- *"Is the churn_risk_detected correctly identified based on the customer's behavior?"*
- *"Did the quality score professionalism_score exceed 6?"*

---

## 📁 Project Structure

```
AI Call Center Assistant/
├── agents/
│   ├── intake_agent.py           # Input validation & metadata extraction
│   ├── transcription_agent.py    # Audio-to-text (Whisper) or text passthrough
│   ├── summarization_agent.py    # GPT-4o call summarization
│   ├── quality_score_agent.py    # GPT-4o rubric-based quality scoring
│   └── routing_agent.py          # LangGraph orchestrator
├── config/
│   ├── rubrics.json              # Dynamic quality scoring rubric
│   └── mcp.yaml                  # Model Control Plane configuration
├── data/
│   └── sample_transcripts/
│       └── samples.json          # Sample call transcripts for testing
├── tests/
│   └── test_closed_ended_validation.py  # LLM-as-judge evaluation suite
├── ui/
│   └── streamlit_app.py          # Streamlit web interface
├── utils/
│   ├── logger.py                 # Centralized logging
│   └── validation.py             # Pydantic models (CallMetadata, QualityScore, etc.)
├── .env                          # API keys & environment config
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Docker deployment config
└── README.md                     # This document
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-openai` | LLM prompting & chain composition |
| `langgraph` | Multi-agent graph orchestration |
| `openai` | GPT-4o + Whisper API access |
| `langsmith` | Tracing, monitoring & evaluation |
| `pydantic` | Structured data validation |
| `streamlit` | Web UI |
| `python-dotenv` | Environment variable management |

---

## 🔮 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Async throughout** | All agents use `async/await` + LangGraph `ainvoke` so summarization and quality scoring run in parallel |
| **Externalized rubric** | `rubrics.json` allows scoring criteria to be updated without code changes |
| **Pydantic output parsing** | Enforces structured, type-safe LLM outputs that flow cleanly between agents |
| **LLM-as-Judge testing** | Uses a second independent LLM to objectively evaluate pipeline output quality |
| **LangSmith integration** | Provides deep observability into every node, prompt, and API call |

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

The `docker-compose.yml` is configured to run the Streamlit application in a container.

---

*Built with LangGraph, GPT-4o, OpenAI Whisper, LangSmith, and Streamlit.*
