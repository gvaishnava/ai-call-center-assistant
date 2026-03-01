import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import get_logger

from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from agents.intake_agent import IntakeAgent
from agents.transcription_agent import TranscriptionAgent
from agents.summarization_agent import SummarizationAgent
from agents.quality_score_agent import QualityScoreAgent
from utils.validation import CallMetadata, TranscriptionResult, CallSummary, QualityScore

logger = get_logger(__name__)

class GraphState(TypedDict):
    raw_input: Dict[str, Any]
    metadata: Optional[CallMetadata]
    transcription: Optional[TranscriptionResult]
    summary: Optional[CallSummary]
    quality_scores: Optional[QualityScore]
    error: Optional[str]

class RoutingAgent:
    def __init__(self):
        self.intake = IntakeAgent()
        self.transcriber = TranscriptionAgent()
        self.summarizer = SummarizationAgent()
        self.scorer = QualityScoreAgent()
        
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(GraphState)

        # Define nodes
        workflow.add_node("intake", self._node_intake)
        workflow.add_node("transcription", self._node_transcription)
        workflow.add_node("summarization", self._node_summarization)
        workflow.add_node("quality_scoring", self._node_quality_scoring)

        # Build edges
        workflow.set_entry_point("intake")
        workflow.add_edge("intake", "transcription")
        workflow.add_edge("transcription", "summarization")
        workflow.add_edge("transcription", "quality_scoring")
        workflow.add_edge("summarization", END)
        workflow.add_edge("quality_scoring", END)

        return workflow.compile()

    async def _node_intake(self, state: GraphState) -> Dict:
        logger.info("Executing node: intake")
        try:
            metadata = await self.intake.process(state["raw_input"])
            return {"metadata": metadata}
        except Exception as e:
            logger.error(f"Error in intake node: {e}")
            return {"error": f"Intake error: {str(e)}"}

    async def _node_transcription(self, state: GraphState) -> Dict:
        if state.get("error"): return {}
        logger.info("Executing node: transcription")
        try:
            raw_input = state["raw_input"]
            if "audio_path" in raw_input:
                transcription = await self.transcriber.transcribe(raw_input["audio_path"])
            else:
                transcription = await self.transcriber.process_text(raw_input.get("text", ""))
            return {"transcription": transcription}
        except Exception as e:
            logger.error(f"Error in transcription node: {e}")
            return {"error": f"Transcription error: {str(e)}"}

    async def _node_summarization(self, state: GraphState) -> Dict:
        if state.get("error"): return {}
        logger.info("Executing node: summarization")
        try:
            summary = await self.summarizer.summarize(state["transcription"].text)
            return {"summary": summary}
        except Exception as e:
            logger.error(f"Error in summarization node: {e}")
            return {"error": f"Summarization error: {str(e)}"}

    async def _node_quality_scoring(self, state: GraphState) -> Dict:
        if state.get("error"): return {}
        logger.info("Executing node: quality_scoring")
        try:
            scores = await self.scorer.score(state["transcription"].text)
            return {"quality_scores": scores}
        except Exception as e:
            logger.error(f"Error in quality_scoring node: {e}")
            return {"error": f"Quality Scoring error: {str(e)}"}

    async def run(self, raw_input: Dict[str, Any]):
        logger.info(f"Starting routing workflow for input: {raw_input.keys()}")
        initial_state = {
            "raw_input": raw_input,
            "metadata": None,
            "transcription": None,
            "summary": None,
            "quality_scores": None,
            "error": None
        }
        return await self.workflow.ainvoke(initial_state)

if __name__ == "__main__":
    import asyncio
    
    # Test text-based flow
    async def main():
        agent = RoutingAgent()
        sample_input = {
            "customer_name": "Jane Smith",
            "text": "Hello, I am calling because my internet is slow for three days. Can you help? Agent: Yes, let me check your connection... It seems there is a line issue. I will send a technician tomorrow. Jane: Thank you."
        }
        result = await agent.run(sample_input)
        print(result)
        
    asyncio.run(main())
