from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from utils.validation import CallSummary
from dotenv import load_dotenv
import os
from utils.logger import get_logger

load_dotenv(override=True)
logger = get_logger(__name__)

class SummarizationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=CallSummary)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert call center analyst. "
                       "Analyze the following transcript and provide a structured summary.\n"
                       "You must extract:\n"
                       "- A concise one-line summary.\n"
                       "- Key discussion points.\n"
                       "- The overall sentiment.\n"
                       "- Action items: Be VERY generous in capturing action items. You MUST include:\n"
                       "  * Actions already taken during the call (e.g. 'Agent checked physical connections').\n"
                       "  * Explicit future promises made by agent or customer (e.g. 'Agent will send technician tomorrow').\n"
                       "  * Conditional future actions (e.g. 'Will escalate if issue persists').\n"
                       "  * Scheduled callbacks or follow-ups (e.g. 'Customer will call back after the holiday', 'Agent to follow up next week').\n"
                       "  * Any mention of a future date, callback, or pending resolution — these are ALWAYS action items.\n"
                       "  If a customer says they will call back, or an agent says they will follow up, those ARE action items. "
                       "Do NOT leave this list empty unless the call was purely informational with zero actions, promises, or follow-ups of any kind.\n\n"
                       "{format_instructions}"),
            ("user", "Transcript: {transcript}")
        ])

    async def summarize(self, transcript: str) -> CallSummary:
        """
        Generates a structured summary from the transcript.
        """
        logger.info("Starting summarization process.")
        try:
            chain = self.prompt | self.llm | self.parser
            
            summary = await chain.ainvoke({
                "transcript": transcript,
                "format_instructions": self.parser.get_format_instructions()
            })
            logger.info("Successfully completed summarization process.")
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {e}", exc_info=True)
            raise
