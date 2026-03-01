from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from utils.validation import QualityScore
from dotenv import load_dotenv
import os
from utils.logger import get_logger

load_dotenv(override=True)
logger = get_logger(__name__)

import json

class QualityScoreAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.parser = PydanticOutputParser(pydantic_object=QualityScore)
        
        # Load external rubric
        rubric_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'rubrics.json')
        try:
            with open(rubric_path, 'r') as f:
                self.rubric = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load rubric from {rubric_path}: {e}")
            self.rubric = {}
            
        self.rubric_str = json.dumps(self.rubric, indent=2)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Quality Assurance Specialist at a call center. "
                       "Evaluate the call based on the transcript provided.\n\n"
                       "You MUST base your scoring entirely on the following structured Rubric Store. "
                       "For each category, determine the appropriate 1-10 score by finding the description bracket that best fits the agent's behavior. "
                       "Justify your scores in the `rubric_notes` field by directly quoting or referencing the bracket requirements.\n\n"
                       "### Rubric Criteria ###\n"
                       "{rubric_str}\n\n"
                       "Extract the required sentiment analysis insights (like emotion, tone, and churn risk) as requested.\n"
                       "{format_instructions}"),
            ("user", "Transcript: {transcript}")
        ])

    async def score(self, transcript: str) -> QualityScore:
        """
        Evaluates the quality of the call based on the transcript.
        """
        logger.info("Starting quality scoring processing.")
        try:
            chain = self.prompt | self.llm | self.parser
            
            scores = await chain.ainvoke({
                "transcript": transcript,
                "format_instructions": self.parser.get_format_instructions(),
                "rubric_str": self.rubric_str
            })
            logger.info("Successfully completed quality scoring.")
            return scores
        except Exception as e:
            logger.error(f"Error during quality scoring: {e}", exc_info=True)
            raise
