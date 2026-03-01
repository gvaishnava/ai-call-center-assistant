import sys
import os
import json
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.routing_agent import RoutingAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class ValidationResult(BaseModel):
    passed: bool = Field(..., description="Whether the main model output passed the validation question.")
    reason: str = Field(..., description="The reasoning behind the pass/fail judgment.")

class LLMJudge:
    def __init__(self):
        # We can use gpt-4o as a strong judge model.
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=ValidationResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI evaluator judging the output of a call center assistant. "
                       "You will be given the original call transcript, the assistant's generated output, and a specific yes/no validation question. "
                       "Your job is to answer the question based on the provided text, and provide a detailed reason for your decision.\n"
                       "{format_instructions}"),
            ("user", "Transcript:\n{transcript}\n\nAssistant Output:\n{output}\n\nValidation Question: {question}")
        ])
        
    def evaluate(self, transcript: str, output: str, question: str) -> ValidationResult:
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({
            "transcript": transcript,
            "output": output,
            "question": question,
            "format_instructions": self.parser.get_format_instructions()
        })

async def run_closed_ended_evaluations():
    print("Initializing components...")
    judge = LLMJudge()
    agent = RoutingAgent()
    
    samples_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_transcripts', 'samples.json')
    try:
        with open(samples_path, 'r') as f:
            samples = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find samples.json at {samples_path}")
        return

    # Define test questions for our samples.
    # In a real test suite, these would be mapped by an ID.
    # Since samples.json is a list, we map by index.
    test_cases = {
        0: [
            "Does the one_line_summary mention an issue with the internet cutting out or fluctuating?",
            "Do the action items include checking physical connections or scheduling a technician?",
            "Is the customer_sentiment_overall Positive or Neutral? (Customer was patient and thanked the agent)",
            "Did the quality score soft_skills_score exceed 5? (The agent apologized and empathized with working from home)",
            "Did the quality score technical_score exceed 6? (The agent correctly diagnosed upstream power issues and suggested checking the splitter/cables)"
        ],
        1: [
            "Does the one_line_summary mention a $50 charge or billing dispute?",
            "Do the action items state that a credit will be issued or the charge removed?",
            "Did the quality score customer_primary_emotion capture frustration, anger, or annoyance?",
            "Was the churn_risk_detected correctly identified as True or False based on the customer's mild frustration but resolution?",
            "Did the quality score professionalism_score exceed 6? (The agent apologized for the confusion and remained calm despite the customer's frustration)"
        ]
    }

    total_tests = 0
    passed_tests = 0

    print("\nStarting Evaluations...\n" + "-"*50)
    for index, sample in enumerate(samples):
        if index not in test_cases:
            continue
            
        print(f"\nProcessing Sample {index + 1}: {sample.get('customer', 'Unknown')}")
        transcript = sample['transcript']
        
        # Run the primary agent pipeline
        result = await agent.run({"text": transcript, "customer_name": sample.get('customer')})
        if result.get("error"):
            print(f"Agent Pipeline Error for Sample {index + 1}: {result['error']}")
            continue
            
        # Serialize the outputs we care about
        # We use Pydantic's model_dump_json for clean serialization to strings
        output_data = {
            "summary": result.get("summary").model_dump() if result.get("summary") else None,
            "quality_scores": result.get("quality_scores").model_dump() if result.get("quality_scores") else None
        }
        output_str = json.dumps(output_data, indent=2)
        
        # Run judge evaluations
        for question in test_cases[index]:
            total_tests += 1
            print(f"\n  Q: {question}")
            try:
                eval_result = judge.evaluate(transcript, output_str, question)
                status = "[PASS]" if eval_result.passed else "[FAIL]"
                print(f"  {status}")
                print(f"  Reason: {eval_result.reason}")
                
                if eval_result.passed:
                    passed_tests += 1
            except Exception as e:
                print(f"  [ERROR] evaluating question: {e}")

    print("\n" + "="*50)
    print(f"EVALUATION SUMMARY")
    print(f"Passed {passed_tests} / {total_tests} test cases ({(passed_tests/max(total_tests, 1))*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_closed_ended_evaluations())
