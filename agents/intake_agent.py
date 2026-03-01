from utils.validation import CallMetadata
import uuid
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

class IntakeAgent:
    def __init__(self):
        pass

    async def process(self, raw_input: dict) -> CallMetadata:
        """
        Validates the input and extracts metadata.
        If minimal data is provided, it generates a call ID and timestamp.
        """
        logger.info("Starting intake processing.")
        try:
            if "call_id" not in raw_input:
                raw_input["call_id"] = str(uuid.uuid4())
            
            if "timestamp" not in raw_input:
                raw_input["timestamp"] = datetime.now()
                
            metadata = CallMetadata(**raw_input)
            logger.info(f"Successfully processed intake for call_id: {metadata.call_id}")
            return metadata
        except Exception as e:
            logger.error(f"Error during intake processing: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = IntakeAgent()
        sample_input = {"customer_name": "John Doe", "agent_name": "Alice"}
        metadata = await agent.process(sample_input)
        print(f"Validated Metadata: {metadata}")
        
    asyncio.run(main())
