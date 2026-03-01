from openai import AsyncOpenAI
import os
from utils.validation import TranscriptionResult
from dotenv import load_dotenv
from utils.logger import get_logger
from langsmith.wrappers import wrap_openai

load_dotenv(override=True)
logger = get_logger(__name__)

class TranscriptionAgent:
    def __init__(self):
        self.client = wrap_openai(AsyncOpenAI())

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Converts audio to text using OpenAI Whisper API.
        """
        logger.info(f"Starting audio transcription for file: {audio_path}")
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text"
                )
            logger.info("Successfully completed audio transcription.")
            return TranscriptionResult(text=transcript, language="en")
        except Exception as e:
            logger.error(f"Error during audio transcription: {e}", exc_info=True)
            raise

    async def process_text(self, text: str) -> TranscriptionResult:
        """
        Fallback for when text is provided directly instead of audio.
        """
        logger.info("Processing direct text input for transcription fallback.")
        return TranscriptionResult(text=text, language="en")
