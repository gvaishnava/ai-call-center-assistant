import time
import threading
import queue
import wave
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from RealtimeSTT import AudioToTextRecorder
except ImportError:
    print("RealtimeSTT not installed. Skipping test.")
    sys.exit(0)

def stream_wav_file(wav_path, chunk_duration_ms=100):
    """
    Simulates a WebRTC stream by reading a WAV file and feeding it 
    to RealtimeSTT in chunks of `chunk_duration_ms`.
    """
    if not os.path.exists(wav_path):
        print(f"Test file not found: {wav_path}")
        print("Please place a 16kHz Mono WAV file at this location to test.")
        return

    print(f"--- Starting Streaming Test for {wav_path} ---")
    
    # 1. Initialize RealtimeSTT with optimized parameters
    text_queue = queue.Queue()
    realtime_queue = queue.Queue()
    
    def on_realtime_update(text):
        if text.strip():
            realtime_queue.put(text)
            
    print("Initializing Model...")
    recorder = AudioToTextRecorder(
        use_microphone=False,
        model="base.en",
        spinner=False,
        language="en",
        enable_realtime_transcription=True,
        on_realtime_transcription_update=on_realtime_update,
        realtime_processing_pause=0.2,
        post_speech_silence_duration=2.0
    )
    
    # Start the continuous worker
    def stt_worker():
        while True:
            text = recorder.text()
            if text and text.strip():
                text_queue.put(text)
                realtime_queue.put("") # clear realtime buffer
                
    t = threading.Thread(target=stt_worker, daemon=True)
    t.start()
    
    # 2. Open WAV file
    wf = wave.open(wav_path, 'rb')
    sample_rate = wf.getframerate()
    channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    
    if sample_rate != 16000 or channels != 1 or sampwidth != 2:
        print("Warning: RealtimeSTT expects 16kHz, 16-bit Mono WAV files.")
        
    chunk_size = int(sample_rate * (chunk_duration_ms / 1000.0))
    
    print("Streaming audio chunks...")
    data = wf.readframes(chunk_size)
    
    finalized_text = ""
    
    while data:
        # Feed exactly like the WebRTC AudioProcessor does
        recorder.feed_audio(data)
        
        # Pull finalized text
        while not text_queue.empty():
            finalized_text += text_queue.get_nowait() + " "
            print(f"\n[FINALIZED]: {finalized_text}")
            
        # Pull realtime text
        current_rt = ""
        while not realtime_queue.empty():
            current_rt = realtime_queue.get_nowait()
            
        if current_rt:
            print(f"\r[REALTIME]: {finalized_text} *{current_rt}*", end="", flush=True)
            
        # Simulate realtime delay
        time.sleep(chunk_duration_ms / 1000.0)
        data = wf.readframes(chunk_size)
        
    print("\n\nFinished streaming file. Waiting for final transcription to settle...")
    time.sleep(3) # Wait for VAD silence timeout to finalize the last sentence
    
    while not text_queue.empty():
        finalized_text += text_queue.get_nowait() + " "
        
    print(f"\n--- TEST COMPLETE ---")
    print(f"Final Transcript: {finalized_text}")
    
    # Clean up
    recorder.shutdown()

if __name__ == "__main__":
    # If a file is passed as an argument, use it. Otherwise use a default name.
    test_file = sys.argv[1] if len(sys.argv) > 1 else "data/test_audio.wav"
    stream_wav_file(test_file)
