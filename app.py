"""Simple terminal entry point for production inference."""
import argparse
from src.inference.inference import InferenceEngine
from src.events.event_detection import generate_events
from src.utils.config import CONFIDENCE_THRESHOLD, MIN_CONSECUTIVE_CLIPS

def main():
    parser = argparse.ArgumentParser(description="Real-World Anomaly Detection System")
    parser.add_argument("video", nargs="?", help="Path to an input video")
    args = parser.parse_args()
    video = args.video or input("Enter video path: ").strip()
    print("Loading model...")
    engine = InferenceEngine()
    print("Processing video...")
    predictions = engine.predict_video(video)
    print("Generating predictions and aggregating events...")
    events = generate_events(predictions, CONFIDENCE_THRESHOLD, MIN_CONSECUTIVE_CLIPS)
    print("\nRESULT\n----------------------------------------")
    if not events:
        print("Status: No potential incident generated")
    for event in events:
        print(f"Status: {event['status']}")
        print(f"Activity: {event['activity']}")
        print(f"Confidence: {event['confidence'] * 100:.1f}%")
        print(f"Start: {event['start_time']:.2f}s")
        print(f"End: {event['end_time']:.2f}s")
        print(f"Duration: {event['duration']:.2f}s")
        print("Requires Human Review")
        print("----------------------------------------")

if __name__ == "__main__": main()
