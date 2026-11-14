"""Post-training accuracy/demo runner for the official IDRiD test set."""
import argparse
from src.pipeline import evaluate, predict
def main():
    p=argparse.ArgumentParser(description="Test trained DR model")
    p.add_argument("--accuracy",action="store_true",help="evaluate labeled official test set")
    p.add_argument("--validation",action="store_true",help="score a checkpoint on the validation partition only")
    p.add_argument("--image",help="optional fundus image for prediction and Grad-CAM")
    p.add_argument("--checkpoint")
    a=p.parse_args()
    if a.accuracy and a.validation: p.error("Choose only one evaluation split")
    if a.accuracy: evaluate(a.checkpoint,"test")
    if a.validation: evaluate(a.checkpoint,"validation")
    if a.image: predict(a.image,a.checkpoint)
    if not a.accuracy and not a.image: p.error("Use --accuracy and/or --image PATH")
if __name__=="__main__": main()
