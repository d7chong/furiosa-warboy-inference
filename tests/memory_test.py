from torchinfo import summary
import argparse
from ultralytics import YOLO
import torch

def check_memory_usage(model_name, batch_size=1):
    if model_name[-3] != ".pt":
        model_name += ".pt"
    model = YOLO(model=model_name)
    dummy_input = torch.randn(batch_size, 3, 640, 640)  # Create a dummy input tensor
    summary(model.model, input_data=dummy_input)  # Pass the model's internal model and dummy input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="yolov5s")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    check_memory_usage(args.model_name, args.batch_size)