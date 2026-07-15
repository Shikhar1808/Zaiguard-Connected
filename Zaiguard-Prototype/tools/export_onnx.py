# tools/export_onnx.py
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX")
    parser.add_argument("--model", type=str, default="yolov8n", help="Model name (e.g. yolov8n, yolov8s)")
    parser.add_argument("--size", type=int, default=640, help="Input image size")
    parser.add_argument("--output-dir", type=str, default="models/onnx", help="Output directory")
    args = parser.parse_args()

    from ultralytics import YOLO

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    output_path = output_dir / f"{args.model}.onnx"
    print(f"Exporting to ONNX (imgsz={args.size})...")
    model.export(
        format="onnx",
        imgsz=args.size,
        dynamic=False,          # static shapes — better for TensorRT later
        simplify=True,          # onnx-simplifier pass
        opset=17,               # opset 17 = safe for TRT 8.6+
        half=False,             # FP32 for now; quantize at TRT step
    )

    # ultralytics exports next to the .pt file, move it to output_dir
    default_export = Path(model_name).with_suffix(".onnx")
    if default_export.exists() and not output_path.exists():
        default_export.rename(output_path)
        print(f"Moved to {output_path}")
    else:
        print(f"Exported to {output_path}")

if __name__ == "__main__":
    main()