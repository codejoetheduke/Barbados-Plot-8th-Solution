import argparse
import pandas as pd
from utils.config import Config
from utils.model_manager import ModelManager
from inference import BarbadosInference
from train import BarbadosTrainer, DatasetOrganizer


class BarbadosPipeline:
    def __init__(self, mode="inference", config=None):
        self.mode = mode
        self.config = config or Config()
        self.config.set_seed()

    def run_inference(self, test_csv, image_dir, output_path, format_type="wkt", use_pretrained=True, model_path=None):
        print("🚀 Running Inference Pipeline...")

        # Load model
        if use_pretrained:
            model = ModelManager.load_pretrained(self.config.PRETRAINED_MODEL)
        else:
            if model_path is None:
                raise ValueError("❌ Please provide model_path when use_pretrained=False")
            model = ModelManager.load_local(model_path)

        # Load test data
        test_df = pd.read_csv(test_csv)
        inference = BarbadosInference(model, self.config)

        # Run inference
        submission_df = inference.generate_submission(test_df, image_dir, output_path, format_type)
        print(f"✅ Inference complete! Results saved to: {output_path}")
        return submission_df

    def run_training(self, train_csv, image_dir, label_dir, output_dir, epochs=100, batch=8, device=[0]):
        print("🚀 Running Training Pipeline...")
        train_df = pd.read_csv(train_csv)

        # Organize dataset
        organizer = DatasetOrganizer(image_dir, label_dir, output_dir, train_df, self.config.VAL_SPLIT)
        data_yaml = organizer.organize()

        # Train
        trainer = BarbadosTrainer(config=self.config)
        trainer.train(data_yaml, epochs, batch, device)


def parse_args():
    parser = argparse.ArgumentParser(description="🏝 Barbados Survey Plan Pipeline")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Choose mode: train or inference")

    # -------------------------------
    # Inference arguments
    # -------------------------------
    infer_parser = subparsers.add_parser("inference", help="Run inference mode")
    infer_parser.add_argument("--test_csv", required=True, help="Path to Test.csv file")
    infer_parser.add_argument("--image_dir", required=True, help="Path to test images")
    infer_parser.add_argument("--output_path", required=True, help="Path to save submission.csv")
    infer_parser.add_argument("--format_type", default="wkt", choices=["wkt", "list"], help="Output format type")
    infer_parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained model from Hugging Face")
    infer_parser.add_argument("--model_path", default=None, help="Path to local model if not using pretrained")

    # -------------------------------
    # Training arguments
    # -------------------------------
    train_parser = subparsers.add_parser("train", help="Run training mode")
    train_parser.add_argument("--train_csv", required=True, help="Path to Train.csv file")
    train_parser.add_argument("--image_dir", required=True, help="Path to training images")
    train_parser.add_argument("--label_dir", required=True, help="Path to YOLO label directory")
    train_parser.add_argument("--output_dir", required=True, help="Path to save organized dataset")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    train_parser.add_argument("--device", nargs="+", type=int, default=[0], help="GPU devices list")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = BarbadosPipeline(mode=args.mode)

    if args.mode == "inference":
        pipeline.run_inference(
            test_csv=args.test_csv,
            image_dir=args.image_dir,
            output_path=args.output_path,
            format_type=args.format_type,
            use_pretrained=args.use_pretrained,
            model_path=args.model_path
        )

    elif args.mode == "train":
        pipeline.run_training(
            train_csv=args.train_csv,
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device
        )
