import argparse

def predict_input_args():
    parser = argparse.ArgumentParser(
        description='Arguments to classify single image')
    
    parser.add_argument("--input_image", help = "Enter path to image")
    parser.add_argument("--category_names", help = "Enter file with labels (.json)")
    parser.add_argument("--checkpoint_path", help = "Model file (.tar)")
    parser.add_argument("--top_k", type = int, help = "Number of probabilities to show")
    parser.add_argument("--device", help = "Choices: cpu gpu")
      
    return parser.parse_args()