import argparse

def train_input_args():
    
    parser = argparse.ArgumentParser(
        description='Image Classifier Parameters')
    
    parser.add_argument('--train_dir', help= 'Enter path to training data')
    parser.add_argument('--test_dir', help= 'Enter path to testing data')
    parser.add_argument('--valid_dir', help= 'Enter path to validation data')
    
    parser.add_argument('--arch', help= 'Choices: vgg16 resnet18 alexnet')
    parser.add_argument('--hidden_units', nargs = 2, type= int, help = 'Enter 2 hidden layer sizes')
    parser.add_argument('--output_units', type= int, help = 'Enter output layer size')
    parser.add_argument('--dropout_prob', type= float, help = 'Enter decimal number between 0 and 1')
    
    parser.add_argument('--epochs', type = int)
    parser.add_argument('--learning_rate', type = float)
    parser.add_argument('--print_every', type = int )
    parser.add_argument('--device', help = 'Choices: cpu gpu')
    parser.add_argument('--checkpoint_label', help = 'Label for checkpoint (not including file extension)')
    return parser.parse_args()
