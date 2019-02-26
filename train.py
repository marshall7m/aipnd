from train_input_args import train_input_args
from process_images import process_train_test
from train_model import load_pretrained_model, classifier_param, deep_train, test_model, model_accuracy_test, model_checkpoint

def main():
    
    in_arg = train_input_args()
    
    trainloader, validloader, testloader, train_data = process_train_test(in_arg.train_dir, in_arg.valid_dir, in_arg.test_dir)
    
    pretrained_model, pretrained_output = load_pretrained_model(in_arg.arch)
    
    model = classifier_param(pretrained_model, in_arg.arch, pretrained_output, in_arg.hidden_units, in_arg.output_units, in_arg.dropout_prob)
    
    train_model = deep_train(model, in_arg.arch, trainloader, validloader, in_arg.learning_rate, in_arg.epochs, in_arg.print_every, in_arg.device)
    
    model_accuracy_test(train_model, testloader, in_arg.device)

    save_model = model_checkpoint(in_arg.checkpoint_label, train_model, in_arg.arch, train_data, in_arg.epochs, in_arg.learning_rate, in_arg.dropout_prob)
    
if __name__ == "__main__":
    main()
    
    
    #Example arguments
    # python train.py --train_dir flowers/train --test_dir flowers/test --valid_dir flowers/test --arch resnet18 --hidden_units 500 200 --output_units 102 --dropout_prob .2 --epochs 1 --learning_rate .001 --print_every 50 --device gpu --checkpoint_label checkpoint
    
    
    
    
    