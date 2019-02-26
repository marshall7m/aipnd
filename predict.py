from predict_input_args import predict_input_args
from process_images import process_image
from predict_image import model_load, predict

predict_arg = predict_input_args()

model = model_load(predict_arg.checkpoint_path)

predict(predict_arg.input_image, predict_arg.category_names, model, predict_arg.top_k, predict_arg.device)

#Example arguments
# python predict.py --input_image flowers/valid/1/image_06739.jpg --category_names cat_to_name.json --checkpoint checkpoint.tar --top_k 5 --device gpu