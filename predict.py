import argparse
import utils

parser = argparse.ArgumentParser(description='predict module')

parser.add_argument('checkpoint')
parser.add_argument('img_path')
parser.add_argument('--gpu', dest="gpu",default="gpu")
parser.add_argument('--top_k', dest="top_k",default=5,type=int)
parser.add_argument('--category_names', dest="category_names",default="cat_to_name.json")
args = parser.parse_args()
checkpoint = args.checkpoint
img_path = args.img_path
core = args.gpu
top_k=args.top_k
category=args.category_names

model = utils.load_model(checkpoint)
probs,classes = utils.predict(img_path, model,core,category,top_k)
print(probs)
print(classes)

