import argparse
import utils
parser = argparse.ArgumentParser(description='train module')

parser.add_argument('data_dir')
parser.add_argument('--save_dir', dest="save_dir", default="./checkpoint.pth")
parser.add_argument('--arch', dest="arch",default="vgg16")
parser.add_argument('--learning_rate', dest="learning_rate",type=float,default=0.001)
parser.add_argument('--hidden_units', dest="hidden_units",type=int,default=512)
parser.add_argument('--epochs', dest="epochs",type=int,default=15)
parser.add_argument('--gpu', dest="gpu",default="gpu")

args = parser.parse_args()
data_path = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
core = args.gpu


train_data,train_dataloader,test_dataloader,valid_dataloader = utils.load_data(data_path)
model,criterion,optimizer = utils.build_model(102,[hidden_units],learning_rate,arch,core,dropout=0.5)
utils.train_model(model,train_dataloader,valid_dataloader,criterion,optimizer,epochs,core)
utils.save_checkpoint(model,arch,save_dir,train_data,learning_rate,core)

