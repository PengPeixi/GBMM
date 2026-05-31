import os
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from my_dataset import MyDataSet,MyDataSet_5mm
from utils import train_one_epoch, evaluate
from graph_transformer_net import GraphTransformerNet
from monai.transforms import Compose, Resize, NormalizeIntensity, ScaleIntensityRange
import random
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 保存预测分数和标签到CSV文件
def save_predictions_to_csv(predictions, file_path):
    df = pd.DataFrame(predictions, columns=['File Path', 'Label 0 Score', 'Label 1 Score', 'True Label'])
    df.to_csv(file_path, index=False)
    print(f"Saved predictions to {file_path}")
    
def main(args):
    set_seed(611)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./p3_results"):
        os.makedirs("./p3_results")

    # Load the training/validation and test datasets
    # test_csv_path = '/root/autodl-tmp/test/3d/data/standardized_features_new.csv'
    test_csv_path = '/root/autodl-tmp/new_data/p3/3d_data/standardized_features.csv'

    test_data = pd.read_csv(test_csv_path)

    # Extract image paths, labels, and clinical data for training/validation set

    # Extract image paths, labels, and clinical data for test set
    test_images_path = test_data['File Path'].values
    test_images_label = test_data['Label'].values.astype(int)
    test_clinical_data = test_data.iloc[:, 2:].values

    # Generate mask paths for test set
    # test_images_path = np.array(["/root/autodl-tmp/test/3d/data/tumor/cropped_image_tumor/" + path for path in test_images_path])
    test_images_path = np.array(["/root/autodl-tmp/new_data/p3/3d_data/crop/image/" + path for path in test_images_path])
    test_masks_path = np.array([path.replace("image","mask").replace('_0000.nii.gz', '.nii.gz') for path in test_images_path])

    auc_results = []  # To store AUC results for each fold

    for fold in range(0,2):
        print(f"Fold {fold}")
        
        image_transform = {
            "val": Compose([Resize((64, 64, 64)), ScaleIntensityRange(a_min=63.18, a_max=106.23, b_min=0.0, b_max=1.0, clip=True)])
        }

        mask_transform = {
            "val": Compose([Resize((64, 64, 64))])
        }

        
        test_dataset = MyDataSet_5mm(images_path=test_images_path,
                                 masks_path=test_masks_path,
                                 clinical_data=test_clinical_data,
                                 images_class=test_images_label,
                                 image_transform=image_transform["val"], mask_transform=mask_transform["val"])

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        print('Using {} dataloader workers every process'.format(nw))
        
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=test_dataset.collate_fn)

        # Model initialization
        model = GraphTransformerNet().to(device)
        # model.load_state_dict(torch.load("/root/weights/pretrained_model.pth"))

        # for epoch in range(12,15):
        if fold == 0:
            epoch=22
        else:
            epoch=8
        model.load_state_dict(torch.load(f"./weights/teacher/model_fold_{fold + 1}_epoch_{epoch}.pth"))
        # Test and collect AUC
        test_loss, test_acc, test_auc, l0, l1, gt = evaluate(model=model,
                                   data_loader=test_loader,
                                   device=device,
                                   epoch=epoch)  # -1 to denote testing           

        # 保存测试集的预测分数和标签到CSV文件
        predictions = [(test_images_path[i], l0[i], l1[i], gt[i]) for i in range(len(test_images_path))]
        csv_file_path = f"./p3_results/fold_{fold + 1}_epoch_{epoch}_test_predictions.csv"
        save_predictions_to_csv(predictions, csv_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data-path', type=str, default="/data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)