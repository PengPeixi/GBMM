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
from my_dataset import MyDataSet
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


def pretrain(args, device, test_images_path):
    set_seed(42)

    # CSV file path
    csv_file_path = '/root/autodl-tmp/3ddata/standardized_features.csv'

    # Use pandas to read data
    data = pd.read_csv(csv_file_path)

    # Extract original data paths and labels
    images_path = data['File Path'].values
    masks_path = np.array([path.replace('_image', '_mask') for path in images_path])
    clinical_data = data.iloc[:, 2:].values
    images_label = np.ones(len(images_path), dtype=int)  # Original data labels set to 0

    # Prepare new data paths and labels for pretraining
    new_images_path = np.array(
        [path.replace('/root/autodl-tmp/standardized_features', '/root/autodl-tmp/esophagus/esophagus_image') for path
         in images_path])
    new_masks_path = np.array(
        [path.replace('/root/autodl-tmp/standardized_features', '/root/autodl-tmp/esophagus/esophagus_mask') for path in
         masks_path])
    new_images_label = np.zeros(len(new_images_path), dtype=int)  # New data labels set to 1

    # Combine original and new data
    combined_images_path = np.concatenate((images_path, new_images_path))
    combined_masks_path = np.concatenate((masks_path, new_masks_path))
    combined_clinical_data = np.concatenate((clinical_data, clinical_data))
    combined_images_label = np.concatenate((images_label, new_images_label))

    #     # Remove overlapping samples with the test set
    #     test_set = set(test_images_path)
    #     non_overlapping_indices = [i for i, path in enumerate(combined_images_path) if path not in test_set]

    #     combined_images_path = combined_images_path[non_overlapping_indices]
    #     combined_masks_path = combined_masks_path[non_overlapping_indices]
    #     combined_clinical_data = combined_clinical_data[non_overlapping_indices]
    #     combined_images_label = combined_images_label[non_overlapping_indices]

    # data_transform = {
    #     "train": Compose([Resize((64, 64, 64))])
    # }
    image_transform = {
        "train": Compose(
            [Resize((64, 64, 64)), ScaleIntensityRange(a_min=63.18, a_max=106.23, b_min=0.0, b_max=1.0, clip=True)]),
    }

    mask_transform = {
        "train": Compose([Resize((64, 64, 64))]),
    }

    # Instantiate pretraining dataset
    pretrain_dataset = MyDataSet(images_path=combined_images_path,
                                 masks_path=combined_masks_path,
                                 clinical_data=combined_clinical_data,
                                 images_class=combined_images_label,
                                 image_transform=image_transform["train"], mask_transform=mask_transform["train"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # Number of workers
    print('Using {} dataloader workers every process'.format(nw))
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=pretrain_dataset.collate_fn)

    # Model initialization
    model = GraphTransformerNet().to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    lf = lambda x: ((1 + math.cos(x * math.pi / 20)) / 2) * (1 - args.lrf) + args.lrf  # Cosine schedule for 20 epochs
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    tb_writer = SummaryWriter(log_dir="./runs/pretraining")

    # Pretrain for 20 epochs
    for epoch in range(10):
        train_loss, train_acc, train_auc = train_one_epoch(model=model,
                                                           optimizer=optimizer,
                                                           data_loader=pretrain_loader,
                                                           device=device,
                                                           epoch=epoch)

        scheduler.step()

        # Log results to TensorBoard
        tags = ["pretrain_loss", "pretrain_acc", "pretrain_auc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], train_auc, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        print("Pretrain Epoch: {} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}".format(epoch + 1, train_loss, train_acc,
                                                                               train_auc))

    # Save pretrained weights
    torch.save(model.state_dict(), f"./weights/pretrained_model.pth")
    print("Pretraining completed and model saved.")
    tb_writer.close()

    return model


# 保存预测分数和标签到CSV文件
def save_predictions_to_csv(predictions, file_path):
    df = pd.DataFrame(predictions, columns=['File Path', 'Label 0 Score', 'Label 1 Score', 'True Label'])
    df.to_csv(file_path, index=False)
    print(f"Saved predictions to {file_path}")


def main(args):
    set_seed(611)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Load the training/validation and test datasets
    train_val_csv_path = '/root/autodl-tmp/v1/3d/data/split_data/train_data.csv'
    test_csv_path = '/root/autodl-tmp/v1/3d/data/split_data/test_data.csv'

    train_val_data = pd.read_csv(train_val_csv_path)
    test_data = pd.read_csv(test_csv_path)

    # Extract image paths, labels, and clinical data for training/validation set
    images_path = train_val_data['File Path'].values
    images_label = train_val_data['Label'].values.astype(int)
    clinical_data = train_val_data.iloc[:, 2:].values

    # Extract image paths, labels, and clinical data for test set
    test_images_path = test_data['File Path'].values
    test_images_label = test_data['Label'].values.astype(int)
    test_clinical_data = test_data.iloc[:, 2:].values

    # Generate mask paths for training/validation set
    images_path = np.array(["/root/autodl-tmp/v1/3d/data/tumor/tumor_image/" + path for path in images_path])
    masks_path = np.array([path.replace('_image', '_mask') for path in images_path])

    # Generate mask paths for test set
    test_images_path = np.array(["/root/autodl-tmp/v1/3d/data/tumor/tumor_image/" + path for path in test_images_path])
    test_masks_path = np.array([path.replace('_image', '_mask') for path in test_images_path])

    # Pretrain the model
    # pretrained_model = pretrain(args, device, test_images_path)

    # 80% training and 20% validation using StratifiedKFold
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=611)

    auc_results = []  # To store AUC results for each fold

    for fold, (train_index, val_index) in enumerate(skf.split(images_path, images_label)):
        print(f"Fold {fold + 1}")

        # Create a separate TensorBoard writer for each fold
        tb_writer = SummaryWriter(log_dir=f"./runs/fold_{fold + 1}")

        # Split the dataset into training and validation sets
        train_images_path, val_images_path = images_path[train_index], images_path[val_index]
        train_masks_path, val_masks_path = masks_path[train_index], masks_path[val_index]
        train_clinical_data, val_clinical_data = clinical_data[train_index], clinical_data[val_index]
        train_images_label, val_images_label = images_label[train_index], images_label[val_index]

        print("Train labels distribution:", np.bincount(train_images_label))
        print("Validation labels distribution:", np.bincount(val_images_label))

        image_transform = {
            "train": Compose([Resize((64, 64, 64)),
                              ScaleIntensityRange(a_min=63.18, a_max=106.23, b_min=0.0, b_max=1.0, clip=True)]),
            "val": Compose(
                [Resize((64, 64, 64)), ScaleIntensityRange(a_min=63.18, a_max=106.23, b_min=0.0, b_max=1.0, clip=True)])
        }

        mask_transform = {
            "train": Compose([Resize((64, 64, 64))]),
            "val": Compose([Resize((64, 64, 64))])
        }

        # Instantiate training dataset
        train_dataset = MyDataSet(images_path=train_images_path,
                                  masks_path=train_masks_path,
                                  clinical_data=train_clinical_data,
                                  images_class=train_images_label,
                                  image_transform=image_transform["train"], mask_transform=mask_transform["train"])

        # Instantiate validation dataset
        val_dataset = MyDataSet(images_path=val_images_path,
                                masks_path=val_masks_path,
                                clinical_data=val_clinical_data,
                                images_class=val_images_label,
                                image_transform=image_transform["val"], mask_transform=mask_transform["val"])

        test_dataset = MyDataSet(images_path=test_images_path,
                                 masks_path=test_masks_path,
                                 clinical_data=test_clinical_data,
                                 images_class=test_images_label,
                                 image_transform=image_transform["val"], mask_transform=mask_transform["val"])

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        print('Using {} dataloader workers every process'.format(nw))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=test_dataset.collate_fn)

        # Model initialization
        model = GraphTransformerNet().to(device)
        # model.load_state_dict(torch.load("./weights_print/pretrained_model.pth"))

        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        best_metric = -1
        best_metric_epoch = -1

        for epoch in range(args.epochs):
            # Train
            train_loss, train_acc, train_auc = train_one_epoch(model=model,
                                                               optimizer=optimizer,
                                                               data_loader=train_loader,
                                                               device=device,
                                                               epoch=epoch)

            scheduler.step()

            # Validate
            val_loss, val_acc, val_auc, _, _, _ = evaluate(model=model,
                                                           data_loader=val_loader,
                                                           device=device,
                                                           epoch=epoch)

            # Test and collect AUC
            test_loss, test_acc, test_auc, l0, l1, gt = evaluate(model=model,
                                                                 data_loader=test_loader,
                                                                 device=device,
                                                                 epoch=epoch)  # -1 to denote testing

            # 保存测试集的预测分数和标签到CSV文件
            predictions = [(test_images_path[i], l0[i], l1[i], gt[i]) for i in range(len(test_images_path))]
            csv_file_path = f"./results/fold_{fold + 1}_epoch_{epoch + 1}_test_predictions.csv"
            save_predictions_to_csv(predictions, csv_file_path)

            torch.save(model.state_dict(), f"./weights/model_fold_{fold + 1}_epoch_{epoch + 1}.pth")

            # Log results to TensorBoard for the current fold
            tags = ["train_loss", "train_acc", "train_auc", "val_loss", "val_acc", "val_auc", "learning_rate",
                    "test_loss", "test_acc", "test_auc"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], train_auc, epoch)
            tb_writer.add_scalar(tags[3], val_loss, epoch)
            tb_writer.add_scalar(tags[4], val_acc, epoch)
            tb_writer.add_scalar(tags[5], val_auc, epoch)
            tb_writer.add_scalar(tags[6], optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar(tags[7], test_loss, epoch)
            tb_writer.add_scalar(tags[8], test_acc, epoch)
            tb_writer.add_scalar(tags[9], test_auc, epoch)

            if test_auc >= best_metric:
                best_metric = test_auc
                best_metric_epoch = epoch + 1
                # torch.save(model.state_dict(), f"./weights/model_best_fold_{fold + 1}.pth")
                # print("saved new best metric model for fold {}".format(fold + 1))

            print(
                "Fold {} Epoch: {} Current AUC: {:.4f} Best AUC: {:.4f} at Epoch {}".format(
                    fold + 1, epoch + 1, test_auc, best_metric, best_metric_epoch
                )
            )

        auc_results.append(best_metric)
        print(f"Fold {fold + 1} Test AUC: {test_auc:.4f}")

        tb_writer.close()

    # Report average AUC from all folds
    average_auc = np.mean(auc_results)
    print(f"Average Test AUC over {len(auc_results)} folds: {average_auc:.4f}")


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