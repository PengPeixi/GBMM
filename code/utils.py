import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
    
def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    
def train_one_epoch(model, optimizer, data_loader, data_loader_2, device, epoch, teacher, it,momentum_schedule):
    model.train()
    # class_weights = torch.tensor([2.29, 1.]).to(device)
    # loss_function = torch.nn.CrossEntropyLoss(class_weights)
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    preds = []  # 存储所有批次的预测值
    gts = []    # 存储所有批次的ground truth
    
    MSE = torch.nn.MSELoss()
    coff = torch.tensor((momentum_schedule[it] - 0.9) * 5)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, masks, clinical_datas, labels = data
        sample_num += images.shape[0]

        pred, pred_emb = model(images.to(device), masks.to(device), clinical_datas.to(device))
        # pred_t, pred_emb_t = teacher(images.to(device), masks.to(device), clinical_datas.to(device))
        # loss_const = MSE(pred_emb, pred_emb_t)

        # 提取每个预测值中正类的概率
        try:
            preds.append(torch.softmax(pred,dim=1).detach()[:, 1].cpu())  # 只取第二列 (正类的概率)，并移到CPU
        except:
            pred = pred.unsqueeze(0)
            preds.append(torch.softmax(pred,dim=1).detach()[:, 1].cpu())  # 只取第二列 (正类的概率)，并移到CPU
        gts.append(labels.detach().cpu())            # 移到CPU
        
        pred_classes = torch.max(torch.softmax(pred,dim=1), dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        # loss = coff * loss + (1-coff) * loss_const
        # loss =  model.loss(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        
    data_loader_2 = tqdm(data_loader_2, file=sys.stdout)
    for step, data in enumerate(data_loader_2):
        images, masks, clinical_datas, labels = data
        sample_num += images.shape[0]

        pred,pred_emb = model(images.to(device), masks.to(device), clinical_datas.to(device))
        # pred_t, pred_emb_t = teacher(images.to(device), masks.to(device), clinical_datas.to(device))
        # loss_const = MSE(pred_emb, pred_emb_t)

        # 提取每个预测值中正类的概率
        try:
            preds.append(torch.softmax(pred,dim=1).detach()[:, 1].cpu())  # 只取第二列 (正类的概率)，并移到CPU
        except:
            pred = pred.unsqueeze(0)
            preds.append(torch.softmax(pred,dim=1).detach()[:, 1].cpu())  # 只取第二列 (正类的概率)，并移到CPU
        gts.append(labels.detach().cpu())            # 移到CPU
        
        pred_classes = torch.max(torch.softmax(pred,dim=1), dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        # loss = coff * loss + (1-coff) * loss_const
        # loss =  model.loss(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader_2.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 将所有批次的预测值和ground truth连接起来
    all_preds = torch.cat(preds).numpy()  # 转换为numpy数组
    all_gts = torch.cat(gts).numpy()      # 转换为numpy数组
    # 计算AUC
    auc = roc_auc_score(all_gts, all_preds)
    
    ema_update_teacher(model, teacher, momentum_schedule, it)
    it += 1
    
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, data_loader_2=None, degrad=False):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0

    preds = []  # 存储所有批次的预测值
    gts = []    # 存储所有批次的ground truth
    l0_scores = []
    
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, masks, clinical_datas, labels = data
        sample_num += images.shape[0]

        pred,_ = model(images.to(device), masks.to(device), clinical_datas.to(device))
        pred = torch.softmax(pred,dim=1)

        try:
        # 提取每个预测值中正类的概率
            preds.append(pred[:, 1].cpu())  # 只取第二列 (正类的概率)，并移到CPU
            l0_scores.append(pred[:, 0].cpu())
        except:
            pred = pred.unsqueeze(0)
            preds.append(pred[:, 1].cpu())  # 只取第二列 (正类的概率)，并移到CPU
            l0_scores.append(pred[:, 0].cpu())
        
        gts.append(labels.cpu())            # 移到CPU
        
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        # loss =  model.loss(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    
    if degrad==True:
 
        data_loader_2 = tqdm(data_loader_2, file=sys.stdout)
        for step, data in enumerate(data_loader_2):
            images, masks, clinical_datas, labels = data
            sample_num += images.shape[0]

            pred,_ = model(images.to(device), masks.to(device), clinical_datas.to(device))
            pred = torch.softmax(pred,dim=1)

            try:
            # 提取每个预测值中正类的概率
                preds.append(pred[:, 1].cpu())  # 只取第二列 (正类的概率)，并移到CPU
                l0_scores.append(pred[:, 0].cpu())
            except:
                pred = pred.unsqueeze(0)
                preds.append(pred[:, 1].cpu())  # 只取第二列 (正类的概率)，并移到CPU
                l0_scores.append(pred[:, 0].cpu())

            gts.append(labels.cpu())            # 移到CPU

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(device))
            # loss =  model.loss(pred, labels.to(device))
            accu_loss += loss

            data_loader_2.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num)

    # 将所有批次的预测值和ground truth连接起来
    all_preds = torch.cat(preds).numpy()  # 转换为numpy数组
    all_gts = torch.cat(gts).numpy()      # 转换为numpy数组
    all_l0 = torch.cat(l0_scores).numpy() 
    # 计算AUC
    auc = roc_auc_score(all_gts, all_preds)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, all_l0, all_preds, all_gts
    # return accu_loss.item() / (step + 1),auc
