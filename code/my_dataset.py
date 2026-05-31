from PIL import Image
import torch
from torch.utils.data import Dataset
import nibabel
import numpy as np
# from monai.transforms import MaskIntensity
import torch.nn.functional as F

class MyDataSet(Dataset):
    """基础数据集，处理单个样本"""
    
    def __init__(self, images_path: list, masks_path: list, clinical_data: list, 
                 images_class: list, image_transform=None, mask_transform=None,
                 apply_degradation=False):
        self.images_path = images_path
        self.masks_path = masks_path
        self.clinical_data = clinical_data
        self.images_class = images_class
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.apply_degradation = apply_degradation

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 加载原始图像数据
        img = nibabel.load(self.images_path[item])
        img_data = np.array(img.dataobj)
        
        # 加载掩码
        mask = nibabel.load(self.masks_path[item])
        mask_data = np.array(mask.dataobj)
        
        # 应用退化-复原处理（如果启用）
        if self.apply_degradation:
            depth = img_data.shape[2]  # 假设维度顺序为 (H, W, D)
            keep_indices = torch.arange(0, depth, 2)
            
            # 对图像进行退化-复原处理
            degraded_img = img_data[:, :, keep_indices]
            degraded_tensor = torch.from_numpy(degraded_img).float().unsqueeze(0).unsqueeze(0)
            restored_img_tensor = F.interpolate(
                degraded_tensor, 
                size=(img_data.shape[0], img_data.shape[1], depth),
                mode='trilinear',
                align_corners=False
            )
            img_data = restored_img_tensor.squeeze().numpy()
            
            # 对掩码进行同样的退化-复原处理
            degraded_mask = mask_data[:, :, keep_indices]
            degraded_mask_tensor = torch.from_numpy(degraded_mask).float().unsqueeze(0).unsqueeze(0)
            restored_mask_tensor = F.interpolate(
                degraded_mask_tensor, 
                size=(mask_data.shape[0], mask_data.shape[1], depth),
                mode='trilinear',
                align_corners=False
            )
            mask_data = restored_mask_tensor.squeeze().numpy()
        
        # 转换为张量并调整维度
        img_tensor = torch.as_tensor(img_data).unsqueeze(0)  # 添加通道维度
        img_tensor = img_tensor.float().permute(0, 3, 1, 2)  # (C, D, H, W)
        
        mask_tensor = torch.as_tensor(mask_data).unsqueeze(0)
        mask_tensor = mask_tensor.float().permute(0, 3, 1, 2)
        
        label = self.images_class[item]
        clinical_data = torch.as_tensor(self.clinical_data[item]).float()

        # 应用变换（如果存在）
        if self.image_transform is not None:
            img_tensor = self.image_transform(img_tensor)
            mask_tensor = self.mask_transform(mask_tensor)

        return img_tensor, mask_tensor, clinical_data, label

    @staticmethod
    def collate_fn(batch):
        images, masks, clinical_datas, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        clinical_datas = torch.stack(clinical_datas, dim=0)
        labels = torch.as_tensor(labels)
        return images, masks, clinical_datas, labels


    
class MyDataSet_5mm(Dataset):
    """基础数据集，处理单个样本"""
    
    def __init__(self, images_path: list, masks_path: list, clinical_data: list, 
                 images_class: list, image_transform=None, mask_transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.clinical_data = clinical_data
        self.images_class = images_class
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 加载原始图像数据
        img = nibabel.load(self.images_path[item])
        img_data = np.array(img.dataobj)
        
        # 加载掩码
        mask = nibabel.load(self.masks_path[item])
        mask_data = np.array(mask.dataobj)
        
        # 应用退化-复原处理（如果启用）
        depth = img_data.shape[2]*2  # 假设维度顺序为 (H, W, D)
        # keep_indices = torch.arange(0, depth, 2)

        # 对图像进行退化-复原处理
        # degraded_img = img_data[:, :, keep_indices]
        # degraded_tensor = torch.from_numpy(degraded_img).float().unsqueeze(0).unsqueeze(0)
        degraded_tensor  = torch.from_numpy(img_data).float().unsqueeze(0).unsqueeze(0)
        restored_img_tensor = F.interpolate(
            degraded_tensor, 
            size=(img_data.shape[0], img_data.shape[1], depth),
            mode='trilinear',
            align_corners=False
        )
        img_data = restored_img_tensor.squeeze().numpy()

        # 对掩码进行同样的退化-复原处理
        # degraded_mask = mask_data[:, :, keep_indices]
        # degraded_mask_tensor = torch.from_numpy(degraded_mask).float().unsqueeze(0).unsqueeze(0)
        degraded_mask_tensor = torch.from_numpy(mask_data).float().unsqueeze(0).unsqueeze(0)
        restored_mask_tensor = F.interpolate(
            degraded_mask_tensor, 
            size=(mask_data.shape[0], mask_data.shape[1], depth),
            mode='trilinear',
            align_corners=False
        )
        mask_data = restored_mask_tensor.squeeze().numpy()
        
        # 转换为张量并调整维度
        img_tensor = torch.as_tensor(img_data).unsqueeze(0)  # 添加通道维度
        img_tensor = img_tensor.float().permute(0, 3, 1, 2)  # (C, D, H, W)
        
        mask_tensor = torch.as_tensor(mask_data).unsqueeze(0)
        mask_tensor = mask_tensor.float().permute(0, 3, 1, 2)
        
        label = self.images_class[item]
        clinical_data = torch.as_tensor(self.clinical_data[item]).float()

        # 应用变换（如果存在）
        if self.image_transform is not None:
            img_tensor = self.image_transform(img_tensor)
            mask_tensor = self.mask_transform(mask_tensor)

        return img_tensor, mask_tensor, clinical_data, label

    @staticmethod
    def collate_fn(batch):
        images, masks, clinical_datas, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        clinical_datas = torch.stack(clinical_datas, dim=0)
        labels = torch.as_tensor(labels)
        return images, masks, clinical_datas, labels

    