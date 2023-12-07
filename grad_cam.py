import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_model(model_path):
        model = resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(model_path))
        return model


def preprocess_image(image_path):
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path)
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0)
        return input_tensor


def visualize_cam(image_path, cam):
        rgb_img = Image.open(image_path)
        rgb_img = np.asarray(rgb_img)
        grayscale_cam = cam[0, :]
        visualization = show_cam_on_image(rgb_img / 255., grayscale_cam, use_rgb=True)
        pid = image_path.split('/')[-1].split('.')[0]
        plt.imsave('fundus_grad_cam_none_{}.png'.format(pid), visualization)
        return visualization
        

def compare_cam(image_path, cam1, cam2):
        rgb_img = Image.open(image_path)
        rgb_img = np.asarray(rgb_img)
        grayscale_cam1 = cam1[0, :]
        grayscale_cam2 = cam2[0, :]
        visualization1 = show_cam_on_image(rgb_img / 255., grayscale_cam1, use_rgb=True)
        visualization2 = show_cam_on_image(rgb_img / 255., grayscale_cam2, use_rgb=True)
        pid = image_path.split('/')[-1].split('.')[0]
        
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(visualization1)
        ax1.set_title('none cbam')
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(visualization2)
        ax2.set_title('with cbam')
        plt.savefig('fundus_grad_cam_compare_{}.png'.format(pid))


def main():
        # none cbam
        model_path = './checkpoint/best_model_none.pt'
        image_path = sys.argv[1]
        
        model = load_model(model_path)
        target_layers = [model.features[0]]
        input_tensor = preprocess_image(image_path)
        
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        cbam_gradcam = visualize_cam(image_path, grayscale_cam)
        
        # with cbam
        model_path = './checkpoint/best_model_cbam.pt'
        image_path = sys.argv[1]
        
        model = load_model(model_path)
        target_layers = [model.features[0]]
        input_tensor = preprocess_image(image_path)
        
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        cbam_gradcam = visualize_cam(image_path, grayscale_cam)
        
        # compare and save 2 images in one
        compare_cam(image_path, grayscale_cam1, grayscale_cam2)
        

if __name__ == '__main__':
        main()
