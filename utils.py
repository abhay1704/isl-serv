import torch
from PIL import Image as PIMAGE
from torchvision import transforms, models
import matplotlib.pyplot as plt
import requests
from io import BytesIO
# import  kagglehub 
import os
from handDetector import detect_and_crop_hands
import cv2
import numpy as np

# current_dir = os.getcwd()

def load_model():

    model_path = os.path.join(os.getcwd(), 'assets/model_resnet50_2.pth')

    checkpoint = torch.load(model_path, 'cpu')
    model = models.resnet50(weights= None)
    model.fc = checkpoint['classifier']

    model.load_state_dict(checkpoint['model_state'])
    idx_to_class = checkpoint['idx_to_class']

    return model, idx_to_class




def process(image):
    # image = None
    try:
        image = PIMAGE.open(BytesIO(image.read())).convert('RGB')

    except:
        image = PIMAGE.open(image).convert('RGB')
    
    # finally:
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    try:

        image = detect_and_crop_hands(image)
    except Exception as e:
        print(e)
        raise Exception('No hands detected in the image')

    image = PIMAGE.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
    ])

    # image.save('assets/processed_image.jpg')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predict_1(model, image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image, model = image.to(device), model.to(device)
    
    with torch.no_grad():
        model.eval()
        logp = model.forward(image)
        
    probs = torch.exp(logp)
    pred_prob, pred_class  = probs.topk(3, dim=1)
    return pred_prob[0].tolist(), pred_class[0].tolist()

