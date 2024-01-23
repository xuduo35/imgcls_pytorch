import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from model import get_net
import gradio as gr

f = open("./exp/labels.txt", "r")
lines = [line.strip() for line in f.readlines()]
f.close()

labels = []

for line in lines:
    blankidx = line.find(' ')
    labels.append(line[:blankidx])

classnum = len(labels)
ckptpath = "exp/ckpts/resnet101-epoch007.pth"
backbone = os.path.basename(ckptpath)[:os.path.basename(ckptpath).rfind('-')]

net = get_net(backbone, 3, classnum)
state_dict = torch.load(ckptpath)
net.load_state_dict(state_dict)
net.to('cuda')
net.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict(inp):
    inp = transform(inp).unsqueeze(0).to('cuda')
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(net(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(len(labels))}
    return confidences

demo = gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=5),
             examples=["./examples/茯苓.jpg", "./examples/防己.jpg", "./examples/瓜蒌皮.jpg"],
             theme="default",
             css=".footer{display:none !important}")

server_port = 7860 if 'PORT' not in os.environ else int(os.environ['PORT'])
# set quiet to block Event not found exception
demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=server_port, show_api=True, quiet=True)

