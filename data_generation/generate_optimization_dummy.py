import os
import torch
import torchvision
import open_clip

import torch.nn.functional as F

from PIL import Image


class DummyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(in_features=5, out_features=3*1024*1024)

    def forward(self, x):
        x = self.layer(x)
        x = torch.reshape(x, shape=(3, 1024, 1024))
        return x


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    toPIL = torchvision.transforms.ToPILImage()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name='ViT-H-14-quickgelu', pretrained='dfn5b')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name='ViT-H-14-quickgelu')

    openai_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
    openai_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)

    # Dummy direction to optimize
    dummy_network = DummyNetwork().to(device)
    text = "Something"
    ttext = tokenizer([text, "Noise"]).to(device)
    direction = torch.nn.Parameter(torch.randn(size=(1, 5)).to(device))

    # Optimizer
    optimizer = torch.optim.SGD(params=[direction], lr=0.1)
    # normalized_img = (img - openai_mean) / openai_std

    for i in range(100):
        print(f"{i}==========")
        print(direction)
        optimizer.zero_grad()

        image = dummy_network(direction)

        # Pre-process
        pre_img = toPIL(image)
        pre_img = preprocess(pre_img)
        pre_img = pre_img.to(device)
        pre_img = pre_img.unsqueeze(0)

        # Feature extraction
        feature = model.encode_image(image=pre_img)
        text = model.encode_text(text=ttext)

        featuref = feature / feature.norm(dim=-1, keepdim=True)
        textf = text / text.norm(dim=-1, keepdim=True)

        text_probs = featuref @ textf.T
        tobp = text_probs[0, 0]

        tobp.backward()
        optimizer.step()
        print(direction)

