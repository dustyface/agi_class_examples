''' test CLIP model to regonize image '''
from typing import List
from transformers import CLIPProcessor, CLIPModel
from PIL import Image as PILImage
from IPython.display import Image as IImage, display
import torch
import requests
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from transformers import AutoProcessor, AutoModel


def _display_image(image_path: str):
    """ display the provided image, this is not for terminal, it's for Jupyter Notebook """
    display(IImage(filename=image_path))


def clip_parse_image(image_path: str, cls_list: List[str]):
    """ parse image using CLIP """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = PILImage.open(image_path)
    inputs = processor(text=cls_list, images=image,
                       return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return outputs


def get_device():
    """ return cuda or cpu """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    return device


def print_cn_clip_available_models():
    """ return available models """
    return print("Available models=", available_models())


def cn_clip_parse_image(image_path: str, cls_list: List[str]):
    """ parse image using CN_CLIP """
    device = get_device()
    model, preprocess = load_from_name(
        "ViT-B-16", device=get_device(), download_root="./")
    model.eval()
    image = preprocess(PILImage.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(cls_list).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        # 对特征归一化, 应该使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits_per_image, logits_per_text = model.get_similarity(image, text)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()

    for i, cls_item in enumerate(cls_list):
        print(f"{cls_item = !r}: {probs[0][i]}")


# 以下siglip的例子没有跑通
def siglip_parse_image(image_path: str, cls_list: List[str]):
    """ paerse image using SIGLIP """
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    processor = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    image = PILImage.open(image_path)
    inputs = processor(text=cls_list, images=image,
                       padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image)
    print("probs=", probs)
    for i, value in enumerate(cls_list):
        print(f"{value = !r}: at {probs[0][i]:.1%} possibility")
