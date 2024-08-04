""" Test multi-modal functions """

from zhihu.multi_modal.clip import (
    clip_parse_image, print_cn_clip_available_models, get_device,
    cn_clip_parse_image, siglip_parse_image
)


def test_parse_image():
    """ test parse image using CLIP
        pytest tests/zhihu/test_multimodel.py::test_parse_image
    """
    image1 = "data/images/truck.jpg"
    cls_list = [
        "dog", "woman", "main", "car", "truck", "a black truck", "bird", "a white truck"
    ]
    outputs = clip_parse_image(image1, cls_list)
    print("outputs.keys()=", outputs.keys())
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    for i, value in enumerate(cls_list):
        print(f"{value}: {probs[0][i]}")


def test_cn_clip_available_models():
    """
    pytest tests/zhihu/test_multimodal.py::test_cn_clip_available_models

    B 代表 Big，L代表 Large, H代表 Huge;
    B L H 后面紧跟的数字代表图像 patch 化时，每个 patch 的分辨率大小，14 代表图像是按照 14x14 的分辨率被划分成相互没有 overlap 的图像块。
    -336 表示，输入图像被 resize 到 336x336 分辨率后进行的处理；默认是 224x224 的分辨率。
    RN50 表示 ResNet50
    """
    print_cn_clip_available_models()


def test_check_cuda_support():
    """
    pytest tests/zhihu/test_multimodal.py::test_check_cuda_support
    """
    get_device()


def test_cn_clip_parse_image():
    """
    pytest tests/zhihu/test_multimodal.py::test_cn_clip_parse_image
    """
    image1 = "data/images/truck.jpg"
    cls_list = ["狗", "汽车", "白色皮卡", "火车", "皮卡"]
    cn_clip_parse_image(image1, cls_list)


def test_siglip_parse_image():
    """
    pytest tests/zhihu/test_multimodal.py::test_siglip_parse_image
    """
    image1 = "data/images/truck.jpg"
    cls_list = ["dog", "woman", "man", "car", "truck",
                "a black truck", "a white truck", "cat"]
    siglip_parse_image(image1, cls_list)
