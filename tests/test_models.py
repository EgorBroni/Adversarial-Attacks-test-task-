import torch

from imagenette_stage0.models import make_model


@torch.no_grad()
def test_convnext_forward_shape():
    model = make_model("convnext_tiny", num_classes=3, pretrained=False)
    output = model(torch.rand(1, 3, 64, 64))

    assert output.shape == (1, 3)
