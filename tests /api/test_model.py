"""
Unit tests for DogCatClassifier model architecture.
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.efficientnet import (
    DogCatClassifier,
    LabelSmoothingCrossEntropy,
    MixUpAugmentation,
    get_model,
)


@pytest.fixture(scope="module")
def model():
    m = DogCatClassifier(pretrained=False, dropout_rate=0.3)
    m.eval()
    return m

@pytest.fixture
def dummy_batch():
    return torch.randn(4, 3, 380, 380)

@pytest.fixture
def dummy_labels():
    return torch.randint(0, 2, (4,))


class TestModelArchitecture:
    def test_model_instantiation(self):
        assert DogCatClassifier(pretrained=False) is not None

    def test_output_shape(self, model, dummy_batch):
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 2)

    def test_parameter_count(self, model):
        params = model.count_parameters()
        assert 10_000_000 < params < 30_000_000

    def test_freeze_backbone(self):
        m = DogCatClassifier(pretrained=False, freeze_backbone=True)
        for p in m.backbone.parameters():
            assert not p.requires_grad
        for p in m.classifier.parameters():
            assert p.requires_grad

    def test_unfreeze_backbone(self):
        m = DogCatClassifier(pretrained=False, freeze_backbone=True)
        m.unfreeze_backbone()
        for p in m.backbone.parameters():
            assert p.requires_grad

    def test_factory_function(self):
        m = get_model({"model_name": "efficientnet_b4", "pretrained": False})
        assert isinstance(m, DogCatClassifier)

    def test_deterministic_eval(self, model, dummy_batch):
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_batch)
            out2 = model(dummy_batch)
        assert torch.allclose(out1, out2)

    def test_no_nan_output(self, model, dummy_batch):
        with torch.no_grad():
            out = model(dummy_batch)
        assert not torch.isnan(out).any()


class TestPredictions:
    def test_proba_sums_to_one(self, model, dummy_batch):
        probs = model.predict_proba(dummy_batch)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_predict_valid_class(self, model, dummy_batch):
        cls, conf = model.predict(dummy_batch)
        assert ((cls == 0) | (cls == 1)).all()
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_predicted_class_matches_argmax(self, model, dummy_batch):
        probs = model.predict_proba(dummy_batch)
        cls, _ = model.predict(dummy_batch)
        assert (cls == probs.argmax(dim=1)).all()


class TestLabelSmoothing:
    def test_loss_positive(self):
        crit = LabelSmoothingCrossEntropy(0.1)
        loss = crit(torch.randn(8, 2), torch.randint(0, 2, (8,)))
        assert loss.item() > 0

    def test_correct_lower_than_wrong(self):
        crit = LabelSmoothingCrossEntropy(0.1)
        labels = torch.tensor([1, 0])
        wrong   = crit(torch.tensor([[10.0, 0.01], [0.01, 10.0]]), labels)
        correct = crit(torch.tensor([[0.01, 10.0], [10.0, 0.01]]), labels)
        assert correct < wrong

    def test_gradient_flows(self):
        crit = LabelSmoothingCrossEntropy(0.1)
        logits = torch.randn(4, 2, requires_grad=True)
        crit(logits, torch.randint(0, 2, (4,))).backward()
        assert logits.grad is not None and not torch.isnan(logits.grad).any()


class TestMixUp:
    def test_output_shape(self, dummy_batch, dummy_labels):
        mixup = MixUpAugmentation(0.4)
        mixed, la, lb, lam = mixup(dummy_batch, dummy_labels)
        assert mixed.shape == dummy_batch.shape
        assert 0.0 <= lam <= 1.0

    def test_criterion_valid(self, dummy_labels):
        mixup = MixUpAugmentation(0.4)
        crit = LabelSmoothingCrossEntropy()
        _, la, lb, lam = mixup(torch.randn(4, 3, 32, 32), dummy_labels)
        loss = MixUpAugmentation.mixup_criterion(crit, torch.randn(4, 2), la, lb, lam)
        assert not torch.isnan(loss) and loss.item() > 0


class TestCheckpoint:
    def test_save_load_same_output(self, tmp_path, model, dummy_batch):
        ckpt = str(tmp_path / "ckpt.pth")
        model.save_checkpoint(ckpt, epoch=5, metrics={"val_acc": 99.2})
        loaded = DogCatClassifier.load_checkpoint(ckpt, device="cpu")
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(model(dummy_batch), loaded(dummy_batch), atol=1e-5)

    def test_metadata_stored(self, tmp_path, model):
        ckpt = str(tmp_path / "meta.pth")
        model.save_checkpoint(ckpt, epoch=10, metrics={"accuracy": 99.2})
        data = torch.load(ckpt, map_location="cpu")
        assert data["epoch"] == 10
        assert data["metrics"]["accuracy"] == 99.2
        assert data["model_name"] == "efficientnet_b4"
