# 🤝 Contributing to DogCat Vision

Thank you for your interest in contributing! This project follows MIT-lab-grade engineering standards.

---

## 🚀 Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/Dog-Cat-image-classification-project.git
cd Dog-Cat-image-classification-project
```

### 2. Set Up Dev Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

### 3. Create Your Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

---

## 📝 Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code refactoring |
| `test:` | Adding tests |
| `chore:` | Build/tooling changes |
| `perf:` | Performance improvements |

Examples:
```
feat: add ViT-B/16 model option
fix: correct Grad-CAM target layer for EfficientNet-B7
docs: add webcam streaming tutorial
```

---

## ✅ Pull Request Checklist

Before submitting:

- [ ] Code passes `ruff check src/` and `black --check src/`
- [ ] All existing tests pass: `pytest tests/ -v`
- [ ] New features include tests
- [ ] Docstrings updated for changed functions
- [ ] README updated if user-facing changes

---

## 🧪 Running Tests

```bash
# Full suite with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Only fast unit tests
pytest tests/unit/ -v -m "not slow"
```

---

## 💡 Areas for Contribution

- 🐕 Breed classification (Stanford Dogs dataset)
- 📱 Mobile model (MobileNet/TFLite export)
- 🎥 Webcam real-time inference
- 🌍 Multi-language web UI
- 📊 More explainability methods (LIME, SHAP)
- 🧪 Additional test coverage

---

## 📬 Questions?

Open an [issue](https://github.com/Aranya2801/Dog-Cat-image-classification-project/issues) or start a [discussion](https://github.com/Aranya2801/Dog-Cat-image-classification-project/discussions).
