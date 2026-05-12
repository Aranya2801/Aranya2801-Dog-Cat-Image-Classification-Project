"""
DogCat Vision — Package Setup
"""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="dogcat-vision",
    version="2.1.0",
    author="Aranya",
    author_email="",
    description="Advanced Dog vs Cat Image Classifier — EfficientNet-B4 + FastAPI + Grad-CAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aranya2801/Dog-Cat-image-classification-project",
    project_urls={
        "Bug Tracker": "https://github.com/Aranya2801/Dog-Cat-image-classification-project/issues",
        "API Docs": "http://localhost:8000/docs",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    package_dir={"": "."},
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "albumentations>=1.3.1",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "grad-cam>=1.4.8",
        "mlflow>=2.8.0",
        "loguru>=0.7.2",
        "rich>=13.6.0",
        "pydantic>=2.4.0",
    ],
    entry_points={
        "console_scripts": [
            "dogcat-predict=src.predict:main",
            "dogcat-train=src.train:main",
            "dogcat-evaluate=src.evaluate:main",
            "dogcat-api=src.api.app:main",
        ],
    },
    include_package_data=True,
)
