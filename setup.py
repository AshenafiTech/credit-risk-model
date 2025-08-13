from setuptools import setup, find_packages

setup(
    name="credit-risk-model",
    version="1.0.0",
    description="Credit Risk Scoring Model for Bati Bank",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "mlflow>=2.0.0",
        "pydantic>=1.8.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "flake8>=4.0.0", "jupyter>=1.0.0"],
        "api": ["fastapi>=0.68.0", "uvicorn>=0.15.0"],
    },
)