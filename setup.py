from setuptools import setup, find_packages

setup(
    name="ai_hospital_os",
    version="1.0.0",
    description="AI-Powered Hospital Operating System — Clinical Decision Support Platform",
    author="Hospital OS Team",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "integration_tests*", "benchmarks*"]),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.2.0",
        "pyarrow>=14.0.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "ml": ["xgboost>=2.0.0", "lightgbm>=4.1.0", "shap>=0.43.0"],
        "deep": ["torch>=2.1.0"],
        "db":  ["psycopg2-binary>=2.9.9", "sqlalchemy>=2.0.0"],
        "api": ["fastapi>=0.110.0", "uvicorn>=0.27.0"],
        "dashboard": ["dash>=2.16.0", "plotly>=5.19.0"],
        "tracking": ["mlflow>=2.11.0"],
        "ws": ["websockets>=12.0"],
        "all": [
            "xgboost>=2.0.0", "lightgbm>=4.1.0", "shap>=0.43.0",
            "torch>=2.1.0", "psycopg2-binary>=2.9.9", "sqlalchemy>=2.0.0",
            "fastapi>=0.110.0", "uvicorn>=0.27.0", "dash>=2.16.0",
            "plotly>=5.19.0", "mlflow>=2.11.0", "websockets>=12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hospital-cli=cli.hospital_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
