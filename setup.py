from setuptools import setup, find_packages

setup(
    name='flash',
    version='1.0.0',
    description='Accelerating LVMs ',
    packages=find_packages(),
    install_requires=[
        "torch",
        "accelerate == 1.6.0",
        "transformers == 4.51.1",
        "fschat == 0.2.31",
        "gradio == 3.50.2",
        "openai == 0.28.0",
        "anthropic == 0.5.0",
        "sentencepiece == 0.1.99",
        "protobuf == 3.19.0",
        "wandb"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)