import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yolact",
    version="1.2",
    author="Daniel Bolya",
    author_email="dbolya@ucdavis.edu",
    description="A simple, fully convolutional model for real-time instance segmentation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbolya/yolact",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "opencv-python",
        "pycocotools",
        "matplotlib",
    ],
    include_package_data=True,
    package_data={
        "": ["*.pyx"]
    }
)
