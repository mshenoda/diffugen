# DiffuGen: Adaptable Approach for Generating Labeled Image Datasets using Stable Diffusion Models

## [**Paper**](https://arxiv.org/abs/) | [**Project Page**](https://arxiv.org/abs/)

To address challenges associated with dataset generation, we introduce "DiffuGen," a simple and adaptable approach that harnesses the power of stable diffusion models to create labeled image datasets efficiently. By leveraging stable diffusion models, our approach not only ensures the quality of generated datasets but also provides a versatile solution for label generation. In this paper, we present the methodology behind DiffuGen, which combines the capabilities of diffusion models with two distinct labeling techniques: unsupervised and supervised. Distinctively, DiffuGen employs prompt templating for adaptable image generation and textual inversion to enhance diffusion model capabilities. 

<p align="center">
<img src="resources/data_overview.jpg" style="width:100%;"/>
</p>

## Framework
DiffuGen provides a robust framework that integrates pre-trained stable diffusion models, the versatility of prompt templating, and a range of diffusion tasks. By using an input configuration JSON, users can specify parameters to generate image datasets using three primary stable diffusion tasks. Each of these tasks not only benefits from the prompt templating mechanism, ensuring adaptability and richness, but also comes with its dedicated integral labeling pipeline. This design allows DiffuGen to provide both supervised and unsupervised labeling methods tailored to the specific needs of each task, ensuring a well-aligned and efficient labeling process for diverse application needs.

<p align="center">
<img src="resources/diffugen_overview.png" style="width:100%;"/>
</p>

## Installation

### Clone
```
git clone https://github.com/mshenoda/diffugen.git
```
### Install requirements

### PyTorch with CUDA Dependency
```
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```
### LabelDiffusion (Core Module) and other packages
```
pip3 install -r requirements.txt
```

## Generate Datasets

### Starting with Text-to-Image Pipeline
```
python generate.py txt2img datasets\cars.json
```

### Then Generate Dataset with Image-to-Image Pipeline
```
python generate.py img2img datasets\cars.json
```

### And/Or Generate Dataset Inpainting Pipeline
```
python generate.py inpaint datasets\cars.json
```