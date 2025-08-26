# Image Processing & Deep Learning - Complete Study Guide

## Part A - CNN and Computer Vision (Alexandre Meyer)

### 1. Image Style Transfer Using CNNs (Gatys et al., CVPR 2016)

#### Core Concept
- **Objective**: Transfer artistic style from one image to another while preserving content
- **Key Innovation**: Uses pre-trained CNN (VGG) to separate and recombine content and style

#### VGG Network Role
- **Purpose**: Feature extraction at multiple layers
- **Why VGG?**: 
  - Pre-trained on ImageNet, captures rich hierarchical features
  - Lower layers: edges, textures
  - Higher layers: semantic content
- **Equivalent Networks**: ResNet, Inception, MobileNet (any pre-trained CNN)

#### Content vs Style Extraction
- **Content Representation**: 
  - Extracted from higher layers (conv4_2 or conv5_2)
  - Preserves spatial structure and semantic information
  - Loss: MSE between feature maps of content and generated image

- **Style Representation**:
  - Uses Gram matrices across multiple layers
  - Gram matrix: G[i,j] = Σ F[i,k] * F[j,k] (correlation between feature maps)
  - Captures texture patterns independent of spatial location
  - Loss: MSE between Gram matrices

#### Optimization Process
```
Total Loss = α * Content_Loss + β * Style_Loss
- Initialize with white noise or content image
- Optimize pixel values using gradient descent (L-BFGS)
- Not training the network, optimizing the image
```

### 2. Everybody Dance Now (Chan et al., CVPR 2019)

#### Objective
- Transfer dance movements from source video to target person
- **Input**: Source video with dancer, target person video/images
- **Output**: Target person performing source dance

#### Method Overview
1. **Pose Detection**:
   - OpenPose to extract 2D skeleton keypoints
   - Creates pose stick figures from source video

2. **Pose Transfer Pipeline**:
   - Source video → Pose extraction → Pose stick figures
   - Pose stick figures → pix2pixHD GAN → Target person dancing

3. **Key Components**:
   - **Skeleton Detection**: OpenPose for 2D keypoint extraction
   - **Generation**: Conditional GAN (pix2pixHD) 
   - Maps pose representation to realistic human appearance
   - **Temporal Coherence**: Face GAN for consistent facial features
   - **Training**: Requires video of target person for personalized model

### 3. YOLO (You Only Look Once)

#### Problem Type
- **Real-time object detection**: Localization + Classification simultaneously
- Single-stage detector (vs two-stage like R-CNN)

#### Core Algorithm Principle
1. **Grid Division**: 
   - Divide image into S×S grid
   - Each cell responsible for detecting objects whose center falls in it

2. **Predictions per Cell**:
   - B bounding boxes (x, y, w, h, confidence)
   - C class probabilities
   - Output: S × S × (B × 5 + C) tensor

3. **Single Forward Pass**:
   - End-to-end detection in one network evaluation
   - Extremely fast (45-155 FPS depending on version)

4. **Loss Function**:
   - Combines: localization loss + confidence loss + classification loss
   - Multi-task learning approach

5. **Non-Max Suppression**:
   - Post-processing to remove duplicate detections

### 4. CNN Fundamentals

#### Convolutional Layers
- **Purpose**: Feature extraction through learnable filters
- **Key Properties**:
  - Parameter sharing (same filter across image)
  - Translation invariance
  - Sparse connectivity

#### Pooling Layers
- **Purpose**: Downsampling and invariance
- **Types**: Max pooling, Average pooling
- **Benefits**: Reduces parameters, adds translation invariance

### 5. Autoencoders (AE)

#### Basic Structure
- **Encoder**: Compresses input to latent representation
- **Decoder**: Reconstructs from latent code
- **Bottleneck**: Forces information compression

#### Variants
- **Denoising AE**: Trained to reconstruct clean images from noisy inputs
- **Variational AE (VAE)**: Probabilistic, learns distribution in latent space
- **Convolutional AE**: Uses CNN layers for image data

### 6. U-Net Architecture

#### Key Features
- **Skip Connections**: Direct connections from encoder to decoder
- **Purpose**: Preserves fine-grained spatial information
- **Applications**: Segmentation, especially medical imaging
- **Architecture**: Symmetric encoder-decoder with concatenation

### 7. GANs (Generative Adversarial Networks)

#### Core Concept
- **Two Networks**: Generator (G) and Discriminator (D)
- **Adversarial Training**: Min-max game
- **Objective**: G generates realistic data, D distinguishes real from fake

#### Conditional GANs
- **Input Conditioning**: Both G and D receive additional information
- **Applications**: Image-to-image translation, controlled generation
- **Examples**: pix2pix, CycleGAN

---

## Part J. Digne - Generative Models & Neural Geometry

### 1. Neural Priors for Images (Deep Image Priors)

#### Core Principle
- CNN architecture itself provides implicit regularization
- Can restore images without training data
- Network structure encodes natural image statistics

#### Implementation in TP
- **Network Type**: Typically encoder-decoder or U-Net
- **Loss Function**: MSE between output and corrupted/partial input
- **Denoising Application**:
  - Input: Random noise
  - Target: Noisy image
  - Network learns to produce clean image

### 2. Deep Geometric Learning

#### Challenges for Geometric Data
- Non-Euclidean structure (meshes, graphs, point clouds)
- No regular grid like images
- Permutation invariance requirements

#### PointNet Principle
- **Key Innovation**: Symmetric aggregation function
- **Architecture**:
  1. Shared MLP on each point independently
  2. Max pooling for permutation invariance
  3. Global feature vector
- **Applications**: Classification, segmentation of point clouds

#### Categories for PointNet Implementation
- **Classification**: Entire point cloud → single label
- **Part Segmentation**: Each point → part label
- **Semantic Segmentation**: Each point → semantic class

### 3. Implicit Neural Representations

#### Gaussian Splatting Principle
- **Goal**: Represent 3D scenes as collection of 3D Gaussians
- **Input Data**: Multi-view images with camera poses
- **Output**: 3D Gaussian primitives (position, covariance, opacity, color)

#### Surface Reconstruction via INR
- **Principle**: Neural network as continuous function F(x,y,z) → SDF/occupancy
- **Implementation**:
  - Network inputs 3D coordinates
  - Outputs signed distance or occupancy value
  - Surface extracted at zero level set

#### From Implicit Function to Surface Mesh
- **Marching Cubes**: Extract isosurface from volumetric grid
- **Process**:
  1. Query network on 3D grid
  2. Find zero-crossings
  3. Generate triangle mesh

### 4. Encoder-Decoder Networks

#### Purpose in Generative Setting
- **Encoder**: Maps data to latent space
- **Decoder**: Generates data from latent codes
- **Latent Space**: Compact representation enabling interpolation

#### Limitations
- **Reconstruction Quality**: Limited by bottleneck size
- **Mode Collapse**: In GANs, generator produces limited variety
- **Training Instability**: Especially in GANs

---

## Part N. Bonneel - Diffusion Models & Optimal Transport

### 1. Diffusion Models

#### Forward Process
- Gradually add Gaussian noise over T steps
- q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

#### Reverse Process
- Learn to denoise: p_θ(x_{t-1}|x_t)
- Network predicts noise or clean image
- Iterative refinement from pure noise to image

#### Training
- **Objective**: Learn to predict noise ε added at each step
- **Loss**: MSE between predicted and actual noise
- **Sampling**: Start from Gaussian noise, iteratively denoise

#### Advantages over GANs
- More stable training
- Better mode coverage
- High quality generation

### 2. Optimal Transport

#### Core Problem
- Find minimum cost to transport mass from distribution P to Q
- Wasserstein distance measures difference between distributions

#### Applications in ML
- **Loss Functions**: Wasserstein loss for comparing distributions
- **Domain Adaptation**: Transport source to target domain
- **Style Transfer**: Transport style statistics

#### Connection to Image Processing
- Histogram matching
- Color transfer
- Texture synthesis

---

## Key Implementation Details from TPs

### General PyTorch Patterns

```python
# Basic training loop structure
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Common Loss Functions
- **MSE**: Reconstruction tasks (AE, style transfer)
- **BCE**: Binary classification, GAN discriminator
- **Cross-Entropy**: Multi-class classification
- **Perceptual Loss**: Feature matching in VGG space
- **Adversarial Loss**: GAN training

### Optimization Tips
- **Learning Rate Scheduling**: Reduce LR on plateau
- **Batch Normalization**: Stabilize training
- **Data Augmentation**: Improve generalization
- **Early Stopping**: Prevent overfitting

---

## Exam Preparation Checklist

### Must Understand
- [ ] Style transfer: content vs style separation
- [ ] VGG features and why they work
- [ ] Pose transfer pipeline (OpenPose → GAN)
- [ ] YOLO single-shot detection principle
- [ ] GAN training dynamics
- [ ] Diffusion forward/reverse process
- [ ] PointNet symmetry handling
- [ ] Implicit neural representations concept
- [ ] Optimal transport basics

### Implementation Skills
- [ ] Write basic CNN architecture
- [ ] Implement training loop
- [ ] Understand loss functions
- [ ] Debug common issues (gradient vanishing, overfitting)

### Key Papers Understanding
- [ ] Gatys et al. - Neural Style Transfer mechanics
- [ ] Chan et al. - Video-to-video synthesis pipeline
- [ ] Core principles, not architectural details

### Practice Questions
1. Explain how Gram matrices capture style
2. Why does U-Net work well for segmentation?
3. How does YOLO achieve real-time performance?
4. What makes GAN training unstable?
5. How do diffusion models avoid mode collapse?
6. Why is PointNet permutation invariant?