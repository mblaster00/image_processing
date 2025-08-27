# Complete Deep Learning & Image Processing Study Guide

---

## Part A - Alexandre Meyer: CNNs and Computer Vision

### 1. Convolutional Neural Networks (CNNs)

#### What are CNNs and When to Use Them?

**Definition**: CNNs are specialized neural networks designed for processing grid-structured data, particularly images. They use convolutional layers to automatically learn spatial hierarchies of features, from edges to complex objects.

**When to use CNNs:**
- **Image Classification**: Determining what's in an image (ImageNet, CIFAR)
- **Object Detection**: Finding and localizing objects (medical imaging, autonomous driving)
- **Semantic Segmentation**: Pixel-wise classification (medical diagnosis, satellite imagery)
- **Any spatial data**: Time series (1D CNN), videos (3D CNN), graphs (Graph CNN)

**Why CNNs over traditional methods:**
- **Automatic feature learning**: No need for hand-crafted features like SIFT/HOG
- **Translation invariance**: Recognizes patterns regardless of position
- **Parameter efficiency**: Shared weights across spatial dimensions
- **Hierarchical learning**: Builds complex features from simple ones

#### 1.1 Convolution Operation - Mathematical Foundation

**Core Mathematical Operation:**
```
(f * g)(t) = ∑_{m=-∞}^{∞} f(m) · g(t - m)  // 1D discrete
(f * g)(x,y) = ∑∑ f(m,n) · g(x-m, y-n)     // 2D discrete
```

**In CNNs:**
- Input: I (image/feature map)
- Kernel: K (learnable filter)
- Output: O = I * K
- With bias: O[i,j] = (I * K)[i,j] + b

**Output Size Formula:**
```
Output_size = ⌊(Input_size - Kernel_size + 2×Padding) / Stride⌋ + 1
```

**Example**: 224×224 image, 3×3 kernel, padding=1, stride=1 → 224×224 output (same size)

#### 1.2 Why CNNs Work - Three Fundamental Properties

**1. Local Connectivity (Sparse Interactions):**
- Each neuron connects only to local region (receptive field)
- Dramatically reduces parameters: FC 1000×1000 = 1M params vs Conv 5×5 = 25 params
- Biological inspiration: Neurons in visual cortex respond to local stimuli

**2. Parameter Sharing (Weight Tying):**
- Same kernel applied across entire image
- If it detects vertical edges at (10,10), it can detect at (100,100)
- Provides translation invariance naturally

**3. Hierarchical Feature Learning:**
```
Layer 1 (3×3 RF): Edges, colors, gradients
Layer 2 (5×5 RF): Corners, simple textures, curves
Layer 3 (11×11 RF): Parts (eyes, wheels), patterns
Layer 4 (23×23 RF): Objects (faces, cars), complex textures
Layer 5 (47×47 RF): Object compositions, full scenes
```

#### 1.3 Common CNN Architectures Evolution

**LeNet-5 (1998)**: Pioneer for digit recognition
- Small network: 60K parameters
- Introduced conv→pool→conv→pool→FC pattern

**AlexNet (2012)**: ImageNet breakthrough
- 60M parameters, ReLU activation, Dropout
- Proved deep learning superiority

**VGG (2014)**: Simplicity and depth
- Only 3×3 convolutions, very deep (16-19 layers)
- Showed smaller filters + deeper = better

**ResNet (2015)**: Skip connections revolution
- Solved vanishing gradient for very deep networks
- Enabled 100+ layer networks

### 2. Neural Style Transfer (Gatys et al., 2016)

#### What is Style Transfer and Its Applications?

**Definition**: Style Transfer is a technique that reimagines one image (content) in the artistic style of another image (style), creating a new image that maintains the content's structure but adopts the style's artistic characteristics.

**Real-world Applications:**
- **Artistic Creation**: Transform photos into paintings (Prisma app)
- **Video Stylization**: Apply consistent artistic style to videos
- **Data Augmentation**: Generate stylized training data
- **Design Tools**: Quick mockups in different artistic styles
- **Cultural Preservation**: Recreate art in different historical styles

**Why This Method is Revolutionary:**
- First to separate content and style using deep learning
- No need for explicit style rules or artist knowledge
- Works with any content-style pair
- Opened the field of neural artistic creation

#### 2.1 Content Representation - Preserving What

**Content Loss at layer l:**
```
L_content^l = 1/2 ∑_{i,j} (F_{ij}^l - P_{ij}^l)²
```
Where:
- F^l = feature response of generated image at layer l
- P^l = feature response of content image at layer l

**Layer Choice for Content (typically conv4_2 or conv5_2):**
- **Too shallow (conv1_1)**: Preserves exact pixels, not semantic content
- **Just right (conv4_2)**: Preserves object arrangement and composition
- **Too deep (conv5_4)**: Too abstract, loses spatial information

**Intuition**: Higher layers recognize "there's a building here" rather than specific pixels

#### 2.2 Style Representation - Capturing How

**Gram Matrix - The Key Innovation:**
```
G_{ij}^l = ∑_k F_{ik}^l × F_{jk}^l
```
- Captures which features activate together
- Removes spatial information (sum over k)
- Size: C×C where C = number of channels

**What Gram Matrix Actually Captures:**
- Texture patterns: "vertical edges often appear with blue"
- Color distributions: "warm tones dominate"
- Brush strokes: "diagonal patterns are common"
- Overall mood: correlations create artistic feel

**Multi-layer Style (why use multiple layers):**
```
L_style = ∑_l w_l × E_l
```
- conv1_1: Color and simple textures
- conv2_1: Small patterns and strokes
- conv3_1: Larger textural elements
- conv4_1: High-level style patterns
- conv5_1: Overall composition style

#### 2.3 The Optimization Process

**Not Training a Network - Optimizing an Image:**
```python
# Start with noise or content image
generated = content_image.clone() or torch.randn_like(content_image)
generated.requires_grad = True

# Optimize the pixels directly
optimizer = LBFGS([generated])

for iteration in range(500):
    # Forward pass through VGG
    content_features = vgg(generated)[content_layers]
    style_features = vgg(generated)[style_layers]
    
    # Compute losses
    content_loss = MSE(content_features, target_content)
    style_loss = sum([gram_loss(s, target_style) for s in style_features])
    total_loss = α * content_loss + β * style_loss
    
    # Backward pass updates the image pixels
    optimizer.step()
```

**Hyperparameter Balance:**
- α/β ratio determines content vs style strength
- Typical: α=1, β=1000 (style needs larger weight)

### 3. YOLO - You Only Look Once

#### What is YOLO and Why It Matters?

**Definition**: YOLO is a real-time object detection system that frames detection as a single regression problem, directly from image pixels to bounding box coordinates and class probabilities in one evaluation.

**Revolutionary Aspects:**
- **Speed**: 45-155 FPS vs 0.5 FPS for R-CNN
- **Global Context**: Sees entire image at once (unlike sliding window)
- **Unified Architecture**: Single network does everything
- **End-to-end Training**: Direct optimization for detection

**When to Use YOLO:**
- **Real-time applications**: Autonomous driving, surveillance, robotics
- **Video processing**: When you need to process every frame
- **Edge devices**: Efficient enough for mobile/embedded systems
- **Multiple object scenarios**: Naturally handles multiple objects

**Trade-offs:**
- Faster but less accurate than two-stage detectors
- Struggles with small objects in groups
- Better for "good enough quickly" than "perfect eventually"

#### 3.1 Core Innovation - Grid-Based Detection

**The Grid System:**
```
Image → S×S grid (e.g., 7×7 or 13×13)
Each cell responsible for objects whose center falls in it
```

**What Each Grid Cell Outputs:**
```
For each of B bounding boxes:
- (x, y): Center position relative to cell
- (w, h): Size relative to entire image  
- confidence: P(Object) × IoU

For the cell (shared):
- C class probabilities: P(Class_i | Object)

Total: S × S × (B × 5 + C) predictions
Example: 7 × 7 × (2 × 5 + 20) = 1470 values
```

#### 3.2 Single Shot Detection Philosophy

**Traditional Approach (R-CNN family):**
1. Generate ~2000 region proposals
2. Run CNN on each region (2000 forward passes!)
3. Post-process with NMS
4. Speed: ~0.5 FPS

**YOLO Approach:**
1. Single forward pass through network
2. Direct prediction of all boxes and classes
3. NMS post-processing
4. Speed: 45+ FPS

**Why This Works:**
- Implicit region proposals via grid
- Shared computation across entire image
- Learn to suppress duplicates during training

#### 3.3 Loss Function Design

**Multi-task Loss Components:**
```
L = λ_coord × L_coord      # Localization (most important)
  + λ_obj × L_obj          # Objects present
  + λ_noobj × L_noobj      # No objects (many cells empty)
  + λ_class × L_class      # Classification
```

**Key Design Choices:**
- √w, √h in loss: Small objects matter more
- λ_coord = 5: Localization is critical
- λ_noobj = 0.5: Most cells are empty (class imbalance)

### 4. Everybody Dance Now - Motion Transfer

#### What is Motion Transfer and Its Applications?

**Definition**: "Everybody Dance Now" is a deep learning pipeline that transfers dance moves from a source person in a video to a target person, making the target appear to perform the same dance with realistic motion and appearance.

**Breakthrough Applications:**
- **Entertainment**: Virtual performances, dance tutorials
- **Film Industry**: Stunt double replacement, choreography preview
- **Education**: Learning complex movements, sports training
- **Accessibility**: Allowing anyone to "perform" complex dances
- **Virtual Reality**: Avatar animation from real performances

**Why This is Challenging:**
- Preserve target's identity while changing pose
- Maintain temporal coherence across frames
- Handle occlusions and complex poses
- Generate realistic details (face, hands)

#### 4.1 The Three-Stage Pipeline

**Stage 1: Motion Extraction (OpenPose)**
```
Source Video → 2D Pose Keypoints → Stick Figure Representation
```
- Extracts 18-25 body keypoints per frame
- Creates intermediate pose representation
- Person-agnostic (just skeleton)

**Stage 2: Appearance Transfer (pix2pixHD)**
```
Pose Stick Figures → Conditional GAN → Target Person in Pose
```
- Learns mapping from pose to specific person
- Requires training video of target person
- Personalized model per target

**Stage 3: Enhancement (Face GAN + Temporal)**
```
Coarse Result → Face Enhancement → Temporal Smoothing → Final Video
```
- Dedicated face network for identity preservation
- Temporal discriminator for smooth motion

#### 4.2 Why Specialized Components?

**Main Network (pix2pixHD):**
- Resolution: 512×256 for full body
- Good for overall pose and clothing
- But: Face is only ~50×50 pixels (too small!)

**Face GAN (Essential for Realism):**
- Operates at 128×128 for face region
- Preserves identity-critical features
- Humans very sensitive to face artifacts

**Temporal Coherence:**
- Video needs smooth frame transitions
- Temporal discriminator ensures consistency
- Optical flow warping as initialization

### 5. U-Net Architecture

#### What is U-Net and When to Use It?

**Definition**: U-Net is a convolutional neural network architecture designed for precise segmentation tasks, featuring a symmetric encoder-decoder structure with skip connections that preserve fine-grained spatial information.

**Originally Designed For:**
- **Biomedical Image Segmentation**: Cell boundaries, organ segmentation
- Works with very few training images (data augmentation critical)
- Needs precise localization (pixel-level accuracy)

**Now Widely Used For:**
- **Medical Imaging**: Tumor detection, organ segmentation
- **Satellite Imagery**: Land use classification, road extraction
- **Industrial Inspection**: Defect detection, quality control
- **Image Restoration**: Denoising, super-resolution
- **Any Dense Prediction**: Where every pixel needs a label

**Why U-Net Over FCN or DeepLab:**
- Better at preserving fine details (blood vessels, cell boundaries)
- More parameter efficient than FCN
- Works with limited training data
- Skip connections prevent information loss

#### 5.1 The U-Shape Architecture

**Contracting Path (Encoder - Left Side):**
```
Input → [Conv-BN-ReLU-Conv-BN-ReLU] → MaxPool → ... → Bottleneck
572×572 → 570×570 → 284×284 → ... → 28×28
```
- Captures context and "what" is in the image
- Increases feature channels (64→128→256→512→1024)
- Reduces spatial dimensions

**Expanding Path (Decoder - Right Side):**
```
Bottleneck → UpConv → Concat(skip) → [Conv-BN-ReLU-Conv-BN-ReLU] → ... → Output
28×28 → 56×56 → ... → 388×388
```
- Enables precise localization
- Combines semantic information with high-resolution features

**Skip Connections (The Key Innovation):**
```python
decoder_features = concatenate(
    upsampled_deep_features,  # "what" (semantic)
    encoder_features[level]    # "where" (details)
)
```

#### 5.2 Why Skip Connections are Critical

**Problem with Standard Encoder-Decoder:**
- Bottleneck loses spatial information
- Decoder must reconstruct details from compressed representation
- Results in blurry, imprecise boundaries

**Skip Connections Solve This:**
1. **Direct gradient flow**: Easier training, no vanishing gradient
2. **Multi-resolution processing**: Combines coarse and fine features
3. **Preserves boundaries**: High-resolution features maintain edges
4. **Reduces parameters**: Decoder doesn't need to learn to reconstruct details

### 6. Generative Adversarial Networks (GANs)

#### What are GANs and Their Impact?

**Definition**: GANs are a framework where two neural networks compete in a game: a Generator creates fake data trying to fool a Discriminator, while the Discriminator tries to distinguish real from fake data.

**Revolutionary Impact (2014-present):**
- **Photorealistic Image Generation**: StyleGAN faces indistinguishable from real
- **Creative AI**: Art generation, design tools
- **Data Augmentation**: Generate training data for rare cases
- **Domain Translation**: Day→Night, Sketch→Photo, Horse→Zebra
- **Super-resolution**: Enhance low-resolution images

**When to Use GANs:**
- Need high-quality, sharp samples (vs VAE's blurry outputs)
- No need for explicit density estimation
- Have computational resources for careful tuning
- Creative/generative applications

**When NOT to Use GANs:**
- Need stable, predictable training
- Require likelihood estimates
- Limited computational budget
- Need to cover entire data distribution

#### 6.1 The Adversarial Game

**The Minimax Objective:**
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1-D(G(z)))]
```

**Intuitive Explanation:**
- **Discriminator wants**: D(real)=1, D(fake)=0
- **Generator wants**: D(fake)=1
- **Equilibrium**: D(everything)=0.5 (can't tell difference)

**Training Dynamics:**
```python
# Not training simultaneously - alternating optimization
for epoch in epochs:
    # Step 1: Train Discriminator (fix G)
    for k_steps:  # Often k=5
        d_loss = -log(D(real)) - log(1-D(fake))
        optimize_D(d_loss)
    
    # Step 2: Train Generator (fix D)
    g_loss = -log(D(fake))  # Want D to think fake is real
    optimize_G(g_loss)
```

#### 6.2 Common Problems and Solutions

**Mode Collapse:**
- **Problem**: Generator produces limited variety
- **Example**: Always generates same face regardless of input
- **Solutions**: Minibatch discrimination, unrolled GANs, WGAN

**Vanishing Gradients:**
- **Problem**: When D is too good, gradient for G becomes zero
- **Original fix**: max log(D(G(z))) instead of min log(1-D(G(z)))
- **Modern fix**: Wasserstein loss, spectral normalization

**Training Instability:**
- **Problem**: Loss oscillates, no clear convergence
- **Solutions**: 
  - Progressive growing (start small, increase resolution)
  - Spectral normalization (constrain Lipschitz constant)
  - Self-attention (better long-range dependencies)

#### 6.3 Important GAN Variants

**Conditional GAN (cGAN):**
```python
G(z, condition) → fake_image
D(image, condition) → real/fake
```
Applications: Text-to-image, image editing, style control

**CycleGAN (Unpaired Translation):**
- No paired training data needed
- Two generators: G:X→Y, F:Y→X
- Cycle consistency: F(G(x))≈x

**StyleGAN (High-Quality Generation):**
- Separate style and content
- Progressive growing
- Style injection at multiple scales

### 7. Autoencoders Family

#### What are Autoencoders and Their Purpose?

**Definition**: Autoencoders are neural networks trained to copy their input to their output through a bottleneck, learning compressed representations in an unsupervised manner.

**Core Applications:**
- **Dimensionality Reduction**: Nonlinear PCA alternative
- **Anomaly Detection**: Reconstruction error identifies outliers
- **Denoising**: Clean corrupted data
- **Feature Learning**: Pretrain networks, learn representations
- **Generation**: VAEs for controllable generation

**Autoencoder Spectrum:**
- **Vanilla AE**: Basic compression and reconstruction
- **Denoising AE**: Robust feature learning
- **Sparse AE**: Interpretable features
- **VAE**: Probabilistic generation
- **VQ-VAE**: Discrete latent codes

#### 7.1 Vanilla Autoencoder

**Architecture and Loss:**
```
x → Encoder → z (bottleneck) → Decoder → x̂
Loss: L = ||x - x̂||² (MSE)
```

**Why Bottleneck is Essential:**
- Forces compression: dim(z) << dim(x)
- Prevents trivial identity function
- Learns most important features

**Comparison to PCA:**
- PCA: Linear, orthogonal components
- AE: Nonlinear, more flexible
- Both: Unsupervised dimensionality reduction

#### 7.2 Denoising Autoencoder (DAE)

**Training Strategy:**
```python
# Add noise to input, reconstruct clean version
x_noisy = x + noise  # or dropout(x) or mask(x)
x_reconstructed = Decoder(Encoder(x_noisy))
loss = ||x - x_reconstructed||²  # Compare to CLEAN x
```

**Why This Works Better:**
- Learns robust features that survive corruption
- Implicit regularization prevents overfitting
- Forces learning of data manifold structure
- Better for downstream tasks

#### 7.3 Variational Autoencoder (VAE)

**Probabilistic Framework:**
```
Encoder: q(z|x) = N(μ(x), σ²(x))
Decoder: p(x|z) = N(f(z), σ²I)
Prior: p(z) = N(0, I)
```

**The VAE Loss (ELBO):**
```
L = -E[log p(x|z)] + KL(q(z|x) || p(z))
  = Reconstruction + Regularization
```

**Why VAE > Regular AE for Generation:**
1. **Continuous latent space**: Interpolation is meaningful
2. **Regularized structure**: No "holes" in latent space
3. **Probabilistic**: Can sample new data points
4. **Disentanglement**: Often learns interpretable factors

---

## Part B - Nicolas Bonneel: Advanced Generative Models

### 1. Diffusion Models

#### What are Diffusion Models and Why They Dominate?

**Definition**: Diffusion models are generative models that learn to gradually denoise data by reversing a process that slowly adds noise, inspired by thermodynamic diffusion processes.

**Current Dominance (2020-present):**
- **DALL-E 2, Stable Diffusion, Midjourney**: Text-to-image revolution
- **Outperform GANs**: Better mode coverage, more stable
- **State-of-the-art quality**: Photorealistic images, videos
- **Controllable**: Classifier-free guidance, conditioning

**When to Use Diffusion Models:**
- **High-quality generation**: When quality matters most
- **Full distribution coverage**: Need diversity, not just quality
- **Stable training**: No GAN-style instabilities
- **Conditional generation**: Text-to-image, image editing

**Trade-offs:**
- Slower sampling than GANs (50-1000 steps vs 1 step)
- Computationally intensive training
- Large model sizes

#### 1.1 The Forward Process - Gradual Destruction

**Mathematical Definition:**
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

**Intuition**: Like dropping ink in water
- t=0: Clear image
- t=T/4: Slightly noisy but recognizable
- t=T/2: Very noisy, some structure visible
- t=T: Pure Gaussian noise

**Closed Form (Key for Training):**
```
x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
where ᾱ_t = ∏(1-β_i), ε ~ N(0,I)
```
This lets us jump to any noise level directly!

**Variance Schedule (Critical for Performance):**
- **Linear**: β_t increases linearly from 0.0001 to 0.02
- **Cosine**: Better for images, maintains SNR
- **Learned**: Optimize schedule for specific data

#### 1.2 The Reverse Process - Learning to Denoise

**The Challenge:**
```
True reverse: p(x_{t-1} | x_t) = ???  # Intractable!
```

**The Solution - Learn It:**
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Network Parameterization Options:**
1. **Predict x_0**: Direct but unstable
2. **Predict ε (noise)**: Most common, stable
3. **Predict score**: ∇log p(x_t)
4. **Predict velocity**: v = αε - σx

**Why Noise Prediction Works Best:**
```python
# At high noise (early timesteps):
# Predicting x_0 is nearly impossible
# Predicting ε is just identifying noise pattern

# At low noise (late timesteps):  
# Both are easy, but ε more consistent
```

#### 1.3 Training - Beautifully Simple

**Training Algorithm:**
```python
def train_diffusion(model, data, T=1000):
    x_0 = sample_batch(data)
    t = random.randint(1, T)  # Random timestep
    ε = torch.randn_like(x_0)  # Random noise
    
    # Add noise to data
    x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
    
    # Predict the noise
    ε_pred = model(x_t, t)
    
    # Simple MSE loss
    loss = MSE(ε, ε_pred)
    return loss
```

**Connection to Denoising Score Matching:**
```
∇_x log p(x) = -1/σ² (x - μ) = -ε/√(1-ᾱ_t)
```
So predicting noise = predicting score function!

#### 1.4 Sampling Strategies

**DDPM (Original, Stochastic):**
```python
x_T ~ N(0, I)
for t in reversed(range(T)):
    z = torch.randn_like(x_t) if t > 0 else 0
    ε_θ = model(x_t, t)
    
    # Denoise step
    x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ) + σ_t * z
```
- 1000 steps for best quality
- Stochastic: different results each time

**DDIM (Deterministic, Faster):**
```python
# Can skip steps!
x_{t-1} = √(ᾱ_{t-1}) * predicted_x_0 + √(1-ᾱ_{t-1}) * ε_θ
```
- 50 steps often sufficient
- Deterministic: same input → same output
- Enables interpolation in latent space

#### 1.5 Classifier-Free Guidance - The Secret Sauce

**Training (10% unconditional):**
```python
if random() < 0.1:
    ε_pred = model(x_t, t, ∅)  # No condition
else:
    ε_pred = model(x_t, t, "a cat")  # With condition
```

**Sampling (Amplify Conditioning):**
```python
ε_guided = (1 + w) * model(x_t, t, cond) - w * model(x_t, t, ∅)
```
- w=0: No guidance (more diverse, lower quality)
- w=1-3: Balanced
- w=7-10: High quality but less diverse

### 2. Optimal Transport

#### What is Optimal Transport and Why It Matters?

**Definition**: Optimal Transport (OT) theory studies the most efficient way to transform one probability distribution into another, minimizing a cost function. It's the mathematical framework for "moving dirt" problems.

**Modern ML Applications:**
- **Wasserstein GANs**: More stable GAN training
- **Domain Adaptation**: Transfer learning across domains
- **Style Transfer**: Color/texture matching
- **Generative Models**: Flow matching, diffusion connections
- **Fair ML**: Debiasing algorithms

**When to Use OT:**
- Comparing distributions with different supports
- Need geometrically meaningful distance
- Interpolation between distributions
- When KL divergence fails (disjoint supports)

#### 2.1 The Monge Problem (1781)

**Original Formulation - Moving Dirt:**
```
min_{T:X→Y} ∫ c(x, T(x)) dμ(x)
```
"Move pile of dirt μ to hole ν with minimal effort"

**Constraints:**
- T must be deterministic map
- Mass conservation: T#μ = ν

**Problems:**
- May not exist (discrete → continuous)
- Non-convex optimization
- Too restrictive for many applications

#### 2.2 Kantorovich Relaxation (1942)

**Allow Splitting Mass:**
```
W_p(μ,ν) = (inf_{π∈Π(μ,ν)} ∫∫ ||x-y||^p dπ(x,y))^{1/p}
```

**Key Innovation**: Joint distribution π instead of map T
- Can split mass (one source → multiple targets)
- Always exists
- Convex optimization problem

**Discrete Version (Linear Program):**
```
min_{P} ∑_{i,j} C_{ij} P_{ij}
s.t. ∑_j P_{ij} = μ_i  (row sums)
     ∑_i P_{ij} = ν_j  (column sums)
     P_{ij} ≥ 0
```

#### 2.3 Why Wasserstein > KL Divergence

**Example - Two Gaussians:**
```
μ = N(0, 0.01), ν = N(10, 0.01)
```

**KL Divergence:**
- KL(μ||ν) ≈ ∞ (no support overlap)
- Gradient ≈ 0 (no learning signal)

**Wasserstein Distance:**
- W_2(μ,ν) ≈ 10 (distance between means)
- Smooth gradients pointing toward target

**For Neural Networks:**
- KL: Generated and real distributions often disjoint early in training
- Wasserstein: Always provides meaningful gradients

#### 2.4 Computational Methods

**Sinkhorn Algorithm (Entropic OT):**
```python
def sinkhorn(C, μ, ν, ε=0.1, iters=100):
    K = torch.exp(-C/ε)  # Gibbs kernel
    u = torch.ones_like(μ)
    
    for _ in range(iters):
        v = ν / (K.T @ u)
        u = μ / (K @ v)
    
    return u.reshape(-1,1) * K * v.reshape(1,-1)
```
- Adds entropy regularization
- Makes problem strongly convex
- O(n²) but parallelizable

**Sliced Wasserstein (For High Dimensions):**
```
SW(μ,ν) = E_θ[W_1(P_θ#μ, P_θ#ν)]
```
- Project to random 1D directions
- Compute 1D Wasserstein (O(n log n))
- Average over projections

#### 2.5 Connection to Modern Deep Learning

**Wasserstein GAN:**
```python
# Critic approximates Wasserstein distance
critic_loss = torch.mean(critic(fake)) - torch.mean(critic(real))
# Enforce Lipschitz constraint
gradient_penalty = compute_gradient_penalty(critic, real, fake)
```

**Flow Matching (OT-based Generation):**
```
# Learn optimal transport map directly
v_θ(x,t) ≈ ∇Ψ_t(x)  # Ψ is OT potential
```

**Diffusion Models Connection:**
- DDIM trajectories are optimal transport paths
- Score matching relates to W_2 gradient flow

---

## Part C - Julie Digne: 3D and Geometric Deep Learning

### 1. Neural Radiance Fields (NeRF)

#### What is NeRF and Its Revolutionary Impact?

**Definition**: NeRF represents 3D scenes as continuous volumetric radiance fields learned by neural networks, enabling photorealistic novel view synthesis from a set of 2D images.

**Revolutionary Impact (2020):**
- **View Synthesis**: Generate
