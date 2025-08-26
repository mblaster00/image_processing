# Mock - Deep Learning & Image Processing

---

## Part A - Alexandre Meyer (40 points)

### Question 1: Style Transfer Using CNNs (10 points)

**Q: Explain how the Gatys et al. method separates content from style in an image. Why does VGG network work for this task? What would happen if we used only the first layer or only the last layer for style extraction?**

**Model Answer:**

The Gatys et al. method leverages the hierarchical nature of CNN representations to separate content from style. **Content representation** is extracted from higher convolutional layers (typically conv4_2 or conv5_2) because these layers capture high-level semantic information about objects and their spatial arrangement while being relatively invariant to exact pixel values and textures.

**Style representation** uses Gram matrices computed across multiple layers. The Gram matrix G^l_ij = Σ_k F^l_ik × F^l_jk captures the correlations between different feature maps at layer l. This correlation structure encodes texture patterns independent of spatial location - for instance, if vertical edges often co-occur with blue colors in Van Gogh's style, this correlation will be high in the Gram matrix.

**VGG works well because:**
- It was trained on ImageNet for object recognition, so it has learned rich, hierarchical features from edges to complex objects
- Its architecture progressively increases receptive field size, capturing patterns at multiple scales
- The features are general enough to transfer to artistic style analysis

**Using only first layer:** Would capture only low-level textures (edges, colors, simple patterns) missing the complex artistic patterns that emerge from feature interactions at multiple scales. The style would be too simplistic.

**Using only last layer:** Would capture very high-level, semantic patterns but miss the fine-grained texture details that define artistic style. The result would preserve some mood but lack the characteristic brushstrokes and texture patterns of the style image.

The multi-scale approach (using layers conv1_1, conv2_1, conv3_1, conv4_1, conv5_1) is crucial for capturing style at all levels of abstraction.

---

### Question 2: YOLO Architecture (10 points)

**Q: YOLO divides an image into an S×S grid. Explain what each grid cell predicts and how YOLO achieves real-time performance compared to R-CNN based methods. What are the main limitations of this approach?**

**Model Answer:**

**Each grid cell predicts:**
- **B bounding boxes**, where each box contains:
  - 4 coordinates: (x, y, w, h) - center coordinates relative to grid cell, width and height relative to image
  - 1 confidence score: P(Object) × IoU(pred, truth) - probability that box contains object multiplied by how well it fits
- **C class probabilities**: P(Class_i | Object) - conditional probability for each class given an object exists
- **Total output per cell**: B × 5 + C values
- **Full output tensor**: S × S × (B × 5 + C)

**Real-time performance is achieved through:**

1. **Single forward pass**: Unlike R-CNN which runs CNN on ~2000 region proposals, YOLO processes the entire image once through a single network

2. **Unified architecture**: Detection is framed as regression problem to spatially separated bounding boxes and class probabilities, eliminating complex pipelines

3. **Global context**: Each prediction uses features from the entire image, not just local regions, providing better context for detection

4. **Grid-based design**: Spatial subdivision naturally handles multiple objects without needing separate region proposal network

**Main limitations:**

1. **Spatial constraints**: Each grid cell can only detect one object class, struggling with small objects that appear in groups (like flocks of birds)

2. **Localization errors**: More localization errors compared to Fast R-CNN, especially for small objects

3. **Generalization issues**: Struggles with objects in new or unusual aspect ratios or configurations not seen during training

4. **Coarse features**: Using relatively coarse features from final layer makes it harder to detect small objects

5. **Fixed grid**: The S×S grid is fixed, which can be problematic for images with different object distributions

---

### Question 3: Everybody Dance Now (10 points)

**Q: Describe the complete pipeline of the "Everybody Dance Now" method. How does it handle temporal coherence, and why is a specialized face GAN necessary?**

**Model Answer:**

**Complete Pipeline:**

1. **Pose Extraction Stage:**
   - Extract 2D pose keypoints from source video using OpenPose
   - Create pose stick figures (skeleton representations) for each frame
   - These serve as intermediate representation that's person-agnostic

2. **Training Stage (for target person):**
   - Record target person performing various movements
   - Extract their pose stick figures
   - Train conditional GAN (pix2pixHD) to map: pose stick figures → target person appearance
   - Train separate face GAN for facial details
   - Train temporal coherence module

3. **Transfer Stage:**
   - Take pose stick figures from source dancer
   - Feed through trained GAN to generate target person in same poses
   - Apply face GAN for realistic facial features
   - Apply temporal smoothing

**Temporal Coherence Handling:**

The method addresses temporal coherence through two mechanisms:

1. **Temporal discriminator**: Added to distinguish between temporally coherent and incoherent video sequences. It looks at consecutive frames to ensure smooth transitions.

2. **Flow-based warping**: Uses optical flow to warp previous frame's prediction to current frame, providing initialization that maintains consistency. The loss includes: L_temporal = ||G(p_t) - W(G(p_{t-1}), F_{t-1→t})||

**Why Specialized Face GAN is Necessary:**

1. **Resolution mismatch**: The global person GAN operates at 512×256 resolution, insufficient for facial details. Faces typically occupy <10% of image area, resulting in blurry, unrealistic faces.

2. **Identity preservation**: Facial features are crucial for identity. A specialized 128×128 face GAN trained on cropped face regions preserves identity-specific details like eye shape, nose structure.

3. **Expression complexity**: Facial expressions involve subtle muscle movements that the global GAN cannot capture. The face GAN is trained specifically on facial dynamics.

4. **Perceptual importance**: Humans are extremely sensitive to facial abnormalities (uncanny valley effect). Even minor artifacts in faces are immediately noticeable, while body artifacts are more tolerable.

The face GAN predictions are blended back using Poisson blending to ensure seamless integration with the body.

---

### Question 4: Autoencoders and GANs (10 points)

**Q: Compare Variational Autoencoders (VAE) and GANs for image generation. Include their loss functions, training stability, and quality of generated samples. When would you choose one over the other?**

**Model Answer:**

**Loss Functions:**

**VAE Loss:**
```
L_VAE = -E[log p(x|z)] + KL(q(z|x) || p(z))
      = Reconstruction Loss + KL Regularization
```
- Reconstruction term ensures decoded samples match input
- KL term regularizes latent space to match prior (typically N(0,I))
- Provides explicit likelihood model

**GAN Loss:**
```
L_D = -E[log D(x)] - E[log(1-D(G(z)))]  (Discriminator)
L_G = -E[log D(G(z))]                    (Generator)
```
- Adversarial game between generator and discriminator
- No explicit reconstruction, only "realness" measure
- Implicit likelihood through discriminator

**Training Stability:**

**VAE:**
- ✅ Stable training due to explicit objective
- ✅ Guaranteed convergence with proper optimization
- ✅ No mode collapse issues
- ❌ Can get stuck in local minima producing blurry results

**GAN:**
- ❌ Unstable training - careful balancing of G and D required
- ❌ Mode collapse - generator may produce limited variety
- ❌ Vanishing gradients when discriminator is too good
- ❌ Requires tricks (spectral normalization, progressive growing)

**Quality of Generated Samples:**

**VAE:**
- Tends to produce blurrier samples due to pixel-wise reconstruction loss
- Better coverage of data distribution
- Smooth, meaningful latent space interpolations
- Lower perceptual quality but higher diversity

**GAN:**
- Sharp, high-quality samples with fine details
- May miss modes of data distribution
- Less meaningful latent space structure
- Higher perceptual quality but potentially lower diversity

**When to Choose:**

**Choose VAE when:**
- Need explicit likelihood estimation
- Require meaningful latent representations for downstream tasks
- Want stable, predictable training
- Need to ensure full distribution coverage
- Working with limited computational resources

**Choose GAN when:**
- Perceptual quality is paramount
- Generating high-resolution, photorealistic images
- Don't need explicit density modeling
- Have computational resources for careful hyperparameter tuning
- Can tolerate some mode dropping

**Hybrid approaches** like VAE-GAN combine benefits: VAE's stable training and meaningful latents with GAN's perceptual quality.

---

## Part B - Nicolas Bonneel (30 points)

### Question 5: Diffusion Models (15 points)

**Q: Explain the forward and reverse processes in diffusion models. How does the training objective relate to denoising, and why is this formulation more stable than directly predicting x_0?**

**Model Answer:**

**Forward Process (Diffusion):**

The forward process gradually adds Gaussian noise over T timesteps:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```

Using the reparameterization, we can sample x_t directly:
```
x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
```
where ᾱ_t = Π(1-β_i) and ε ~ N(0, I)

This process is fixed and requires no training - it systematically destroys structure until x_T ~ N(0, I).

**Reverse Process (Generation):**

The reverse process learns to denoise, modeled as:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

The key insight is that for small β_t, the reverse process is also approximately Gaussian, making this parametrization reasonable.

**Training Objective - Denoising Connection:**

The variational lower bound can be simplified to:
```
L_simple = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||²]
```

This means we train the network ε_θ to predict the noise added at each step, not the clean image. This is equivalent to training a **denoising autoencoder at multiple noise levels simultaneously**.

**Why Predicting Noise is More Stable than Predicting x_0:**

1. **Bounded target range**: Noise ε ~ N(0, I) has consistent statistics regardless of input, while x_0 varies greatly across the dataset. The network learns a more uniform mapping.

2. **Gradient flow**: When predicting x_0 directly:
   - Early timesteps (high noise): x_0 prediction is extremely difficult, gradients are noisy
   - Late timesteps (low noise): x_0 ≈ x_t, network learns identity mapping
   
   When predicting noise:
   - The task difficulty is more uniform across timesteps
   - Network always learns meaningful denoising operations

3. **Mathematical equivalence with advantages**: 
   The predicted x_0 can be recovered: x̂_0 = (x_t - √(1-ᾱ_t)ε_θ)/√(ᾱ_t)
   But the noise parametrization has better conditioning

4. **Connection to score matching**: Predicting noise is equivalent to predicting the score function ∇_x log p(x), which has nice theoretical properties and connects to Langevin dynamics.

5. **Empirical performance**: The noise prediction formulation leads to:
   - More stable training curves
   - Better sample quality
   - Faster convergence
   - Less sensitivity to hyperparameters

This formulation transforms an ill-posed generation problem into a well-posed denoising problem at multiple scales.

---

### Question 6: Optimal Transport (15 points)

**Q: Explain the Wasserstein distance and how it differs from KL divergence. Provide a concrete example where Wasserstein distance is more appropriate. How does optimal transport relate to the training of modern generative models?**

**Model Answer:**

**Wasserstein Distance Definition:**

The Wasserstein-p distance (Earth Mover's Distance for p=1) between distributions P and Q:
```
W_p(P,Q) = (inf_{γ∈Γ(P,Q)} E_{(x,y)~γ}[||x-y||^p])^{1/p}
```

where Γ(P,Q) is the set of all joint distributions with marginals P and Q. Intuitively, it measures the minimum "cost" of transporting mass from P to Q.

**Key Differences from KL Divergence:**

**KL Divergence**: KL(P||Q) = E_P[log(P/Q)]
- Not symmetric: KL(P||Q) ≠ KL(Q||P)
- Undefined when supports don't overlap (Q(x)=0 but P(x)>0)
- Doesn't consider metric structure of space
- Can be infinite for disjoint distributions

**Wasserstein Distance**:
- Considers geometric structure (distance between points matters)
- Always finite for distributions with finite moments
- Symmetric with appropriate ground metric
- Provides meaningful gradient even for disjoint distributions

**Concrete Example:**

Consider two Gaussian distributions in 1D:
- P = N(0, 0.01) (narrow Gaussian at origin)
- Q = N(10, 0.01) (narrow Gaussian at x=10)

**KL Divergence**: Nearly infinite because supports barely overlap. Gradient would be essentially zero for intermediate positions - no learning signal.

**Wasserstein Distance**: W_1(P,Q) ≈ 10 (the distance between means). Provides smooth gradient pointing from one distribution toward the other. If we move Q gradually toward P, Wasserstein distance decreases linearly.

This is critical in early GAN training when generator and real distributions may be disjoint.

**Relation to Modern Generative Models:**

1. **Wasserstein GANs (WGAN)**:
   - Replace JS divergence with Wasserstein distance
   - Discriminator becomes a "critic" computing Wasserstein distance
   - Solves mode collapse and vanishing gradients
   - Requires Lipschitz constraint (weight clipping or gradient penalty)

2. **Diffusion Models Connection**:
   - DDIM sampling can be viewed as optimal transport
   - The probability flow ODE follows the optimal transport path
   - Score-based models minimize Wasserstein gradient flow

3. **Flow Matching**:
   - Directly learns optimal transport maps between noise and data
   - Straighter trajectories than diffusion models
   - Better sample efficiency

4. **Style Transfer and Domain Adaptation**:
   - Optimal transport for color matching preserves geometry
   - Sliced Wasserstein distance for efficient high-dimensional transport
   - Preserves neighborhood relationships unlike moment matching

5. **Training Advantages**:
   - Provides meaningful gradients even when distributions don't overlap
   - More stable training dynamics
   - Better mode coverage
   - Principled interpolation in latent spaces

The key insight: Wasserstein distance respects the **geometry** of the data space, making it ideal for high-dimensional generation tasks where traditional divergences fail.

---

## Part C - Julie Digne (30 points)

### Question 7: Neural Radiance Fields (NeRF) (10 points)

**Q: Explain the principle of NeRF including the volume rendering equation. Why is positional encoding crucial, and what problems would occur without it?**

**Model Answer:**

**NeRF Principle:**

NeRF represents a 3D scene as a continuous volumetric function learned by a neural network:
```
F_θ: (x, y, z, θ, φ) → (r, g, b, σ)
```
- Input: 3D position (x,y,z) and viewing direction (θ,φ)
- Output: RGB color (r,g,b) and volume density σ

The network learns this mapping from a set of 2D images with known camera poses, effectively encoding the entire 3D scene in the network weights.

**Volume Rendering Equation:**

To render a pixel, we cast a ray r(t) = o + td and integrate color along it:

```
C(r) = ∫[t_n to t_f] T(t) · σ(r(t)) · c(r(t), d) dt
```

where transmittance: `T(t) = exp(-∫[t_n to t] σ(r(s)) ds)`

In practice, this is discretized using quadrature:
```
Ĉ(r) = Σ_i T_i · (1 - exp(-σ_i · δ_i)) · c_i
```
where:
- T_i = exp(-Σ_{j<i} σ_j · δ_j) is accumulated transmittance
- δ_i = t_{i+1} - t_i is distance between samples
- (1 - exp(-σ_i · δ_i)) is alpha value (opacity)

**Why Positional Encoding is Crucial:**

Neural networks have an inherent **spectral bias** toward low-frequency functions. The positional encoding maps coordinates to higher-dimensional space:

```
γ(x) = [sin(2^0πx), cos(2^0πx), ..., sin(2^{L-1}πx), cos(2^{L-1}πx)]
```

**Problems without Positional Encoding:**

1. **Over-smoothing**: Network cannot represent high-frequency details like sharp edges, textures, or fine geometric features. Results appear blurry and lack detail.

2. **Poor convergence**: Network struggles to fit training data because it cannot represent the rapid spatial variations in real scenes.

3. **Limited expressiveness**: The network acts as a low-pass filter, fundamentally limited in the frequencies it can represent. Mathematical analysis shows vanilla networks correspond to kernels with rapid frequency falloff.

4. **Inability to overfit**: Even given infinite capacity and training time, the network cannot perfectly reproduce training views due to frequency limitations.

**Example**: A checkerboard pattern requires high frequencies to represent sharp transitions between black and white. Without positional encoding, the network would produce a blurry gray gradient instead of distinct squares.

The positional encoding essentially lifts the input to a higher-dimensional space where the network can learn linear combinations of sinusoids at multiple frequencies, enabling representation of both smooth regions and sharp details.

---

### Question 8: 3D Gaussian Splatting (10 points)

**Q: Compare 3D Gaussian Splatting with NeRF in terms of representation, rendering speed, and training. What are the key innovations that enable real-time rendering?**

**Model Answer:**

**Representation Comparison:**

**NeRF (Implicit)**:
- Continuous function: MLP network F_θ(x,y,z,θ,φ) → (rgb, σ)
- Scene encoded in network weights (~5MB)
- Infinite resolution in theory
- Requires network evaluation for any query point

**3D Gaussian Splatting (Explicit)**:
- Millions of 3D Gaussians (primitives)
- Each Gaussian has: position μ, covariance Σ, opacity α, color c (spherical harmonics)
- Scene encoded as primitive parameters (~30-300MB)
- Fixed resolution determined by Gaussian count

**Rendering Speed:**

**NeRF**:
- ~5-30 seconds per frame (original)
- Requires 100-300 network evaluations per ray
- ~100K-1M network evaluations per image
- Bottlenecked by sequential MLP evaluations

**3D Gaussian Splatting**:
- 30-140 FPS (real-time)
- Direct rasterization on GPU
- Parallel processing of all Gaussians
- 1000× faster than vanilla NeRF

**Training Comparison:**

**NeRF**:
- 1-2 days training on single GPU
- Photometric loss on randomly sampled rays
- Requires careful sampling strategies
- Slow convergence due to indirect supervision

**Gaussian Splatting**:
- 30-60 minutes training
- Photometric loss on full images
- Adaptive densification and pruning
- Direct gradient flow to primitive parameters

**Key Innovations Enabling Real-Time Rendering:**

1. **Differentiable Rasterization**:
   - Project 3D Gaussians to 2D screen space: Σ' = JΣJ^T (J is Jacobian)
   - Sort by depth and α-blend
   - Fully differentiable for end-to-end optimization
   - Leverages GPU rasterization pipeline

2. **Adaptive Density Control**:
   - Start with sparse SfM points
   - Clone Gaussians in under-reconstructed areas (high gradient)
   - Split large Gaussians exceeding threshold
   - Prune low-opacity Gaussians
   - Automatically adjusts primitive count to scene complexity

3. **Spherical Harmonics for View-Dependence**:
   - Color represented as SH coefficients instead of RGB
   - Efficiently encodes view-dependent effects
   - 4th order SH (48 coefficients) captures specularities
   - Avoids network evaluation for appearance

4. **Anisotropic Gaussians**:
   - Full 3D covariance (rotation + non-uniform scale)
   - Better scene representation with fewer primitives
   - Can model thin structures (surfaces) efficiently
   - Reduces primitive count by 10-100×

5. **Tile-Based Rendering**:
   - Divide screen into tiles (16×16 pixels)
   - Cull Gaussians per tile
   - Process tiles in parallel on GPU
   - Minimize memory bandwidth

**Trade-offs**:
- Gaussian Splatting sacrifices memory efficiency (30-300MB vs 5MB) for massive speed gains
- Less compact representation but orders of magnitude faster
- Better for real-time applications, interactive editing
- NeRF better for storage-constrained scenarios

---

### Question 9: Deep Geometric Learning (10 points)

**Q: PointNet solves the problem of learning on point clouds. Explain the fundamental challenges of point cloud processing and how PointNet addresses them, especially permutation invariance. What are its limitations?**

**Model Answer:**

**Fundamental Challenges of Point Cloud Processing:**

1. **Unordered structure**: Unlike images with regular grids, point clouds are sets with no canonical ordering. The same object can be represented by N! different orderings.

2. **Permutation invariance requirement**: F(x₁,...,xₙ) must equal F(xπ(1),...,xπ(n)) for any permutation π.

3. **Varying point count**: Different point clouds have different numbers of points, unlike fixed-size images.

4. **No local connectivity**: No explicit neighborhood structure like mesh edges or image pixels.

5. **Irregular sampling**: Point density varies across the surface, dense in some areas, sparse in others.

**How PointNet Addresses These Challenges:**

**Core Architecture:**
```
Input Points → Shared MLP → MaxPool → Global Feature → Task-specific layers
x_i → h_i = MLP(x_i) → g = max{h_1,...,h_n} → output
```

**Key Solutions:**

1. **Symmetric Function for Permutation Invariance**:
   - Uses max pooling: max{h₁,...,hₙ} is invariant to input order
   - Theoretical foundation: Any permutation-invariant function can be approximated as: f(X) = γ(max{h(x_i)})
   - Max pooling aggregates features while preserving invariance

2. **Shared MLP (Point-wise Processing)**:
   - Same MLP applied independently to each point
   - Learns per-point features before aggregation
   - Handles varying point counts naturally

3. **T-Net (Transformation Networks)**:
   - Mini-PointNets that predict 3×3 and 64×64 transformation matrices
   - Input transform: Aligns point cloud to canonical pose
   - Feature transform: Aligns features for better invariance
   - Regularization: ||I - AA^T||² ensures orthogonality

4. **Global + Local Features**:
   - Concatenate global features (from max pool) with point features
   - Enables both classification (global) and segmentation (local) tasks

**Limitations of PointNet:**

1. **No Local Context**: 
   - Processes each point independently before aggregation
   - Cannot capture local geometric patterns or neighborhoods
   - Misses fine-grained local structures

2. **Limited by Global Aggregation**:
   - Max pooling selects only most salient features
   - Information bottleneck at global feature
   - Cannot model hierarchical structures

3. **Sensitivity to Outliers**:
   - Max pooling can be dominated by outlier points
   - No robustness to noise or sampling artifacts

4. **No Multi-scale Features**:
   - Single-scale processing
   - Cannot capture patterns at different granularities
   - PointNet++ addresses this with hierarchical grouping

5. **Computational Inefficiency for Large Clouds**:
   - Processes all points regardless of importance
   - No downsampling or attention mechanisms
   - Later methods add sampling and grouping

**Solutions in PointNet++:**
- Hierarchical point set learning
- Multi-scale grouping
- Farthest point sampling for coverage
- Local neighborhood aggregation with ball queries

**Example limitation**: Given a point cloud of a building, PointNet might recognize it's a building (global shape) but miss architectural details like window patterns or decorative elements that require understanding local point relationships.

---

## Bonus Question (10 points)

### Question 10: Cross-Topic Integration

**Q: Propose a method that combines diffusion models with 3D Gaussian Splatting for text-to-3D generation. What would be the main challenges and how would you address them?**

**Model Answer:**

**Proposed Method: DiffusionSplat**

**Architecture Overview:**

1. **Text Encoding**: CLIP/T5 encoder → text embeddings
2. **3D Diffusion Backbone**: Operates on Gaussian parameters instead of pixels
3. **Rendering Bridge**: Differentiable splatting for 2D supervision
4. **Multi-view Consistency**: Cross-view attention mechanisms

**Key Components:**

**1. Gaussian Parameter Diffusion:**
Instead of diffusing pixels, diffuse Gaussian parameters:
```
G_t = √(ᾱ_t)G_0 + √(1-ᾱ_t)ε
```
where G = {μ, Σ, α, c} represents Gaussian parameters.

**2. Hybrid Training Objective:**
```
L = λ_diff L_diffusion + λ_render L_render + λ_reg L_regularization
```
- L_diffusion: Standard diffusion loss on Gaussian parameters
- L_render: 2D reconstruction loss after splatting
- L_regularization: Prevent degenerate Gaussians

**Main Challenges and Solutions:**

**Challenge 1: Irregular Structure**
- Unlike images (regular grid) or point clouds (unordered set), Gaussians have varying dimensions (position, scale, rotation, appearance)
- **Solution**: Normalize parameters to common scale, use separate diffusion processes for geometric and appearance attributes

**Challenge 2: Multi-view Consistency**
- Diffusion might generate Gaussians that look good from one view but inconsistent from others
- **Solution**: 
  - Score Distillation Sampling from pretrained 2D diffusion model
  - Render multiple views during training, enforce consistency loss
  - Cross-view attention in denoising network

**Challenge 3: Text-to-3D Alignment**
- No large-scale text-3D datasets like LAION for 2D
- **Solution**:
  - Bootstrap from 2D diffusion models using SDS loss
  - Use CLIP-based similarity in 3D space
  - Progressive training: 2D → 2.5D → full 3D

**Challenge 4: Computational Efficiency**
- Millions of Gaussians × thousands of diffusion steps = massive computation
- **Solution**:
  - Hierarchical approach: Generate coarse Gaussians first, refine progressively
  - Latent diffusion: Operate in compressed Gaussian space
  - Adaptive sampling: Focus computation on visible Gaussians

**Challenge 5: Mode Collapse**
- Model might generate similar 3D structures for different prompts
- **Solution**:
  - Classifier-free guidance with varying scales
  - Diverse initialization strategies
  - Condition on multiple text encoders (CLIP + T5)

**Implementation Pipeline:**

1. **Stage 1**: Train VAE to encode/decode Gaussian scenes to latent space
2. **Stage 2**: Train diffusion model in latent space conditioned on text
3. **Stage 3**: Fine-tune with rendering losses for quality

**Advantages of This Approach:**
- Real-time rendering once generated (inherit Gaussian Splatting speed)
- Explicit 3D representation (can edit, compose)
- Leverages both 2D and 3D supervision
- Controllable generation through text conditioning

**Expected Results:**
- Generation time: 30-60 seconds (vs hours for NeRF-based)
- Rendering: Real-time (100+ FPS)
- Quality: Comparable to DreamFusion but 100× faster inference

This integration leverages diffusion models' generation quality with Gaussian Splatting's efficiency, potentially revolutionizing text-to-3D generation.

---

## Grading Rubric

- **Conceptual Understanding** (40%): Clear explanation of principles
- **Mathematical Formulation** (20%): Correct equations and notation
- **Critical Analysis** (20%): Discussion of limitations and trade-offs
- **Practical Application** (20%): Real-world examples and use cases

**Note**: Focus on explaining WHY methods work, not just describing WHAT they do. Mathematical rigor is appreciated but intuitive understanding is essential.
