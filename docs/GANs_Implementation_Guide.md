# Generative Adversarial Networks (GANs) Implementation Guide

This document provides a comprehensive overview of the GAN implementations available in AiDotNet, addressing [Issue #416](https://github.com/ooples/AiDotNet/issues/416).

## Table of Contents
- [Overview](#overview)
- [Core GAN Architectures](#core-gan-architectures)
- [Conditional GANs](#conditional-gans)
- [Training Techniques and Layers](#training-techniques-and-layers)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [References](#references)

## Overview

Generative Adversarial Networks (GANs) are a class of machine learning frameworks where two neural networks contest with each other in a zero-sum game framework. This implementation provides multiple GAN architectures ranging from the foundational vanilla GAN to advanced variants.

### What's New in This Release

This implementation adds the following GAN architectures and components to AiDotNet:

#### Core GANs (CRITICAL) ✅
- **DCGAN** (Deep Convolutional GAN)
- **WGAN** (Wasserstein GAN)
- **WGAN-GP** (Wasserstein GAN with Gradient Penalty)
- **Vanilla GAN** (already existed, enhanced)

#### Conditional GANs (HIGH) ✅
- **cGAN** (Conditional GAN)
- **AC-GAN** (Auxiliary Classifier GAN)
- **InfoGAN** (Information Maximizing GAN)

#### Advanced GANs (HIGH) ✅
- **Pix2Pix** (Paired Image-to-Image Translation)
- **CycleGAN** (Unpaired Image-to-Image Translation)
- **StyleGAN** (Style-Based Generator Architecture)
- **Progressive GAN** (Progressively Growing GAN)
- **BigGAN** (Large-Scale GAN Training)
- **SAGAN** (Self-Attention GAN)

#### Training Techniques (HIGH) ✅
- **Spectral Normalization Layer**
- **Self-Attention Layer** (for SAGAN)

#### Evaluation Metrics (HIGH) ✅
- **FID** (Fréchet Inception Distance)
- **IS** (Inception Score)

## Core GAN Architectures

### 1. Vanilla GAN (GenerativeAdversarialNetwork)

**Location**: `src/NeuralNetworks/GenerativeAdversarialNetwork.cs`

The original GAN implementation with a generator and discriminator competing against each other.

**Key Features**:
- Binary cross-entropy loss
- Adam optimizer with momentum
- Adaptive learning rate
- Mode collapse detection

**When to Use**:
- Learning GAN basics
- Simple image generation tasks
- Baseline comparisons

### 2. DCGAN (Deep Convolutional GAN)

**Location**: `src/NeuralNetworks/DCGAN.cs`

An architecture that uses convolutional and transposed convolutional layers with specific design guidelines for stable training.

**Key Features**:
- Strided convolutions instead of pooling
- Batch normalization in both generator and discriminator
- ReLU activation in generator (Tanh for output)
- LeakyReLU activation in discriminator
- No fully connected hidden layers

**Architectural Guidelines**:
```csharp
var dcgan = new DCGAN<double>(
    latentSize: 100,
    imageChannels: 3,      // RGB images
    imageHeight: 64,
    imageWidth: 64,
    generatorFeatureMaps: 64,
    discriminatorFeatureMaps: 64,
    initialLearningRate: 0.0002
);
```

**When to Use**:
- Image generation tasks
- When you need stable training
- As a strong baseline for comparison
- Generation of 64x64 or 128x128 images

**Reference**: Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2015)

### 3. WGAN (Wasserstein GAN)

**Location**: `src/NeuralNetworks/WGAN.cs`

Uses Wasserstein distance (Earth Mover's distance) instead of Jensen-Shannon divergence for more stable training.

**Key Features**:
- Critic instead of discriminator (no sigmoid output)
- Wasserstein loss correlates with image quality
- Weight clipping to enforce Lipschitz constraint
- Can train critic multiple times per generator update
- RMSprop optimizer (recommended)

**Architectural Guidelines**:
```csharp
var wgan = new WGAN<double>(
    generatorArchitecture,
    criticArchitecture,
    InputType.Image,
    initialLearningRate: 0.00005,
    weightClipValue: 0.01,
    criticIterations: 5
);
```

**Advantages**:
- More stable training
- Loss value is meaningful (lower = better)
- Reduced mode collapse
- Can train critic more without instability

**When to Use**:
- When vanilla GAN training is unstable
- When you need meaningful loss metrics
- For research and experimentation

**Reference**: Arjovsky et al., "Wasserstein GAN" (2017)

### 4. WGAN-GP (Wasserstein GAN with Gradient Penalty)

**Location**: `src/NeuralNetworks/WGANGP.cs`

An improved version of WGAN that uses gradient penalty instead of weight clipping.

**Key Features**:
- Gradient penalty on interpolated samples
- Smoother and more stable than weight clipping
- Better performance and convergence
- Adam optimizer with beta1=0 (paper recommendation)
- No weight clipping artifacts

**Architectural Guidelines**:
```csharp
var wgangp = new WGANGP<double>(
    generatorArchitecture,
    criticArchitecture,
    InputType.Image,
    initialLearningRate: 0.0001,
    gradientPenaltyCoefficient: 10.0,
    criticIterations: 5
);
```

**Gradient Penalty**:
The gradient penalty is computed as:
```
GP = λ * E[(||∇_x D(x)||₂ - 1)²]
```
where x is sampled uniformly along straight lines between real and generated samples.

**Advantages Over WGAN**:
- No weight clipping (which can cause pathological behavior)
- More stable gradients
- Better image quality
- Easier to tune (fewer hyperparameters)

**When to Use**:
- Preferred over WGAN in most cases
- High-quality image generation
- When training stability is crucial
- State-of-the-art GAN baseline

**Reference**: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)

## Conditional GANs

### Conditional GAN (cGAN)

**Location**: `src/NeuralNetworks/ConditionalGAN.cs`

Generates data conditioned on additional information such as class labels or attributes.

**Key Features**:
- Both generator and discriminator receive conditioning information
- Controlled generation (e.g., "generate a specific digit")
- One-hot encoded class labels
- Flexible conditioning on any additional information

**Architectural Guidelines**:
```csharp
var cgan = new ConditionalGAN<double>(
    generatorArchitecture,
    discriminatorArchitecture,
    numConditionClasses: 10,  // e.g., 10 digits for MNIST
    InputType.Image,
    initialLearningRate: 0.0002
);

// Generate images of a specific class
var noise = cgan.GenerateRandomNoiseTensor(batchSize: 16, noiseSize: 100);
var conditions = cgan.CreateOneHotCondition(batchSize: 16, classIndex: 7);
var images = cgan.GenerateConditional(noise, conditions);
```

**Use Cases**:
- Class-conditional image generation (e.g., MNIST, CIFAR-10)
- Attribute-conditional generation (e.g., faces with glasses, smiling, etc.)
- Text-to-image generation
- Image-to-image translation with labels

**How It Works**:
1. **Generator**: Takes noise + condition → generates image
2. **Discriminator**: Takes image + condition → determines if pair is authentic
3. **Training**: Both networks learn to respect the conditioning information

**When to Use**:
- When you need control over what is generated
- Multi-class datasets
- When labels or attributes are available
- Interactive generation applications

**Reference**: Mirza and Osindero, "Conditional Generative Adversarial Nets" (2014)

### AC-GAN (Auxiliary Classifier GAN)

**Location**: `src/NeuralNetworks/ACGAN.cs`

An extension of conditional GANs where the discriminator also predicts the class label, providing stronger gradients and better image quality.

**Key Features**:
- Discriminator performs two tasks: authenticity + classification
- Stronger gradient signals for class-conditional generation
- Better image quality than basic cGAN
- Improved class separability

**Architectural Guidelines**:
```csharp
var acgan = new ACGAN<double>(
    generatorArchitecture,     // Takes noise + class label
    discriminatorArchitecture, // Outputs: [authenticity, class_probs...]
    numClasses: 10,
    InputType.Image,
    initialLearningRate: 0.0002
);

// Train with class labels
var (discLoss, genLoss) = acgan.TrainStep(realImages, realLabels, noise, fakeLabels);

// Generate specific class
var noise = acgan.GenerateRandomNoiseTensor(16, 100);
var labels = acgan.CreateOneHotLabels(16, classIndex: 7);
var images = acgan.GenerateConditional(noise, labels);
```

**Advantages Over cGAN**:
- Discriminator learns better features (multi-task)
- Higher quality class-conditional images
- Better class consistency
- More stable training

**When to Use**:
- When image quality is critical
- Multi-class conditional generation
- When you have labeled data
- Research requiring strong baselines

**Reference**: Odena et al., "Conditional Image Synthesis with Auxiliary Classifier GANs" (2017)

### InfoGAN (Information Maximizing GAN)

**Location**: `src/NeuralNetworks/InfoGAN.cs`

Learns disentangled representations in an unsupervised manner by maximizing mutual information between latent codes and generated observations.

**Key Features**:
- Automatic discovery of interpretable features
- No labeled data required
- Disentangled latent codes
- Auxiliary Q network for code prediction
- Mutual information maximization

**Architectural Guidelines**:
```csharp
var infogan = new InfoGAN<double>(
    generatorArchitecture,      // Takes noise z + latent codes c
    discriminatorArchitecture,
    qNetworkArchitecture,       // Predicts c from generated images
    latentCodeSize: 10,         // Number of codes to learn
    InputType.Image,
    initialLearningRate: 0.0002,
    mutualInfoCoefficient: 1.0
);

// Train
var noise = infogan.GenerateRandomNoiseTensor(batchSize, 100);
var codes = infogan.GenerateRandomLatentCodes(batchSize);
var (discLoss, genLoss, miLoss) = infogan.TrainStep(realImages, noise, codes);

// Generate with specific codes (e.g., control rotation, width, etc.)
var controlledCodes = new Tensor<double>(new int[] { 1, 10 });
controlledCodes[0, 0] = 0.8;  // First code = 0.8 (might control rotation)
var image = infogan.Generate(noise, controlledCodes);
```

**What It Learns** (Examples from MNIST):
- Code 1: Digit rotation
- Code 2: Stroke width
- Code 3: Digit style
- All discovered automatically!

**When to Use**:
- Discovering latent structure in data
- Controllable generation without labels
- Disentangled representation learning
- Feature manipulation tasks
- Research on interpretability

**Reference**: Chen et al., "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets" (2016)

## Advanced GANs for Image-to-Image Translation

### Pix2Pix

**Location**: `src/NeuralNetworks/Pix2Pix.cs`

A conditional GAN for paired image-to-image translation using U-Net generator and PatchGAN discriminator.

**Key Features**:
- Requires paired training data (input-output pairs)
- U-Net generator with skip connections
- PatchGAN discriminator (classifies patches)
- L1 + adversarial loss
- Preserves spatial information

**Architectural Guidelines**:
```csharp
var pix2pix = new Pix2Pix<double>(
    generatorArchitecture,      // U-Net architecture
    discriminatorArchitecture,  // PatchGAN
    InputType.Image,
    initialLearningRate: 0.0002,
    l1Lambda: 100.0             // Weight for L1 loss
);

// Train with paired data
var (discLoss, genLoss, l1Loss) = pix2pix.TrainStep(inputImages, targetImages);

// Translate images
var translated = pix2pix.Translate(inputImages);
```

**Applications**:
- Edges → Photos
- Sketches → Realistic images
- Day → Night scenes
- Semantic labels → Photos
- Black-and-white → Color
- Maps → Satellite imagery

**Key Insight**: PatchGAN focuses on local patches rather than the whole image, encouraging sharp high-frequency details.

**When to Use**:
- Paired training data available
- Image-to-image transformation tasks
- When spatial correspondence is important
- Applications requiring sharp details

**Reference**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (2017)

### CycleGAN

**Location**: `src/NeuralNetworks/CycleGAN.cs`

Enables unpaired image-to-image translation using cycle consistency loss, eliminating the need for paired training data.

**Key Features**:
- No paired training data required
- Two generators (A→B and B→A)
- Two discriminators (for domains A and B)
- Cycle consistency loss: A→B→A ≈ A
- Identity loss for color preservation

**Architectural Guidelines**:
```csharp
var cyclegan = new CycleGAN<double>(
    generatorAtoB,              // Translates A → B
    generatorBtoA,              // Translates B → A
    discriminatorA,
    discriminatorB,
    InputType.Image,
    initialLearningRate: 0.0002,
    cycleConsistencyLambda: 10.0,  // Cycle loss weight
    identityLambda: 5.0             // Identity loss weight
);

// Train with unpaired data
var (discLoss, genLoss, cycleLoss) = cyclegan.TrainStep(imagesA, imagesB);

// Translate between domains
var horsesToZebras = cyclegan.TranslateAtoB(horseImages);
var zebrasToHorses = cyclegan.TranslateBtoA(zebraImages);
```

**Applications**:
- Style transfer (Photo ↔ Monet, Photo ↔ Van Gogh)
- Season transfer (Summer ↔ Winter)
- Object transfiguration (Horse ↔ Zebra)
- Domain adaptation
- Photo enhancement

**Cycle Consistency**: The key innovation is enforcing that translating A→B→A should return to A, preventing mode collapse and maintaining content.

**When to Use**:
- Paired data NOT available
- Style transfer tasks
- Domain adaptation
- When you have two separate image collections
- Exploratory style experiments

**Reference**: Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (2017)

## Training Techniques and Layers

### Spectral Normalization Layer

**Location**: `src/NeuralNetworks/Layers/SpectralNormalizationLayer.cs`

A weight normalization technique that constrains the Lipschitz constant of neural network layers by dividing weights by their spectral norm (largest singular value).

**Key Features**:
- Stabilizes GAN training
- Prevents discriminator from becoming too powerful
- Computationally efficient (uses power iteration)
- Helps prevent mode collapse

**How to Use**:
```csharp
// Wrap any layer with spectral normalization
var convLayer = new ConvolutionalLayer<double>(...);
var normalizedLayer = new SpectralNormalizationLayer<double>(
    convLayer,
    powerIterations: 1  // Usually 1 is sufficient
);
```

**How It Works**:
1. Computes the largest singular value (spectral norm) of weight matrix
2. Divides all weights by this value
3. Uses power iteration for efficient computation
4. Updated during each forward pass

**Benefits**:
- More stable discriminator training
- Prevents extreme gradients
- Works well with other GAN techniques
- Can be applied to any layer type

**When to Use**:
- In discriminator/critic networks
- When training is unstable
- For high-quality image generation
- In combination with self-attention (SAGAN)

**Reference**: Miyato et al., "Spectral Normalization for Generative Adversarial Networks" (2018)

### Self-Attention Layer

**Location**: `src/NeuralNetworks/Layers/SelfAttentionLayer.cs`

Allows the model to attend to different spatial locations in feature maps, enabling better modeling of long-range dependencies.

**Key Features**:
- Captures global dependencies in images
- Query-Key-Value attention mechanism
- Learnable gamma parameter for gradual introduction
- Efficient channel reduction

**How It Works**:
1. Projects features into Query, Key, and Value representations
2. Computes attention map: softmax(Q^T @ K)
3. Applies attention to values
4. Residual connection with learnable gamma

**Benefits**:
- Better global coherence in generated images
- Improved generation of complex structures
- Multi-class object generation
- Complementary to convolution (local patterns)

**When to Use**:
- High-resolution image generation
- Complex scenes with multiple objects
- When local convolution is insufficient
- In combination with spectral normalization (SAGAN)

**Reference**: Zhang et al., "Self-Attention Generative Adversarial Networks" (2019)

## High-Resolution and Large-Scale GANs

### 10. StyleGAN

**Location**: `src/NeuralNetworks/StyleGAN.cs`

StyleGAN introduces a style-based generator architecture for high-quality image synthesis with unprecedented control over generated images.

**Key Features**:
- Mapping network (Z → W latent space)
- Style-based generator with AdaIN
- Style mixing for fine-grained control
- Progressive architecture
- High-quality image generation

**Architecture Components**:
```csharp
var stylegan = new StyleGAN<double>(
    latentSize: 512,
    imageChannels: 3,
    imageHeight: 1024,
    imageWidth: 1024,
    enableStyleMixing: true,
    styleMixingProbability: 0.9
);
```

**Innovations**:
1. **Mapping Network**: Transforms latent code Z into intermediate latent space W
2. **AdaIN (Adaptive Instance Normalization)**: Injects style at each layer
3. **Style Mixing**: Combines coarse and fine features from different sources
4. **Progressive Growth**: Can start from low resolution and increase

**When to Use**:
- High-resolution image generation (512×512, 1024×1024+)
- When you need fine-grained control over generation
- Face generation and synthesis
- Creative applications requiring style control
- State-of-the-art image quality required

**Reference**: Karras et al., "A Style-Based Generator Architecture for Generative Adversarial Networks" (2019)

### 11. Progressive GAN

**Location**: `src/NeuralNetworks/ProgressiveGAN.cs`

Progressive GAN grows the generator and discriminator progressively during training, starting from low resolution and gradually increasing to high resolution.

**Key Features**:
- Progressive growing architecture
- Smooth layer fade-in with alpha blending
- Minibatch standard deviation
- Pixel normalization in generator
- Equalized learning rate
- Supports very high resolutions (1024×1024)

**Architecture**:
```csharp
var progan = new ProgressiveGAN<double>(
    latentSize: 512,
    imageChannels: 3,
    maxResolutionLevel: 8,  // Up to 1024x1024
    baseFeatureMaps: 512
);

// Start training at 4x4
// Gradually grow to higher resolutions
while (progan.CurrentResolutionLevel < progan.MaxResolutionLevel)
{
    // Train at current resolution
    TrainForEpochs(progan, epochs: 1000);

    // Grow to next resolution
    progan.GrowNetworks();
    progan.Alpha = 0.0;  // Start fade-in

    // Gradually increase alpha from 0 to 1
    for (int i = 0; i < fadeInSteps; i++)
    {
        progan.Alpha = (double)i / fadeInSteps;
        TrainForEpochs(progan, epochs: 100);
    }
}
```

**When to Use**:
- Very high-resolution image generation (512×512+)
- When training stability at high resolution is important
- Limited GPU memory (start small, grow gradually)
- Face and texture generation
- Progressive training paradigm preferred

**Reference**: Karras et al., "Progressive Growing of GANs for Improved Quality, Stability, and Variation" (2018)

### 12. BigGAN

**Location**: `src/NeuralNetworks/BigGAN.cs`

BigGAN scales up GAN training using large batch sizes, increased model capacity, and class-conditional generation for state-of-the-art results.

**Key Features**:
- Large batch training (256-2048 samples)
- Spectral normalization in G and D
- Self-attention layers
- Class-conditional generation with projection discriminator
- Truncation trick for quality-diversity tradeoff
- Orthogonal initialization

**Architecture**:
```csharp
var biggan = new BigGAN<double>(
    latentSize: 120,
    numClasses: 1000,  // ImageNet
    classEmbeddingDim: 128,
    imageHeight: 128,
    imageWidth: 128,
    generatorChannels: 96,
    discriminatorChannels: 96
);

// Enable truncation for higher quality (lower diversity)
biggan.UseTruncation = true;
biggan.TruncationThreshold = 0.5;  // Lower = higher quality

// Generate images of specific class
var noise = GenerateNoise(batchSize);
var classIndices = new int[] { 281, 281, 281, 281 };  // Class 281: "cat"
var catImages = biggan.Generate(noise, classIndices);
```

**When to Use**:
- Large-scale conditional image generation
- High-fidelity natural image synthesis
- When you have large computational resources
- Class-conditional generation (ImageNet, etc.)
- State-of-the-art image quality on complex datasets

**Training Tips**:
- Use large batch sizes (128-512 minimum, 256-2048 optimal)
- Different learning rates for G and D (D typically 4× G)
- Spectral normalization in both G and D
- Monitor training with FID and IS metrics

**Reference**: Brock et al., "Large Scale GAN Training for High Fidelity Natural Image Synthesis" (2019)

### 13. SAGAN (Self-Attention GAN)

**Location**: `src/NeuralNetworks/SAGAN.cs`

SAGAN combines self-attention mechanisms with spectral normalization for modeling long-range dependencies in generated images.

**Key Features**:
- Self-attention layers at multiple resolutions
- Spectral normalization in both G and D
- Hinge loss for stable training
- TTUR (Two Time-Scale Update Rule)
- Both conditional and unconditional generation

**Architecture**:
```csharp
var sagan = new SAGAN<double>(
    latentSize: 128,
    imageChannels: 3,
    imageHeight: 64,
    imageWidth: 64,
    numClasses: 10,  // 0 for unconditional
    attentionLayers: new[] { 2, 3 },  // At 16x16 and 32x32
    useSpectralNormalization: true
);

// Unconditional generation
var images = sagan.Generate(numImages: 16);

// Conditional generation
var classIndices = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
var conditionalImages = sagan.Generate(10, classIndices);
```

**How Self-Attention Helps**:
- Captures long-range dependencies (e.g., symmetry between object parts)
- Better global structure coherence
- Improved multi-class image generation
- Complementary to convolutional layers

**When to Use**:
- When generated images need global coherence
- Complex scenes with multiple objects
- Geometric patterns requiring long-range consistency
- Class-conditional generation with diverse categories

**Training Configuration**:
- Generator LR: 1e-4
- Discriminator LR: 4e-4 (TTUR)
- Adam with β1=0, β2=0.9
- Hinge loss for both G and D
- Spectral normalization in all layers

**Reference**: Zhang et al., "Self-Attention Generative Adversarial Networks" (2019)

## Evaluation Metrics

### Fréchet Inception Distance (FID)

**Location**: `src/Metrics/FrechetInceptionDistance.cs`

FID measures the distance between the distributions of real and generated images using features from a pre-trained Inception network.

**Key Features**:
- Objective quality metric
- Captures both quality and diversity
- Correlates well with human judgment
- Industry standard for GAN evaluation

**Usage**:
```csharp
using AiDotNet.Metrics;

// Create FID calculator (optionally provide pre-trained Inception network)
var fid = new FrechetInceptionDistance<double>(
    inceptionNetwork: pretrainedInception,
    featureDimension: 2048
);

// Compute FID between real and generated images
var realImages = LoadRealImages();
var generatedImages = gan.Generate(numImages: 10000);
var fidScore = fid.ComputeFID(realImages, generatedImages);

Console.WriteLine($"FID Score: {fidScore:F2}");
// Lower is better: <10 excellent, 10-20 good, 20-50 moderate, >50 poor
```

**How It Works**:
1. Extract features from real images using Inception network
2. Extract features from generated images
3. Compute mean and covariance for both distributions
4. Calculate Fréchet distance: ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))

**When to Use**:
- Primary metric for GAN evaluation
- Comparing different GAN architectures
- Monitoring training progress
- Research and publication

**Typical Scores**:
- **< 10**: Excellent quality (state-of-the-art)
- **10-20**: Good quality
- **20-50**: Moderate quality
- **> 50**: Poor quality

**Reference**: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (2017)

### Inception Score (IS)

**Location**: `src/Metrics/InceptionScore.cs`

Inception Score evaluates both the quality (confident predictions) and diversity (variety of classes) of generated images.

**Key Features**:
- Measures quality and diversity simultaneously
- Uses KL divergence between conditional and marginal distributions
- Provides mean and standard deviation across splits
- Widely used in GAN research

**Usage**:
```csharp
using AiDotNet.Metrics;

// Create IS calculator
var is = new InceptionScore<double>(
    inceptionNetwork: pretrainedInception,
    numClasses: 1000,  // ImageNet
    numSplits: 10
);

// Compute IS with uncertainty
var generatedImages = gan.Generate(numImages: 50000);
var (mean, std) = is.ComputeISWithUncertainty(generatedImages);

Console.WriteLine($"Inception Score: {mean:F2} ± {std:F2}");
// Higher is better: >10 excellent, 5-10 good, 2-5 moderate, <2 poor

// Compute both IS and FID together
var (isMean, isStd, fidScore) = is.ComputeComprehensiveMetrics(
    generatedImages,
    realImages
);
```

**How It Works**:
1. Pass generated images through Inception classifier
2. Get probability distributions p(y|x) for each image
3. Compute marginal distribution p(y) = mean(p(y|x))
4. Calculate IS = exp(E[KL(p(y|x) || p(y))])

**Interpretation**:
- **High IS**: Images are recognizable (quality) AND diverse (variety)
- **Low IS**: Images are ambiguous OR lack diversity

**When to Use**:
- Secondary metric alongside FID
- Evaluating class-conditional GANs
- Quick quality check during training
- Research comparisons

**Typical Scores (ImageNet)**:
- **> 10**: Excellent (near real images)
- **5-10**: Good quality
- **2-5**: Moderate quality
- **< 2**: Poor (approaching random noise ≈ 1)

**Important Notes**:
- IS can be fooled by mode collapse (same high-quality image repeated)
- FID is generally more reliable
- Scores only comparable within same dataset
- Use both IS and FID for comprehensive evaluation

**Reference**: Salimans et al., "Improved Techniques for Training GANs" (2016)

## Usage Examples

### Example 1: Training a DCGAN

```csharp
using AiDotNet.NeuralNetworks;

// Create DCGAN
var dcgan = new DCGAN<double>(
    latentSize: 100,
    imageChannels: 3,
    imageHeight: 64,
    imageWidth: 64,
    generatorFeatureMaps: 64,
    discriminatorFeatureMaps: 64,
    initialLearningRate: 0.0002
);

// Training loop
for (int epoch = 0; epoch < numEpochs; epoch++)
{
    foreach (var batch in dataLoader)
    {
        // Get real images
        Tensor<double> realImages = batch.Images;

        // Generate noise
        var noise = dcgan.Generator.GenerateRandomNoiseTensor(
            batchSize: realImages.Shape[0],
            noiseSize: 100
        );

        // Train step
        var (discriminatorLoss, generatorLoss) = dcgan.TrainStep(realImages, noise);

        Console.WriteLine($"Epoch {epoch}, D Loss: {discriminatorLoss}, G Loss: {generatorLoss}");
    }

    // Evaluate periodically
    if (epoch % 10 == 0)
    {
        var metrics = dcgan.EvaluateModel(sampleSize: 100);
        Console.WriteLine($"Average Discriminator Score: {metrics["AverageDiscriminatorScore"]}");
    }
}
```

### Example 2: Conditional Generation

```csharp
// Create Conditional GAN
var cgan = new ConditionalGAN<double>(
    generatorArchitecture,
    discriminatorArchitecture,
    numConditionClasses: 10,
    InputType.Image,
    initialLearningRate: 0.0002
);

// Generate images of specific class (e.g., digit 7)
int targetClass = 7;
int numImages = 16;

var noise = cgan.GenerateRandomNoiseTensor(numImages, 100);
var conditions = cgan.CreateOneHotCondition(numImages, targetClass);
var generatedImages = cgan.GenerateConditional(noise, conditions);

// Generated images will all be of class 7
```

### Example 3: Using WGAN-GP for Stable Training

```csharp
// Create WGAN-GP
var wgangp = new WGANGP<double>(
    generatorArchitecture,
    criticArchitecture,
    InputType.Image,
    initialLearningRate: 0.0001,
    gradientPenaltyCoefficient: 10.0,
    criticIterations: 5
);

// Training with gradient penalty
foreach (var batch in dataLoader)
{
    var (criticLoss, generatorLoss) = wgangp.TrainStep(batch.Images, noise);

    // WGAN-GP losses are meaningful:
    // - Critic loss should decrease (Wasserstein distance decreasing)
    // - Lower values = better image quality
    Console.WriteLine($"Wasserstein Distance: {-criticLoss}");
}
```

## Best Practices

### 1. Choosing the Right GAN

- **For beginners**: Start with **DCGAN** - it's stable and well-understood
- **For stability**: Use **WGAN-GP** - best training stability and performance
- **For control**: Use **Conditional GAN** - when you need specific outputs
- **For research**: **WGAN** - meaningful loss metrics for experimentation

### 2. Training Tips

**Learning Rates**:
- DCGAN: 0.0002 (default from paper)
- WGAN: 0.00005 (lower than vanilla GAN)
- WGAN-GP: 0.0001 (slightly higher than WGAN)
- cGAN: 0.0002 (same as DCGAN)

**Batch Sizes**:
- Use batch size 64-128 for best stability
- Larger batches (128-256) can improve WGAN-GP
- Smaller batches may require adjusting learning rate

**Monitoring Training**:
- Check discriminator/critic vs generator loss balance
- Look for mode collapse (low diversity in outputs)
- Monitor evaluation metrics regularly
- Save checkpoints frequently

**Common Issues**:
- **Mode Collapse**: Generator produces limited diversity
  - Solution: Try WGAN-GP, adjust learning rates, increase critic iterations
- **Training Instability**: Losses fluctuate wildly
  - Solution: Lower learning rate, use spectral normalization, try WGAN-GP
- **Poor Image Quality**: Blurry or noisy outputs
  - Solution: Train longer, adjust architecture, use WGAN-GP

### 3. Architecture Design

**Generator**:
- Start with small latent size (64-128)
- Use transposed convolutions for upsampling
- Apply batch normalization (except output layer)
- Use ReLU (hidden) and Tanh (output)

**Discriminator/Critic**:
- Mirror generator architecture
- Use strided convolutions for downsampling
- Apply batch normalization (except first layer)
- Use LeakyReLU activations
- Consider spectral normalization for stability

### 4. Data Preparation

- Normalize images to [-1, 1] (to match Tanh output)
- Apply data augmentation carefully
- Ensure balanced class distribution
- Use appropriate image sizes (powers of 2: 64, 128, 256)

## Implemented Architectures Summary

This implementation provides **13 GAN architectures**:

1. **Vanilla GAN** - Original GAN (already existed)
2. **DCGAN** - Deep Convolutional GAN with architectural guidelines
3. **WGAN** - Wasserstein GAN with weight clipping
4. **WGAN-GP** - WGAN with Gradient Penalty
5. **Conditional GAN** - Class-conditional generation
6. **AC-GAN** - Auxiliary Classifier GAN
7. **InfoGAN** - Information Maximizing GAN
8. **Pix2Pix** - Paired image-to-image translation
9. **CycleGAN** - Unpaired image-to-image translation
10. **StyleGAN** - Style-based generator architecture for high-quality images
11. **Progressive GAN** - Progressively growing architecture for high-resolution generation
12. **BigGAN** - Large-scale GAN training with class conditioning
13. **SAGAN** - Self-Attention GAN for modeling long-range dependencies

Plus supporting components:
- **Spectral Normalization Layer** - For training stability
- **Self-Attention Layer** - For SAGAN support
- **FID Metric** - Fréchet Inception Distance for quality evaluation
- **IS Metric** - Inception Score for quality and diversity evaluation

## References

1. Goodfellow et al., "Generative Adversarial Networks" (2014)
2. Radford et al., "Unsupervised Representation Learning with Deep Convolutional GANs" (2015)
3. Arjovsky et al., "Wasserstein GAN" (2017)
4. Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
5. Mirza and Osindero, "Conditional Generative Adversarial Nets" (2014)
6. Odena et al., "Conditional Image Synthesis with Auxiliary Classifier GANs" (2017)
7. Chen et al., "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets" (2016)
8. Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
9. Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (2017)
10. Miyato et al., "Spectral Normalization for Generative Adversarial Networks" (2018)
11. Zhang et al., "Self-Attention Generative Adversarial Networks" (2019)
12. Karras et al., "Progressive Growing of GANs for Improved Quality, Stability, and Variation" (2018)
13. Karras et al., "A Style-Based Generator Architecture for Generative Adversarial Networks" (2019)
14. Brock et al., "Large Scale GAN Training for High Fidelity Natural Image Synthesis" (2019)
15. Salimans et al., "Improved Techniques for Training GANs" (2016)
16. Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (2017)

## Contributing

Contributions to improve GAN implementations or add new architectures are welcome! Please follow the project guidelines and ensure all code includes comprehensive documentation.

## License

This implementation is part of the AiDotNet library. Please refer to the main project license.
