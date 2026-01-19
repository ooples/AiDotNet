---
layout: default
title: Neural Networks
parent: Reference
nav_order: 1
permalink: /reference/neural-networks/
---

# Neural Network Architectures
{: .no_toc }

Complete reference for all 100+ neural network architectures in AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Convolutional Networks (CNN)

### Standard Architectures

| Architecture | Parameters | Use Case |
|:-------------|:-----------|:---------|
| `ConvolutionalNetwork<T>` | Configurable | General image tasks |
| `ResNet<T>` | 11M-60M | Image classification, feature extraction |
| `VGG<T>` | 138M | Deep feature extraction |
| `Inception<T>` | 23M | Multi-scale features |
| `DenseNet<T>` | 8M-20M | Dense connections |
| `EfficientNet<T>` | 5M-66M | Efficient scaling |

```csharp
var model = new ResNet<float>(
    variant: ResNetVariant.ResNet50,
    numClasses: 1000,
    pretrained: true);
```

### Lightweight Networks

| Architecture | Parameters | Use Case |
|:-------------|:-----------|:---------|
| `MobileNet<T>` | 3.4M | Mobile/edge deployment |
| `ShuffleNet<T>` | 2M | Efficient channel shuffle |
| `SqueezeNet<T>` | 1.2M | Ultra-compact |
| `GhostNet<T>` | 5M | Ghost features |

### Attention-Enhanced CNNs

| Architecture | Description |
|:-------------|:------------|
| `SENet<T>` | Squeeze-and-Excitation blocks |
| `CBAM<T>` | Convolutional Block Attention |
| `SKNet<T>` | Selective Kernel Networks |
| `ECANet<T>` | Efficient Channel Attention |

---

## Recurrent Networks (RNN)

### Core Architectures

| Architecture | Description | Bidirectional |
|:-------------|:------------|:--------------|
| `RNN<T>` | Basic recurrent network | Yes |
| `LSTM<T>` | Long Short-Term Memory | Yes |
| `GRU<T>` | Gated Recurrent Unit | Yes |
| `IndRNN<T>` | Independently Recurrent | Yes |

```csharp
var model = new LSTM<float>(
    inputSize: 100,
    hiddenSize: 256,
    numLayers: 2,
    dropout: 0.2f,
    bidirectional: true);
```

### Advanced RNNs

| Architecture | Description |
|:-------------|:------------|
| `StackedLSTM<T>` | Multi-layer LSTM |
| `PeepholeLSTM<T>` | LSTM with peephole connections |
| `AttentionLSTM<T>` | LSTM with attention mechanism |
| `ConvLSTM<T>` | Convolutional LSTM for sequences |

---

## Transformers

### Text Transformers

| Architecture | Parameters | Use Case |
|:-------------|:-----------|:---------|
| `Transformer<T>` | Configurable | Seq-to-seq tasks |
| `BERT<T>` | 110M-340M | Language understanding |
| `GPT<T>` | 117M-175B | Text generation |
| `T5<T>` | 60M-11B | Text-to-text |
| `RoBERTa<T>` | 125M-355M | Robust BERT |
| `ALBERT<T>` | 12M-235M | Factorized embeddings |
| `XLNet<T>` | 110M-340M | Permutation language model |

```csharp
var model = new Transformer<float>(
    vocabSize: 30522,
    dModel: 768,
    numHeads: 12,
    numLayers: 12,
    dFf: 3072,
    maxSeqLen: 512);
```

### Vision Transformers

| Architecture | Description |
|:-------------|:------------|
| `ViT<T>` | Vision Transformer |
| `DeiT<T>` | Data-efficient Image Transformer |
| `Swin<T>` | Shifted Window Transformer |
| `BEiT<T>` | BERT for images |
| `CvT<T>` | Convolutional Vision Transformer |

### Multi-Modal Transformers

| Architecture | Modalities |
|:-------------|:-----------|
| `CLIP<T>` | Image + Text |
| `BLIP<T>` | Image + Text |
| `Flamingo<T>` | Image + Text |
| `ImageBind<T>` | Multiple modalities |

---

## Generative Adversarial Networks (GAN)

### Standard GANs

| Architecture | Description |
|:-------------|:------------|
| `GAN<T>` | Basic GAN |
| `DCGAN<T>` | Deep Convolutional GAN |
| `WGAN<T>` | Wasserstein GAN |
| `WGANGP<T>` | WGAN with Gradient Penalty |
| `SNGAN<T>` | Spectral Normalization GAN |

```csharp
var gan = new DCGAN<float>(
    latentDim: 100,
    imageSize: 64,
    numChannels: 3);
```

### Conditional GANs

| Architecture | Description |
|:-------------|:------------|
| `ConditionalGAN<T>` | Class-conditional |
| `InfoGAN<T>` | Information-maximizing |
| `ACGAN<T>` | Auxiliary Classifier |

### Style GANs

| Architecture | Description |
|:-------------|:------------|
| `StyleGAN<T>` | Style-based generator |
| `StyleGAN2<T>` | Improved StyleGAN |
| `StyleGAN3<T>` | Alias-free generation |

### Image-to-Image

| Architecture | Description |
|:-------------|:------------|
| `Pix2Pix<T>` | Paired image translation |
| `CycleGAN<T>` | Unpaired image translation |
| `StarGAN<T>` | Multi-domain translation |

---

## Variational Autoencoders (VAE)

| Architecture | Description |
|:-------------|:------------|
| `VAE<T>` | Standard VAE |
| `CVAE<T>` | Conditional VAE |
| `BetaVAE<T>` | Disentangled VAE |
| `VQVAE<T>` | Vector Quantized VAE |
| `VQVAE2<T>` | Hierarchical VQ-VAE |
| `NVAE<T>` | Nouveau VAE |

```csharp
var vae = new VAE<float>(
    inputDim: 784,
    latentDim: 32,
    hiddenDims: [512, 256]);
```

---

## Diffusion Models

| Architecture | Description |
|:-------------|:------------|
| `DDPM<T>` | Denoising Diffusion Probabilistic |
| `DDIM<T>` | Denoising Diffusion Implicit |
| `ScoreSDE<T>` | Score-based SDE |
| `StableDiffusion<T>` | Latent diffusion |
| `DiT<T>` | Diffusion Transformer |
| `Consistency<T>` | Consistency Models |

```csharp
var diffusion = new DDPM<float>(
    imageSize: 256,
    timesteps: 1000,
    betaSchedule: BetaSchedule.Linear);
```

---

## Graph Neural Networks (GNN)

| Architecture | Description |
|:-------------|:------------|
| `GCN<T>` | Graph Convolutional Network |
| `GAT<T>` | Graph Attention Network |
| `GraphSAGE<T>` | Sampling and Aggregation |
| `GIN<T>` | Graph Isomorphism Network |
| `MPNN<T>` | Message Passing NN |
| `PNA<T>` | Principal Neighborhood Aggregation |

```csharp
var gnn = new GAT<float>(
    inputDim: 64,
    hiddenDim: 128,
    outputDim: 7,
    numHeads: 8);
```

---

## Capsule Networks

| Architecture | Description |
|:-------------|:------------|
| `CapsuleNetwork<T>` | Standard capsule network |
| `DynamicRouting<T>` | Dynamic routing between capsules |
| `EMRouting<T>` | Expectation-Maximization routing |

---

## Neural Radiance Fields (NeRF)

| Architecture | Description |
|:-------------|:------------|
| `NeRF<T>` | Original NeRF |
| `InstantNGP<T>` | Instant Neural Graphics |
| `TensoRF<T>` | Tensorial Radiance Fields |
| `Plenoxels<T>` | Plenoptic Voxels |

---

## Physics-Informed Neural Networks

| Architecture | Description |
|:-------------|:------------|
| `PINN<T>` | Physics-Informed NN |
| `DeepONet<T>` | Deep Operator Network |
| `FNO<T>` | Fourier Neural Operator |
| `PhysicsNet<T>` | Generic physics network |

---

## Specialized Architectures

### Sequence Models

| Architecture | Description |
|:-------------|:------------|
| `TCN<T>` | Temporal Convolutional Network |
| `WaveNet<T>` | Dilated causal convolutions |
| `Mamba<T>` | State Space Model |
| `RWKV<T>` | Receptance Weighted Key Value |

### Memory Networks

| Architecture | Description |
|:-------------|:------------|
| `MemoryNetwork<T>` | End-to-end memory network |
| `NTM<T>` | Neural Turing Machine |
| `DNC<T>` | Differentiable Neural Computer |

### Attention Mechanisms

| Module | Description |
|:-------|:------------|
| `MultiHeadAttention<T>` | Standard multi-head attention |
| `FlashAttention<T>` | Memory-efficient attention |
| `LinearAttention<T>` | Linear complexity attention |
| `CrossAttention<T>` | Cross-modal attention |

---

## Usage Examples

### Image Classification

```csharp
var model = new EfficientNet<float>(
    variant: EfficientNetVariant.B4,
    numClasses: 100);

var result = await new AiModelBuilder<float, Tensor<float>, int>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new AdamWOptimizer<float>())
    .ConfigureGpuAcceleration(new GpuAccelerationConfig { Enabled = true })
    .BuildAsync(images, labels);
```

### Text Generation

```csharp
var model = new GPT<float>(
    vocabSize: 50257,
    dModel: 768,
    numHeads: 12,
    numLayers: 12);

var generated = model.Generate(
    prompt: "Once upon a time",
    maxTokens: 100,
    temperature: 0.7f);
```

### Graph Classification

```csharp
var model = new GIN<float>(
    inputDim: 32,
    hiddenDim: 64,
    outputDim: 10,
    numLayers: 5);

var prediction = model.Classify(graph);
```
