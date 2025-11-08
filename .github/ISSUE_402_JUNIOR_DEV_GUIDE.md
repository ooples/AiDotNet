# Issue #402: Self-Supervised Learning (SimCLR, BYOL, MAE) - Junior Developer Implementation Guide

## Overview

Self-Supervised Learning (SSL) enables learning powerful representations from unlabeled data by creating pretext tasks. Instead of requiring expensive labeled datasets, SSL methods learn by solving auxiliary tasks like predicting image transformations, masked patches, or contrastive learning.

This guide covers three influential SSL methods:
- **SimCLR**: Contrastive learning with data augmentation
- **BYOL**: Self-supervised learning without negative samples
- **MAE (Masked Autoencoder)**: Predicting masked image patches

**Learning Value**: Understanding how to extract meaningful features without labels, data augmentation strategies, and contrastive learning principles.

**Estimated Complexity**: Advanced (20-30 hours)

**Prerequisites**:
- Deep neural networks (CNNs, Transformers)
- Image processing and data augmentation
- Embedding spaces and similarity metrics
- Batch normalization and gradient descent

---

## Educational Objectives

By implementing SSL algorithms, you will learn:

1. **Contrastive Learning**: Learning by comparing similar and dissimilar examples
2. **Data Augmentation**: Creating multiple views of the same data
3. **Embedding Spaces**: Representing data in learned feature spaces
4. **Momentum Encoders**: Slowly updating target networks
5. **InfoNCE Loss**: Noise contrastive estimation for representation learning
6. **Masked Prediction**: Learning from reconstructing hidden information
7. **Self-Distillation**: Learning from model's own predictions

---

## Self-Supervised Learning Background

### The SSL Paradigm

Traditional supervised learning:
```
Input X -> Model -> Label Y
Requires: (X, Y) pairs
```

Self-supervised learning:
```
Input X -> Create views (X1, X2) -> Model -> Match representations
Requires: Only X (no labels!)
```

### Why SSL?

**Advantages**:
- No expensive labeling required
- Can leverage unlimited unlabeled data
- Often learns better features than supervised learning
- Transfers well to downstream tasks

**Applications**:
- Pre-training for image classification
- Few-shot learning
- Anomaly detection
- Transfer learning

---

## Architecture Design

### Core Interfaces

```csharp
namespace AiDotNet.SelfSupervised
{
    /// <summary>
    /// Base interface for self-supervised learning methods.
    /// </summary>
    /// <typeparam name="T">Data type (float, double)</typeparam>
    public interface ISSLMethod<T> where T : struct
    {
        /// <summary>
        /// Trains the model on unlabeled data.
        /// </summary>
        /// <param name="data">Unlabeled training data [batch, channels, height, width]</param>
        /// <returns>Training loss</returns>
        double Train(Tensor<T> data);

        /// <summary>
        /// Extracts learned representations from data.
        /// </summary>
        /// <param name="data">Input data</param>
        /// <returns>Feature embeddings [batch, embeddingDim]</returns>
        Matrix<T> GetEmbeddings(Tensor<T> data);

        /// <summary>
        /// The encoder network that produces representations.
        /// </summary>
        IEncoder<T> Encoder { get; }
    }

    /// <summary>
    /// Encoder network that maps inputs to embeddings.
    /// </summary>
    public interface IEncoder<T> where T : struct
    {
        /// <summary>
        /// Encodes input to feature representation.
        /// </summary>
        Matrix<T> Encode(Tensor<T> input);

        /// <summary>
        /// Dimension of output embeddings.
        /// </summary>
        int EmbeddingDimension { get; }
    }

    /// <summary>
    /// Projection head that maps representations to contrastive space.
    /// Used in SimCLR and BYOL.
    /// </summary>
    public interface IProjectionHead<T> where T : struct
    {
        /// <summary>
        /// Projects embeddings to contrastive learning space.
        /// </summary>
        Matrix<T> Project(Matrix<T> embeddings);

        int ProjectionDimension { get; }
    }
}
```

### Data Augmentation Interface

```csharp
namespace AiDotNet.SelfSupervised.Augmentation
{
    /// <summary>
    /// Applies data augmentation to create multiple views.
    /// </summary>
    /// <typeparam name="T">Data type</typeparam>
    public interface IAugmentation<T> where T : struct
    {
        /// <summary>
        /// Applies random augmentation to input.
        /// </summary>
        Tensor<T> Augment(Tensor<T> input);

        /// <summary>
        /// Creates two different augmented views of the same input.
        /// </summary>
        (Tensor<T> view1, Tensor<T> view2) CreateViews(Tensor<T> input);
    }

    /// <summary>
    /// Composition of multiple augmentation operations.
    /// </summary>
    public class AugmentationPipeline<T> : IAugmentation<T> where T : struct
    {
        private readonly List<IAugmentation<T>> _augmentations;
        private readonly Random _random;

        public AugmentationPipeline()
        {
            _augmentations = new List<IAugmentation<T>>();
            _random = new Random();
        }

        public void Add(IAugmentation<T> augmentation)
        {
            _augmentations.Add(augmentation);
        }

        public Tensor<T> Augment(Tensor<T> input)
        {
            var output = input.Clone();

            foreach (var aug in _augmentations)
            {
                if (_random.NextDouble() < 0.8) // Apply with 80% probability
                {
                    output = aug.Augment(output);
                }
            }

            return output;
        }

        public (Tensor<T> view1, Tensor<T> view2) CreateViews(Tensor<T> input)
        {
            // Create two independently augmented views
            return (Augment(input), Augment(input));
        }
    }
}
```

---

## Algorithm 1: SimCLR (Simple Contrastive Learning)

### Theory

SimCLR learns representations by maximizing agreement between differently augmented views of the same image.

**Key Idea**: Pull positive pairs (same image, different augmentations) together while pushing negative pairs (different images) apart.

**Training Process**:
1. Sample a batch of N images
2. Create 2 augmented views for each → 2N total images
3. Encode all views to embeddings
4. For each view, its positive pair is the other view of same image
5. All other 2N-2 views are negative pairs
6. Minimize contrastive loss

**NT-Xent Loss** (Normalized Temperature-scaled Cross Entropy):
```
L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]
```

Where:
- `z_i, z_j` are embeddings of positive pair
- `sim(u,v) = u·v / (||u|| ||v||)` is cosine similarity
- `τ` is temperature parameter
- Sum is over all negative pairs

### Implementation

**File**: `src/SelfSupervised/SimCLR/SimCLR.cs`

```csharp
public class SimCLR<T> : ISSLMethod<T> where T : struct
{
    private readonly IEncoder<T> _encoder;
    private readonly IProjectionHead<T> _projectionHead;
    private readonly IAugmentation<T> _augmentation;
    private readonly IOptimizer<T> _optimizer;

    private readonly double _temperature;
    private readonly int _batchSize;

    public SimCLR(
        IEncoder<T> encoder,
        IProjectionHead<T> projectionHead,
        IAugmentation<T> augmentation,
        double temperature = 0.5,
        int batchSize = 256)
    {
        _encoder = encoder;
        _projectionHead = projectionHead;
        _augmentation = augmentation;
        _temperature = temperature;
        _batchSize = batchSize;

        _optimizer = new AdamOptimizer<T>(learningRate: 0.001);
    }

    public IEncoder<T> Encoder => _encoder;

    public double Train(Tensor<T> data)
    {
        // 1. Create augmented views
        var (view1, view2) = CreateBatchViews(data);

        // 2. Encode both views
        var h1 = _encoder.Encode(view1); // [batchSize, embeddingDim]
        var h2 = _encoder.Encode(view2);

        // 3. Project to contrastive space
        var z1 = _projectionHead.Project(h1); // [batchSize, projectionDim]
        var z2 = _projectionHead.Project(h2);

        // 4. Normalize embeddings
        z1 = L2Normalize(z1);
        z2 = L2Normalize(z2);

        // 5. Compute contrastive loss
        var loss = ComputeNTXentLoss(z1, z2);

        // 6. Backpropagation
        _optimizer.Step(_encoder, _projectionHead, loss);

        return Convert.ToDouble(loss);
    }

    private (Tensor<T>, Tensor<T>) CreateBatchViews(Tensor<T> data)
    {
        var batchSize = data.Shape[0];
        var view1List = new List<Tensor<T>>();
        var view2List = new List<Tensor<T>>();

        for (int i = 0; i < batchSize; i++)
        {
            var image = data[i]; // Single image
            var (v1, v2) = _augmentation.CreateViews(image);
            view1List.Add(v1);
            view2List.Add(v2);
        }

        var view1 = Tensor<T>.Stack(view1List);
        var view2 = Tensor<T>.Stack(view2List);

        return (view1, view2);
    }

    private T ComputeNTXentLoss(Matrix<T> z1, Matrix<T> z2)
    {
        var batchSize = z1.Rows;
        var totalLoss = default(T);

        // Concatenate z1 and z2 to form full batch of 2N samples
        var z = ConcatenateRows(z1, z2); // [2*batchSize, projectionDim]

        // Compute similarity matrix: [2N, 2N]
        var similarities = ComputeCosineSimilarity(z, z); // z @ z^T

        // Scale by temperature
        similarities = DivideScalar(similarities, _temperature);

        // For each sample in z1 and z2
        for (int i = 0; i < batchSize; i++)
        {
            // Positive pairs: (i, i+batchSize) and (i+batchSize, i)
            totalLoss = Add(totalLoss, ComputeContrastiveLoss(similarities, i, i + batchSize, 2 * batchSize));
            totalLoss = Add(totalLoss, ComputeContrastiveLoss(similarities, i + batchSize, i, 2 * batchSize));
        }

        return Divide(totalLoss, Convert.ChangeType(2 * batchSize, typeof(T)));
    }

    private T ComputeContrastiveLoss(Matrix<T> similarities, int i, int positiveIdx, int totalSamples)
    {
        // Numerator: exp(sim(z_i, z_positive))
        var numerator = Exp(similarities[i, positiveIdx]);

        // Denominator: sum of exp(sim(z_i, z_k)) for all k != i
        var denominator = default(T);
        for (int k = 0; k < totalSamples; k++)
        {
            if (k != i) // Exclude self
            {
                denominator = Add(denominator, Exp(similarities[i, k]));
            }
        }

        // Loss: -log(numerator / denominator)
        var ratio = Divide(numerator, denominator);
        return Negate(Log(ratio));
    }

    private Matrix<T> ComputeCosineSimilarity(Matrix<T> a, Matrix<T> b)
    {
        // Already normalized, so dot product = cosine similarity
        return MatrixMultiply(a, Transpose(b));
    }

    private Matrix<T> L2Normalize(Matrix<T> matrix)
    {
        var normalized = new Matrix<T>(matrix.Rows, matrix.Columns);

        for (int i = 0; i < matrix.Rows; i++)
        {
            var norm = default(T);
            for (int j = 0; j < matrix.Columns; j++)
            {
                norm = Add(norm, Multiply(matrix[i, j], matrix[i, j]));
            }
            norm = Sqrt(Add(norm, Convert.ChangeType(1e-8, typeof(T)))); // Add epsilon for stability

            for (int j = 0; j < matrix.Columns; j++)
            {
                normalized[i, j] = Divide(matrix[i, j], norm);
            }
        }

        return normalized;
    }

    public Matrix<T> GetEmbeddings(Tensor<T> data)
    {
        // For downstream tasks, use encoder only (not projection head)
        return _encoder.Encode(data);
    }

    // Generic arithmetic helpers
    private T Add(T a, T b) => (dynamic)a + (dynamic)b;
    private T Multiply(T a, T b) => (dynamic)a * (dynamic)b;
    private T Divide(T a, T b) => (dynamic)a / (dynamic)b;
    private T Negate(T a) => -(dynamic)a;
    private T Exp(T a) => (T)Convert.ChangeType(Math.Exp(Convert.ToDouble(a)), typeof(T));
    private T Log(T a) => (T)Convert.ChangeType(Math.Log(Convert.ToDouble(a)), typeof(T));
    private T Sqrt(T a) => (T)Convert.ChangeType(Math.Sqrt(Convert.ToDouble(a)), typeof(T));
}
```

### SimCLR Augmentation Strategy

**File**: `src/SelfSupervised/Augmentation/SimCLRAugmentation.cs`

```csharp
public class SimCLRAugmentation<T> : IAugmentation<T> where T : struct
{
    private readonly AugmentationPipeline<T> _pipeline;

    public SimCLRAugmentation(int imageSize)
    {
        _pipeline = new AugmentationPipeline<T>();

        // SimCLR augmentation composition (from paper)
        _pipeline.Add(new RandomCrop<T>(imageSize));
        _pipeline.Add(new RandomHorizontalFlip<T>());
        _pipeline.Add(new ColorJitter<T>(
            brightness: 0.4,
            contrast: 0.4,
            saturation: 0.4,
            hue: 0.1));
        _pipeline.Add(new RandomGrayscale<T>(probability: 0.2));
        _pipeline.Add(new GaussianBlur<T>(kernelSize: imageSize / 10));
    }

    public Tensor<T> Augment(Tensor<T> input)
    {
        return _pipeline.Augment(input);
    }

    public (Tensor<T> view1, Tensor<T> view2) CreateViews(Tensor<T> input)
    {
        return _pipeline.CreateViews(input);
    }
}
```

**Key Augmentations** (from SimCLR paper):
1. **Random Crop + Resize**: Crop random patch and resize to original size
2. **Color Jitter**: Randomly change brightness, contrast, saturation, hue
3. **Random Grayscale**: Convert to grayscale with 20% probability
4. **Gaussian Blur**: Apply blur with random kernel size

**Why These Work**:
- **Crop**: Forces model to recognize objects from partial views
- **Color**: Prevents model from relying on color alone
- **Blur**: Encourages learning of spatial structure
- **Combination**: No single augmentation is sufficient - composition is key

---

## Algorithm 2: BYOL (Bootstrap Your Own Latent)

### Theory

BYOL learns representations without negative samples, using a momentum encoder.

**Key Innovation**: Asymmetric architecture with:
- **Online network**: Updated by gradient descent
- **Target network**: Momentum-updated copy of online network

**Training Process**:
1. Online network predicts target network's representation
2. Target network is slowly updated (momentum average)
3. No negative pairs needed!

**Loss Function**:
```
L = MSE(predictor(online(x1)), target(x2))
```

Where x1 and x2 are different augmentations of the same image.

**Why It Works** (surprising!):
- Momentum encoder provides slowly changing targets
- Predictor prevents collapse (online ≠ target architecturally)
- Asymmetry is crucial

### Implementation

**File**: `src/SelfSupervised/BYOL/BYOL.cs`

```csharp
public class BYOL<T> : ISSLMethod<T> where T : struct
{
    private readonly IEncoder<T> _onlineEncoder;
    private readonly IProjectionHead<T> _onlineProjection;
    private readonly IPredictor<T> _predictor;

    private readonly IEncoder<T> _targetEncoder;
    private readonly IProjectionHead<T> _targetProjection;

    private readonly IAugmentation<T> _augmentation;
    private readonly IOptimizer<T> _optimizer;

    private readonly double _momentumCoefficient; // τ (tau)
    private readonly double _baseCoefficient;
    private int _step;

    public BYOL(
        IEncoder<T> encoder,
        IProjectionHead<T> projectionHead,
        IPredictor<T> predictor,
        IAugmentation<T> augmentation,
        double baseMomentum = 0.996)
    {
        // Online network (trainable)
        _onlineEncoder = encoder;
        _onlineProjection = projectionHead;
        _predictor = predictor;

        // Target network (momentum-updated)
        _targetEncoder = encoder.Clone();
        _targetProjection = projectionHead.Clone();

        _augmentation = augmentation;
        _optimizer = new AdamOptimizer<T>(learningRate: 0.001);

        _baseCoefficient = baseMomentum;
        _momentumCoefficient = baseMomentum;
        _step = 0;
    }

    public IEncoder<T> Encoder => _onlineEncoder;

    public double Train(Tensor<T> data)
    {
        // 1. Create two augmented views
        var (view1, view2) = CreateBatchViews(data);

        // 2. Forward pass through online network
        var onlineProj1 = _onlineProjection.Project(_onlineEncoder.Encode(view1));
        var onlineProj2 = _onlineProjection.Project(_onlineEncoder.Encode(view2));

        var onlinePred1 = _predictor.Predict(onlineProj1);
        var onlinePred2 = _predictor.Predict(onlineProj2);

        // 3. Forward pass through target network (no gradients)
        var targetProj1 = _targetProjection.Project(_targetEncoder.Encode(view1));
        var targetProj2 = _targetProjection.Project(_targetEncoder.Encode(view2));

        // Normalize projections
        targetProj1 = L2Normalize(targetProj1);
        targetProj2 = L2Normalize(targetProj2);
        onlinePred1 = L2Normalize(onlinePred1);
        onlinePred2 = L2Normalize(onlinePred2);

        // 4. Compute symmetric loss
        var loss1 = MeanSquaredError(onlinePred1, targetProj2); // Predict view2 from view1
        var loss2 = MeanSquaredError(onlinePred2, targetProj1); // Predict view1 from view2

        var totalLoss = Add(loss1, loss2);

        // 5. Update online network
        _optimizer.Step(_onlineEncoder, _onlineProjection, _predictor, totalLoss);

        // 6. Update target network with momentum
        UpdateTargetNetwork();

        _step++;

        return Convert.ToDouble(totalLoss);
    }

    private void UpdateTargetNetwork()
    {
        // Cosine schedule for momentum coefficient
        _momentumCoefficient = 1 - (1 - _baseCoefficient) * (Math.Cos(Math.PI * _step / 1000) + 1) / 2;

        // θ_target = τ * θ_target + (1 - τ) * θ_online
        UpdateParameters(_targetEncoder, _onlineEncoder, _momentumCoefficient);
        UpdateParameters(_targetProjection, _onlineProjection, _momentumCoefficient);
    }

    private void UpdateParameters(INetwork<T> target, INetwork<T> online, double momentum)
    {
        var targetParams = target.GetParameters();
        var onlineParams = online.GetParameters();

        for (int i = 0; i < targetParams.Length; i++)
        {
            // EMA update: θ_target = τ * θ_target + (1-τ) * θ_online
            targetParams[i] = Add(
                Multiply(Convert.ChangeType(momentum, typeof(T)), targetParams[i]),
                Multiply(Convert.ChangeType(1 - momentum, typeof(T)), onlineParams[i]));
        }

        target.SetParameters(targetParams);
    }

    private (Tensor<T>, Tensor<T>) CreateBatchViews(Tensor<T> data)
    {
        // Same as SimCLR
        var batchSize = data.Shape[0];
        var view1List = new List<Tensor<T>>();
        var view2List = new List<Tensor<T>>();

        for (int i = 0; i < batchSize; i++)
        {
            var (v1, v2) = _augmentation.CreateViews(data[i]);
            view1List.Add(v1);
            view2List.Add(v2);
        }

        return (Tensor<T>.Stack(view1List), Tensor<T>.Stack(view2List));
    }

    private T MeanSquaredError(Matrix<T> predictions, Matrix<T> targets)
    {
        var sum = default(T);
        var count = predictions.Rows * predictions.Columns;

        for (int i = 0; i < predictions.Rows; i++)
        {
            for (int j = 0; j < predictions.Columns; j++)
            {
                var error = Subtract(predictions[i, j], targets[i, j]);
                sum = Add(sum, Multiply(error, error));
            }
        }

        return Divide(sum, Convert.ChangeType(count, typeof(T)));
    }

    public Matrix<T> GetEmbeddings(Tensor<T> data)
    {
        return _onlineEncoder.Encode(data);
    }

    // Arithmetic helpers
    private T Add(T a, T b) => (dynamic)a + (dynamic)b;
    private T Subtract(T a, T b) => (dynamic)a - (dynamic)b;
    private T Multiply(T a, T b) => (dynamic)a * (dynamic)b;
    private T Divide(T a, T b) => (dynamic)a / (dynamic)b;
}
```

### Predictor Network

**File**: `src/SelfSupervised/BYOL/Predictor.cs`

```csharp
public interface IPredictor<T> where T : struct
{
    /// <summary>
    /// Predicts target representation from online projection.
    /// </summary>
    Matrix<T> Predict(Matrix<T> onlineProjection);
}

public class MLPPredictor<T> : IPredictor<T> where T : struct
{
    private readonly List<ILayer<T>> _layers;

    public MLPPredictor(int inputDim, int hiddenDim, int outputDim)
    {
        // Simple 2-layer MLP
        _layers = new List<ILayer<T>>
        {
            new DenseLayer<T>(inputDim, hiddenDim),
            new BatchNormLayer<T>(hiddenDim),
            new ReLUActivation<T>(),
            new DenseLayer<T>(hiddenDim, outputDim)
        };
    }

    public Matrix<T> Predict(Matrix<T> onlineProjection)
    {
        var output = onlineProjection;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }
}
```

**Why Predictor Prevents Collapse**:
- Without predictor, online and target would learn trivial constant representations
- Predictor creates asymmetry: online must predict target, not vice versa
- This asymmetry is sufficient to prevent collapse (proven empirically)

---

## Algorithm 3: Masked Autoencoder (MAE)

### Theory

MAE learns representations by reconstructing masked patches of images.

**Key Idea**: Mask large portions (75%) of image patches, reconstruct from remaining visible patches.

**Architecture**:
1. **Encoder**: Processes only visible patches (efficient!)
2. **Decoder**: Reconstructs all patches from encoded representation + mask tokens

**Loss**: MSE between reconstructed and original patches (only on masked patches)

**Why It Works**:
- Forces encoder to learn semantic features (not just local patterns)
- High masking ratio makes task hard, preventing trivial solutions
- Reconstruction requires understanding of object structure

### Implementation

**File**: `src/SelfSupervised/MAE/MaskedAutoencoder.cs`

```csharp
public class MaskedAutoencoder<T> : ISSLMethod<T> where T : struct
{
    private readonly IVisionTransformer<T> _encoder;
    private readonly ITransformerDecoder<T> _decoder;
    private readonly IOptimizer<T> _optimizer;

    private readonly int _patchSize;
    private readonly double _maskingRatio;
    private readonly Random _random;

    public MaskedAutoencoder(
        IVisionTransformer<T> encoder,
        ITransformerDecoder<T> decoder,
        int patchSize = 16,
        double maskingRatio = 0.75)
    {
        _encoder = encoder;
        _decoder = decoder;
        _patchSize = patchSize;
        _maskingRatio = maskingRatio;
        _random = new Random();

        _optimizer = new AdamOptimizer<T>(learningRate: 0.0001);
    }

    public IEncoder<T> Encoder => _encoder;

    public double Train(Tensor<T> data)
    {
        // 1. Divide images into patches
        var patches = DivideIntoPatches(data); // [batch, numPatches, patchDim]

        // 2. Random masking
        var (visiblePatches, maskedPatches, maskIndices) = ApplyRandomMasking(patches);

        // 3. Encode visible patches
        var encodedFeatures = _encoder.Encode(visiblePatches);

        // 4. Decode to reconstruct all patches
        var reconstructedPatches = _decoder.Decode(encodedFeatures, maskIndices);

        // 5. Compute reconstruction loss (only on masked patches)
        var loss = ComputeReconstructionLoss(reconstructedPatches, maskedPatches, maskIndices);

        // 6. Backpropagation
        _optimizer.Step(_encoder, _decoder, loss);

        return Convert.ToDouble(loss);
    }

    private Tensor<T> DivideIntoPatches(Tensor<T> images)
    {
        // images: [batch, channels, height, width]
        var batchSize = images.Shape[0];
        var channels = images.Shape[1];
        var height = images.Shape[2];
        var width = images.Shape[3];

        var numPatchesH = height / _patchSize;
        var numPatchesW = width / _patchSize;
        var numPatches = numPatchesH * numPatchesW;
        var patchDim = _patchSize * _patchSize * channels;

        var patches = new Tensor<T>(new[] { batchSize, numPatches, patchDim });

        for (int b = 0; b < batchSize; b++)
        {
            int patchIdx = 0;
            for (int i = 0; i < numPatchesH; i++)
            {
                for (int j = 0; j < numPatchesW; j++)
                {
                    // Extract patch
                    int pixelIdx = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int ph = 0; ph < _patchSize; ph++)
                        {
                            for (int pw = 0; pw < _patchSize; pw++)
                            {
                                int h = i * _patchSize + ph;
                                int w = j * _patchSize + pw;
                                patches[b, patchIdx, pixelIdx] = images[b, c, h, w];
                                pixelIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        return patches;
    }

    private (Tensor<T> visible, Tensor<T> masked, List<int> maskIndices) ApplyRandomMasking(Tensor<T> patches)
    {
        var batchSize = patches.Shape[0];
        var numPatches = patches.Shape[1];
        var patchDim = patches.Shape[2];

        var numMasked = (int)(numPatches * _maskingRatio);
        var numVisible = numPatches - numMasked;

        var visiblePatches = new Tensor<T>(new[] { batchSize, numVisible, patchDim });
        var maskedPatches = new Tensor<T>(new[] { batchSize, numMasked, patchDim });
        var maskIndices = new List<int>();

        for (int b = 0; b < batchSize; b++)
        {
            // Random shuffle patch indices
            var indices = Enumerable.Range(0, numPatches).OrderBy(_ => _random.Next()).ToList();

            var visibleIndices = indices.Take(numVisible).ToList();
            var maskedIndicesLocal = indices.Skip(numVisible).ToList();

            maskIndices.AddRange(maskedIndicesLocal);

            // Split patches into visible and masked
            for (int i = 0; i < numVisible; i++)
            {
                int patchIdx = visibleIndices[i];
                for (int d = 0; d < patchDim; d++)
                {
                    visiblePatches[b, i, d] = patches[b, patchIdx, d];
                }
            }

            for (int i = 0; i < numMasked; i++)
            {
                int patchIdx = maskedIndicesLocal[i];
                for (int d = 0; d < patchDim; d++)
                {
                    maskedPatches[b, i, d] = patches[b, patchIdx, d];
                }
            }
        }

        return (visiblePatches, maskedPatches, maskIndices);
    }

    private T ComputeReconstructionLoss(Tensor<T> reconstructed, Tensor<T> original, List<int> maskIndices)
    {
        // MSE loss on masked patches only
        var sum = default(T);
        var count = 0;

        var batchSize = reconstructed.Shape[0];
        var numMasked = reconstructed.Shape[1];
        var patchDim = reconstructed.Shape[2];

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numMasked; i++)
            {
                for (int d = 0; d < patchDim; d++)
                {
                    var error = Subtract(reconstructed[b, i, d], original[b, i, d]);
                    sum = Add(sum, Multiply(error, error));
                    count++;
                }
            }
        }

        return Divide(sum, Convert.ChangeType(count, typeof(T)));
    }

    public Matrix<T> GetEmbeddings(Tensor<T> data)
    {
        var patches = DivideIntoPatches(data);
        // Use all patches (no masking) for inference
        return _encoder.Encode(patches);
    }

    // Arithmetic helpers
    private T Add(T a, T b) => (dynamic)a + (dynamic)b;
    private T Subtract(T a, T b) => (dynamic)a - (dynamic)b;
    private T Multiply(T a, T b) => (dynamic)a * (dynamic)b;
    private T Divide(T a, T b) => (dynamic)a / (dynamic)b;
}
```

### Vision Transformer Encoder

**File**: `src/SelfSupervised/MAE/VisionTransformer.cs`

```csharp
public interface IVisionTransformer<T> : IEncoder<T> where T : struct
{
    /// <summary>
    /// Encodes visible patches to representations.
    /// </summary>
    Matrix<T> Encode(Tensor<T> patches);
}

public class VisionTransformer<T> : IVisionTransformer<T> where T : struct
{
    private readonly IEmbeddingLayer<T> _patchEmbedding;
    private readonly IPositionalEncoding<T> _positionalEncoding;
    private readonly List<ITransformerBlock<T>> _transformerBlocks;
    private readonly int _embeddingDim;

    public VisionTransformer(
        int patchDim,
        int embeddingDim,
        int numHeads,
        int numLayers,
        int ffnDim)
    {
        _embeddingDim = embeddingDim;

        // Learnable linear projection of patches
        _patchEmbedding = new LinearEmbedding<T>(patchDim, embeddingDim);

        // Learnable positional embeddings
        _positionalEncoding = new LearnedPositionalEncoding<T>(embeddingDim);

        // Transformer blocks
        _transformerBlocks = new List<ITransformerBlock<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            _transformerBlocks.Add(new TransformerBlock<T>(
                embeddingDim,
                numHeads,
                ffnDim));
        }
    }

    public Matrix<T> Encode(Tensor<T> patches)
    {
        // patches: [batch, numPatches, patchDim]

        // 1. Linear projection
        var embeddings = _patchEmbedding.Embed(patches); // [batch, numPatches, embeddingDim]

        // 2. Add positional embeddings
        embeddings = _positionalEncoding.AddPositions(embeddings);

        // 3. Transformer blocks
        foreach (var block in _transformerBlocks)
        {
            embeddings = block.Forward(embeddings);
        }

        // 4. Global average pooling for final representation
        return GlobalAveragePool(embeddings);
    }

    private Matrix<T> GlobalAveragePool(Tensor<T> embeddings)
    {
        var batchSize = embeddings.Shape[0];
        var numPatches = embeddings.Shape[1];
        var pooled = new Matrix<T>(batchSize, _embeddingDim);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _embeddingDim; d++)
            {
                var sum = default(T);
                for (int p = 0; p < numPatches; p++)
                {
                    sum = Add(sum, embeddings[b, p, d]);
                }
                pooled[b, d] = Divide(sum, Convert.ChangeType(numPatches, typeof(T)));
            }
        }

        return pooled;
    }

    public int EmbeddingDimension => _embeddingDim;

    private T Add(T a, T b) => (dynamic)a + (dynamic)b;
    private T Divide(T a, T b) => (dynamic)a / (dynamic)b;
}
```

---

## Augmentation Implementations

### Random Crop

**File**: `src/SelfSupervised/Augmentation/RandomCrop.cs`

```csharp
public class RandomCrop<T> : IAugmentation<T> where T : struct
{
    private readonly int _outputSize;
    private readonly Random _random;

    public RandomCrop(int outputSize)
    {
        _outputSize = outputSize;
        _random = new Random();
    }

    public Tensor<T> Augment(Tensor<T> input)
    {
        // input: [channels, height, width]
        var channels = input.Shape[0];
        var height = input.Shape[1];
        var width = input.Shape[2];

        // Random crop position
        var maxTop = height - _outputSize;
        var maxLeft = width - _outputSize;

        var top = _random.Next(0, maxTop + 1);
        var left = _random.Next(0, maxLeft + 1);

        // Extract crop
        var output = new Tensor<T>(new[] { channels, _outputSize, _outputSize });

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < _outputSize; h++)
            {
                for (int w = 0; w < _outputSize; w++)
                {
                    output[c, h, w] = input[c, top + h, left + w];
                }
            }
        }

        return output;
    }

    public (Tensor<T>, Tensor<T>) CreateViews(Tensor<T> input)
    {
        return (Augment(input), Augment(input));
    }
}
```

### Color Jitter

**File**: `src/SelfSupervised/Augmentation/ColorJitter.cs`

```csharp
public class ColorJitter<T> : IAugmentation<T> where T : struct
{
    private readonly double _brightness;
    private readonly double _contrast;
    private readonly double _saturation;
    private readonly double _hue;
    private readonly Random _random;

    public ColorJitter(
        double brightness = 0.4,
        double contrast = 0.4,
        double saturation = 0.4,
        double hue = 0.1)
    {
        _brightness = brightness;
        _contrast = contrast;
        _saturation = saturation;
        _hue = hue;
        _random = new Random();
    }

    public Tensor<T> Augment(Tensor<T> input)
    {
        var output = input.Clone();

        // Apply transformations in random order
        var transforms = new List<Action<Tensor<T>>>
        {
            AdjustBrightness,
            AdjustContrast,
            AdjustSaturation,
            AdjustHue
        };

        foreach (var transform in transforms.OrderBy(_ => _random.Next()))
        {
            transform(output);
        }

        return output;
    }

    private void AdjustBrightness(Tensor<T> image)
    {
        var factor = 1.0 + (_random.NextDouble() * 2 - 1) * _brightness;

        for (int c = 0; c < image.Shape[0]; c++)
        {
            for (int h = 0; h < image.Shape[1]; h++)
            {
                for (int w = 0; w < image.Shape[2]; w++)
                {
                    var value = Convert.ToDouble(image[c, h, w]);
                    value = Math.Clamp(value * factor, 0.0, 1.0);
                    image[c, h, w] = (T)Convert.ChangeType(value, typeof(T));
                }
            }
        }
    }

    private void AdjustContrast(Tensor<T> image)
    {
        var factor = 1.0 + (_random.NextDouble() * 2 - 1) * _contrast;

        // Compute mean per channel
        var means = new double[image.Shape[0]];
        for (int c = 0; c < image.Shape[0]; c++)
        {
            var sum = 0.0;
            var count = image.Shape[1] * image.Shape[2];
            for (int h = 0; h < image.Shape[1]; h++)
            {
                for (int w = 0; w < image.Shape[2]; w++)
                {
                    sum += Convert.ToDouble(image[c, h, w]);
                }
            }
            means[c] = sum / count;
        }

        // Adjust contrast around mean
        for (int c = 0; c < image.Shape[0]; c++)
        {
            for (int h = 0; h < image.Shape[1]; h++)
            {
                for (int w = 0; w < image.Shape[2]; w++)
                {
                    var value = Convert.ToDouble(image[c, h, w]);
                    value = means[c] + (value - means[c]) * factor;
                    value = Math.Clamp(value, 0.0, 1.0);
                    image[c, h, w] = (T)Convert.ChangeType(value, typeof(T));
                }
            }
        }
    }

    // AdjustSaturation and AdjustHue similar (convert to HSV, modify, convert back)

    public (Tensor<T>, Tensor<T>) CreateViews(Tensor<T> input)
    {
        return (Augment(input), Augment(input));
    }
}
```

---

## Training and Evaluation

### Pre-training Loop

**File**: `src/SelfSupervised/Training/SSLTrainer.cs`

```csharp
public class SSLTrainer<T> where T : struct
{
    private readonly ISSLMethod<T> _sslMethod;

    public SSLTrainer(ISSLMethod<T> sslMethod)
    {
        _sslMethod = sslMethod;
    }

    public void PreTrain(IDataLoader<T> dataLoader, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            var epochLoss = 0.0;
            var batchCount = 0;

            foreach (var batch in dataLoader)
            {
                var loss = _sslMethod.Train(batch);
                epochLoss += loss;
                batchCount++;
            }

            var avgLoss = epochLoss / batchCount;
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}: Loss = {avgLoss:F4}");
        }
    }
}
```

### Linear Evaluation Protocol

**File**: `src/SelfSupervised/Evaluation/LinearEvaluation.cs`

```csharp
public class LinearEvaluator<T> where T : struct
{
    private readonly ISSLMethod<T> _sslMethod;
    private readonly ILinearClassifier<T> _classifier;

    public LinearEvaluator(ISSLMethod<T> sslMethod, int numClasses)
    {
        _sslMethod = sslMethod;

        // Linear classifier on top of frozen embeddings
        var embeddingDim = sslMethod.Encoder.EmbeddingDimension;
        _classifier = new LinearClassifier<T>(embeddingDim, numClasses);
    }

    public double Evaluate(
        IDataLoader<T> trainLoader,
        IDataLoader<T> testLoader,
        int epochs)
    {
        // 1. Freeze SSL encoder
        _sslMethod.Encoder.Freeze();

        // 2. Train linear classifier
        var optimizer = new SGDOptimizer<T>(learningRate: 0.1);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var (data, labels) in trainLoader)
            {
                // Extract frozen embeddings
                var embeddings = _sslMethod.GetEmbeddings(data);

                // Train classifier
                var predictions = _classifier.Forward(embeddings);
                var loss = CrossEntropyLoss(predictions, labels);

                optimizer.Step(_classifier, loss);
            }
        }

        // 3. Evaluate on test set
        var correct = 0;
        var total = 0;

        foreach (var (data, labels) in testLoader)
        {
            var embeddings = _sslMethod.GetEmbeddings(data);
            var predictions = _classifier.Forward(embeddings);

            for (int i = 0; i < predictions.Rows; i++)
            {
                var predicted = predictions.Row(i).ArgMax();
                if (predicted == labels[i])
                {
                    correct++;
                }
                total++;
            }
        }

        return (double)correct / total;
    }
}
```

**Linear Evaluation Rationale**:
- Measures quality of learned representations
- If representations are good, simple linear classifier should achieve high accuracy
- Standard benchmark: CIFAR-10, ImageNet linear evaluation

---

## Testing Strategy

### Unit Tests

**File**: `tests/SelfSupervised/SimCLRTests.cs`

```csharp
[TestClass]
public class SimCLRTests
{
    [TestMethod]
    public void TestSimCLR_PositivePairsSimilarity()
    {
        // Positive pairs should have higher similarity than negative pairs

        var encoder = new SimpleEncoder<double>(inputDim: 784, embeddingDim: 128);
        var projection = new MLPProjection<double>(128, 64);
        var augmentation = new SimCLRAugmentation<double>(28);

        var simclr = new SimCLR<double>(encoder, projection, augmentation);

        // Create batch with 2 identical images
        var batch = CreateIdenticalBatch(2, 1, 28, 28);

        simclr.Train(batch);

        // Extract embeddings
        var embeddings = simclr.GetEmbeddings(batch);

        // Positive pair similarity
        var positiveSim = CosineSimilarity(embeddings.Row(0), embeddings.Row(1));

        // Should be high (close to 1.0) since they're the same image
        Assert.IsTrue(positiveSim > 0.5, $"Positive similarity too low: {positiveSim}");
    }

    [TestMethod]
    public void TestNTXentLoss_Decreases()
    {
        var encoder = new SimpleEncoder<double>(inputDim: 784, embeddingDim: 128);
        var projection = new MLPProjection<double>(128, 64);
        var augmentation = new SimCLRAugmentation<double>(28);

        var simclr = new SimCLR<double>(encoder, projection, augmentation, batchSize: 32);

        var batch = GenerateRandomBatch(32, 1, 28, 28);

        // Train for multiple steps
        var initialLoss = simclr.Train(batch);

        for (int i = 0; i < 100; i++)
        {
            simclr.Train(batch);
        }

        var finalLoss = simclr.Train(batch);

        // Loss should decrease
        Assert.IsTrue(finalLoss < initialLoss,
            $"Loss did not decrease: {initialLoss} -> {finalLoss}");
    }

    [TestMethod]
    public void TestAugmentation_CreatesDifferentViews()
    {
        var augmentation = new SimCLRAugmentation<double>(28);
        var image = GenerateRandomImage(1, 28, 28);

        var (view1, view2) = augmentation.CreateViews(image);

        // Views should be different
        var difference = ComputeDifference(view1, view2);
        Assert.IsTrue(difference > 0.1,
            $"Views too similar: difference = {difference}");
    }
}
```

---

## Common Pitfalls

### 1. Representation Collapse

**Problem**: All embeddings collapse to constant vector

**Symptoms**: Perfect loss (0.0) but embeddings are identical

**Solutions**:
- **SimCLR**: Ensure sufficient negative samples (large batch size: 256+)
- **BYOL**: Verify predictor is asymmetric and momentum is working
- **MAE**: Check masking ratio is high enough (75%+)

### 2. Weak Augmentations

**Problem**: Augmentations too simple, task becomes trivial

**Solution**: Use strong composition (crop + color + blur)

### 3. Batch Size Too Small

**Problem**: SimCLR needs large batches for enough negatives

**Solution**: Use batch size ≥ 256, or gradient accumulation

### 4. Temperature Parameter

**Problem**: Wrong temperature makes training unstable

**Solution**: Start with τ = 0.5 for SimCLR, tune if needed

### 5. Momentum Too High/Low

**Problem**: BYOL target network updates too fast or too slow

**Solution**: Use τ = 0.996 with cosine schedule

---

## Advanced Topics

### 1. Multi-Crop Strategy

Use multiple crops of different scales:
```
2x large crops (224x224) + 4x small crops (96x96)
```

Improves performance and efficiency.

### 2. SwAV (Swapped Assignment Views)

Cluster-based contrastive learning without pairwise comparisons.

### 3. DINO (Self-Distillation with No Labels)

Self-supervised Vision Transformers with cross-entropy loss.

### 4. MoCo (Momentum Contrast)

Queue-based negative sampling with momentum encoder.

---

## Performance Optimization

### 1. Mixed Precision Training

Use FP16 for faster training with minimal accuracy loss.

### 2. Gradient Checkpointing

Save memory by recomputing activations during backward pass.

### 3. Efficient Augmentation

Use GPU for augmentation (TorchVision, DALI).

### 4. Distributed Training

Multi-GPU training with synchronized batch norm.

---

## Validation and Verification

### Checklist

- [ ] SimCLR loss decreases consistently
- [ ] BYOL doesn't collapse (embeddings have variance)
- [ ] MAE reconstructs masked patches visually
- [ ] Linear evaluation achieves >80% on CIFAR-10
- [ ] Embeddings transfer to downstream tasks

### Benchmark Datasets

1. **CIFAR-10**: 60k images, 10 classes (good starting point)
2. **ImageNet**: 1.3M images, 1000 classes (standard benchmark)
3. **STL-10**: 100k unlabeled images (designed for SSL)

---

## Resources

### Papers
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, 2020)
- Grill et al., "Bootstrap Your Own Latent" (BYOL, 2020)
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE, 2021)

### Code References
- SimCLR official implementation (TensorFlow)
- PyTorch implementations (lightly.ai)

---

## Success Metrics

### Functionality
- [ ] SimCLR achieves competitive linear evaluation accuracy
- [ ] BYOL trains without collapse
- [ ] MAE reconstructs images visually

### Code Quality
- [ ] Modular augmentation pipeline
- [ ] Comprehensive unit tests
- [ ] Clean separation of encoder/projection/predictor

### Performance
- [ ] Training completes in reasonable time
- [ ] GPU memory usage optimized

---

## Next Steps

After mastering SSL:
1. Apply to **downstream tasks** (detection, segmentation)
2. Explore **multi-modal SSL** (CLIP, ALIGN)
3. Study **SSL for other domains** (NLP, audio, video)
4. Investigate **semi-supervised learning** (combining labeled + unlabeled)

**Congratulations!** You've learned three state-of-the-art self-supervised learning methods that power modern representation learning.
