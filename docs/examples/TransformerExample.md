# Transformer Model Usage Guide

This guide demonstrates how to build and use Transformer models with AiDotNet.

## Overview

Transformers are the foundation of modern NLP and increasingly used in computer vision. AiDotNet provides:
- Pre-built transformer architectures
- Multi-head self-attention layers
- Positional encodings
- Encoder-decoder structures

## Quick Start: Text Classification

```csharp
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Models;

// Configure transformer for text classification
var config = new TransformerConfig<float>
{
    VocabSize = 30000,
    MaxSequenceLength = 512,
    EmbeddingDim = 256,
    NumHeads = 8,
    NumLayers = 4,
    FeedForwardDim = 1024,
    Dropout = 0.1f,
    NumClasses = 5
};

// Create model
var transformer = new TransformerClassifier<float>(config);

// Train
var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = await builder
    .ConfigureModel(transformer)
    .ConfigureOptimizer(new AdamWOptimizer<float>(learningRate: 1e-4f, weightDecay: 0.01f))
    .BuildAsync(tokenizedTexts, labels);

// Predict
var prediction = builder.Predict(newText, result);
```

## Transformer Architecture

### Components

1. **Token Embeddings**: Convert tokens to vectors
2. **Positional Encodings**: Add position information
3. **Multi-Head Attention**: Learn relationships between tokens
4. **Feed-Forward Networks**: Process each position
5. **Layer Normalization**: Stabilize training

### Building Custom Transformer

```csharp
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;

// Transformer encoder block
public class TransformerEncoderBlock<T> where T : struct, IFloatingPoint<T>
{
    private readonly MultiHeadAttentionLayer<T> _attention;
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly DenseLayer<T> _ff1;
    private readonly DenseLayer<T> _ff2;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly DropoutLayer<T> _dropout;

    public TransformerEncoderBlock(int embedDim, int numHeads, int ffDim, float dropout)
    {
        _attention = new MultiHeadAttentionLayer<T>(embedDim, numHeads, dropout);
        _norm1 = new LayerNormalizationLayer<T>(embedDim);
        _ff1 = new DenseLayer<T>(embedDim, ffDim, new GELUActivation<T>());
        _ff2 = new DenseLayer<T>(ffDim, embedDim);
        _norm2 = new LayerNormalizationLayer<T>(embedDim);
        _dropout = new DropoutLayer<T>(dropout);
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? mask = null)
    {
        // Self-attention with residual
        var attnOutput = _attention.Forward(x, x, x, mask);
        x = _norm1.Forward(x + _dropout.Forward(attnOutput));

        // Feed-forward with residual
        var ffOutput = _ff2.Forward(_ff1.Forward(x));
        x = _norm2.Forward(x + _dropout.Forward(ffOutput));

        return x;
    }
}
```

## Multi-Head Attention

### How It Works

```csharp
// Multi-head attention layer
var attention = new MultiHeadAttentionLayer<float>(
    embedDim: 256,    // Embedding dimension
    numHeads: 8,      // Number of attention heads
    dropout: 0.1f     // Attention dropout
);

// Forward pass
// Query, Key, Value all same for self-attention
var output = attention.Forward(
    query: embeddings,
    key: embeddings,
    value: embeddings,
    mask: attentionMask  // Optional: mask padding tokens
);
```

### Attention Mask

```csharp
// Create padding mask for variable-length sequences
public Tensor<float> CreatePaddingMask(int[] sequenceLengths, int maxLen)
{
    var batchSize = sequenceLengths.Length;
    var mask = Tensor<float>.Zeros(batchSize, maxLen);

    for (int i = 0; i < batchSize; i++)
    {
        for (int j = sequenceLengths[i]; j < maxLen; j++)
        {
            mask[i, j] = float.NegativeInfinity;  // Masked positions
        }
    }

    return mask;
}

// Create causal mask for autoregressive models (decoder)
public Tensor<float> CreateCausalMask(int seqLen)
{
    var mask = Tensor<float>.Zeros(seqLen, seqLen);

    for (int i = 0; i < seqLen; i++)
    {
        for (int j = i + 1; j < seqLen; j++)
        {
            mask[i, j] = float.NegativeInfinity;  // Can't attend to future
        }
    }

    return mask;
}
```

## Positional Encoding

### Sinusoidal Encoding (Original Transformer)

```csharp
public class SinusoidalPositionalEncoding<T> where T : struct, IFloatingPoint<T>
{
    private readonly Tensor<T> _encodings;

    public SinusoidalPositionalEncoding(int maxLen, int embedDim)
    {
        _encodings = Tensor<T>.Zeros(maxLen, embedDim);

        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < embedDim; i++)
            {
                var angle = pos / Math.Pow(10000, (2 * (i / 2)) / (double)embedDim);

                if (i % 2 == 0)
                    _encodings[pos, i] = T.CreateChecked(Math.Sin(angle));
                else
                    _encodings[pos, i] = T.CreateChecked(Math.Cos(angle));
            }
        }
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var seqLen = x.Shape[1];
        return x + _encodings[..seqLen, ..];
    }
}
```

### Learned Positional Encoding

```csharp
// Learned positional embeddings (often better for shorter sequences)
var posEmbedding = new EmbeddingLayer<float>(
    numEmbeddings: maxSequenceLength,
    embeddingDim: embedDim
);

// In forward pass
var positions = Tensor<int>.Arange(0, seqLen);
var posEmbed = posEmbedding.Forward(positions);
var embeddings = tokenEmbeddings + posEmbed;
```

## Complete Transformer Encoder

```csharp
public class TransformerEncoder<T> where T : struct, IFloatingPoint<T>
{
    private readonly EmbeddingLayer<T> _tokenEmbedding;
    private readonly SinusoidalPositionalEncoding<T> _posEncoding;
    private readonly List<TransformerEncoderBlock<T>> _layers;
    private readonly LayerNormalizationLayer<T> _finalNorm;
    private readonly DropoutLayer<T> _dropout;

    public TransformerEncoder(TransformerConfig<T> config)
    {
        _tokenEmbedding = new EmbeddingLayer<T>(config.VocabSize, config.EmbeddingDim);
        _posEncoding = new SinusoidalPositionalEncoding<T>(config.MaxSequenceLength, config.EmbeddingDim);
        _dropout = new DropoutLayer<T>(config.Dropout);
        _finalNorm = new LayerNormalizationLayer<T>(config.EmbeddingDim);

        _layers = new List<TransformerEncoderBlock<T>>();
        for (int i = 0; i < config.NumLayers; i++)
        {
            _layers.Add(new TransformerEncoderBlock<T>(
                config.EmbeddingDim,
                config.NumHeads,
                config.FeedForwardDim,
                config.Dropout
            ));
        }
    }

    public Tensor<T> Forward(Tensor<int> tokenIds, Tensor<T>? mask = null)
    {
        // Token embeddings
        var x = _tokenEmbedding.Forward(tokenIds);

        // Add positional encoding
        x = _posEncoding.Forward(x);
        x = _dropout.Forward(x);

        // Pass through encoder layers
        foreach (var layer in _layers)
        {
            x = layer.Forward(x, mask);
        }

        return _finalNorm.Forward(x);
    }
}
```

## Text Classification Example

```csharp
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.Text;

// Tokenize text data
var tokenizer = new BPETokenizer(vocabSize: 30000);
tokenizer.Train(trainingTexts);

var tokenizedTexts = trainingTexts
    .Select(text => tokenizer.Encode(text, maxLength: 128))
    .ToArray();

// Create labels (one-hot encoded)
var labels = CreateOneHotLabels(rawLabels, numClasses: 5);

// Configure model
var config = new TransformerConfig<float>
{
    VocabSize = tokenizer.VocabSize,
    MaxSequenceLength = 128,
    EmbeddingDim = 128,
    NumHeads = 4,
    NumLayers = 3,
    FeedForwardDim = 512,
    Dropout = 0.1f,
    NumClasses = 5
};

var model = new TransformerClassifier<float>(config);

// Train
var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = await builder
    .ConfigureModel(model)
    .ConfigureOptimizer(new AdamWOptimizer<float>(
        learningRate: 2e-4f,
        weightDecay: 0.01f
    ))
    .ConfigureLossFunction(new CrossEntropyLoss<float>())
    .ConfigureTraining(new TrainingConfig
    {
        Epochs = 10,
        BatchSize = 32,
        ValidationSplit = 0.1f
    })
    .ConfigureLearningRateScheduler(new WarmupLinearScheduler(
        warmupSteps: 1000,
        totalSteps: 10000
    ))
    .BuildAsync(tokenizedTexts, labels);

Console.WriteLine($"Validation Accuracy: {result.ValidationAccuracy:P2}");

// Predict on new text
var newText = "This product is amazing!";
var tokenized = tokenizer.Encode(newText, maxLength: 128);
var prediction = builder.Predict(tokenized, result);
var predictedClass = prediction.ArgMax();
Console.WriteLine($"Predicted class: {predictedClass}");
```

## Encoder-Decoder (Seq2Seq)

For tasks like translation:

```csharp
public class TransformerSeq2Seq<T> where T : struct, IFloatingPoint<T>
{
    private readonly TransformerEncoder<T> _encoder;
    private readonly TransformerDecoder<T> _decoder;
    private readonly DenseLayer<T> _outputProjection;

    public TransformerSeq2Seq(Seq2SeqConfig<T> config)
    {
        _encoder = new TransformerEncoder<T>(config.EncoderConfig);
        _decoder = new TransformerDecoder<T>(config.DecoderConfig);
        _outputProjection = new DenseLayer<T>(
            config.DecoderConfig.EmbeddingDim,
            config.TargetVocabSize
        );
    }

    public Tensor<T> Forward(Tensor<int> srcTokens, Tensor<int> tgtTokens,
                             Tensor<T>? srcMask = null, Tensor<T>? tgtMask = null)
    {
        // Encode source
        var encoderOutput = _encoder.Forward(srcTokens, srcMask);

        // Decode with cross-attention to encoder output
        var decoderOutput = _decoder.Forward(tgtTokens, encoderOutput, tgtMask);

        // Project to vocabulary
        return _outputProjection.Forward(decoderOutput);
    }
}
```

## Vision Transformer (ViT)

Transformers for image classification:

```csharp
public class VisionTransformer<T> where T : struct, IFloatingPoint<T>
{
    private readonly PatchEmbedding<T> _patchEmbed;
    private readonly EmbeddingLayer<T> _posEmbed;
    private readonly Tensor<T> _clsToken;
    private readonly List<TransformerEncoderBlock<T>> _layers;
    private readonly LayerNormalizationLayer<T> _norm;
    private readonly DenseLayer<T> _classifier;

    public VisionTransformer(ViTConfig<T> config)
    {
        int numPatches = (config.ImageSize / config.PatchSize) *
                         (config.ImageSize / config.PatchSize);

        _patchEmbed = new PatchEmbedding<T>(
            config.ImageSize, config.PatchSize, config.Channels, config.EmbeddingDim);
        _posEmbed = new EmbeddingLayer<T>(numPatches + 1, config.EmbeddingDim);
        _clsToken = Tensor<T>.RandomNormal(1, 1, config.EmbeddingDim);

        _layers = new List<TransformerEncoderBlock<T>>();
        for (int i = 0; i < config.NumLayers; i++)
        {
            _layers.Add(new TransformerEncoderBlock<T>(
                config.EmbeddingDim, config.NumHeads, config.FeedForwardDim, config.Dropout));
        }

        _norm = new LayerNormalizationLayer<T>(config.EmbeddingDim);
        _classifier = new DenseLayer<T>(config.EmbeddingDim, config.NumClasses);
    }

    public Tensor<T> Forward(Tensor<T> images)
    {
        var batchSize = images.Shape[0];

        // Create patch embeddings
        var x = _patchEmbed.Forward(images);  // [batch, numPatches, embedDim]

        // Prepend CLS token
        var clsTokens = _clsToken.Expand(batchSize, -1, -1);
        x = Tensor<T>.Concatenate(clsTokens, x, axis: 1);

        // Add positional embeddings
        var positions = Tensor<int>.Arange(0, x.Shape[1]);
        x = x + _posEmbed.Forward(positions);

        // Transformer layers
        foreach (var layer in _layers)
        {
            x = layer.Forward(x);
        }

        // Classify using CLS token
        var clsOutput = _norm.Forward(x[.., 0, ..]);  // [batch, embedDim]
        return _classifier.Forward(clsOutput);
    }
}
```

## Training Tips

### Learning Rate Warmup

```csharp
// Important for transformer training stability
.ConfigureLearningRateScheduler(new WarmupLinearScheduler(
    warmupSteps: 1000,    // Gradually increase LR
    totalSteps: 50000,    // Total training steps
    peakLr: 1e-4f,        // Maximum learning rate
    endLr: 1e-6f          // Final learning rate
))
```

### Gradient Clipping

```csharp
// Prevent gradient explosion
.ConfigureOptimizer(new AdamWOptimizer<float>(
    learningRate: 1e-4f,
    weightDecay: 0.01f,
    gradientClipNorm: 1.0f  // Clip gradients by norm
))
```

### Mixed Precision Training

```csharp
// Use FP16 for faster training with lower memory
.ConfigureMixedPrecision(new MixedPrecisionConfig
{
    Enabled = true,
    LossScale = 1024f
})
```

## Summary

This guide covered:
- Transformer architecture components
- Multi-head attention mechanism
- Positional encodings
- Building encoder and encoder-decoder models
- Text classification with transformers
- Vision Transformer (ViT) for images
- Training best practices

Transformers are versatile and powerful - start with pre-trained models when possible, and fine-tune for your specific task.
