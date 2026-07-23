---
title: "Neural Networks"
description: "Reference for AiDotNet's neural network architectures."
order: 1
section: "Reference"
---


Reference for AiDotNet's neural network architectures. Every architecture trains through the same facade — `ConfigureModel(...)` + `ConfigureDataLoader(...)` + `BuildAsync()` — and the general-purpose `NeuralNetwork<T>` is configured from a `NeuralNetworkArchitecture<T>`. Specialized architectures take their own architecture/options type but build and predict the same way.

---

## Convolutional Networks (CNN)

| Architecture | Use Case |
|:-------------|:---------|
| `ConvolutionalNeuralNetwork<T>` | General image tasks |
| `ResNet<T>` | Image classification, feature extraction |
| `VGG<T>` | Deep feature extraction |
| `DenseNet<T>` | Dense connections |
| `EfficientNet<T>` | Efficient scaling |
| `MobileNet<T>` | Mobile / edge deployment |
| `SqueezeNet<T>` | Ultra-compact |

## Recurrent Networks (RNN)

| Architecture | Description |
|:-------------|:------------|
| `RNN<T>` | Basic recurrent network |
| `LSTMNeuralNetwork<T>` | Long Short-Term Memory |
| `GRUNeuralNetwork<T>` | Gated Recurrent Unit |
| `ConvLSTM<T>` | Convolutional LSTM for sequences |

## Transformers

| Architecture | Use Case |
|:-------------|:---------|
| `Transformer<T>` | Seq-to-seq tasks |
| `VisionTransformer<T>` | Image classification |
| `BERT<T>` | Language understanding |
| `GPT<T>` | Text generation |

## Generative

| Architecture | Description |
|:-------------|:------------|
| `GAN<T>`, `DCGAN<T>`, `WGAN<T>` | Generative adversarial networks |
| `ConditionalGAN<T>`, `CycleGAN<T>` | Conditional / image-to-image GANs |
| `VAE<T>`, `ConditionalVAE<T>`, `VQVAE<T>` | Variational autoencoders |

## Graph & Specialized

| Architecture | Description |
|:-------------|:------------|
| `GraphConvolutionalNetwork<T>` | Graph convolution (GCN) |
| `GraphAttentionNetwork<T>` | Graph attention (GAT) |
| `CapsuleNetwork<T>` | Capsule network |
| `NeuralRadianceField<T>` | NeRF |
| `TemporalConvolutionalNetwork<T>` | Sequence modeling (TCN) |

> Tip: discover concrete architecture types under the `AiDotNet.NeuralNetworks` namespace; each pairs with an architecture or options type in the same namespace.

---

## Building a Neural Network

The general-purpose `NeuralNetwork<T>` builds an appropriate stack from a `NeuralNetworkArchitecture<T>` (input feature count, class count, and a complexity preset).

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// 200 samples of 64 features, one-hot labels for 10 classes.
var rng = new Random(42);
var trainX = new Tensor<double>(new[] { 200, 64 });
var trainY = new Tensor<double>(new[] { 200, 10 });
for (int i = 0; i < 200; i++)
{
    for (int j = 0; j < 64; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 10 }] = 1.0;
}

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 64, numClasses: 10, complexity: NetworkComplexity.Medium));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

var scores = result.Predict(trainX);
Console.WriteLine($"Output shape: [{string.Join(", ", scores.Shape)}]");
Console.WriteLine($"Layers: {result.LayerCount}, params: {result.TotalTrainableParameters:N0}");
```

## Choosing Complexity

`NetworkComplexity` controls depth/width without hand-specifying layers.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var trainX = new Tensor<double>(new[] { 64, 16 });
var trainY = new Tensor<double>(new[] { 64, 2 });
for (int i = 0; i < 64; i++) { trainX[new[] { i, 0 }] = i / 64.0; trainY[new[] { i, i % 2 }] = 1.0; }

foreach (var complexity in new[] { NetworkComplexity.Simple, NetworkComplexity.Medium })
{
    var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
        inputFeatures: 16, numClasses: 2, complexity: complexity));

    var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
        .ConfigureModel(model)
        .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
        .BuildAsync();

    Console.WriteLine($"{complexity}: {result.TotalTrainableParameters:N0} parameters");
}
```

## GPU Acceleration

Add `ConfigureGpuAcceleration()` — it uses the GPU when available and falls back to CPU otherwise.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var trainX = new Tensor<float>(new[] { 64, 32 });
var trainY = new Tensor<float>(new[] { 64, 4 });
for (int i = 0; i < 64; i++) { trainX[new[] { i, 0 }] = i / 64f; trainY[new[] { i, i % 4 }] = 1f; }

var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
    inputFeatures: 32, numClasses: 4, complexity: NetworkComplexity.Simple));

var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureGpuAcceleration()
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Trained; output [{string.Join(", ", result.Predict(trainX).Shape)}]");
```
