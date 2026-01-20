# Neural Network Training Guide

This guide demonstrates how to build and train neural networks with AiDotNet.

## Overview

AiDotNet provides a flexible neural network API that supports:
- Feed-forward networks
- Convolutional neural networks (CNNs)
- Recurrent neural networks (RNNs)
- Transformers
- Custom architectures

## Quick Start: MNIST Classification

```csharp
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;

// Load MNIST data (28x28 images, 10 classes)
var (trainImages, trainLabels) = LoadMNIST("train");
var (testImages, testLabels) = LoadMNIST("test");

// Define architecture
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    inputSize: 784,  // 28x28 flattened
    outputSize: 10   // 10 digit classes
);

// Create model
var model = new FeedForwardNeuralNetwork<float>(architecture);

// Train with AiModelBuilder
var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = await builder
    .ConfigureModel(model)
    .ConfigureOptimizer(new AdamOptimizer<float>(learningRate: 0.001f))
    .ConfigureLossFunction(new CrossEntropyLoss<float>())
    .ConfigureTraining(new TrainingConfig
    {
        Epochs = 10,
        BatchSize = 64,
        ValidationSplit = 0.1f
    })
    .BuildAsync(trainImages, trainLabels);

// Evaluate
Console.WriteLine($"Training Accuracy: {result.TrainingAccuracy:P2}");
Console.WriteLine($"Validation Accuracy: {result.ValidationAccuracy:P2}");
```

## Building Custom Architectures

### Layer-by-Layer Construction

```csharp
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;

// Create layers manually
var layers = new List<ILayer<float>>
{
    // Input: 784 features
    new DenseLayer<float>(784, 256, new ReLUActivation<float>()),
    new DropoutLayer<float>(0.3f),

    new DenseLayer<float>(256, 128, new ReLUActivation<float>()),
    new DropoutLayer<float>(0.3f),

    new DenseLayer<float>(128, 64, new ReLUActivation<float>()),

    // Output: 10 classes with softmax
    new DenseLayer<float>(64, 10, new SoftmaxActivation<float>())
};

// Create architecture with custom layers
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    inputSize: 784,
    outputSize: 10,
    layers: layers
);

var model = new FeedForwardNeuralNetwork<float>(architecture);
```

### Convolutional Neural Network (CNN)

```csharp
// CNN for image classification
var cnnLayers = new List<ILayer<float>>
{
    // Input: 28x28x1 grayscale image
    // Conv block 1
    new Conv2DLayer<float>(
        inputChannels: 1,
        outputChannels: 32,
        kernelSize: 3,
        padding: 1,
        activation: new ReLUActivation<float>()
    ),
    new MaxPooling2DLayer<float>(poolSize: 2),

    // Conv block 2
    new Conv2DLayer<float>(
        inputChannels: 32,
        outputChannels: 64,
        kernelSize: 3,
        padding: 1,
        activation: new ReLUActivation<float>()
    ),
    new MaxPooling2DLayer<float>(poolSize: 2),

    // Flatten and classify
    new FlattenLayer<float>(),
    new DenseLayer<float>(64 * 7 * 7, 128, new ReLUActivation<float>()),
    new DropoutLayer<float>(0.5f),
    new DenseLayer<float>(128, 10, new SoftmaxActivation<float>())
};

var cnnArchitecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.Image,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    inputSize: 784,  // 28x28
    outputSize: 10,
    inputHeight: 28,
    inputWidth: 28,
    inputChannels: 1,
    layers: cnnLayers
);

var cnn = new ConvolutionalNeuralNetwork<float>(cnnArchitecture);
```

## Available Layers

### Dense (Fully Connected)

```csharp
new DenseLayer<float>(
    inputSize: 256,
    outputSize: 128,
    activation: new ReLUActivation<float>(),
    useBias: true
)
```

### Convolutional

```csharp
// 2D Convolution
new Conv2DLayer<float>(
    inputChannels: 3,
    outputChannels: 64,
    kernelSize: 3,
    stride: 1,
    padding: 1,
    activation: new ReLUActivation<float>()
)

// 1D Convolution (for sequences)
new Conv1DLayer<float>(
    inputChannels: 128,
    outputChannels: 256,
    kernelSize: 3
)
```

### Pooling

```csharp
// Max pooling
new MaxPooling2DLayer<float>(poolSize: 2, stride: 2)

// Average pooling
new AveragePooling2DLayer<float>(poolSize: 2, stride: 2)

// Global average pooling
new GlobalAveragePooling2DLayer<float>()
```

### Regularization

```csharp
// Dropout
new DropoutLayer<float>(rate: 0.5f)

// Batch normalization
new BatchNormalizationLayer<float>(numFeatures: 64)

// Layer normalization
new LayerNormalizationLayer<float>(normalizedShape: 128)
```

### Recurrent

```csharp
// LSTM
new LSTMLayer<float>(
    inputSize: 128,
    hiddenSize: 256,
    numLayers: 2,
    bidirectional: true,
    dropout: 0.1f
)

// GRU
new GRULayer<float>(
    inputSize: 128,
    hiddenSize: 256
)
```

### Attention

```csharp
// Multi-head attention
new MultiHeadAttentionLayer<float>(
    embedDim: 512,
    numHeads: 8,
    dropout: 0.1f
)
```

## Activation Functions

```csharp
// Common activations
new ReLUActivation<float>()
new LeakyReLUActivation<float>(alpha: 0.01f)
new ELUActivation<float>(alpha: 1.0f)
new SELUActivation<float>()
new SiLUActivation<float>()  // Swish
new GELUActivation<float>()

// Sigmoid family
new SigmoidActivation<float>()
new TanhActivation<float>()
new HardSigmoidActivation<float>()

// Output activations
new SoftmaxActivation<float>()
new LogSoftmaxActivation<float>()
```

## Optimizers

### Adam (Recommended Default)

```csharp
var adam = new AdamOptimizer<float>(
    learningRate: 0.001f,
    beta1: 0.9f,
    beta2: 0.999f,
    epsilon: 1e-8f,
    weightDecay: 0.01f
);
```

### SGD with Momentum

```csharp
var sgd = new SGDOptimizer<float>(
    learningRate: 0.01f,
    momentum: 0.9f,
    nesterov: true
);
```

### Other Optimizers

```csharp
// AdamW (Adam with decoupled weight decay)
var adamw = new AdamWOptimizer<float>(learningRate: 0.001f, weightDecay: 0.01f);

// RMSprop
var rmsprop = new RMSpropOptimizer<float>(learningRate: 0.001f, alpha: 0.99f);

// Adagrad
var adagrad = new AdagradOptimizer<float>(learningRate: 0.01f);
```

## Loss Functions

### Classification

```csharp
// Cross-entropy for multi-class
var crossEntropy = new CrossEntropyLoss<float>();

// Binary cross-entropy
var bce = new BinaryCrossEntropyLoss<float>();

// Focal loss (for imbalanced classes)
var focal = new FocalLoss<float>(alpha: 0.25f, gamma: 2.0f);
```

### Regression

```csharp
// Mean squared error
var mse = new MeanSquaredErrorLoss<float>();

// Mean absolute error
var mae = new MeanAbsoluteErrorLoss<float>();

// Huber loss (robust to outliers)
var huber = new HuberLoss<float>(delta: 1.0f);
```

## Training Configuration

### Basic Training

```csharp
.ConfigureTraining(new TrainingConfig
{
    Epochs = 50,
    BatchSize = 32,
    ValidationSplit = 0.2f,  // 20% for validation
    ShuffleData = true,
    RandomSeed = 42
})
```

### Learning Rate Scheduling

```csharp
.ConfigureLearningRateScheduler(new StepLRScheduler(
    stepSize: 10,  // Decay every 10 epochs
    gamma: 0.1f    // Multiply LR by 0.1
))

// Or cosine annealing
.ConfigureLearningRateScheduler(new CosineAnnealingScheduler(
    tMax: 50,      // Total epochs
    etaMin: 1e-6f  // Minimum learning rate
))
```

### Early Stopping

```csharp
.ConfigureEarlyStopping(new EarlyStoppingConfig
{
    Patience = 10,         // Stop if no improvement for 10 epochs
    MinDelta = 0.001f,     // Minimum change to qualify as improvement
    Monitor = "val_loss",  // Metric to monitor
    RestoreBestWeights = true
})
```

### Callbacks

```csharp
.ConfigureCallbacks(new List<ICallback>
{
    new ModelCheckpoint("best_model.bin", saveWeightsOnly: true, monitor: "val_accuracy", mode: "max"),
    new ReduceLROnPlateau(factor: 0.5f, patience: 5),
    new TensorBoardLogger("logs/run1")
})
```

## GPU Training

```csharp
// Enable GPU acceleration
.ConfigureGpuAcceleration(new GpuConfig
{
    DeviceId = 0,              // GPU device ID
    MemoryFraction = 0.9f,     // Use 90% of GPU memory
    AllowGrowth = true         // Allocate memory as needed
})
```

## Data Augmentation

```csharp
// Image augmentation
.ConfigureDataAugmentation(new ImageAugmentationConfig
{
    RandomHorizontalFlip = true,
    RandomRotation = 15,  // degrees
    RandomCrop = new CropConfig { Height = 28, Width = 28, Padding = 4 },
    Normalize = new NormalizeConfig { Mean = new[] { 0.485f }, Std = new[] { 0.229f } }
})
```

## Complete Training Example

```csharp
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Optimizers;
using AiDotNet.LossFunctions;

// Prepare data
var (trainData, trainLabels) = PrepareData("train");
var (testData, testLabels) = PrepareData("test");

// Build model
var layers = new List<ILayer<float>>
{
    new DenseLayer<float>(784, 512, new ReLUActivation<float>()),
    new BatchNormalizationLayer<float>(512),
    new DropoutLayer<float>(0.3f),

    new DenseLayer<float>(512, 256, new ReLUActivation<float>()),
    new BatchNormalizationLayer<float>(256),
    new DropoutLayer<float>(0.3f),

    new DenseLayer<float>(256, 10, new SoftmaxActivation<float>())
};

var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    inputSize: 784,
    outputSize: 10,
    layers: layers
);

var model = new FeedForwardNeuralNetwork<float>(architecture);

// Train
var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = await builder
    .ConfigureModel(model)
    .ConfigureOptimizer(new AdamOptimizer<float>(learningRate: 0.001f))
    .ConfigureLossFunction(new CrossEntropyLoss<float>())
    .ConfigureTraining(new TrainingConfig
    {
        Epochs = 50,
        BatchSize = 64,
        ValidationSplit = 0.1f
    })
    .ConfigureLearningRateScheduler(new CosineAnnealingScheduler(tMax: 50))
    .ConfigureEarlyStopping(new EarlyStoppingConfig
    {
        Patience = 10,
        Monitor = "val_loss",
        RestoreBestWeights = true
    })
    .ConfigureGpuAcceleration()
    .BuildAsync(trainData, trainLabels);

// Print training history
Console.WriteLine("\nTraining History:");
for (int i = 0; i < result.History.Epochs.Count; i++)
{
    var epoch = result.History.Epochs[i];
    Console.WriteLine($"Epoch {i+1}: loss={epoch.Loss:F4}, acc={epoch.Accuracy:P2}, " +
                     $"val_loss={epoch.ValLoss:F4}, val_acc={epoch.ValAccuracy:P2}");
}

// Evaluate on test set
var predictions = builder.Predict(testData, result);
var testAccuracy = ComputeAccuracy(predictions, testLabels);
Console.WriteLine($"\nTest Accuracy: {testAccuracy:P2}");

// Save model
builder.SaveModel(result, "mnist_model.bin");
Console.WriteLine("Model saved to mnist_model.bin");
```

## Monitoring Training

### Training Progress

```csharp
// Access training history
var history = result.History;

Console.WriteLine("Training Metrics:");
Console.WriteLine($"  Final Loss: {history.Epochs.Last().Loss:F4}");
Console.WriteLine($"  Final Accuracy: {history.Epochs.Last().Accuracy:P2}");
Console.WriteLine($"  Best Val Accuracy: {history.Epochs.Max(e => e.ValAccuracy):P2}");
Console.WriteLine($"  Epochs Trained: {history.Epochs.Count}");
```

### Visualizing Loss Curves

```csharp
// Export history for plotting
var losses = history.Epochs.Select(e => e.Loss).ToArray();
var valLosses = history.Epochs.Select(e => e.ValLoss).ToArray();

// Use your preferred plotting library
PlotLossCurves(losses, valLosses, "training_curves.png");
```

## Best Practices

1. **Start Simple**: Begin with a small network and increase complexity as needed

2. **Use Batch Normalization**: Helps with training stability and often improves results

3. **Apply Dropout**: Prevents overfitting, especially in dense layers

4. **Monitor Validation Loss**: The key indicator for generalization

5. **Use Learning Rate Scheduling**: Helps fine-tune convergence

6. **Save Checkpoints**: Don't lose progress if training is interrupted

7. **Normalize Inputs**: Scale features to zero mean and unit variance

## Common Issues

### Overfitting
- Add dropout layers
- Use data augmentation
- Reduce model size
- Add weight decay

### Underfitting
- Increase model capacity
- Train longer
- Reduce regularization
- Check data preprocessing

### Training Instability
- Reduce learning rate
- Add batch normalization
- Use gradient clipping
- Check for NaN values in data

## Summary

Training neural networks with AiDotNet involves:
1. Define architecture (layers, activations)
2. Choose optimizer and loss function
3. Configure training parameters
4. Use callbacks for monitoring and early stopping
5. Evaluate and save the model

The AiModelBuilder provides a fluent API that makes this process straightforward while allowing full customization when needed.
