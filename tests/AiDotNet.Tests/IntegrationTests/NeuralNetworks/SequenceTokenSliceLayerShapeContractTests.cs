using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Guards the per-sample shape contract of <see cref="SequenceTokenSliceLayer{T}"/>.
/// Runtime tensors include a leading batch dimension, but layer metadata must not
/// retain it because architecture validation and cloning compare sample shapes.
/// </summary>
[Collection("NonParallelIntegration")]
public sealed class SequenceTokenSliceLayerShapeContractTests
{
    private const int BatchSize = 7;
    private const int SequenceLength = 8;
    private const int FeatureCount = 16;
    private const int ClassCount = 2;

    [Fact]
    public void Forward_ResolvesPerSampleShapesWithoutBatchDimension()
    {
        var layer = new SequenceTokenSliceLayer<float>();
        var input = new Tensor<float>([BatchSize, SequenceLength, FeatureCount]);

        Tensor<float> output = layer.Forward(input);

        Assert.Equal([BatchSize, FeatureCount], output.Shape.ToArray());
        Assert.Equal([SequenceLength, FeatureCount], layer.GetInputShape());
        Assert.Equal([FeatureCount], layer.GetOutputShape());
    }

    [Fact]
    public void DeepCopy_AfterBatchedForward_PreservesSliceToDenseShapeContract()
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>([SequenceLength, FeatureCount]),
            new SequenceTokenSliceLayer<float>(),
            new DenseLayer<float>(ClassCount),
            new ActivationLayer<float>((IVectorActivationFunction<float>)new SoftmaxActivation<float>()),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: SequenceLength,
            inputWidth: FeatureCount,
            outputSize: ClassCount,
            layers: layers);
        var network = new FeedForwardNeuralNetwork<float>(architecture);
        var input = new Tensor<float>([BatchSize, SequenceLength, FeatureCount]);

        Tensor<float> prediction = network.Predict(input);
        var copy = (FeedForwardNeuralNetwork<float>)network.DeepCopy();

        Assert.Equal([BatchSize, ClassCount], prediction.Shape.ToArray());
        Assert.NotSame(network, copy);
        Assert.Equal([SequenceLength, FeatureCount], network.Layers[1].GetInputShape());
        Assert.Equal([FeatureCount], network.Layers[1].GetOutputShape());
        Assert.Equal([ClassCount], network.Layers[3].GetInputShape());
        Assert.Equal([ClassCount], network.Layers[3].GetOutputShape());
        Assert.Equal(network.Layers.Select(layer => layer.GetType()),
            copy.Layers.Select(layer => layer.GetType()));
    }

    [Fact(Timeout = 120_000)]
    public async Task BuildAsync_WithSequenceSliceAndDenseHead_CompletesOptimizerCopyPath()
    {
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                MaxIterations = 1,
                InitialLearningRate = 1e-3,
                UseAdaptiveLearningRate = false,
            });
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>([SequenceLength, FeatureCount]),
            new SequenceTokenSliceLayer<float>(),
            new DenseLayer<float>(ClassCount),
            new ActivationLayer<float>((IVectorActivationFunction<float>)new SoftmaxActivation<float>()),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: SequenceLength,
            inputWidth: FeatureCount,
            outputSize: ClassCount,
            layers: layers);
        var network = new FeedForwardNeuralNetwork<float>(
            architecture,
            optimizer,
            new CategoricalCrossEntropyLoss<float>());
        var inputs = new Tensor<float>([48, SequenceLength, FeatureCount]);
        var targets = new Tensor<float>([48, ClassCount]);
        Span<float> inputValues = inputs.AsWritableSpan();
        Span<float> targetValues = targets.AsWritableSpan();
        for (int sample = 0; sample < inputs.Shape[0]; sample++)
        {
            for (int feature = 0; feature < SequenceLength * FeatureCount; feature++)
            {
                inputValues[sample * SequenceLength * FeatureCount + feature] =
                    ((sample + feature) % 17 - 8) / 8f;
            }

            targetValues[sample * ClassCount + sample % ClassCount] = 1;
        }

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(network)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(DataLoaders.FromTensors(inputs, targets))
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Equal([SequenceLength, FeatureCount], network.Layers[1].GetInputShape());
        Assert.Equal([FeatureCount], network.Layers[1].GetOutputShape());
        Assert.Equal([ClassCount], network.Layers[3].GetInputShape());
        Assert.Equal([ClassCount], network.Layers[3].GetOutputShape());
    }
}
