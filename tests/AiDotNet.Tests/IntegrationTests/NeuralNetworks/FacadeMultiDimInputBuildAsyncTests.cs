using System;
using System.Collections.Generic;
using System.Threading.Tasks;
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

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression tests for issue #1468: <c>AiModelBuilder.BuildAsync</c> threw
/// <see cref="ArgumentOutOfRangeException"/> ("Feature index N exceeds the input dimension D")
/// for any prebuilt neural network whose first layer has a non-flat input shape
/// (CNN, GRU/LSTM, Vision Transformer, etc.).
///
/// Root cause: the optimizer's feature-selection pass derives feature indices from the
/// flattened per-sample input size (e.g. 3·8·8 = 192 for a 3-channel 8×8 image), but a
/// multi-dimensional NN validates/handles "features" against the FIRST axis of its first
/// layer's input shape (depth=3 for a CNN, sequence-length for an RNN). The counts diverge,
/// so an index ≥ that axis throws — and even the "use all features" path then breaks the
/// spatial/sequence tensor in <c>GetColumnVectors</c>/<c>SelectFeatures</c> (which index axis 1).
///
/// "Feature selection" (selecting a subset of input columns) is a tabular concept that does
/// not apply to spatial/sequence input, so for these models the builder must train on the
/// full input and skip feature subsetting entirely (generalizing the embedding-only #1113 fix).
/// </summary>
public class FacadeMultiDimInputBuildAsyncTests
{
    private static (Tensor<double> x, Tensor<double> y) GenerateImages(
        int samples, int side, int channels, int classes, int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<double>(new[] { samples, channels, side, side });
        var y = new Tensor<double>(new[] { samples, classes });
        for (int n = 0; n < samples; n++)
        {
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < side; h++)
                    for (int w = 0; w < side; w++)
                        x[n, c, h, w] = rng.NextDouble();

            int label = rng.Next(classes);
            for (int k = 0; k < classes; k++)
                y[n, k] = k == label ? 1.0 : 0.0;
        }
        return (x, y);
    }

    private static ConvolutionalNeuralNetwork<double> BuildCnn(int side, int channels, int classes)
    {
        var layers = new List<ILayer<double>>
        {
            new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 3, stride: 1, padding: 1,
                activationFunction: new ReLUActivation<double>()),
            new MaxPoolingLayer<double>(poolSize: 2, stride: 2),
            new FlattenLayer<double>(),
            new DenseLayer<double>(outputSize: classes, activationFunction: new SoftmaxActivation<double>()),
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputHeight: side, inputWidth: side, inputDepth: channels,
            outputSize: classes, layers: layers);
        return new ConvolutionalNeuralNetwork<double>(architecture);
    }

    [Fact(Timeout = 180000)]
    public async Task BuildAsync_PrebuiltCnn_MultiDimInput_DoesNotThrow_AndPredicts()
    {
        const int side = 8, channels = 3, classes = 3, samples = 60;
        var (trainX, trainY) = GenerateImages(samples, side, channels, classes, seed: 1468);

        var cnn = BuildCnn(side, channels, classes);
        var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 5
            });

        // Before the #1468 fix this threw ArgumentOutOfRangeException:
        // "Feature index 3 exceeds the input dimension 3."
        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(cnn)
            .ConfigureOptimizer(adam)
            .ConfigureLossFunction(new CategoricalCrossEntropyLoss<double>())
            .ConfigureDataLoader(
                new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trainX, trainY))
            .BuildAsync();

        Assert.NotNull(result);
        Assert.NotNull(result.Model);

        // The facade must also predict on new multi-dim input without throwing.
        var (testX, _) = GenerateImages(4, side, channels, classes, seed: 9999);
        var predictions = result.Predict(testX);
        Assert.NotNull(predictions);
        Assert.Equal(4, predictions.Shape[0]);
        Assert.Equal(classes, predictions.Shape[predictions.Shape.Length - 1]);
    }

    [Fact(Timeout = 60000)]
    public async Task SetActiveFeatureIndices_MultiDimInputModel_DoesNotThrow()
    {
        await Task.CompletedTask;

        const int side = 8, channels = 3, classes = 3;
        var cnn = BuildCnn(side, channels, classes);

        // Resolve the lazy conv input shape ([-1,-1,-1] → [3,8,8]) via one forward pass,
        // so GetInputShape()[0] == 3 and the pre-fix validation would fire.
        var (oneImage, _) = GenerateImages(1, side, channels, classes, seed: 7);
        _ = cnn.Predict(oneImage);

        // The optimizer selects flat feature indices (0..191 for a 3×8×8 image). On a
        // multi-dim model these must be treated as a no-op, NOT validated against axis 0 (=3).
        var flatIndices = new List<int>();
        for (int i = 0; i < channels * side * side; i++)
            flatIndices.Add(i);

        var ex = Record.Exception(() => cnn.SetActiveFeatureIndices(flatIndices));
        Assert.Null(ex);
    }
}
