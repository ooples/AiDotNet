using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SelfSupervisedLearning.Losses;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

[Collection("FusedTrainingSerial")]
public class GraFPrintPaperFidelityTests
{
    [Fact]
    public async Task OfficialRecipeDefaults_ArePreserved()
    {
        await Task.Yield();

        var options = new GraFPrintOptions();

        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(64, options.NumMels);
        Assert.Equal(512, options.HopLength);
        Assert.Equal(3, options.KNeighbors);
        Assert.Equal(8, options.PeakFilters);
        Assert.Equal(2, options.PeakStride);
        Assert.Equal(7, options.PeakKernelSize);
        Assert.Equal(6, options.NumGnnLayers);
        Assert.Equal(1024, options.EncoderEmbeddingDim);
        Assert.Equal(32, options.ProjectionExpansion);
        Assert.Equal(8e-5, options.LearningRate, 12);
        Assert.Equal(7e-7, options.MinimumLearningRate, 12);
        Assert.Equal(400, options.LRSchedulerTMax);
        Assert.Equal(0.05, options.Temperature, 12);
        Assert.Equal(0.0, options.MaxGradNorm, 12);
    }

    [Fact]
    public async Task CoordinateAugmentation_EmitsTimeFrequencyAndNormalizedSpectrogram()
    {
        await Task.Yield();

        var input = new Tensor<double>([1, 2, 3]);
        for (int i = 0; i < input.Length; i++) input[i] = 2.0 + i * 2.0;
        var layer = new GraFPrintCoordinateAugmentationLayer<double>();

        var output = layer.Forward(input);

        Assert.Equal(new[] { 3, 2, 3 }, output.Shape.ToArray());
        Assert.Equal(0.0, output[0, 0, 0], 12);
        Assert.Equal(0.5, output[0, 0, 1], 12);
        Assert.Equal(1.0, output[0, 0, 2], 12);
        Assert.Equal(0.0, output[1, 0, 1], 12);
        Assert.Equal(1.0, output[1, 1, 1], 12);
        Assert.Equal(0.0, output[2, 0, 0], 12);
        Assert.Equal(0.4, output[2, 0, 2], 12);
        Assert.Equal(1.0, output[2, 1, 2], 12);
    }

    [Fact]
    public async Task CoordinateAugmentation_ConstantSpectrogramFallbackPreservesInputLevel()
    {
        await Task.Yield();

        var low = new Tensor<double>([1, 2, 3]);
        var high = new Tensor<double>([1, 2, 3]);
        low.Fill(0.1);
        high.Fill(0.9);
        var layer = new GraFPrintCoordinateAugmentationLayer<double>();

        var lowOutput = layer.Forward(low);
        var highOutput = layer.Forward(high);

        for (int h = 0; h < 2; h++)
        {
            for (int w = 0; w < 3; w++)
            {
                Assert.Equal(lowOutput[0, h, w], highOutput[0, h, w]);
                Assert.Equal(lowOutput[1, h, w], highOutput[1, h, w]);
                Assert.Equal(0.1, lowOutput[2, h, w], 12);
                Assert.Equal(0.9, highOutput[2, h, w], 12);
            }
        }
    }

    [Fact]
    public async Task FrequencyStride_PreservesTimeAxisAndSelectsEveryFrequencyRow()
    {
        await Task.Yield();

        var input = new Tensor<double>([1, 5, 2]);
        for (int i = 0; i < input.Length; i++) input[i] = i;
        var layer = new GraFPrintFrequencyStrideLayer<double>(2);

        var output = layer.Forward(input);

        Assert.Equal(new[] { 1, 3, 2 }, output.Shape.ToArray());
        Assert.Equal(new[] { 0d, 1d, 4d, 5d, 8d, 9d }, output.ToArray());
    }

    [Fact]
    public async Task EncoderFactory_ContainsPublishedGraphAndProjectionTopology()
    {
        await Task.Yield();

        var layers = LayerHelper<float>.CreateDefaultGraFPrintLayers(
            numMels: 16,
            gnnHiddenDim: 16,
            numGnnLayers: 6,
            embeddingDim: 4,
            dropoutRate: 0.1,
            kNeighbors: 3,
            peakFilters: 4,
            encoderEmbeddingDim: 16,
            projectionExpansion: 2).ToList();

        Assert.IsType<GraFPrintCoordinateAugmentationLayer<float>>(layers[0]);
        Assert.IsType<ConvolutionalLayer<float>>(layers[1]);
        Assert.IsType<GraFPrintFrequencyStrideLayer<float>>(layers[2]);

        var graphBlocks = layers.OfType<GraFPrintGraphBlockLayer<float>>().ToArray();
        Assert.Equal(12, graphBlocks.Length);
        Assert.All(graphBlocks, block => Assert.Equal(3, block.K));
        Assert.Equal(
            new[] { 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32 },
            graphBlocks.Select(block => block.Channels).ToArray());
        Assert.Equal(2, layers.OfType<DenseLayer<float>>().Count());
        Assert.Contains(layers, layer => layer is GlobalPoolingLayer<float>);
        Assert.Contains(layers, layer => layer is FlattenLayer<float>);
    }

    [Fact]
    public async Task GraphBlock_RebuildsKnnFromCurrentFeatures_AndReplaysExactly()
    {
        await Task.Yield();

        var layer = new GraFPrintGraphBlockLayer<double>(channels: 2, k: 2, dilation: 1, dropPathRate: 0.0)
        {
            RandomSeed = 42,
        };
        layer.SetTrainingMode(false);

        var first = BuildFourNodeInput([0.0, 1.0, 10.0, 11.0]);
        var second = BuildFourNodeInput([0.0, 10.0, 11.0, 1.0]);

        var firstOutput = layer.Forward(first);
        int[,,] firstGraph = (int[,,])layer.LastNeighborIndices!.Clone();

        for (int node = 0; node < 4; node++)
            Assert.Equal(node, firstGraph[0, node, 0]);

        var replay = layer.Forward(first);
        int[,,] replayGraph = (int[,,])layer.LastNeighborIndices!.Clone();

        Assert.Equal(firstGraph.Cast<int>(), replayGraph.Cast<int>());
        AssertBitwiseEqual(firstOutput, replay);

        var secondOutput = layer.Forward(second);
        int[,,] secondGraph = layer.LastNeighborIndices!;

        Assert.False(firstGraph.Cast<int>().SequenceEqual(secondGraph.Cast<int>()));
        Assert.All(secondOutput.ToArray(), value =>
            Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
    }

    [Fact(Timeout = 60_000)]
    public async Task CompiledTraining_RebuildsKnnForEachInput()
    {
        await Task.Yield();

        var (model, graphBlock) = CreateSingleGraphBlockModel();
        using var compiledModel = model;
        var (eagerModel, _) = CreateSingleGraphBlockModel();
        using var eagerModelLifetime = eagerModel;
        var initialParameters = compiledModel.GetParameters();
        eagerModel.UpdateParameters(initialParameters);
        double[] initialValues = initialParameters.ToArray();
        var first = BuildFourNodeInput([0.0, 1.0, 10.0, 11.0]);
        var second = BuildFourNodeInput([0.0, 10.0, 11.0, 1.0]);
        var target = new Tensor<double>([1, 2, 2, 2]);

        bool previousCompilation = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;
        try
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = true;
            AiDotNet.Training.CompiledTapeTrainingStep<double>.Invalidate();
            AiDotNet.Training.CompiledTapeTrainingStep<double>.ResetFusedStepCount();

            compiledModel.Train(first, target);
            Assert.Equal(1, AiDotNet.Training.CompiledTapeTrainingStep<double>.GetFusedStepCount());
            int[,,] firstGraph = (int[,,])graphBlock.LastNeighborIndices!.Clone();

            compiledModel.Train(second, target);
            Assert.Equal(2, AiDotNet.Training.CompiledTapeTrainingStep<double>.GetFusedStepCount());
            int[,,] secondGraph = graphBlock.LastNeighborIndices!;

            Assert.False(firstGraph.Cast<int>().SequenceEqual(secondGraph.Cast<int>()),
                "Compiled replay reused the first input's k-NN topology for a different input.");

            double[] compiledValues = compiledModel.GetParameters().ToArray();
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = false;
            AiDotNet.Training.CompiledTapeTrainingStep<double>.Invalidate();
            eagerModel.Train(first, target);
            eagerModel.Train(second, target);
            double[] eagerValues = eagerModel.GetParameters().ToArray();

            Assert.Equal(compiledValues.Length, eagerValues.Length);
            double maxUpdate = 0.0;
            double maxDivergence = 0.0;
            for (int i = 0; i < compiledValues.Length; i++)
            {
                maxUpdate = Math.Max(maxUpdate, Math.Abs(compiledValues[i] - initialValues[i]));
                maxDivergence = Math.Max(
                    maxDivergence, Math.Abs(compiledValues[i] - eagerValues[i]));
            }

            Assert.True(maxUpdate > 1e-12,
                $"GraFPrint parameters did not update; parity would be vacuous (max update {maxUpdate:E3}).");
            Assert.True(maxDivergence < 1e-9,
                $"Compiled dynamic k-NN backward diverged from eager training by {maxDivergence:E3}.");
        }
        finally
        {
            AiDotNet.Training.CompiledTapeTrainingStep<double>.Invalidate();
            AiDotNet.Training.CompiledTapeTrainingStep<double>.ResetFusedStepCount();
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = previousCompilation;
        }
    }

    private static (GraFPrint<double> Model, GraFPrintGraphBlockLayer<double> GraphBlock)
        CreateSingleGraphBlockModel()
    {
        var graphBlock = new GraFPrintGraphBlockLayer<double>(
            channels: 2, k: 2, dilation: 1, dropPathRate: 0.0)
        {
            RandomSeed = 42,
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 2,
            inputWidth: 2,
            inputDepth: 2,
            outputSize: 8,
            layers: [graphBlock])
        {
            RandomSeed = 42,
        };
        var model = new GraFPrint<double>(architecture, new GraFPrintOptions
        {
            EmbeddingDim = 8,
            DropoutRate = 0.0,
            LearningRate = 1e-4,
        });
        return (model, graphBlock);
    }

    [Fact]
    public async Task GraphBlock_KnnIsInvariantToPositivePerNodeScaling()
    {
        await Task.Yield();

        var layer = new GraFPrintGraphBlockLayer<double>(channels: 2, k: 2, dilation: 1, dropPathRate: 0.0)
        {
            RandomSeed = 42,
        };
        layer.SetTrainingMode(false);

        var input = BuildFourNodeInput([0.0, 1.0, 10.0, 11.0]);
        var scaled = new Tensor<double>(input.ToArray(), input.Shape.ToArray());
        double[] scales = [0.25, 2.0, 7.0, 0.5];
        for (int node = 0; node < scales.Length; node++)
        {
            int h = node / 2;
            int w = node % 2;
            for (int channel = 0; channel < 2; channel++)
                scaled[0, channel, h, w] *= scales[node];
        }

        _ = layer.Forward(input);
        int[,,] originalGraph = (int[,,])layer.LastNeighborIndices!.Clone();
        _ = layer.Forward(scaled);
        int[,,] scaledGraph = layer.LastNeighborIndices!;

        Assert.Equal(originalGraph.Cast<int>(), scaledGraph.Cast<int>());
    }

    [Fact(Timeout = 120000)]
    public async Task ContrastiveTraining_UsesConnectedNtxentGradient()
    {
        await Task.Yield();

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 4)
        {
            RandomSeed = 17,
        };
        var model = new GraFPrint<double>(architecture, new GraFPrintOptions
        {
            NumMels = 8,
            GnnHiddenDim = 8,
            NumGnnLayers = 1,
            KNeighbors = 2,
            PeakFilters = 4,
            EncoderEmbeddingDim = 8,
            ProjectionExpansion = 2,
            DropoutRate = 0.0,
            LearningRate = 1e-3,
            MinimumLearningRate = 1e-5,
            LRSchedulerTMax = 20,
        });
        var first = BuildContrastiveBatch(phase: 0.0);
        var second = BuildContrastiveBatch(phase: 0.03);

        double initialLoss = model.ComputeContrastiveLoss(first, second);
        var parametersBefore = model.GetParameters().ToArray();
        double trainingLoss = model.TrainContrastive(first, second);
        var parametersAfter = model.GetParameters().ToArray();

        Assert.True(!double.IsNaN(initialLoss) && !double.IsInfinity(initialLoss));
        Assert.True(!double.IsNaN(trainingLoss) && !double.IsInfinity(trainingLoss));
        Assert.True(parametersBefore.Zip(parametersAfter, (a, b) => a != b).Any(changed => changed),
            "NT-Xent backward did not update any GraFPrint parameter.");

        double best = model.ComputeContrastiveLoss(first, second);
        for (int i = 0; i < 5; i++)
        {
            model.TrainContrastive(first, second);
            best = Math.Min(best, model.ComputeContrastiveLoss(first, second));
        }
        Assert.True(best < initialLoss,
            $"Connected NT-Xent training never improved the objective: initial={initialLoss:R}, best={best:R}.");
    }

    [Fact]
    public async Task NtxentNegativeControl_RejectsIncorrectPairing()
    {
        await Task.Yield();

        var first = new Tensor<double>(new[]
        {
            1d, 0d, 0d,
            0d, 1d, 0d,
            0d, 0d, 1d,
        }, [3, 3]);
        var matched = new Tensor<double>(first.ToArray(), [3, 3]);
        var mismatched = new Tensor<double>(new[]
        {
            0d, 1d, 0d,
            0d, 0d, 1d,
            1d, 0d, 0d,
        }, [3, 3]);
        var loss = new NTXentLoss<double>(temperature: 0.05, normalize: true);

        double matchedLoss = loss.ComputeLoss(first, matched)[0];
        double mismatchedLoss = loss.ComputeLoss(first, mismatched)[0];

        Assert.True(matchedLoss < mismatchedLoss,
            $"NT-Xent did not distinguish correct pairs: matched={matchedLoss:R}, mismatched={mismatchedLoss:R}.");
    }

    private static Tensor<double> BuildFourNodeInput(double[] values)
    {
        var tensor = new Tensor<double>([1, 2, 2, 2]);
        for (int node = 0; node < values.Length; node++)
        {
            int h = node / 2;
            int w = node % 2;
            tensor[0, 0, h, w] = values[node];
            tensor[0, 1, h, w] = 1.0;
        }
        return tensor;
    }

    private static Tensor<double> BuildContrastiveBatch(double phase)
    {
        var tensor = new Tensor<double>([3, 1, 8, 8]);
        for (int b = 0; b < 3; b++)
            for (int h = 0; h < 8; h++)
                for (int w = 0; w < 8; w++)
                    tensor[b, 0, h, w] =
                        Math.Sin((b + 1) * (h + 1) * 0.17 + w * 0.11 + phase);
        return tensor;
    }

    private static void AssertBitwiseEqual(Tensor<double> expected, Tensor<double> actual)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected[i]), BitConverter.DoubleToInt64Bits(actual[i]));
    }
}
