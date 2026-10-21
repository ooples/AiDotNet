using AiDotNet.ComputerVision.Segmentation.OpenVocabulary;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// The persistence probe samples the head of every trained tensor before and after a step. A streamed weight holds an
/// empty vector until it is paged back, and the probe's Reshape inspected that storage before anything paged it in:
/// OpenVocabSAM, which streams at its default size, threw "View exceeds storage bounds" on its first training step.
/// </summary>
[Collection("FoundationScaleSerial")]
public class FusedTrainingSessionStreamedWeightTests
{
    [Fact]
    public void TrainingAStreamedModel_ProbesItsPagedOutWeights()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Regression, NetworkComplexity.Deep, 0, 32, 32, 3, 0);
        try
        {
            using var model = CreateSmall(architecture);
            // Stream at any size, so this small model takes the path the full-size one takes by default.
            model.ApplyAutoDetectThresholdOverride(1);
            var input = Seeded(42, 1, 3, 32, 32);
            var predicted = model.Predict(input);
            Assert.True(model.WeightStreamingAutoDetected, "the threshold override did not turn streaming on");

            var target = Seeded(7, predicted.Shape.ToArray());
            Assert.Null(Record.Exception(() => model.Train(input, target)));
        }
        finally
        {
            NeuralNetworkBase<float>.ResetWeightStreamingForTests();
        }
    }

    // The first Train can run the probe before any layer boundary has registered the weights with the pool. An
    // unregistered weight still holds its resident vector (RegisterWeight is what empties it), so the probe reads it
    // directly; registering it from a read-only probe would change pool ownership as a side effect.
    [Fact]
    public void TrainingAStreamedModel_WithoutAPrecedingPredict_ProbesItsWeights()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Regression, NetworkComplexity.Deep, 0, 32, 32, 3, 0);
        try
        {
            var input = Seeded(42, 1, 3, 32, 32);
            int[] outputShape;
            using (var shapeProbe = CreateSmall(architecture))
                outputShape = shapeProbe.Predict(input).Shape.ToArray();
            NeuralNetworkBase<float>.ResetWeightStreamingForTests();

            using var model = CreateSmall(architecture);
            model.ApplyAutoDetectThresholdOverride(1);
            var target = Seeded(7, outputShape);
            Assert.Null(Record.Exception(() => model.Train(input, target)));
            Assert.Null(Record.Exception(() => model.Train(input, target)));
            Assert.True(model.WeightStreamingAutoDetected, "the threshold override did not turn streaming on");
        }
        finally
        {
            NeuralNetworkBase<float>.ResetWeightStreamingForTests();
        }
    }

    private static OpenVocabSAM<float> CreateSmall(NeuralNetworkArchitecture<float> architecture)
        => new(architecture, options: new OpenVocabSAMOptions
        {
            NumClasses = 5,
            ChannelDimensions = [32, 64, 128, 256],
            StageDepths = [1, 1, 1, 1],
            NeckEmbeddingDimension = 64,
            DecoderDimension = 32,
        });

    private static Tensor<float> Seeded(int seed, params int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        var rng = RandomHelper.CreateSeededRandom(seed);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (float)rng.NextDouble();
        return tensor;
    }
}
