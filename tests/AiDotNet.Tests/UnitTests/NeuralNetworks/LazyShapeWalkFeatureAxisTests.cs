using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SpeechRecognition.ConformerFamily;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Pins that the lazy shape walk carries a per-feature layer's width into the sequence shape it falls back to (#2387).
/// </summary>
/// <remarks>
/// <para>
/// A dense or normalization layer declares rank-1 shapes but runs along the last axis of a [time, features] sequence.
/// The walk kept only rank-2+ outputs as its fallback, so after EfficientConformer's stage-0 projection (64 -> 32
/// features) the fallback was still the stage-0 splicer's [32, 64]. The stage-1 splicer could not take the rank-1
/// [32] and resolved from that stale fallback: it declared [16, 128] while computing [16, 64], so the first training
/// forward failed its shape check. Predict never verifies declared shapes, so inference looked healthy.
/// </para>
/// <para>
/// The walk runs when anything reads the parameters before the first forward (GetParameters, the parameter layout,
/// an optimizer being set up), so the first test forces it that way. The second follows the census exactly: float,
/// a first Predict, then the training forward.
/// </para>
/// </remarks>
public class LazyShapeWalkFeatureAxisTests
{
    [Fact]
    public void Splicers_resolved_by_the_shape_walk_match_the_real_forward()
    {
        using var model = CreateModel();
        model.GetParameters();

        // 64 frames x 32 mels -> [32, 64] -> project to 32 -> [16, 64] -> project to 44 -> [8, 88].
        int[][] computed = { new[] { 32, 64 }, new[] { 16, 64 }, new[] { 8, 88 } };
        var splicers = model.Layers.OfType<TemporalFrameSplicingLayer<float>>().ToArray();
        Assert.Equal(computed.Length, splicers.Length);
        for (int i = 0; i < splicers.Length; i++)
        {
            // A splicer the walk left lazy resolves from the real forward; one it resolved must match that forward.
            int[] declared = splicers[i].GetOutputShape();
            if (declared.All(d => d > 0))
                Assert.Equal(computed[i], declared);
        }

        model.SetTrainingMode(true);
        var output = model.ForwardForTraining(Tensor<float>.CreateRandom(new[] { 1, 64, 32 }));
        Assert.Equal(new[] { 1, 8, 4 }, output.Shape.ToArray());
    }

    private static EfficientConformer<float> CreateModel() => new(
        new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputHeight: 64, inputWidth: 32, inputDepth: 1, outputSize: 4),
        new EfficientConformerOptions
        {
            EncoderDim = 32, NumEncoderLayers = 3, NumAttentionHeads = 2,
            FeedForwardExpansionFactor = 2, ConvKernelSize = 5, InitialAttentionGroupSize = 2,
            DownsamplingFactor = 8, NumMels = 32, VocabSize = 4,
            DropoutRate = 0.0, UseLayerNormalization = true, WarmupSteps = 100,
        });

    [Fact]
    public void Splicers_declare_what_they_compute_after_a_projection()
    {
        using var model = CreateModel();
        var input = Tensor<float>.CreateRandom(new[] { 1, 64, 32 });
        model.SetTrainingMode(false);
        model.Predict(input);

        var splicers = model.Layers.OfType<TemporalFrameSplicingLayer<float>>().ToArray();
        Assert.Equal(3, splicers.Length);
        // 64 frames x 32 mels -> [32, 64] -> project to 32 -> [16, 64] -> project to 44 -> [8, 88].
        Assert.Equal(new[] { 32, 64 }, splicers[0].GetOutputShape());
        Assert.Equal(new[] { 16, 64 }, splicers[1].GetOutputShape());
        Assert.Equal(new[] { 8, 88 }, splicers[2].GetOutputShape());
    }

    [Fact]
    public void Training_forward_after_predict_keeps_the_declared_shapes()
    {
        using var model = CreateModel();
        var input = Tensor<float>.CreateRandom(new[] { 1, 64, 32 });
        model.SetTrainingMode(false);
        model.Predict(input);

        model.SetTrainingMode(true);
        var output = model.ForwardForTraining(input);

        Assert.Equal(new[] { 1, 8, 4 }, output.Shape.ToArray());
    }
}
