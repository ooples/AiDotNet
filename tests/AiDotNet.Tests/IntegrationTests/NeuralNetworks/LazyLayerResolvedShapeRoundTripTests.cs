using System.IO;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Pins the batch-axis contract between a layer's persisted <c>InputShape</c> and the rank its
/// <c>OnFirstForward</c> hook demands.
/// </summary>
/// <remarks>
/// <para>
/// <c>InputShape</c> is stored PER-SAMPLE -- it carries no batch axis -- but <c>OnFirstForward</c>
/// is written against the tensor a real <c>Forward</c> hands it, which is batched. Layers whose
/// hook accepts the unbatched rank (<c>DenseLayer</c> and <c>FeedForwardLayer</c> take rank>=1)
/// round-tripped correctly and hid the defect. Layers that pin an exact batched rank did not:
/// <c>Conv1DLayer</c> and <c>Conv1DTransposeLayer</c> demand rank-3 [B,C,T] and
/// <c>SwinPatchEmbeddingLayer</c> rank-4 [B,C,H,W], so the persisted [C,T] / [C,H,W] arrived one
/// axis short, <c>ResolveFromShape</c> threw <c>ArgumentException</c>, and the restore path
/// swallowed it by design. The layer stayed unresolved and every restored weight was discarded.
/// </para>
/// <para>
/// Measured before the fix: Conv1DLayer serialized 28 parameters into a 288-byte payload and
/// deserialized to 0, with <c>InputShape</c> back at its [-1,-1] construction sentinel. The bytes
/// were on disk the whole time; they had nowhere to land.
/// </para>
/// </remarks>
public class LazyLayerResolvedShapeRoundTripTests
{
    private static Tensor<double> Ramp(int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = 0.01 * (i + 1);
        return t;
    }

    private static void AssertRoundTripsParameters(
        LayerBase<double> original, LayerBase<double> fresh, Tensor<double> input, int expectedCount)
    {
        original.SetTrainingMode(false);
        original.Forward(input);

        Assert.Equal(expectedCount, original.GetParameters().Length);

        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            original.Serialize(writer);

        ms.Position = 0;
        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            fresh.Deserialize(reader);

        var originalParameters = original.GetParameters();
        var restoredParameters = fresh.GetParameters();

        Assert.True(originalParameters.Length == restoredParameters.Length,
            $"parameter count changed across the round trip: " +
            $"{originalParameters.Length} -> {restoredParameters.Length}");

        for (int i = 0; i < originalParameters.Length; i++)
        {
            Assert.True(originalParameters[i] == restoredParameters[i],
                $"parameter[{i}] differs: {originalParameters[i]:G17} vs {restoredParameters[i]:G17}");
        }
    }

    [Fact]
    public void Conv1DLayer_RankThreeHook_RestoresEveryParameter()
    {
        // 4 output channels x 2 input channels x kernel 3 = 24 weights, + 4 biases = 28.
        AssertRoundTripsParameters(
            new Conv1DLayer<double>(outputChannels: 4, kernelSize: 3),
            new Conv1DLayer<double>(outputChannels: 4, kernelSize: 3),
            Ramp([1, 2, 8]),
            expectedCount: 28);
    }

    [Fact]
    public void Conv1DTransposeLayer_RankThreeHook_RestoresEveryParameter()
    {
        var probe = new Conv1DTransposeLayer<double>(outputChannels: 4, kernelSize: 3);
        probe.SetTrainingMode(false);
        probe.Forward(Ramp([1, 2, 8]));
        int expected = probe.GetParameters().Length;
        Assert.True(expected > 0, "the probe layer reported no parameters, so the test proves nothing");

        AssertRoundTripsParameters(
            new Conv1DTransposeLayer<double>(outputChannels: 4, kernelSize: 3),
            new Conv1DTransposeLayer<double>(outputChannels: 4, kernelSize: 3),
            Ramp([1, 2, 8]),
            expected);
    }

    /// <summary>
    /// A composite is NOT itself lazy, so it returned early from the root-level shape restore and
    /// handed its lazy children straight to the rebind -- where their registered-slot count was 0,
    /// the rebind was skipped as designed, and the values never landed.
    /// </summary>
    /// <remarks>
    /// CifAlignmentLayer is the minimal case: one DenseLayer child and nothing else. It restored
    /// 18 parameters as 0 while that same DenseLayer round-tripped perfectly standing alone, which
    /// is what located the defect in the NESTING rather than in DenseLayer. Recording the resolved
    /// input shape per layout node instead of once at the root is what lets a child at any depth
    /// rebuild itself.
    /// </remarks>
    [Fact]
    public void Composite_WithLazyChild_RestoresChildParameters()
    {
        const int encoderDim = 4;
        var input = Ramp([1, 3, encoderDim]);

        var probe = new CifAlignmentLayer<double>(encoderDim);
        probe.SetTrainingMode(false);
        probe.Forward(input);
        int expected = probe.GetParameters().Length;
        Assert.True(expected > 0,
            "the composite reported no parameters at all, so this test would pass vacuously");

        AssertRoundTripsParameters(
            new CifAlignmentLayer<double>(encoderDim),
            new CifAlignmentLayer<double>(encoderDim),
            input,
            expected);
    }

    /// <summary>
    /// The negative control: a shape the layer genuinely cannot accept must still be refused rather
    /// than forced through by the batch retry, so the retry cannot mask a real mismatch.
    /// </summary>
    [Fact]
    public void Conv1DLayer_RejectsRankTwoInputAtForward()
    {
        var layer = new Conv1DLayer<double>(outputChannels: 4, kernelSize: 3);
        layer.SetTrainingMode(false);
        Assert.Throws<System.ArgumentException>(() => layer.Forward(Ramp([2, 8])));
    }
}
