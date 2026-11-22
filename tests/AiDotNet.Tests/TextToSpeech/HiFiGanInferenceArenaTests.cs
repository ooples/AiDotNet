using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanInferenceArenaTests
{
    private static HiFiGanGenerator<double> Create(int residualDepth, bool condition = false)
    {
        var generator = new HiFiGanGenerator<double>(AiDotNetEngine.Current, 4, 16,
            new[] { 2 }, new[] { 4 }, new[] { 3, 5, 7 },
            Enumerable.Range(0, 3).Select(_ => Enumerable.Repeat(1, residualDepth).ToArray()).ToArray(),
            resblockType1: true, conditionChannels: condition ? 3 : 0,
            conditionInEachUpsamplingLayer: condition);
        foreach (var layer in generator.Layers) layer.SetTrainingMode(false);
        return generator;
    }

    [Fact]
    public void ResidualBranchesMatchUnscopedForwardAndPreserveCallerBuffers()
    {
        var generator = Create(3, condition: true);
        var input = new Tensor<double>(new[] { 2, 4, 11 }, new Vector<double>(
            Enumerable.Range(0, 88).Select(i => Math.Sin(i * .13)).ToArray()));
        var condition = new Tensor<double>(new[] { 2, 3, 1 }, new Vector<double>(
            Enumerable.Range(0, 6).Select(i => .2 + i * .1).ToArray()));
        var originalInput = input.ToArray();
        var originalCondition = condition.ToArray();
        double[] expected;
        using (new NoGradScope<double>()) expected = generator.Forward(input, condition).ToArray();

        using var callerArena = TensorArena.Create();
        var sentinel = TensorAllocator.Rent<double>(new[] { 2, 8, 22 });
        sentinel.Fill(123.0);
        using var noGrad = new NoGradScope<double>();
        for (int repeat = 0; repeat < 6; repeat++)
        {
            var actual = generator.Forward(input, condition).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < actual.Length; i++) Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-12);
            Assert.All(sentinel.ToArray(), value => Assert.Equal(123.0, value));
        }
        Assert.Equal(originalInput, input.ToArray());
        Assert.Equal(originalCondition, condition.ToArray());
    }

    [Fact]
    public void ResidualScratchPeakDoesNotGrowWithResidualDepth()
    {
        bool savedRecycle = InferenceArenaSettings.RecycleLayerScratch;
        try
        {
            InferenceArenaSettings.RecycleLayerScratch = true;
            var shallow = Create(1);
            var deep = Create(8);
            var input = new Tensor<double>(new[] { 1, 4, 128 });
            input.Fill(.1);
            using var callerArena = TensorArena.Create();
            using var noGrad = new NoGradScope<double>();
            _ = shallow.Forward(input).ToArray();
            long shallowPeak = TensorArena.LastDisposedPeakBackingBytes;
            _ = deep.Forward(input).ToArray();
            long deepPeak = TensorArena.LastDisposedPeakBackingBytes;
            Assert.True(shallowPeak > 0);
            Assert.True(deepPeak <= shallowPeak * 1.15 + 65536,
                $"Residual scratch grew with depth: shallow={shallowPeak}, deep={deepPeak}.");
        }
        finally { InferenceArenaSettings.RecycleLayerScratch = savedRecycle; }
    }
}
