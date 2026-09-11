using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Pins the FPN / PANet top-down pathway: each pyramid level merges the upsampled map of the NEXT
/// deeper level (Lin et al. 2017), so every output depends on every backbone stage at or below it.
/// </summary>
/// <remarks>
/// Both necks used to take the top-down input from <c>list[^1]</c> of a list built with
/// <c>Insert(0, ...)</c> - the deepest level, every time. The second-deepest level then fed nothing,
/// and Faster / Cascade R-CNN, which read only P3, never trained the C4 lateral conv.
/// </remarks>
public class NeckTopDownPathwayTests
{
    private static readonly int[] StageChannels = { 4, 6, 8, 10 };

    private static List<Tensor<double>> Stages()
    {
        var r = new Random(11);
        var stages = new List<Tensor<double>>();
        for (int i = 0; i < StageChannels.Length; i++)
        {
            int size = 16 >> i;
            var t = new Tensor<double>(new[] { 1, StageChannels[i], size, size });
            for (int k = 0; k < t.Length; k++) t[k] = r.NextDouble() * 2 - 1;
            stages.Add(t);
        }

        return stages;
    }

    private static void AssertEveryDeeperStageReaches(NeckBase<double> neck, int level)
    {
        var stages = Stages();
        var engine = AiDotNetEngine.Current;
        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var outputs = neck.Forward(stages);
            var loss = engine.ReduceSum(engine.TensorMultiply(outputs[level], outputs[level]), null);
            gradients = tape.ComputeGradients(loss, stages.ToArray());

            for (int stage = level; stage < stages.Count; stage++)
            {
                Assert.True(gradients.TryGetValue(stages[stage], out var g), $"No gradient from P{level + 2} to C{stage + 2}.");
                double max = 0;
                for (int k = 0; k < g.Length; k++) max = Math.Max(max, Math.Abs(g[k]));
                Assert.True(max > 0, $"P{level + 2} does not depend on C{stage + 2}: the top-down pathway skips it.");
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public async Task Fpn_OutputDependsOnEveryDeeperStage(int level)
    {
        await Task.Yield();
        AssertEveryDeeperStageReaches(new FPN<double>(StageChannels, outputChannels: 8), level);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public async Task PaNet_OutputDependsOnEveryDeeperStage(int level)
    {
        await Task.Yield();
        AssertEveryDeeperStageReaches(new PANet<double>(StageChannels, outputChannels: 8), level);
    }
}
