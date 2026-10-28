using System;
using AiDotNet.Engines;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

public class TapeLstmSequenceTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 5)]
    public void Sequence_MatchesEveryStep(int batch, int length)
    {
        const int width = 3, hidden = 2;
        var weights = new Vector<float>(TapeLstmCell<float>.ParameterCount(width, hidden, 1));
        TapeLstmCell<float>.InitializeUniform(weights, 0, weights.Length, hidden, new Random(31));
        var cell = new TapeLstmCell<float>(weights, 0, width, hidden, 1);
        var input = new Tensor<float>(new[] { batch, length, width });
        for (int i = 0; i < input.Length; i++) input[i] = (i % 11 - 5) * 0.1f;
        var actual = cell.Sequence(input);
        var engine = AiDotNetEngine.Current;
        var h = new Tensor<float>(new[] { batch, hidden });
        var c = new Tensor<float>(new[] { batch, hidden });
        for (int t = 0; t < length; t++)
        {
            var x = engine.Reshape(engine.TensorNarrow(input, 1, t, 1), new[] { batch, width });
            (h, c) = cell.Step(x, new[] { h }, c);
            for (int b = 0; b < batch; b++)
                for (int j = 0; j < hidden; j++)
                    Assert.InRange(Math.Abs(actual[(b * length + t) * hidden + j] - h[b * hidden + j]), 0, 2e-6f);
        }
    }

    [Fact]
    public void Sequence_GradientsMatchUnrolledRecurrence()
    {
        var weights = new Vector<float>(TapeLstmCell<float>.ParameterCount(2, 2, 1));
        TapeLstmCell<float>.InitializeUniform(weights, 0, weights.Length, 2, new Random(42));
        var cell = new TapeLstmCell<float>(weights, 0, 2, 2, 1);
        var input = new Tensor<float>(new[] { 2, 4, 2 });
        for (int i = 0; i < input.Length; i++) input[i] = (i % 7 - 3) * 0.1f;
        var engine = AiDotNetEngine.Current;
        var leaves = cell.Leaves.Concat(new[] { input }).ToArray();
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> Gradient(bool fused)
        {
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
            Tensor<float> output;
            if (fused) output = cell.Sequence(input);
            else
            {
                var h = new Tensor<float>(new[] { 2, 2 });
                var c = new Tensor<float>(new[] { 2, 2 });
                var steps = new Tensor<float>[4];
                for (int t = 0; t < 4; t++)
                {
                    var x = engine.Reshape(engine.TensorNarrow(input, 1, t, 1), new[] { 2, 2 });
                    (h, c) = cell.Step(x, new[] { h }, c);
                    steps[t] = h;
                }
                output = engine.TensorStack(steps, 1);
            }
            return tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(output, output), null), leaves);
        }
        var expected = Gradient(false);
        var actual = Gradient(true);
        foreach (var leaf in leaves)
        {
            Assert.True(expected.ContainsKey(leaf));
            Assert.True(actual.ContainsKey(leaf));
            for (int i = 0; i < leaf.Length; i++)
                Assert.InRange(Math.Abs(actual[leaf][i] - expected[leaf][i]), 0, 3e-5f);
        }
    }

    [Fact]
    public void Sequence_HandlesEmptyTime_AndRejectsAttentionFeedback()
    {
        var plain = new TapeLstmCell<double>(new Vector<double>(TapeLstmCell<double>.ParameterCount(3, 2, 1)), 0, 3, 2, 1);
        Assert.Equal(new[] { 2, 0, 2 }, plain.Sequence(new Tensor<double>(new[] { 2, 0, 3 })).Shape.ToArray());
        var attention = new TapeLstmCell<double>(new Vector<double>(TapeLstmCell<double>.ParameterCount(3, 2, 2)), 0, 3, 2, 2);
        Assert.Throws<InvalidOperationException>(() => attention.Sequence(new Tensor<double>(new[] { 1, 4, 3 })));
    }
}
