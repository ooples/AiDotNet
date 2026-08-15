using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Autodiff;

public class GradientTapeFanOutTests
{
    [Fact(Timeout = 30000)]
    public async Task NestedIdentityFanOut_AccumulatesEachPathExactlyOnce()
    {
        await Task.Yield();

        var input = new Tensor<double>(new[] { 3 });
        input[0] = 0.5;
        input[1] = -1.25;
        input[2] = 2.0;

        using var tape = new GradientTape<double>();
        var left = AiDotNetEngine.Current.TensorAdd(input, input);
        var right = AiDotNetEngine.Current.TensorAdd(input, input);
        var output = AiDotNetEngine.Current.TensorAdd(left, right);
        var objective = AiDotNetEngine.Current.ReduceSum(output, new[] { 0 }, keepDims: false);
        var gradients = tape.ComputeGradients(objective, new[] { input });

        Assert.True(gradients.TryGetValue(input, out var gradient));
        for (int i = 0; i < gradient.Length; i++)
        {
            Assert.True(Math.Abs(gradient[i] - 4.0) < 1e-10,
                $"Fan-out gradient at index {i} should include four paths exactly once; " +
                $"expected 4, actual {gradient[i]:R}.");
        }
    }
}
