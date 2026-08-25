using AiDotNet.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Autodiff;

/// <summary>
/// Covers <see cref="ComputationNode{T}.Backward"/>, the driver for the reverse-mode pass over a
/// <see cref="TensorOperations{T}"/> graph.
/// </summary>
/// <remarks>
/// Before this method existed, <see cref="TensorOperations{T}"/> could build a complete graph —
/// parents recorded, local derivatives captured in each backward function — and no public API
/// could run it. The only code that invoked a backward function was a private helper inside
/// AiDotNet.Autodiff.Testing, so a caller of the library could not obtain a gradient at all.
/// </remarks>
public class ComputationNodeBackwardTests
{
    private static Tensor<double> TensorOf(params double[] values)
    {
        var tensor = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++) tensor[i] = values[i];

        return tensor;
    }

    [Fact]
    public void Backward_OnAProduct_GivesEachFactorTheOtherFactor()
    {
        var a = TensorOperations<double>.Variable(TensorOf(2.0, -3.0, 0.5), "a");
        var b = TensorOperations<double>.Variable(TensorOf(5.0, 7.0, -4.0), "b");

        TensorOperations<double>.ElementwiseMultiply(a, b).Backward();

        // d(a*b)/da = b and d(a*b)/db = a, exactly.
        Assert.NotNull(a.Gradient);
        Assert.NotNull(b.Gradient);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(b.Value[i], a.Gradient![i]);
            Assert.Equal(a.Value[i], b.Gradient![i]);
        }
    }

    [Fact]
    public void Backward_ThroughAChainOfOperations_MatchesTheClosedForm()
    {
        var x = TensorOperations<double>.Variable(TensorOf(0.3, -1.1, 2.0), "x");

        // f = exp(x) * x, so df/dx = exp(x)(1 + x).
        var f = TensorOperations<double>.ElementwiseMultiply(TensorOperations<double>.Exp(x), x);
        f.Backward();

        Assert.NotNull(x.Gradient);

        for (int i = 0; i < 3; i++)
        {
            double expected = Math.Exp(x.Value[i]) * (1.0 + x.Value[i]);
            Assert.True(Math.Abs(x.Gradient![i] - expected) < 1e-12,
                $"at {i}: got {x.Gradient[i]:G17}, closed form {expected:G17}");
        }
    }

    [Fact]
    public void Backward_WhenOneNodeFeedsTwoBranches_CountsBothPaths()
    {
        var x = TensorOperations<double>.Variable(TensorOf(1.5, -2.0), "x");

        // f = x*x + x, so df/dx = 2x + 1. The diamond is the case a driver that walks the graph
        // in the wrong order gets wrong: x's gradient is complete only after BOTH the product and
        // the sum have contributed, which is exactly what reversing the topological order buys.
        var f = TensorOperations<double>.Add(TensorOperations<double>.ElementwiseMultiply(x, x), x);
        f.Backward();

        Assert.NotNull(x.Gradient);
        Assert.Equal(2.0 * 1.5 + 1.0, x.Gradient![0], 12);
        Assert.Equal(2.0 * -2.0 + 1.0, x.Gradient[1], 12);
    }

    [Fact]
    public void Backward_FromAScalarLoss_ReachesEveryParameter()
    {
        var w = TensorOperations<double>.Variable(TensorOf(0.5, -0.25, 2.0), "w");
        var x = TensorOperations<double>.Variable(TensorOf(1.0, 4.0, -0.5), "x", requiresGradient: false);

        // loss = sum(w * x), whose gradient with respect to w is x — the smallest thing that is
        // genuinely a training step: one scalar out, every parameter's derivative in one pass.
        var loss = TensorOperations<double>.Sum(TensorOperations<double>.ElementwiseMultiply(w, x));
        loss.Backward();

        Assert.NotNull(w.Gradient);
        for (int i = 0; i < 3; i++) Assert.Equal(x.Value[i], w.Gradient![i], 12);

        // Declared as not requiring a gradient, so nothing should have been computed for it.
        Assert.Null(x.Gradient);
    }

    [Fact]
    public void Backward_MatchesCentralDifferences()
    {
        var start = TensorOf(0.7, -0.4, 1.3);

        var x = TensorOperations<double>.Variable(start.Clone(), "x");
        var f = TensorOperations<double>.Sum(TensorOperations<double>.Tanh(
            TensorOperations<double>.ElementwiseMultiply(x, x)));
        f.Backward();

        // An independent check built on a different principle. Finite differences are the right
        // tool for verifying a gradient even though they are the wrong tool for computing one.
        const double h = 1e-6;
        for (int i = 0; i < start.Length; i++)
        {
            double original = start[i];

            start[i] = original + h;
            double up = SumOfTanhSquares(start);

            start[i] = original - h;
            double down = SumOfTanhSquares(start);

            start[i] = original;

            double numeric = (up - down) / (2.0 * h);
            Assert.True(Math.Abs(x.Gradient![i] - numeric) < 1e-7,
                $"at {i}: autodiff {x.Gradient[i]:G17}, central difference {numeric:G17}");
        }
    }

    private static double SumOfTanhSquares(Tensor<double> values)
    {
        double total = 0.0;
        for (int i = 0; i < values.Length; i++) total += Math.Tanh(values[i] * values[i]);

        return total;
    }

    [Fact]
    public void Backward_AcceptsASeedOtherThanOnes()
    {
        var x = TensorOperations<double>.Variable(TensorOf(3.0, -1.0), "x");
        var f = TensorOperations<double>.ElementwiseMultiply(x, x);

        // Seeding with 2 rather than 1 asks for twice the derivative: d(x^2)/dx = 2x, so 4x here.
        f.Backward(TensorOf(2.0, 2.0));

        Assert.Equal(4.0 * 3.0, x.Gradient![0], 12);
        Assert.Equal(4.0 * -1.0, x.Gradient[1], 12);
    }

    [Fact]
    public void Backward_AccumulatesUntilGradientsAreCleared()
    {
        var x = TensorOperations<double>.Variable(TensorOf(2.0), "x");
        var f = TensorOperations<double>.ElementwiseMultiply(x, x);

        f.Backward();
        double once = x.Gradient![0];

        f.Backward();
        double twice = x.Gradient[0];

        // d(x^2)/dx = 2x = 4 at x = 2, so each pass contributes 4 and two passes leave 8.
        // Accumulation on the LEAVES is the documented behaviour, because it is what makes
        // gradient accumulation across micro-batches work. The SEED is not accumulated: were it,
        // the second pass would seed with 2 and leave 12, and passes would grow quadratically.
        Assert.Equal(4.0, once, 12);
        Assert.Equal(8.0, twice, 12);

        f.ZeroGradientRecursive();
        f.Backward();
        Assert.Equal(4.0, x.Gradient[0], 12);
    }

    [Fact]
    public void Backward_RejectsASeedOfTheWrongShape()
    {
        var x = TensorOperations<double>.Variable(TensorOf(1.0, 2.0, 3.0), "x");
        var f = TensorOperations<double>.Exp(x);

        var wrong = Assert.Throws<ArgumentException>(() => f.Backward(TensorOf(1.0, 1.0)));
        Assert.Contains("same shape", wrong.Message);
    }
}
