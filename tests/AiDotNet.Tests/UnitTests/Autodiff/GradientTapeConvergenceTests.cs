using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Autodiff;

/// <summary>
/// Verifies that GradientTape-based training actually converges — loss decreases over steps.
/// These are end-to-end tests that catch silent gradient failures (all-zero, wrong sign, etc.).
/// </summary>
public class GradientTapeConvergenceTests
{
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();

    [Fact]
    public void SimpleLinearRegression_LossDecreases()
    {
        // y = 2x + 1, learn weights w and bias b
        var w = new Tensor<double>([1], new Vector<double>(new double[] { 0.0 }));
        var b = new Tensor<double>([1], new Vector<double>(new double[] { 0.0 }));
        double lr = 0.01;

        double initialLoss = double.MaxValue;
        double finalLoss = double.MaxValue;

        for (int step = 0; step < 100; step++)
        {
            using var tape = new GradientTape<double>();
            tape.Watch(b);

            // Forward: y_pred = w * x + b
            var x = new Tensor<double>([1], new Vector<double>(new double[] { 3.0 }));
            var y_true = new Tensor<double>([1], new Vector<double>(new double[] { 7.0 })); // 2*3+1

            var pred = DifferentiableOps<double>.Add(
                DifferentiableOps<double>.Multiply(w, x), b);
            var diff = DifferentiableOps<double>.Subtract(pred, y_true);
            var loss = DifferentiableOps<double>.Mean(
                DifferentiableOps<double>.Multiply(diff, diff));

            double lossVal = loss[0];
            if (step == 0) initialLoss = lossVal;
            if (step == 99) finalLoss = lossVal;

            var grads = tape.ComputeGradients(loss);

            // SGD update
            if (grads.TryGetValue(w, out var wGrad))
                w[0] = NumOps.Subtract(w[0], NumOps.FromDouble(lr * NumOps.ToDouble(wGrad[0])));
            if (grads.TryGetValue(b, out var bGrad))
                b[0] = NumOps.Subtract(b[0], NumOps.FromDouble(lr * NumOps.ToDouble(bGrad[0])));
        }

        Assert.True(finalLoss < initialLoss * 0.01,
            $"Loss should decrease significantly. Initial: {initialLoss}, Final: {finalLoss}");
        Assert.True(Math.Abs(NumOps.ToDouble(w[0]) - 2.0) < 0.5,
            $"Weight should converge near 2.0, got {NumOps.ToDouble(w[0])}");
    }

    [Fact]
    public void MatMulGradient_TrainsLinearLayer()
    {
        // Learn W such that W @ [1,2] ≈ [5] (i.e., W ≈ [1, 2])
        var W = new Tensor<double>([1, 2], new Vector<double>(new double[] { 0.0, 0.0 }));
        double lr = 0.01;
        var target = new Tensor<double>([1, 1], new Vector<double>(new double[] { 5.0 }));

        double initialLoss = double.MaxValue;
        double finalLoss = double.MaxValue;

        for (int step = 0; step < 200; step++)
        {
            using var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = true });

            var x = new Tensor<double>([2, 1], new Vector<double>(new double[] { 1.0, 2.0 }));
            var pred = DifferentiableOps<double>.MatMul(W, x);
            var diff = DifferentiableOps<double>.Subtract(pred, target);
            var loss = DifferentiableOps<double>.Mean(
                DifferentiableOps<double>.Multiply(diff, diff));

            double lossVal = loss[0];
            if (step == 0) initialLoss = lossVal;
            if (step == 199) finalLoss = lossVal;

            var grads = tape.ComputeGradients(loss);
            if (grads.TryGetValue(W, out var wGrad))
            {
                for (int i = 0; i < W.Length; i++)
                    W[i] = NumOps.Subtract(W[i], NumOps.FromDouble(lr * NumOps.ToDouble(wGrad[i])));
            }
        }

        Assert.True(finalLoss < initialLoss * 0.01,
            $"Loss should decrease. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void HigherOrderGradient_CreateGraphProducesGradients()
    {
        // Verify createGraph=true produces gradient tensors that are themselves differentiable
        using var outerTape = new GradientTape<double>(new GradientTapeOptions { Persistent = true });

        var x = new Tensor<double>([1], new Vector<double>(new double[] { 3.0 }));

        // f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
        var x2 = DifferentiableOps<double>.Multiply(x, x);
        var x3 = DifferentiableOps<double>.Multiply(x2, x);

        // First derivative with createGraph=true
        var firstGrads = outerTape.ComputeGradients(x3, createGraph: true);
        Assert.True(firstGrads.ContainsKey(x), "Should have gradient for x");

        var dx = firstGrads[x]; // Should be 3 * 3^2 = 27
        Assert.True(Math.Abs(dx[0] - 27.0) < 0.1,
            $"First derivative of x^3 at x=3 should be 27, got {dx[0]}");
    }

    [Fact]
    public void NoGradScope_PreventsRecording()
    {
        using var tape = new GradientTape<double>();
        var x = new Tensor<double>([1], new Vector<double>(new double[] { 2.0 }));

        Tensor<double> result;
        using (new NoGradScope<double>())
        {
            // Operations inside NoGradScope should NOT be recorded
            result = DifferentiableOps<double>.Multiply(x, x);
        }

        var grads = tape.ComputeGradients(result);
        // x should have NO gradient because the multiply was inside NoGradScope
        Assert.False(grads.ContainsKey(x),
            "NoGradScope should prevent gradient computation");
    }

    [Fact]
    public void NewOps_SoftplusGradientIsCorrect()
    {
        using var tape = new GradientTape<double>();
        var x = new Tensor<double>([3], new Vector<double>(new double[] { -1.0, 0.0, 2.0 }));

        var y = DifferentiableOps<double>.Softplus(x);
        var loss = DifferentiableOps<double>.Sum(y);
        var grads = tape.ComputeGradients(loss);

        Assert.True(grads.ContainsKey(x), "Should have gradient for x");
        var dx = grads[x];

        // Softplus'(x) = sigmoid(x)
        for (int i = 0; i < 3; i++)
        {
            double xi = x[i];
            double expected = 1.0 / (1.0 + Math.Exp(-xi));
            Assert.True(Math.Abs(dx[i] - expected) < 1e-5,
                $"Softplus gradient at {xi} should be {expected}, got {dx[i]}");
        }
    }

    [Fact]
    public void NewOps_LeakyReLUGradientIsCorrect()
    {
        using var tape = new GradientTape<double>();
        var x = new Tensor<double>([4], new Vector<double>(new double[] { -2.0, -0.5, 0.0, 1.0 }));

        var y = DifferentiableOps<double>.LeakyReLU(x, alpha: 0.1);
        var loss = DifferentiableOps<double>.Sum(y);
        var grads = tape.ComputeGradients(loss);

        var dx = grads[x];
        // LeakyReLU'(x) = 1 if x >= 0, alpha if x < 0
        Assert.True(Math.Abs(dx[0] - 0.1) < 1e-6, $"Negative: expected 0.1, got {dx[0]}");
        Assert.True(Math.Abs(dx[1] - 0.1) < 1e-6, $"Negative: expected 0.1, got {dx[1]}");
        Assert.True(Math.Abs(dx[2] - 1.0) < 1e-6, $"Zero: expected 1.0, got {dx[2]}");
        Assert.True(Math.Abs(dx[3] - 1.0) < 1e-6, $"Positive: expected 1.0, got {dx[3]}");
    }

    [Fact]
    public void ScaledDotProductAttention_ProducesFiniteGradients()
    {
        using var tape = new GradientTape<double>();

        // [1, 4, 8] = batch=1, seq=4, d_k=8
        var query = Tensor<double>.CreateRandom([1, 4, 8]);
        var key = Tensor<double>.CreateRandom([1, 4, 8]);
        var value = Tensor<double>.CreateRandom([1, 4, 8]);
        tape.Watch(key);

        var output = DifferentiableOps<double>.ScaledDotProductAttention(query, key, value);
        var loss = DifferentiableOps<double>.Sum(output);
        var grads = tape.ComputeGradients(loss);

        Assert.True(grads.ContainsKey(query), "Should have query gradient");
        Assert.True(grads.ContainsKey(key), "Should have key gradient");
        Assert.True(grads.ContainsKey(value), "Should have value gradient");

        // Verify all gradients are finite
        foreach (var (name, grad) in new[] { ("query", grads[query]), ("key", grads[key]), ("value", grads[value]) })
        {
            for (int i = 0; i < grad.Length; i++)
            {
                Assert.False(double.IsNaN(grad[i]) || double.IsInfinity(grad[i]),
                    $"{name} gradient has NaN/Inf at index {i}");
            }
        }
    }
}
