using System;
using System.Collections.Generic;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// The tape Newton-CG step (Nocedal and Wright, Algorithm 7.1) on a quadratic whose Hessian is known.
/// </summary>
public class NewtonCgTapeStepTests
{
    // L(w) = 0.5 * sum(a_i w_i^2) - sum(b_i w_i): gradient a*w - b, Hessian diag(a), minimiser b / a.
    private static readonly double[] A = { 2.0, 5.0 };
    private static readonly double[] B = { 1.0, -3.0 };

    private static TapeStepContext<double> QuadraticContext(Tensor<double> w, double[] a)
    {
        var engine = AiDotNetEngine.Current;
        var aT = new Tensor<double>(new[] { 2 }, new Vector<double>(a));
        var bT = new Tensor<double>(new[] { 2 }, new Vector<double>(B));
        var grad = new Tensor<double>(new[] { 2 });
        for (int i = 0; i < 2; i++) grad[i] = a[i] * w[i] - B[i];
        return new TapeStepContext<double>(
            parameters: new[] { w },
            gradients: new Dictionary<Tensor<double>, Tensor<double>> { [w] = grad },
            loss: 0.0,
            input: new Tensor<double>(new[] { 1 }),
            target: new Tensor<double>(new[] { 1 }),
            forwardFn: (_, _) => w,
            lossFn: (pred, _) =>
            {
                var quad = engine.TensorMultiplyScalar(engine.TensorMultiply(aT, engine.TensorMultiply(pred, pred)), 0.5);
                var lin = engine.TensorMultiply(bT, pred);
                return engine.ReduceSum(engine.TensorSubtract(quad, lin), new[] { 0 }, keepDims: false);
            });
    }

    [Fact]
    public void Step_OnAQuadratic_LandsOnTheMinimiserInOneFullNewtonStep()
    {
        var optimizer = new NewtonMethodOptimizer<double, Tensor<double>, Tensor<double>>(
            null!, new NewtonMethodOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1.0 });
        var w = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.7, 0.4 }));

        optimizer.Step(QuadraticContext(w, A));

        // CG on a 2-dimensional positive-definite system is exact in two iterations.
        for (int i = 0; i < 2; i++)
            Assert.True(Math.Abs(w[i] - B[i] / A[i]) < 1e-9, $"w[{i}] = {w[i]:R}, minimiser {B[i] / A[i]:R}");
    }

    [Fact]
    public void Step_WithNegativeCurvatureAtTheStart_TakesTheSteepestDescentStep()
    {
        var optimizer = new NewtonMethodOptimizer<double, Tensor<double>, Tensor<double>>(
            null!, new NewtonMethodOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.1 });
        var w = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.7, 0.4 }));
        double[] concave = { -2.0, -5.0 };
        var expected = new double[2];
        for (int i = 0; i < 2; i++) expected[i] = w[i] - 0.1 * (concave[i] * w[i] - B[i]);

        optimizer.Step(QuadraticContext(w, concave));

        for (int i = 0; i < 2; i++)
            Assert.True(Math.Abs(w[i] - expected[i]) < 1e-12, $"w[{i}] = {w[i]:R}, steepest descent {expected[i]:R}");
    }
}
