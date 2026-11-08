using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Conjugate gradient and DFP produce a direction, and their training step (<c>Train</c>, the tape step) had no line
/// search: a fixed-rate step along the direction. On <c>f(x) = 10x^2</c> from <c>x = 1</c> at the default rate that step
/// overshoots. With gradient clipping off by default for these methods (it changes the function their curvature
/// describes), the tape step now backtracks on the Armijo condition, as BFGS and L-BFGS do.
/// </summary>
public class FixedRateQuasiNewtonLineSearchTests
{
    private readonly ITestOutputHelper _output;

    public FixedRateQuasiNewtonLineSearchTests(ITestOutputHelper output) => _output = output;

    private const double Curvature = 10.0;

    // f(w) = 10 w^2 with its gradient 20 w seeded, as a training step would have it.
    private static TapeStepContext<double> Context(Tensor<double> parameter, double gradientSign = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<double> Forward(Tensor<double> _, Tensor<double> __) => parameter;
        Tensor<double> Loss(Tensor<double> prediction, Tensor<double> ___)
            => engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(prediction, prediction)), Curvature);

        double w = parameter[0];
        var gradient = new Tensor<double>(parameter._shape);
        gradient[0] = gradientSign * 2.0 * Curvature * w;
        var unused = new Tensor<double>(new[] { 1 });
        return new TapeStepContext<double>(
            new[] { parameter },
            new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance) { [parameter] = gradient },
            Curvature * w * w,
            unused,
            unused,
            Forward,
            Loss);
    }

    private static IGradientBasedOptimizer<double, Matrix<double>, Vector<double>> Build(string method, bool lineSearch) => method switch
    {
        "CG" => new ConjugateGradientOptimizer<double, Matrix<double>, Vector<double>>(
            null, new ConjugateGradientOptimizerOptions<double, Matrix<double>, Vector<double>> { UseLineSearch = lineSearch }),
        "DFP" => new DFPOptimizer<double, Matrix<double>, Vector<double>>(
            null, new DFPOptimizerOptions<double, Matrix<double>, Vector<double>> { UseLineSearch = lineSearch }),
        _ => throw new ArgumentOutOfRangeException(nameof(method)),
    };

    // The loss after each of `steps` training steps from x = 1, at the method's default options.
    private double[] Losses(string method, bool lineSearch, int steps)
    {
        var optimizer = Build(method, lineSearch);
        var parameter = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var losses = new double[steps];
        for (int step = 0; step < steps; step++)
        {
            optimizer.Step(Context(parameter));
            losses[step] = Curvature * parameter[0] * parameter[0];
            _output.WriteLine($"{method} line search {lineSearch}: step {step + 1} x = {parameter[0]:G6}, loss {losses[step]:G6}");
        }

        return losses;
    }

    /// <summary>
    /// Without the line search the default step does not lower the loss (the overshoot is real, so the test below is
    /// not vacuous); with it, every step lowers the loss.
    /// </summary>
    [Theory]
    [InlineData("CG")]
    [InlineData("DFP")]
    public void TapeStep_BacktracksAnOvershootingStep(string method)
    {
        const int steps = 5;
        var fixedRate = Losses(method, lineSearch: false, steps);
        Assert.True(fixedRate[0] >= Curvature, $"{method}: the fixed-rate first step lowered the loss to {fixedRate[0]:G6}");

        var searched = Losses(method, lineSearch: true, steps);
        double previous = Curvature;
        for (int step = 0; step < steps; step++)
        {
            Assert.True(searched[step] <= previous,
                $"{method}: step {step + 1} raised the loss from {previous:G6} to {searched[step]:G6}");
            previous = searched[step];
        }

        Assert.True(searched[0] < Curvature, $"{method}: the first backtracked step did not lower the loss ({searched[0]:G6})");
    }

    /// <summary>
    /// A step the line search rejects (here an uphill gradient, so every halving raises the loss) restores the
    /// parameters, so its direction was never taken. The next step must start again from steepest descent, as a fresh
    /// optimizer's does. Conjugating against the rejected direction instead (beta = 1 here, the previous direction
    /// +20 against a gradient of 20) gives a zero direction, and the parameter does not move.
    /// </summary>
    [Fact]
    public void ConjugateGradient_RejectedStep_RestartsFromSteepestDescent()
    {
        var rejected = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var optimizer = Build("CG", lineSearch: true);
        optimizer.Step(Context(rejected, gradientSign: -1.0));
        Assert.Equal(1.0, rejected[0]);
        optimizer.Step(Context(rejected));

        var fresh = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        Build("CG", lineSearch: true).Step(Context(fresh));

        _output.WriteLine($"after a rejected step x = {rejected[0]:G6}; a fresh optimizer's first step x = {fresh[0]:G6}");
        Assert.Equal(fresh[0], rejected[0]);
        Assert.NotEqual(1.0, rejected[0]);
    }
}
