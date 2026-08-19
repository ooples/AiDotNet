using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Covers what the second-order optimizers do with a step once they can measure whether it helped.
/// </summary>
/// <remarks>
/// <para>
/// L-BFGS, BFGS, Newton and TrustRegion each carried the same copy-pasted block: re-evaluate the loss and,
/// if it had gone UP, compute and apply a SECOND full step from the new, worse point. The comments above it
/// said "line search" and "adjust the trust region radius". It was neither, and it is not any published
/// method — a direction that overshot does not get better by being followed again from further along.
/// </para>
/// <para>
/// These tests pin the replacements: backtracking on the Armijo condition for the quasi-Newton methods
/// (Nocedal &amp; Wright, Algorithm 3.1) and the actual-to-predicted ratio test for trust region
/// (Algorithm 4.1). Each is written so that the old behaviour would fail it, not merely score worse.
/// </para>
/// </remarks>
public class TapeStepAcceptanceTests
{
    /// <summary>
    /// Builds a context around f(w) = Σ(w − target)², whose minimum is at <c>target</c> and whose gradient
    /// is 2(w − target).
    /// </summary>
    /// <remarks>
    /// The forward function returns the parameter tensor itself, so the loss is a plain quadratic in the
    /// parameters and every quantity the optimizers compute has a closed form to compare against.
    /// </remarks>
    private static TapeStepContext<double> CreateQuadraticContext(
        Tensor<double> parameter, Tensor<double> target)
    {
        var engine = AiDotNetEngine.Current;

        Tensor<double> Forward(Tensor<double> _, Tensor<double> __) => parameter;
        Tensor<double> Loss(Tensor<double> prediction, Tensor<double> tgt)
        {
            var diff = engine.TensorSubtract(prediction, tgt);
            return engine.ReduceSum(engine.TensorMultiply(diff, diff));
        }

        // Seed the initial gradient and loss the way a training step would have.
        double loss = 0.0;
        var gradientTensor = new Tensor<double>(parameter._shape);
        for (int i = 0; i < parameter.Length; i++)
        {
            double d = parameter[i] - target[i];
            gradientTensor[i] = 2.0 * d;
            loss += d * d;
        }

        return new TapeStepContext<double>(
            new[] { parameter },
            new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance)
            {
                [parameter] = gradientTensor,
            },
            loss,
            target,
            target,
            Forward,
            Loss);
    }

    /// <summary>
    /// A first L-BFGS step long enough to overshoot must be shortened, not repeated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// On the first step there is no curvature history, so the direction is −g and the full step is
    /// w − lr·2(w − t). At w = 1, t = 0 and lr = 2 that lands at −3: three times as far from the minimum,
    /// nine times the loss. Backtracking halves twice — the first halving lands at −1, which is symmetric
    /// and therefore not a sufficient decrease — and accepts an Armijo step scale of t = 0.25. That is the
    /// SCALE, not the resulting parameter: the trial point is 1 + 0.25*(−4) = 0, exactly the minimum, which
    /// is what the assertion below checks.
    /// </para>
    /// <para>
    /// The replaced code would have taken a second full step from −3, reaching +9.
    /// </para>
    /// </remarks>
    [Fact]
    public void Lbfgs_LineSearch_ShortensAnOvershootingStepInsteadOfRepeatingIt()
    {
        var parameter = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var target = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 0.0 }));
        var context = CreateQuadraticContext(parameter, target);

        var optimizer = new LBFGSOptimizer<double, Matrix<double>, Vector<double>>(
            null,
            new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 2.0,
                UseLineSearch = true,
                UseAdaptiveLearningRate = false,
            });

        optimizer.Step(context);

        Assert.Equal(0.0, parameter[0], 9);

        // The full step and the rejected "step again" outcome, spelled out so this cannot pass vacuously.
        Assert.NotEqual(-3.0, parameter[0], 3);
        Assert.NotEqual(9.0, parameter[0], 3);
    }

    /// <summary>
    /// With the line search off, the same configuration takes the full step — which is the behaviour the
    /// fused kernel reproduces.
    /// </summary>
    [Fact]
    public void Lbfgs_WithoutLineSearch_TakesTheFullStep()
    {
        var parameter = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var target = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 0.0 }));
        var context = CreateQuadraticContext(parameter, target);

        var optimizer = new LBFGSOptimizer<double, Matrix<double>, Vector<double>>(
            null,
            new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 2.0,
                UseLineSearch = false,
                UseAdaptiveLearningRate = false,
            });

        optimizer.Step(context);

        // w - lr*g = 1 - 2*2 = -3.
        Assert.Equal(-3.0, parameter[0], 9);
    }

    /// <summary>
    /// A step the line search cannot rescue is rejected, leaving the parameters where they were.
    /// </summary>
    /// <remarks>
    /// Not stepping is always available and is never worse than stepping somewhere measured to be worse.
    /// With <c>MaxLineSearchIterations = 1</c> the search gets a single halving, which at lr = 2 lands on
    /// the symmetric point −1 — equal loss, not less — so the step is refused.
    /// </remarks>
    [Fact]
    public void Lbfgs_LineSearch_RejectsAStepItCannotRescue()
    {
        var parameter = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var target = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 0.0 }));
        var context = CreateQuadraticContext(parameter, target);

        var optimizer = new LBFGSOptimizer<double, Matrix<double>, Vector<double>>(
            null,
            new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 2.0,
                UseLineSearch = true,
                MaxLineSearchIterations = 1,
                UseAdaptiveLearningRate = false,
            });

        optimizer.Step(context);

        Assert.Equal(1.0, parameter[0], 9);
    }

    /// <summary>
    /// A trust-region step that increases the loss is rejected and the region shrinks — which is the whole
    /// mechanism by which the method recovers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// At w = 1, t = 0 the gradient is 2 and ‖g‖ = 2. With radius 3 and lr 2 the Cauchy step is
    /// α = min(3/2, 2) = 1.5, so Δ = −3 and the trial point is −2: loss 4 against 1. The predicted
    /// reduction is −(gᵀΔ + ½‖Δ‖²) = −(−6 + 4.5) = 1.5, so ρ = −3/1.5 = −2 — negative, and the step is
    /// refused.
    /// </para>
    /// <para>
    /// The replaced code took a SECOND full step from −2 instead, moving further away every time the
    /// objective got worse.
    /// </para>
    /// </remarks>
    [Fact]
    public void TrustRegion_RejectsAStepThatIncreasesTheLoss()
    {
        var parameter = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var target = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 0.0 }));
        var context = CreateQuadraticContext(parameter, target);

        var optimizer = new TrustRegionOptimizer<double, Matrix<double>, Vector<double>>(
            null,
            new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 2.0,
                InitialTrustRegionRadius = 3.0,
                AdaptTrustRegionRadius = true,
                UseAdaptiveLearningRate = false,
            });

        optimizer.Step(context);

        Assert.Equal(1.0, parameter[0], 9);
        Assert.NotEqual(-2.0, parameter[0], 3);
    }

    /// <summary>
    /// With adaptation off the same step is taken as-is, which is the fixed-radius form the fused kernel
    /// runs.
    /// </summary>
    [Fact]
    public void TrustRegion_WithFixedRadius_TakesTheCauchyStepAsIs()
    {
        var parameter = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var target = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 0.0 }));
        var context = CreateQuadraticContext(parameter, target);

        var optimizer = new TrustRegionOptimizer<double, Matrix<double>, Vector<double>>(
            null,
            new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 2.0,
                InitialTrustRegionRadius = 3.0,
                AdaptTrustRegionRadius = false,
                UseAdaptiveLearningRate = false,
            });

        optimizer.Step(context);

        // alpha = min(3/2, 2) = 1.5, so w = 1 - 1.5*2 = -2.
        Assert.Equal(-2.0, parameter[0], 9);

        // And the radius really is fixed. UpdateParameters adapts it unconditionally from its
        // gradient-norm proxy, so the non-adapting path has to RESTORE the pre-step value rather than just
        // skip the ratio test — otherwise the radius drifts with adaptation switched off, and the fused
        // spec's promise of a constant InitialTrustRegionRadius stops being true after one step.
        Assert.Equal(3.0, GetTrustRegionRadius(optimizer), 12);

        // A second step must not move it either.
        var context2 = CreateQuadraticContext(parameter, target);
        optimizer.Step(context2);
        Assert.Equal(3.0, GetTrustRegionRadius(optimizer), 12);
    }

    private static double GetTrustRegionRadius(
        TrustRegionOptimizer<double, Matrix<double>, Vector<double>> optimizer)
    {
        var field = typeof(TrustRegionOptimizer<double, Matrix<double>, Vector<double>>).GetField(
            "_trustRegionRadius",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(field);
        return (double)field!.GetValue(optimizer)!;
    }

    /// <summary>
    /// A trust-region step that reduces the loss is kept.
    /// </summary>
    /// <remarks>
    /// The rejection path must not be so eager that ordinary progress gets thrown away. At radius 1 the
    /// Cauchy step is α = min(1/2, 2) = 0.5, so Δ = −1 lands exactly on the minimum, ρ = 1, and the step
    /// stands.
    /// </remarks>
    [Fact]
    public void TrustRegion_KeepsAStepThatReducesTheLoss()
    {
        var parameter = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var target = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 0.0 }));
        var context = CreateQuadraticContext(parameter, target);

        var optimizer = new TrustRegionOptimizer<double, Matrix<double>, Vector<double>>(
            null,
            new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 2.0,
                InitialTrustRegionRadius = 1.0,
                AdaptTrustRegionRadius = true,
                UseAdaptiveLearningRate = false,
            });

        optimizer.Step(context);

        Assert.Equal(0.0, parameter[0], 9);
    }
}
