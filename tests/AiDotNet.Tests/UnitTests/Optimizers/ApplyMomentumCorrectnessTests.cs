using System;
using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Guards the classical-momentum recurrence in <c>GradientBasedOptimizerBase.ApplyMomentum</c> (#2008).
/// </summary>
/// <remarks>
/// <para>
/// The defect was <c>v ← v + μ·g</c> instead of <c>v ← μ·v + g</c> — the coefficient on the wrong term,
/// so nothing decayed the velocity and it became an unbounded running sum. That is invisible in a
/// smoke test: the optimizer still "trains", it just takes ever-larger steps the longer it runs.
/// </para>
/// <para>
/// So these assert the ANALYTIC property that distinguishes the two rules. Under a constant gradient
/// <c>g</c>, correct momentum converges to <c>g/(1−μ)</c>; the buggy form grows without bound as
/// <c>g + μ·n·g</c>. At μ=0.9 over 200 steps that is a bounded 10·g versus roughly 180·g — the two are
/// impossible to confuse.
/// </para>
/// </remarks>
public class ApplyMomentumCorrectnessTests
{
    private const double Momentum = 0.9;

    /// <summary>
    /// Under a constant gradient the velocity must converge to g/(1−μ), not grow with the step count.
    /// </summary>
    [Fact]
    public void ApplyMomentum_UnderConstantGradient_ConvergesToBoundedLimit()
    {
        var optimizer = MakeOptimizer();
        var gradient = new Vector<double>(new[] { 1.0 });

        double last = 0.0;
        for (int step = 0; step < 200; step++)
            last = InvokeApplyMomentum(optimizer, gradient)[0];

        double expectedLimit = 1.0 / (1.0 - Momentum);   // = 10
        Assert.InRange(last, expectedLimit - 1e-6, expectedLimit + 1e-6);
    }

    /// <summary>
    /// The specific regression: the old rule grew linearly with the step count, so running longer
    /// changed the answer. Correct momentum converges, so two runs long enough to have converged agree.
    /// </summary>
    /// <remarks>
    /// Both counts are past convergence deliberately. Convergence is geometric — at μ=0.9 twenty steps
    /// is only 88% of the way there (0.9^20 ≈ 0.12), so comparing 20 against 200 would fail on a
    /// CORRECT implementation. 200 vs 400 leaves the residual at 0.9^200, far below the tolerance,
    /// while the buggy accumulating form would give roughly 180 versus 360.
    /// </remarks>
    [Fact]
    public void ApplyMomentum_DoesNotGrowWithStepCount()
    {
        var gradient = new Vector<double>(new[] { 1.0 });

        var shortRun = MakeOptimizer();
        double after200 = 0.0;
        for (int i = 0; i < 200; i++) after200 = InvokeApplyMomentum(shortRun, gradient)[0];

        var longRun = MakeOptimizer();
        double after400 = 0.0;
        for (int i = 0; i < 400; i++) after400 = InvokeApplyMomentum(longRun, gradient)[0];

        // 1e-6, not 1e-9: after 200 steps the geometric residual is still ~7e-9 (10 * 0.9^200 scaled by
        // accumulated float error), so a tighter bound fails on a CORRECT implementation. The buggy
        // form differs by ~180 here, so this bound separates them by eight orders of magnitude.
        Assert.True(Math.Abs(after400 - after200) < 1e-6,
            $"Velocity depends on step count: 200 steps -> {after200}, 400 steps -> {after400}. " +
            "Classical momentum converges; an accumulating sum does not.");
    }

    /// <summary>
    /// Parity with the fused SGDMomentum kernel's recurrence. The compiled path runs
    /// <c>v = μ·v + g; param -= lr·v</c>; the eager path must produce the same velocity sequence, or
    /// the same optimizer trains differently depending on which path is taken.
    /// </summary>
    [Fact]
    public void ApplyMomentum_MatchesTheFusedSgdMomentumRecurrence()
    {
        var optimizer = MakeOptimizer();

        // A varying gradient, so agreement cannot come from a degenerate constant sequence.
        double[] gradients = { 1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 0.0, -0.25 };

        double fusedVelocity = 0.0;
        foreach (double g in gradients)
        {
            double eager = InvokeApplyMomentum(optimizer, new Vector<double>(new[] { g }))[0];

            // Exactly what FusedOptimizer.SgdMomentumUpdateSimd does: v = mu*v + grad.
            fusedVelocity = Momentum * fusedVelocity + g;

            Assert.True(Math.Abs(eager - fusedVelocity) < 1e-9,
                $"Eager velocity {eager} diverged from the fused kernel's {fusedVelocity} at gradient {g}.");
        }
    }

    private static StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>> MakeOptimizer()
    {
        var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = Momentum,
            UseAdaptiveLearningRate = false,
        };
        return new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(null!, options);
    }

    /// <summary>
    /// ApplyMomentum is protected; reflection keeps the test on the real production method rather than
    /// re-deriving the recurrence in a subclass, which would happily pass against a wrong base class.
    /// </summary>
    private static Vector<double> InvokeApplyMomentum(object optimizer, Vector<double> gradient)
    {
        var method = optimizer.GetType().GetMethod(
            "ApplyMomentum", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        return (Vector<double>)method!.Invoke(optimizer, new object[] { gradient })!;
    }
}
