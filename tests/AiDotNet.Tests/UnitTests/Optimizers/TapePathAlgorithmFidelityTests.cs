using System;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Guards the optimizers that silently ran plain SGD whenever they were used to train a neural network.
/// </summary>
/// <remarks>
/// <para>
/// <c>Step(TapeStepContext)</c> calls <c>UpdateParameters</c>. Several optimizers never overrode it, so that
/// resolved to <c>GradientBasedOptimizerBase</c>'s default <c>theta -= lr * g</c> — plain gradient descent.
/// Their real algorithms lived only in <c>Optimize()</c>'s flat-vector loop, which neural-network training
/// never enters. Each of them therefore converged, reported progress, and was not the algorithm requested.
/// </para>
/// <para>
/// Every test here is written to FAIL against plain SGD. That is the point: the previous behaviour looked
/// entirely reasonable, so an assertion that merely checks "loss goes down" would have passed throughout.
/// </para>
/// </remarks>
public class TapePathAlgorithmFidelityTests
{
    private const double Lr = 0.1;

    // ── CoordinateDescent ────────────────────────────────────────────────────

    /// <summary>
    /// CoordinateDescent carries per-coordinate momentum. Under plain SGD a constant gradient produces a
    /// constant step forever; with momentum the steps grow and then level off, so the second step must be
    /// strictly larger than the first.
    /// </summary>
    [Fact]
    public void CoordinateDescent_AppliesPerCoordinateMomentum_NotAConstantSgdStep()
    {
        var optimizer = new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(
            null, new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = Lr,
                InitialMomentum = 0.9,
                UseAdaptiveLearningRate = false,
            });

        var p0 = new Vector<double>(new[] { 0.0 });
        var g = new Vector<double>(new[] { 1.0 });

        var p1 = optimizer.UpdateParameters(p0, g);
        var p2 = optimizer.UpdateParameters(p1, g);

        double step1 = Math.Abs(p1[0] - p0[0]);
        double step2 = Math.Abs(p2[0] - p1[0]);

        // update_i = -(lr*g + m*prevUpdate) => step1 = lr, step2 = lr + 0.9*lr
        Assert.Equal(Lr, step1, 12);
        Assert.Equal(Lr + 0.9 * Lr, step2, 12);
        Assert.True(step2 > step1,
            "second step did not exceed the first — per-coordinate momentum is not being applied (plain SGD).");
    }

    /// <summary>
    /// The update must be exactly the recurrence <c>Optimize()</c> uses, so both paths agree.
    /// </summary>
    [Fact]
    public void CoordinateDescent_MatchesTheSweepRecurrenceExactly()
    {
        const double m = 0.5;
        var optimizer = new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(
            null, new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = Lr,
                InitialMomentum = m,
                UseAdaptiveLearningRate = false,
            });

        var p = new Vector<double>(new[] { 1.0, -2.0 });
        var g1 = new Vector<double>(new[] { 0.4, -0.8 });
        var g2 = new Vector<double>(new[] { 0.2, 0.6 });

        var after1 = optimizer.UpdateParameters(p, g1);
        var after2 = optimizer.UpdateParameters(after1, g2);

        // CalculateUpdate STORES the un-negated accumulator and returns its negation:
        //   stored_t  = lr*g_t + m*stored_{t-1}
        //   applied_t = -stored_t
        // so the momentum term keeps its sign inside the accumulator rather than alternating.
        for (int i = 0; i < p.Length; i++)
        {
            double stored1 = Lr * g1[i];
            double stored2 = Lr * g2[i] + m * stored1;
            Assert.Equal(p[i] - stored1, after1[i], 12);
            Assert.Equal(after1[i] - stored2, after2[i], 12);
        }
    }

    // ── ADMM ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// ADMM's x-update carries the augmented-Lagrangian coupling term rho*(x - z + u). With a zero gradient,
    /// plain SGD would not move at all; ADMM still must, because the coupling to the split variable is
    /// non-zero once z and u have been updated.
    /// </summary>
    [Fact]
    public void Admm_MovesOnAZeroGradient_BecauseOfTheCouplingTerm()
    {
        var optimizer = new ADMMOptimizer<double, Matrix<double>, Vector<double>>(
            null, new ADMMOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = Lr,
                Rho = 1.0,
                UseAdaptiveLearningRate = false,
            });

        var p = new Vector<double>(new[] { 1.0, -1.0 });
        var zeroGradient = new Vector<double>(new[] { 0.0, 0.0 });

        // First step seeds z and u from x; the second must then feel the coupling.
        var after1 = optimizer.UpdateParameters(p, zeroGradient);
        var after2 = optimizer.UpdateParameters(after1, zeroGradient);

        double moved = 0;
        for (int i = 0; i < p.Length; i++) moved = Math.Max(moved, Math.Abs(after2[i] - after1[i]));

        Assert.True(moved > 1e-12,
            "ADMM did not move on a zero gradient — the rho*(x - z + u) coupling is absent, i.e. this is a plain gradient step.");

        foreach (var v in after2)
        {
            Assert.False(double.IsNaN(v));
            Assert.False(double.IsInfinity(v));
        }
    }

    /// <summary>
    /// The dual variable must actually accumulate, which is what makes it dual ASCENT rather than a one-off
    /// projection.
    /// </summary>
    [Fact]
    public void Admm_DualVariableAccumulatesAcrossSteps()
    {
        var optimizer = new ADMMOptimizer<double, Matrix<double>, Vector<double>>(
            null, new ADMMOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = Lr,
                Rho = 2.0,
                UseAdaptiveLearningRate = false,
            });

        var p = new Vector<double>(new[] { 0.7 });
        var g = new Vector<double>(new[] { 0.3 });

        var prev = p;
        var deltas = new double[4];
        for (int s = 0; s < 4; s++)
        {
            var next = optimizer.UpdateParameters(prev, g);
            deltas[s] = next[0] - prev[0];
            prev = next;
        }

        // Under plain SGD every delta would be identical (-lr*g). The dual/coupling terms make them differ.
        bool allEqual = true;
        for (int s = 1; s < deltas.Length; s++)
            if (Math.Abs(deltas[s] - deltas[0]) > 1e-12) allEqual = false;

        Assert.False(allEqual,
            "every ADMM step was identical, which is the signature of a plain -lr*g update with no dual variable.");
    }

    // ── LevenbergMarquardt ───────────────────────────────────────────────────

    /// <summary>
    /// LM cannot be expressed from a gradient alone, so the tape path must refuse rather than substitute.
    /// </summary>
    /// <remarks>
    /// LM solves <c>(J^T J + lambda*diag) delta = J^T r</c>. A tape supplies only <c>grad L = J^T r</c>, and J
    /// is not recoverable from that product — the residual dimension is collapsed. Previously this method
    /// silently ran <c>theta -= lr * g</c>, handing back plain gradient descent under the LM name.
    /// </remarks>
    [Fact]
    public void LevenbergMarquardt_RefusesTheTapePath_RatherThanSilentlyRunningSgd()
    {
        var optimizer = new LevenbergMarquardtOptimizer<double, Matrix<double>, Vector<double>>(
            null, new LevenbergMarquardtOptimizerOptions<double, Matrix<double>, Vector<double>>());

        var ex = Assert.Throws<NotSupportedException>(
            () => optimizer.Step(null!));

        Assert.Contains("Jacobian", ex.Message);
        // The message must point somewhere useful, not just refuse.
        Assert.Contains("Optimize()", ex.Message);
    }
}
