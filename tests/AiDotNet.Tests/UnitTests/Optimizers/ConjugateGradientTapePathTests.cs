using System;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins <see cref="ConjugateGradientOptimizer{T, TInput, TOutput}"/> to actually being conjugate gradient on
/// the path neural-network training uses.
/// </summary>
/// <remarks>
/// <para>
/// It was not. <c>Step(TapeStepContext)</c> calls <c>UpdateParameters</c>, and with no override that resolved
/// to <c>GradientBasedOptimizerBase</c>'s default of plain <c>theta -= lr * g</c>. The conjugate direction was
/// only ever computed inside <c>Optimize()</c>'s flat-vector loop, which neural-network training never enters
/// — so selecting this optimizer to train a network silently gave you SGD, with no error and no diagnostic.
/// </para>
/// <para>
/// The tests below are written so that a regression to plain SGD fails them. That is the whole point: the
/// previous behaviour was a perfectly reasonable-looking optimizer that simply was not the one requested.
/// </para>
/// </remarks>
public class ConjugateGradientTapePathTests
{
    private const double Lr = 0.1;

    private static ConjugateGradientOptimizer<double, Matrix<double>, Vector<double>> CreateOptimizer()
        => new(null, new ConjugateGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = Lr,
            UseAdaptiveLearningRate = false,
        });

    /// <summary>
    /// With no history, the direction is steepest descent, so the first step must equal a plain SGD step.
    /// </summary>
    /// <remarks>
    /// This is what makes the change safe for existing runs: nothing about the first update moves.
    /// </remarks>
    [Fact]
    public void FirstStepEqualsGradientDescent_BecauseTheDirectionStartsAsSteepestDescent()
    {
        var optimizer = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 1.0, -2.0, 0.5 });
        var gradient = new Vector<double>(new[] { 0.4, 0.8, -0.2 });

        var updated = optimizer.UpdateParameters(parameters, gradient);

        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(parameters[i] - Lr * gradient[i], updated[i], 12);
        }
    }

    /// <summary>
    /// The second step must fold in beta * previous direction, so it must NOT equal a plain SGD step.
    /// </summary>
    /// <remarks>
    /// Under the old behaviour every step equalled SGD, so this assertion is exactly the one that separates
    /// "is conjugate gradient" from "claims to be conjugate gradient".
    /// </remarks>
    [Fact]
    public void SecondStepDivergesFromGradientDescent_BecauseTheConjugateTermIsApplied()
    {
        var optimizer = CreateOptimizer();
        var start = new Vector<double>(new[] { 1.0, -2.0, 0.5 });
        var g1 = new Vector<double>(new[] { 0.4, 0.8, -0.2 });
        var g2 = new Vector<double>(new[] { 0.2, 0.5, -0.1 });

        var afterFirst = optimizer.UpdateParameters(start, g1);
        var afterSecond = optimizer.UpdateParameters(afterFirst, g2);

        // Fletcher-Reeves, computed independently from the paper formula.
        double beta = Dot(g2, g2) / Dot(g1, g1);
        var d1 = Negate(g1);
        var d2 = new Vector<double>(new double[g2.Length]);
        for (int i = 0; i < g2.Length; i++) d2[i] = -g2[i] + beta * d1[i];

        for (int i = 0; i < start.Length; i++)
        {
            Assert.Equal(afterFirst[i] + Lr * d2[i], afterSecond[i], 12);
        }

        // And it genuinely differs from the SGD step it used to take.
        double maxGap = 0;
        for (int i = 0; i < start.Length; i++)
            maxGap = Math.Max(maxGap, Math.Abs(afterSecond[i] - (afterFirst[i] - Lr * g2[i])));
        Assert.True(maxGap > 1e-6,
            $"Second step is indistinguishable from plain SGD (max gap {maxGap}) — the conjugate term is not being applied.");
    }

    /// <summary>
    /// A direction that stops pointing downhill must be discarded for steepest descent (Powell restart),
    /// rather than followed uphill.
    /// </summary>
    /// <remarks>
    /// Constructed so the raw Fletcher-Reeves direction has a non-negative inner product with the gradient:
    /// the gradient reverses sign while its magnitude grows, which makes beta large enough that the retained
    /// previous direction dominates and flips the result uphill.
    /// </remarks>
    [Fact]
    public void RestartsToSteepestDescent_WhenTheConjugateDirectionStopsDescending()
    {
        var optimizer = CreateOptimizer();
        var start = new Vector<double>(new[] { 0.0 });
        var g1 = new Vector<double>(new[] { 1.0 });
        var g2 = new Vector<double>(new[] { -4.0 });

        var afterFirst = optimizer.UpdateParameters(start, g1);

        // Raw FR here: beta = 16, d1 = -1, so d2 = 4 + 16*(-1) = -12, and d2.g2 = (-12)(-4) = +48 >= 0,
        // i.e. uphill. The restart must reject it and use -g2 = +4 instead.
        double beta = Dot(g2, g2) / Dot(g1, g1);
        double rawDirection = -g2[0] + beta * (-g1[0]);
        Assert.True(rawDirection * g2[0] >= 0, "fixture no longer produces an ascent direction; the test would prove nothing");

        var afterSecond = optimizer.UpdateParameters(afterFirst, g2);

        Assert.Equal(afterFirst[0] + Lr * (-g2[0]), afterSecond[0], 12);
        Assert.NotEqual(afterFirst[0] + Lr * rawDirection, afterSecond[0], 6);
    }

    /// <summary>
    /// A vanishing previous gradient must not produce a non-finite beta that poisons every later direction.
    /// </summary>
    [Fact]
    public void SurvivesAVanishingPreviousGradient()
    {
        var optimizer = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 1.0, 2.0 });

        var afterZero = optimizer.UpdateParameters(parameters, new Vector<double>(new[] { 0.0, 0.0 }));
        var afterNext = optimizer.UpdateParameters(afterZero, new Vector<double>(new[] { 0.3, -0.7 }));

        foreach (var v in afterNext)
        {
            Assert.False(double.IsNaN(v), "beta divided by a zero previous-gradient norm and produced NaN");
            Assert.False(double.IsInfinity(v), "beta divided by a zero previous-gradient norm and produced Infinity");
        }
    }

    /// <summary>
    /// On a quadratic where CG is supposed to shine, it must reach a lower objective than plain SGD given the
    /// same learning rate and step count.
    /// </summary>
    /// <remarks>
    /// An end-to-end check that the recurrence is wired the right way round. A sign error in the direction
    /// would still pass the algebraic tests above by construction, but would lose to SGD here.
    /// </remarks>
    [Fact]
    public void OutperformsGradientDescentOnAnIllConditionedQuadratic()
    {
        // f(x) = 0.5 * sum(c_i * x_i^2), gradient = c_i * x_i. Curvature spread 1:50 is where the conjugate
        // term earns its keep and plain SGD zig-zags.
        double[] c = { 1.0, 10.0, 50.0 };
        var start = new[] { 1.0, 1.0, 1.0 };
        const int steps = 40;
        const double lr = 0.01;

        var cg = new ConjugateGradientOptimizer<double, Matrix<double>, Vector<double>>(
            null, new ConjugateGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = lr,
                UseAdaptiveLearningRate = false,
            });

        var cgX = new Vector<double>((double[])start.Clone());
        for (int s = 0; s < steps; s++)
        {
            var g = new Vector<double>(new double[c.Length]);
            for (int i = 0; i < c.Length; i++) g[i] = c[i] * cgX[i];
            cgX = cg.UpdateParameters(cgX, g);
        }

        var sgdX = new Vector<double>((double[])start.Clone());
        for (int s = 0; s < steps; s++)
        {
            for (int i = 0; i < c.Length; i++) sgdX[i] -= lr * (c[i] * sgdX[i]);
        }

        double cgLoss = Objective(c, cgX), sgdLoss = Objective(c, sgdX);
        Assert.True(cgLoss < sgdLoss,
            $"Conjugate gradient ({cgLoss:G6}) did not beat plain SGD ({sgdLoss:G6}) on an ill-conditioned quadratic.");
    }

    private static double Objective(double[] c, Vector<double> x)
    {
        double sum = 0;
        for (int i = 0; i < c.Length; i++) sum += 0.5 * c[i] * x[i] * x[i];
        return sum;
    }

    private static double Dot(Vector<double> a, Vector<double> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static Vector<double> Negate(Vector<double> a)
    {
        var r = new Vector<double>(new double[a.Length]);
        for (int i = 0; i < a.Length; i++) r[i] = -a[i];
        return r;
    }
}
