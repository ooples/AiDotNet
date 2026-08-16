using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins each optimizer's fused self-description to what its <c>Step</c> method actually does (#1930).
/// </summary>
/// <remarks>
/// <para>
/// The failure mode these guard against is silent and path-dependent: an optimizer reports one algorithm to the
/// compiled path while its eager path runs another, so the same optimizer with the same settings trains
/// differently depending on which path happens to be taken. Nothing errors.
/// </para>
/// <para>
/// <b>Which eager path counts.</b> Several optimizers have TWO. <c>Optimize()</c> runs a flat-vector loop that
/// drives non-neural models (regression, clustering) and never reaches the compiled plan. <c>Step()</c> is the
/// per-tensor tape path, and that is the one the fused kernel replaces — so that is the one the spec must match.
/// The distinction is not academic: <c>StochasticGradientDescentOptimizer</c> and
/// <c>GradientDescentOptimizer</c> both call <c>ApplyMomentum</c> in <c>Optimize()</c> with
/// <c>InitialMomentum</c> defaulting to 0.9, while their <c>Step()</c> methods apply plain
/// <c>param -= lr * grad</c> (the sparse branch passes an explicit <c>momentum: 0.0</c>). Mapping them to
/// SGDMomentum on the strength of <c>Optimize()</c> would ADD momentum on the fused path that the tape path
/// never applies — manufacturing exactly the divergence this file exists to prevent.
/// </para>
/// </remarks>
public class FusedSpecMatchesEagerBehaviourTests
{
    private static bool TryGetConfig(object optimizer, out FusedOptimizerConfig config)
    {
        config = default;
        return optimizer is IFusedOptimizerSpec spec && spec.TryGetFusedOptimizerConfig(out config);
    }

    /// <summary>
    /// SGD's <c>Step</c> is plain <c>param -= lr * grad</c>, so the spec must say plain SGD — at the default
    /// momentum of 0.9 as much as at zero, because <c>Step</c> ignores momentum either way.
    /// </summary>
    [Theory]
    [InlineData(0.9)]
    [InlineData(0.0)]
    public void Sgd_ReportsPlainSgd_BecauseItsStepAppliesNoMomentum(double initialMomentum)
    {
        var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = initialMomentum,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGD, config.Type);
        Assert.Equal(0f, config.Beta1);
    }

    /// <summary>
    /// GradientDescent gains a spec for the first time here, and it is plain SGD for the same reason.
    /// </summary>
    [Theory]
    [InlineData(0.9)]
    [InlineData(0.0)]
    public void GradientDescent_ReportsPlainSgd_BecauseItsStepAppliesNoMomentum(double initialMomentum)
    {
        var options = new GradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = initialMomentum,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new GradientDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGD, config.Type);
        Assert.Equal(0f, config.Beta1);
    }

    /// <summary>
    /// MiniBatchGD does not call ApplyMomentum on EITHER path, so plain SGD is unambiguous for it.
    /// </summary>
    [Fact]
    public void MiniBatchGradientDescent_ReportsPlainSgd()
    {
        var options = new MiniBatchGradientDescentOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = 0.9,   // present on the base options, unused by this optimizer
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new MiniBatchGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGD, config.Type);
    }

    /// <summary>
    /// RMSprop must keep fusing at the default momentum. Its <c>Step</c> has no momentum term, so it matches
    /// the momentum-free RMSprop kernel exactly; declining on <c>InitialMomentum != 0</c> would drop every
    /// default-configured RMSprop off the fused path for no reason.
    /// </summary>
    [Theory]
    [InlineData(0.9)]
    [InlineData(0.0)]
    public void RmsProp_FusesRegardlessOfInitialMomentum(double initialMomentum)
    {
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = initialMomentum,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config),
            "RMSprop declined fusion; its Step applies no momentum, so it matches the RMSprop kernel.");
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.RMSprop, config.Type);
    }

    /// <summary>
    /// MomentumOptimizer is the one that genuinely carries momentum through <c>Step</c>, so it maps to the
    /// SGDMomentum kernel with the coefficient in beta1. This is the control that keeps the tests above from
    /// being satisfied by a spec that simply always says "plain SGD".
    /// </summary>
    [Fact]
    public void Momentum_ReportsSgdMomentum_WithTheCoefficientInBeta1()
    {
        var options = new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = 0.9,
            UseAdaptiveMomentum = false,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new MomentumOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGDMomentum, config.Type);
        Assert.Equal(0.9f, config.Beta1, 5);
    }

    /// <summary>
    /// Momentum adapting during training cannot be baked into a plan built once, so an adaptive
    /// instance must decline.
    /// </summary>
    [Fact]
    public void Momentum_WithAdaptiveMomentum_DeclinesFusion()
    {
        var options = new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            UseAdaptiveMomentum = true,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new MomentumOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.False(TryGetConfig(optimizer, out _),
            "MomentumOptimizer fused despite adapting its momentum, which the plan bakes in at build time.");
    }

    // ── CoordinateDescent → SGDMomentum ──────────────────────────────────────

    /// <summary>
    /// CoordinateDescent reports SGDMomentum, carrying the momentum in beta1.
    /// </summary>
    [Fact]
    public void CoordinateDescent_ReportsSgdMomentum_CarryingMomentumInBeta1()
    {
        var options = new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.05,
            InitialMomentum = 0.7,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGDMomentum, config.Type);
        Assert.Equal(0.05f, config.LearningRate, 6);
        Assert.Equal(0.7f, config.Beta1, 6);
    }

    /// <summary>
    /// The mapping claims to be EXACT, not approximate, so this reproduces the kernel's recurrence
    /// independently and demands agreement step for step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CoordinateDescent applies <c>u_t = lr*g_t + m*u_{t-1}; x -= u_t</c>. The SGDMomentum kernel applies
    /// <c>v_t = m*v_{t-1} + g_t; x -= lr*v_t</c>. Substituting <c>u = lr*v</c> turns the first into the
    /// second, so the two are the same update with the stored state scaled differently — and both start it
    /// at zero.
    /// </para>
    /// <para>
    /// This is the assertion that would catch the mapping being merely plausible. A spec that named
    /// SGDMomentum while the eager sweep did something subtly different would still train, and would still
    /// pass the type check above.
    /// </para>
    /// </remarks>
    [Fact]
    public void CoordinateDescent_MatchesTheSgdMomentumRecurrenceExactly()
    {
        const double lr = 0.05, m = 0.7;
        var options = new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            InitialMomentum = m,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var gradients = new[]
        {
            new[] { 0.4, -0.8 },
            new[] { 0.2, 0.6 },
            new[] { -0.1, 0.3 },
            new[] { 0.5, -0.2 },
        };

        var actual = new Vector<double>(new[] { 1.0, -2.0 });
        // Independent SGDMomentum reference: v = m*v + g; x -= lr*v.
        var reference = new[] { 1.0, -2.0 };
        var v = new[] { 0.0, 0.0 };

        foreach (var g in gradients)
        {
            actual = optimizer.UpdateParameters(actual, new Vector<double>((double[])g.Clone()));
            for (int i = 0; i < reference.Length; i++)
            {
                v[i] = m * v[i] + g[i];
                reference[i] -= lr * v[i];
            }

            for (int i = 0; i < reference.Length; i++)
                Assert.Equal(reference[i], actual[i], 12);
        }
    }

    /// <summary>
    /// A non-constant schedule breaks the <c>u = lr*v</c> identity the mapping rests on, so the spec must
    /// decline rather than fuse into a silently different trajectory.
    /// </summary>
    [Fact]
    public void CoordinateDescent_DeclinesOnANonConstantSchedule()
    {
        var options = new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.05,
            InitialMomentum = 0.7,
            UseAdaptiveLearningRate = false,
            LearningRateScheduler = new AiDotNet.LearningRateSchedulers.ExponentialLRScheduler(0.05, 0.95),
        };
        var optimizer = new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.False(TryGetConfig(optimizer, out _),
            "CoordinateDescent fused under a moving learning rate; the u = lr*v identity only holds while lr is fixed.");
    }

    // ── NesterovAcceleratedGradient → SGDMomentum + Nesterov ─────────────────

    /// <summary>
    /// NAG must now perform the actual Nesterov look-ahead, not classical momentum.
    /// </summary>
    /// <remarks>
    /// <para>
    /// It previously applied <c>v = mu*v + lr*g; p -= v</c> — classical momentum under a Nesterov name.
    /// The correct form (Sutskever et al. 2013, as in PyTorch's <c>nesterov=True</c>) is
    /// <c>v = mu*v + g; p -= lr*(g + mu*v)</c>.
    /// </para>
    /// <para>
    /// The two agree nowhere after the first step, and even the FIRST step differs: classical gives
    /// <c>p -= lr*g</c> while Nesterov gives <c>p -= lr*(1 + mu)*g</c>, because the look-ahead already
    /// includes the freshly updated velocity. That factor is what this test pins.
    /// </para>
    /// </remarks>
    [Fact]
    public void Nag_PerformsTheNesterovLookAhead_NotClassicalMomentum()
    {
        const double lr = 0.1, mu = 0.9;
        var options = new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            InitialMomentum = mu,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new NesterovAcceleratedGradientOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var start = new Vector<double>(new[] { 0.0 });
        var g = new Vector<double>(new[] { 1.0 });

        var after = optimizer.UpdateParameters(start, g);

        // v = 0.9*0 + 1 = 1; update = 1 + 0.9*1 = 1.9; p -= 0.1*1.9
        Assert.Equal(-lr * (1.0 + mu), after[0], 12);
        // Classical momentum would have moved exactly -lr*g here.
        Assert.NotEqual(-lr, after[0], 6);
    }

    /// <summary>
    /// Exact agreement with an independent transcription of the kernel's nesterov branch, over several
    /// steps — the assertion that would catch a mapping that merely looks right.
    /// </summary>
    [Fact]
    public void Nag_MatchesTheKernelNesterovRecurrenceExactly()
    {
        const double lr = 0.05, mu = 0.8;
        var options = new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            InitialMomentum = mu,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new NesterovAcceleratedGradientOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var gradients = new[]
        {
            new[] { 0.4, -0.8 },
            new[] { 0.2, 0.6 },
            new[] { -0.1, 0.3 },
        };

        var actual = new Vector<double>(new[] { 1.0, -2.0 });
        var reference = new[] { 1.0, -2.0 };
        var v = new[] { 0.0, 0.0 };

        foreach (var g in gradients)
        {
            actual = optimizer.UpdateParameters(actual, new Vector<double>((double[])g.Clone()));
            for (int i = 0; i < reference.Length; i++)
            {
                v[i] = mu * v[i] + g[i];
                reference[i] -= lr * (g[i] + mu * v[i]);
            }

            for (int i = 0; i < reference.Length; i++)
                Assert.Equal(reference[i], actual[i], 12);
        }
    }

    /// <summary>
    /// The spec must request the Nesterov kernel variant, carrying momentum in beta1.
    /// </summary>
    [Fact]
    public void Nag_ReportsSgdMomentum_WithNesterovSet()
    {
        var options = new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.05,
            InitialMomentum = 0.8,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new NesterovAcceleratedGradientOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGDMomentum, config.Type);
        Assert.Equal(0.8f, config.Beta1, 6);
        Assert.NotNull(config.Extras);
        Assert.True(config.Extras!.Nesterov,
            "NAG fused without the Nesterov flag — the kernel would run classical momentum while the eager path runs the look-ahead.");
    }

    // ── FTRL ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// FTRL's hyperparameters live in Extras, and the learning rate is Alpha rather than the base class's
    /// CurrentLearningRate — Alpha is what the eager update actually reads.
    /// </summary>
    /// <remarks>
    /// <c>FtrlBeta</c> is the one that matters most here. The kernel previously assumed beta = 0, computing
    /// <c>sqrt(n)/lr</c> with no beta term; McMahan's paper and this optimizer both default it to 1, so a
    /// spec that omitted it would run a different per-coordinate learning-rate schedule on the fused path.
    /// <c>LrPower = -0.5</c> is what turns the kernel's general <c>n^-p</c> into the paper's <c>sqrt(n)</c>.
    /// </remarks>
    [Fact]
    public void Ftrl_CarriesAlphaBetaAndBothLambdasIntoTheKernel()
    {
        var options = new FTRLOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            Alpha = 0.02,
            Beta = 1.5,
            Lambda1 = 0.3,
            Lambda2 = 0.7,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new FTRLOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.FTRL, config.Type);
        Assert.Equal(0.02f, config.LearningRate, 6);

        Assert.NotNull(config.Extras);
        Assert.Equal(0.3f, config.Extras!.L1, 6);
        Assert.Equal(0.7f, config.Extras.L2, 6);
        Assert.Equal(1.5f, config.Extras.FtrlBeta, 6);
        Assert.Equal(-0.5f, config.Extras.LrPower, 6);

        // A default-constructed Extras would carry beta = 0 and silently change the schedule.
        Assert.NotEqual(0f, config.Extras.FtrlBeta);
    }

    /// <summary>
    /// The eager update must be the FTRL-Proximal formula the kernel implements, so this reproduces it
    /// independently from the paper and demands element-wise agreement.
    /// </summary>
    /// <remarks>
    /// Includes a coordinate that gets thresholded to exactly zero, which is the part of FTRL that makes it
    /// worth choosing and the part a plausible-but-wrong denominator would still get right.
    /// </remarks>
    [Fact]
    public void Ftrl_MatchesTheProximalFormulaExactly()
    {
        const double alpha = 0.1, beta = 1.0, l1 = 0.05, l2 = 0.2;
        var options = new FTRLOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            Alpha = alpha,
            Beta = beta,
            Lambda1 = l1,
            Lambda2 = l2,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new FTRLOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new[] { 0.5, -0.4, 0.02 });
        var gradient = new Vector<double>(new[] { 0.6, -0.9, 0.01 });

        var actual = optimizer.UpdateParameters(parameters, gradient);

        // Independent transcription of McMahan et al. (2013), Algorithm 1.
        var z = new double[3];
        var n = new double[3];
        var expected = new double[3];
        for (int i = 0; i < 3; i++)
        {
            double g = gradient[i];
            double nOld = n[i];
            double nNew = nOld + g * g;
            double sigma = (Math.Sqrt(nNew) - Math.Sqrt(nOld)) / alpha;
            z[i] += g - sigma * parameters[i];
            n[i] = nNew;

            double absZ = Math.Abs(z[i]);
            if (absZ <= l1)
            {
                expected[i] = 0.0;
            }
            else
            {
                double sign = z[i] > 0 ? 1.0 : -1.0;
                expected[i] = -sign * (absZ - l1) / (l2 + (Math.Sqrt(nNew) + beta) / alpha);
            }
        }

        for (int i = 0; i < 3; i++)
            Assert.Equal(expected[i], actual[i], 12);

        // The third coordinate must have been thresholded to exactly zero.
        Assert.Equal(0.0, actual[2], 12);
    }
}
