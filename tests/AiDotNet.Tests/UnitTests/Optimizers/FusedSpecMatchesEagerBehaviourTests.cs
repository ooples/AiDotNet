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
}
