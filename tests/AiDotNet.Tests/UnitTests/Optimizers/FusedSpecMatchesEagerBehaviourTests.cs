using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins each optimizer's fused self-description to what its EAGER loop actually does (#1930).
/// </summary>
/// <remarks>
/// <para>
/// The failure mode these guard against is silent and path-dependent: an optimizer reports one
/// algorithm to the compiled path while its eager loop runs another, so the same optimizer with the
/// same settings trains differently depending on which path happens to be taken. Nothing errors.
/// </para>
/// <para>
/// The specific case: <c>GradientDescentOptimizer</c>, <c>StochasticGradientDescentOptimizer</c> and
/// <c>RootMeanSquarePropagationOptimizer</c> all call <c>ApplyMomentum</c>, and
/// <c>InitialMomentum</c> defaults to 0.9 — yet SGD reported plain <c>SGD</c> with
/// <c>beta1 = 0</c>, i.e. no momentum at all.
/// </para>
/// </remarks>
public class FusedSpecMatchesEagerBehaviourTests
{
    private static bool TryGetConfig(object optimizer, out FusedOptimizerConfig config)
    {
        config = default;
        return optimizer is IFusedOptimizerSpec spec && spec.TryGetFusedOptimizerConfig(out config);
    }

    [Fact]
    public void Sgd_WithDefaultMomentum_ReportsSgdMomentumNotPlainSgd()
    {
        var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = 0.9,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGDMomentum, config.Type);
        Assert.Equal(0.9f, config.Beta1, 5);
    }

    [Fact]
    public void Sgd_WithZeroMomentum_ReportsPlainSgd()
    {
        var options = new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = 0.0,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGD, config.Type);
        Assert.Equal(0f, config.Beta1);
    }

    [Fact]
    public void GradientDescent_WithDefaultMomentum_ReportsSgdMomentum()
    {
        var options = new GradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = 0.9,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new GradientDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGDMomentum, config.Type);
        Assert.Equal(0.9f, config.Beta1, 5);
    }

    /// <summary>
    /// MiniBatchGD does NOT call ApplyMomentum, so plain SGD is the correct description for it. This
    /// is the control: it shows the rule above is "match the eager loop", not "always momentum".
    /// </summary>
    [Fact]
    public void MiniBatchGradientDescent_ReportsPlainSgd_BecauseItAppliesNoMomentum()
    {
        var options = new MiniBatchGradientDescentOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = 0.9,   // present, but unused by this optimizer's loop
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new MiniBatchGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.SGD, config.Type);
    }

    /// <summary>
    /// RMSprop's eager loop applies momentum but the fused RMSprop kernel has no momentum term, and
    /// there is no RMSprop+momentum kernel. It must DECLINE rather than report RMSprop and quietly
    /// lose the momentum on the fused path only.
    /// </summary>
    [Fact]
    public void RmsProp_WithMomentum_DeclinesFusionRatherThanDroppingIt()
    {
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = 0.9,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.False(TryGetConfig(optimizer, out _),
            "RMSprop reported a fused config despite carrying momentum the kernel cannot express.");
    }

    [Fact]
    public void RmsProp_WithoutMomentum_StillFuses()
    {
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialMomentum = 0.0,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        Assert.True(TryGetConfig(optimizer, out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.RMSprop, config.Type);
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
