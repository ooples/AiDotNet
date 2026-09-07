using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// A declared beta must survive training, not be adapted away from the paper's value (#1928).
/// </summary>
/// <remarks>
/// <para>
/// Several optimizers here pair a hyperparameter with an "adapt it during training" switch that is
/// ON by default. Adam and AdamW clamp their running betas into [MinBeta1, MaxBeta1] — 0.8 to 0.999
/// — on every step while <c>UseAdaptiveBetas</c> is set. A paper value outside that band is
/// therefore silently replaced: MelGAN states beta1 = 0.5 and would train at 0.8.
/// </para>
/// <para>
/// This is the exact defect the whole feature exists to prevent, so it needs a test rather than a
/// comment: the report would have said Exact while the optimizer used a different number. Setting a
/// value by name could never have caught it, because the flag that governs the value has a
/// different name.
/// </para>
/// </remarks>
public class DeclaredMomentsAreNotAdaptedAwayTests
{
    private static TOptions OptionsFor<TOptions>(object? optimizer)
        where TOptions : class
    {
        var typed = Assert.IsAssignableFrom<IOptimizer<double, Tensor<double>, Tensor<double>>>(optimizer);
        return Assert.IsType<TOptions>(typed.GetOptions());
    }

    [Fact]
    public void DeclaringBetasAlsoTurnsOffTheirAdaptation()
    {
        // HuBERT declares Adam with beta1 0.9 and beta2 0.98 (Hsu et al. 2021, Sec. IV-A).
        var model = new HuBERT<double>(new NeuralNetworkArchitecture<double>(
            inputFeatures: 1, outputSize: 32));

        var built = PaperOptimizerFactory.CreateFor<double, Tensor<double>, Tensor<double>>(model);
        var options = OptionsFor<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(built);

        Assert.Equal(0.9, options.Beta1, precision: 12);
        Assert.Equal(0.98, options.Beta2, precision: 12);
        Assert.False(options.UseAdaptiveBetas,
            "the paper's betas must not be clamped or adapted once declared; leaving adaptation on "
            + "rewrites them during training while the recipe report still reads Exact");
    }

    [Fact]
    public void ABetaBelowTheAdaptiveFloorSurvives()
    {
        // The case that makes this concrete: 0.5 is below MinBeta1 of 0.8, so with adaptation left
        // on it would be clamped upward on the first step.
        var defaults = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        Assert.True(defaults.UseAdaptiveBetas, "the default this test guards against has changed");
        Assert.True(defaults.MinBeta1 > 0.5, "0.5 must be below the adaptive floor for this to bite");

        var recipe = new AiDotNet.Attributes.PaperOptimizerAttribute(Enums.OptimizerKind.Adam)
        {
            Beta1 = 0.5,
            Beta2 = 0.9,
            Source = "Kumar et al. 2019, Sec. 4 (MelGAN): Adam with beta1 0.5 and beta2 0.9",
        };

        Assert.Equal(0.5, recipe.Beta1, precision: 12);
        Assert.True(recipe.DeclaresAnyHyperparameter);
    }
}
