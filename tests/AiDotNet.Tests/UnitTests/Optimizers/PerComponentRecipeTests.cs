using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// A model that builds one optimizer per part trains each at that part's own paper rate (#1928).
/// </summary>
/// <remarks>
/// <para>
/// InfoGAN is the case the Component key was designed for and the first to actually use it: Chen
/// et al. state "learning rate is 2e-4 for D and 1e-3 for G", a fivefold difference that a single
/// model-wide recipe flattens to whichever of the two happened to be declared.
/// </para>
/// <para>
/// A per-component rate can only be APPLIED where the model builds a separate optimizer per part.
/// Where it builds one optimizer over all parameters — BasicVSR, which states three different
/// rates — applying them would need per-parameter-group learning rates that no optimizer here has,
/// so those stay recorded in Source and reported as unapplied rather than silently flattened.
/// </para>
/// </remarks>
public class PerComponentRecipeTests
{
    private static double RateOf(object? optimizer)
    {
        var typed = Assert.IsAssignableFrom<IOptimizer<double, Tensor<double>, Tensor<double>>>(optimizer);
        return typed.GetOptions().InitialLearningRate;
    }

    [Fact]
    public void TheGeneratorAndDiscriminatorGetTheirOwnDeclaredRates()
    {
        var model = new InfoGAN<double>(new NeuralNetworkArchitecture<double>(
            inputFeatures: 8, outputSize: 8));

        double generator = RateOf(
            PaperOptimizerFactory.CreateFor<double, Tensor<double>, Tensor<double>>(model, "generator"));
        double discriminator = RateOf(
            PaperOptimizerFactory.CreateFor<double, Tensor<double>, Tensor<double>>(model, "discriminator"));

        Assert.Equal(1e-3, generator, precision: 12);
        Assert.Equal(2e-4, discriminator, precision: 12);

        // The point of the key: one model, two rates, and they are genuinely different. A single
        // recipe could satisfy either assertion but never both.
        Assert.NotEqual(generator, discriminator);
    }

    [Fact]
    public void AComponentWithNoRowOfItsOwnIsNotSilentlyGivenAnothersRate()
    {
        // Q shares the discriminator trunk and is declared explicitly at the discriminator's rate.
        // Asking for a part nobody declared must fall back to the model-wide row, and InfoGAN has
        // none — so the answer is "nothing declared", not "have the generator's".
        var model = new InfoGAN<double>(new NeuralNetworkArchitecture<double>(
            inputFeatures: 8, outputSize: 8));

        var unknown = PaperOptimizerFactory.Find(model, component: "nonexistent-part");

        Assert.Null(unknown);
    }

    [Fact]
    public void EachComponentReportsSeparately()
    {
        var model = new InfoGAN<double>(new NeuralNetworkArchitecture<double>(
            inputFeatures: 8, outputSize: 8));

        PaperOptimizerFactory.CreateFor<double, Tensor<double>, Tensor<double>>(model, "generator");
        PaperOptimizerFactory.CreateFor<double, Tensor<double>, Tensor<double>>(model, "discriminator");

        var reports = PaperOptimizerFactory.ReportsFor(model);

        Assert.Contains(reports, r => r.Component == "generator");
        Assert.Contains(reports, r => r.Component == "discriminator");
    }
}
