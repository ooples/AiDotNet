using System.Linq;
using AiDotNet.Audio.Classification;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Tests that a model which builds its own optimizer is CHECKED against its paper, not trusted (#1928).
/// </summary>
/// <remarks>
/// <para>
/// Some models are more faithful than the factory could make them — PANNs deliberately turns off the
/// gradient clipping its options default to, because Kong et al. do not clip — so their optimizer is
/// kept rather than replaced. Keeping it only helps if the declaration then verifies it; a
/// declaration that were merely recorded would make every hand-built model report Exact by
/// construction, which is worse than not reporting at all.
/// </para>
/// <para>
/// These assert the negative case too. A verifier that never disagrees is indistinguishable from one
/// that always returns Exact, and only the disagreeing case proves it is actually comparing.
/// </para>
/// </remarks>
public class HandBuiltRecipeVerificationTests
{
    [Fact]
    public void AHandBuiltOptimizerMatchingItsPaperReportsExact()
    {
        var model = new PANNs<double>(new NeuralNetworkArchitecture<double>(
            inputFeatures: 1, outputSize: 527));

        var report = Assert.Single(PaperOptimizerFactory.ReportsFor(model));

        Assert.Equal(OptimizerKind.Adam, report.PaperOptimizer);
        Assert.Contains("Kong", report.Source);
        Assert.Contains("AdamOptimizer", report.AppliedOptimizer);
        Assert.Equal(RecipeFidelity.Exact, report.Fidelity);
    }

    [Fact]
    public void AHandBuiltOptimizerThatDisagreesWithItsPaperIsReported()
    {
        // The optimizer the caller supplies is kept, exactly as before — but it is 100x the paper's
        // rate, and the point of verification is that this is stated rather than accepted silently.
        var wrong = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            { InitialLearningRate = 0.1 });

        var model = new PANNs<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 527),
            optimizer: wrong);

        var report = Assert.Single(PaperOptimizerFactory.ReportsFor(model));

        Assert.Equal(RecipeFidelity.Deviated, report.Fidelity);
        Assert.Contains(report.Unhonoured, u => u.Contains("LearningRate"));

        // The message names both numbers, so a reader does not have to go and find the paper to
        // learn what the difference actually is.
        Assert.Contains(report.Unhonoured, u => u.Contains("0.001") && u.Contains("0.1"));
    }
}
