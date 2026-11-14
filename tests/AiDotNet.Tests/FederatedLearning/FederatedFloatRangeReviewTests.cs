using System;
using System.Collections.Generic;
using AiDotNet.FederatedLearning.Trainers;
using AiDotNet.FederatedLearning.Unlearning;
using AiDotNet.FederatedLearning.Verification;
using AiDotNet.FederatedLearning.Vertical;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Norms, losses and weighted means whose squares or sample-count products overflow the element type
/// (float: above 3.4e38) when they are formed in T, and the pFedMe defaults of Dinh, Tran and Nguyen (2020).
/// Each expected value is the double reference, which these inputs do not overflow.
/// </summary>
public class FederatedFloatRangeReviewTests
{
    private static Tensor<float> Flat(params float[] values) => new Tensor<float>(values, new[] { values.Length });

    private static void AssertRelative(double expected, double actual, double tolerance = 1e-5)
    {
        Assert.True(Math.Abs(actual - expected) <= tolerance * Math.Abs(expected), $"Expected {expected}, got {actual}.");
    }

    [Fact]
    public void InfluenceScores_UsesTheTwoPassDeviationOfLargeUpdates()
    {
        // Parameter 0 takes 10000, 10001, 10002: |mean| (1 + population std) = 10001 (1 + sqrt(2/3)).
        // Parameter 1 is constant 20000: 20000. Scores are divided by the largest.
        var updates = new List<Tensor<float>> { Flat(10000f, 20000f), Flat(10001f, 20000f), Flat(10002f, 20000f) };

        var scores = UnlearningEngineOps.InfluenceScores(updates, 2);

        double expected = 10001.0 * (1.0 + Math.Sqrt(2.0 / 3.0)) / 20000.0;
        Assert.Equal(expected, scores[0], 4);
        Assert.Equal(1.0, scores[1], 6);
    }

    [Fact]
    public void LabelDifferentialPrivacy_ClipsLargeGradientsToTheMaxNorm()
    {
        // ||[3e19, 4e19]|| = 5e19, clipped to 1e6: [6e5, 8e5]. The noise deviation is
        // 1e6 sqrt(2 ln(1.25 / 1e-5)) / 1e6 = 4.84, far below the tolerance.
        var protector = new LabelDifferentialPrivacy<float>(epsilon: 1e6, delta: 1e-5, maxGradientNorm: 1e6, seed: 7);

        var protectedGradients = protector.ProtectGradients(Flat(3e19f, 4e19f));

        Assert.InRange(protectedGradients[0], 6e5f - 100f, 6e5f + 100f);
        Assert.InRange(protectedGradients[1], 8e5f - 100f, 8e5f + 100f);
    }

    [Fact]
    public void ProxyZkpVerifier_AcceptsALargeUpdateInsideTheNormBound()
    {
        var verifier = new ProxyZKPVerifier<float>(maxNorm: 1e20, maxElementMagnitude: 1e20);
        var update = new Dictionary<string, float[]> { ["w"] = new[] { 3e19f, 4e19f } };
        byte[] nonce = ProxyZKPVerifier<float>.GenerateNonce();
        string commitment = verifier.ComputeCommitment(update, nonce);

        var result = verifier.Verify(update, commitment, nonce);

        Assert.True(result.IsValid, result.Reason);
        Assert.NotNull(result.UpdateNorm);
        AssertRelative(5e19, result.UpdateNorm ?? double.NaN);
    }

    [Fact]
    public void GradientNormRangeProof_ProvesALargeGradientInsideTheBound()
    {
        var proof = new GradientNormRangeProof<float>(new HashCommitmentScheme<float>(), normBound: 1e20);

        Assert.NotNull(proof.GenerateNormProof(Flat(3e19f, 4e19f)));
        Assert.Null(proof.GenerateNormProof(Flat(3e20f, 4e20f)));
    }

    [Fact]
    public void VerticalFederatedUnlearner_InfluenceAndDivergence_MatchTheDoubleReference()
    {
        var unlearner = new VerticalFederatedUnlearner<float>();

        // grad = 2 (pred - label) = [3e19, 4e19], norm 5e19.
        double influence = unlearner.ComputeInfluence(Flat(1f, 1f), Flat(0f, 0f), Flat(1.5e19f, 2e19f));
        AssertRelative(5e19, influence);

        // sqrt((9e38 + 16e38) / 2)
        var verification = unlearner.VerifyUnlearning(Flat(3e19f, 4e19f), Flat(0f, 0f));
        AssertRelative(Math.Sqrt(12.5e38), verification.DivergenceScore);
        Assert.True(verification.Passed);
    }

    [Fact]
    public void VerticalPartyLabelHolder_MseLoss_MatchesTheDoubleReference()
    {
        var holder = new VerticalPartyLabelHolder<float>(
            "labels", new Tensor<float>(new[] { 1f, 2f }, new[] { 2, 1 }), Flat(0f, 0f), new[] { "a", "b" },
            embeddingDimension: 2, seed: 1);

        var (loss, gradient) = holder.ComputeLoss(Flat(3e19f, 4e19f), Flat(0f, 0f));

        // (9e38 + 16e38) / 2; gradient 2 (pred - label) / 2.
        AssertRelative(1.25e39, loss);
        AssertRelative(3e19, gradient[0]);
        AssertRelative(4e19, gradient[1]);
    }

    [Fact]
    public void OneShotEnsembleDiversity_IsTheMeanSquaredDeviationOfLargeClients()
    {
        var trainer = new OneShotFederatedTrainer<float>(aggregationMode: OneShotAggregationMode.EnsembleDistillation);
        var clients = new Dictionary<int, Dictionary<string, float[]>>
        {
            [0] = new Dictionary<string, float[]> { ["w"] = new[] { 1e20f } },
            [1] = new Dictionary<string, float[]> { ["w"] = new[] { -1e20f } }
        };

        trainer.Aggregate(clients, new Dictionary<int, int> { [0] = 1, [1] = 1 });

        // The average is 0; each client is 1e20 away: (1e40 + 1e40) / 2.
        AssertRelative(1e40, trainer.LastEnsembleDiversity);
    }

    [Fact]
    public void SemiAsyncBarrier_NormalizesSampleCountsBeforeWeighting()
    {
        var trainer = new SemiAsyncFederatedTrainer<float>();
        var global = new Dictionary<string, float[]> { ["w"] = new[] { 0f } };
        var clients = new Dictionary<int, Dictionary<string, float[]>>
        {
            [0] = new Dictionary<string, float[]> { ["w"] = new[] { 1e30f } },
            [1] = new Dictionary<string, float[]> { ["w"] = new[] { 3e30f } }
        };
        var counts = new Dictionary<int, int> { [0] = 1_000_000_000, [1] = 2_000_000_000 };

        var merged = trainer.SynchronizationBarrier(global, clients, counts);

        // The global model weighs the average client weight 1.5e9: (1e9 1e30 + 2e9 3e30 + 1.5e9 0) / 4.5e9.
        AssertRelative(7e39 / 4.5e9, merged["w"][0]);
    }

    [Fact]
    public void ClusterWeightedMean_NormalizesSampleCountsBeforeWeighting()
    {
        var members = new List<(Vector<float> Parameters, double Weight)>
        {
            (new Vector<float>(new[] { 1e30f }), 1e9),
            (new Vector<float>(new[] { 3e30f }), 3e9)
        };

        var mean = InMemoryFederatedTrainer<float, Matrix<float>, Vector<float>>.WeightedMean(members, 4e9);

        // (1e9 1e30 + 3e9 3e30) / 4e9
        AssertRelative(2.5e30, mean[0]);
    }

    [Fact]
    public void PFedMeDefaults_AreThePaperSettings()
    {
        // Dinh, Tran and Nguyen (2020), section 5 and the official implementation: lambda 15, K 5,
        // personal learning rate 0.09, eta 0.005, beta 1.
        var options = new FederatedPersonalizationOptions();

        Assert.Equal(15.0, options.PFedMeMu);
        Assert.Equal(5, options.PFedMeInnerSteps);
        Assert.Equal(0.09, options.PFedMePersonalLearningRate);
        Assert.Equal(0.005, options.PFedMeLearningRate);
        Assert.Equal(1.0, options.PFedMeBeta);
    }
}
