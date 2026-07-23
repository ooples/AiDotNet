using System;
using System.Collections.Generic;
using AiDotNet.FederatedLearning;
using AiDotNet.FederatedLearning.DriftDetection;
using AiDotNet.FederatedLearning.Fairness;
using AiDotNet.FederatedLearning.MPC;
using AiDotNet.FederatedLearning.PSI;
using AiDotNet.FederatedLearning.TEE;
using AiDotNet.FederatedLearning.Unlearning;
using AiDotNet.FederatedLearning.Verification;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Drives FederatedAdvancedCapabilities with the real capability implementations to prove that all eight
/// advanced federated capabilities configured via ConfigureFederatedLearning are actually exercised and surfaced.
/// </summary>
public class FederatedAdvancedCapabilitiesTests
{
    private static Vector<double> Vec(params double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) v[i] = values[i];
        return v;
    }

    private static FederatedAdvancedCapabilities<double> BuildCoordinator()
        => new(
            contribution: new ShapleyValueEvaluator<double>(new ContributionEvaluationOptions()),
            fairness: new GroupFairnessConstraint<double>(new FederatedFairnessOptions()),
            drift: new StatisticalDriftDetector<double>(new FederatedDriftOptions()),
            unlearner: new ExactRetrainingUnlearner<double>(new FederatedUnlearningOptions()),
            tee: new SimulatedTeeProvider<double>(),
            zk: new HashCommitmentScheme<double>(),
            mpc: new ArithmeticSecretSharing<double>(numberOfParties: 2),
            psi: new DiffieHellmanPsi());

    [Fact]
    public void AllEightCapabilities_AreExercised_AndSurfaced()
    {
        var coordinator = BuildCoordinator();
        Assert.True(coordinator.AnyConfigured);

        // PSI: two clients that share two of three sample identities → overlap detected.
        var identities = new Dictionary<int, IReadOnlyList<string>>
        {
            [0] = new List<string> { "a", "b", "c" },
            [1] = new List<string> { "b", "c", "d" },
        };
        coordinator.RunPrivateSetIntersection(identities);

        // Two rounds of three-client updates plus a moving global model.
        var global = Vec(0.0, 0.0, 0.0, 0.0);
        for (int round = 0; round < 2; round++)
        {
            var clientParams = new Dictionary<int, Vector<double>>
            {
                [0] = Vec(1.0 + round, 0.5, -0.2, 0.1),
                [1] = Vec(0.9 + round, 0.4, -0.1, 0.2),
                [2] = Vec(1.1 + round, 0.6, -0.3, 0.0),
            };
            global = Vec(1.0 + round, 0.5, -0.2, 0.1);
            coordinator.RecordClientParameters(clientParams);
            coordinator.OnRoundComplete(round, clientParams, global, new Dictionary<int, double>());

            if (round == 1)
            {
                coordinator.Finalize(clientParams, global);
            }
        }

        var meta = coordinator.BuildMetadata();

        // 1. Contribution (Shapley)
        Assert.True(meta.ContributionEvaluated, "contribution not evaluated");
        Assert.Equal(3, meta.ClientContributions.Count);

        // 2. Fairness
        Assert.True(meta.FairnessEvaluated, "fairness not evaluated");

        // 3. Drift (checked both rounds)
        Assert.True(meta.DriftDetectionEnabled);
        Assert.Equal(2, meta.DriftRoundsChecked);

        // 4. PSI overlap (b, c shared)
        Assert.True(meta.PsiEnabled);
        Assert.True(meta.PsiTotalOverlap >= 2, $"psi overlap {meta.PsiTotalOverlap}");

        // 5. TEE attestation (one quote per round)
        Assert.True(meta.TeeEnabled);
        Assert.Equal(2, meta.TeeAttestationCount);

        // 6. Zero-knowledge range proofs (all verify)
        Assert.True(meta.ZkEnabled);
        Assert.Equal(2, meta.ZkProofCount);
        Assert.True(meta.ZkAllVerified, "not all ZK proofs verified");

        // 7. Secure MPC aggregation reconstructs the plaintext sum
        Assert.True(meta.McpEnabled);
        Assert.True(meta.McpSecureSumVerified, "secure MPC sum did not reconstruct");

        // 8. Unlearning available
        Assert.True(meta.UnlearningAvailable);

        // Unlearning actually runs and returns a certificate for a participating client.
        var unlearned = coordinator.UnlearnClient(0, global);
        Assert.NotNull(unlearned);
        Assert.Equal(0, unlearned?.Certificate.TargetClientId);
    }

    [Fact]
    public void NoCapabilities_AnyConfiguredIsFalse()
    {
        var coordinator = new FederatedAdvancedCapabilities<double>(null, null, null, null, null, null, null, null);
        Assert.False(coordinator.AnyConfigured);
        Assert.Null(coordinator.UnlearnClient(0, Vec(1.0)));
    }
}
