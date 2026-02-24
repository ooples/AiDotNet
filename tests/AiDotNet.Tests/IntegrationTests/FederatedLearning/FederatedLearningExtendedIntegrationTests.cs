using AiDotNet.FederatedLearning.DriftDetection;
using AiDotNet.FederatedLearning.PSI;
using AiDotNet.FederatedLearning.TEE;
using AiDotNet.FederatedLearning.Unlearning;
using AiDotNet.FederatedLearning.Verification;
using AiDotNet.FederatedLearning.Vertical;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FederatedLearning;

/// <summary>
/// Extended integration tests for FederatedLearning module classes not covered
/// in FederatedLearningIntegrationTests:
/// PSI results, DriftDetection reports, Unlearning certificates, TEE attestation,
/// Verification proof/constraint, VFL results, and related enums.
/// </summary>
public class FederatedLearningExtendedIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region PsiResult

    [Fact]
    public void PsiResult_DefaultValues()
    {
        var result = new PsiResult();
        Assert.NotNull(result.IntersectionIds);
        Assert.Empty(result.IntersectionIds);
        Assert.Equal(0, result.IntersectionSize);
        Assert.NotNull(result.LocalToSharedIndexMap);
        Assert.Empty(result.LocalToSharedIndexMap);
        Assert.NotNull(result.RemoteToSharedIndexMap);
        Assert.Empty(result.RemoteToSharedIndexMap);
        Assert.Equal(0.0, result.LocalOverlapRatio, Tolerance);
        Assert.Equal(0.0, result.RemoteOverlapRatio, Tolerance);
        Assert.Equal(TimeSpan.Zero, result.ExecutionTime);
        Assert.False(result.IsFuzzyMatch);
        Assert.NotNull(result.FuzzyMatchConfidences);
        Assert.Empty(result.FuzzyMatchConfidences);
    }

    [Fact]
    public void PsiResult_SetProperties()
    {
        var result = new PsiResult
        {
            IntersectionIds = new List<string> { "Alice", "Bob" },
            IntersectionSize = 2,
            LocalToSharedIndexMap = new Dictionary<int, int> { [0] = 0, [2] = 1 },
            RemoteToSharedIndexMap = new Dictionary<int, int> { [1] = 0, [3] = 1 },
            LocalOverlapRatio = 0.67,
            RemoteOverlapRatio = 0.5,
            ExecutionTime = TimeSpan.FromMilliseconds(150),
            ProtocolUsed = PsiProtocol.DiffieHellman,
            IsFuzzyMatch = true,
            FuzzyMatchConfidences = new Dictionary<int, double> { [0] = 0.95, [1] = 0.88 }
        };
        Assert.Equal(2, result.IntersectionIds.Count);
        Assert.Equal(2, result.IntersectionSize);
        Assert.Equal(0, result.LocalToSharedIndexMap[0]);
        Assert.Equal(0, result.RemoteToSharedIndexMap[1]);
        Assert.Equal(0.67, result.LocalOverlapRatio, Tolerance);
        Assert.Equal(0.5, result.RemoteOverlapRatio, Tolerance);
        Assert.True(result.IsFuzzyMatch);
        Assert.Equal(0.95, result.FuzzyMatchConfidences[0], Tolerance);
    }

    #endregion

    #region DriftType Enum

    [Fact]
    public void DriftType_AllValues()
    {
        var values = Enum.GetValues<DriftType>();
        Assert.Equal(5, values.Length);
        Assert.Contains(DriftType.None, values);
        Assert.Contains(DriftType.Warning, values);
        Assert.Contains(DriftType.Sudden, values);
        Assert.Contains(DriftType.Gradual, values);
        Assert.Contains(DriftType.Recurring, values);
    }

    #endregion

    #region DriftAction Enum

    [Fact]
    public void DriftAction_AllValues()
    {
        var values = Enum.GetValues<DriftAction>();
        Assert.Equal(5, values.Length);
        Assert.Contains(DriftAction.None, values);
        Assert.Contains(DriftAction.Monitor, values);
        Assert.Contains(DriftAction.ReduceWeight, values);
        Assert.Contains(DriftAction.SelectiveRetrain, values);
        Assert.Contains(DriftAction.TemporaryExclude, values);
    }

    #endregion

    #region ClientDriftResult

    [Fact]
    public void ClientDriftResult_DefaultValues()
    {
        var result = new ClientDriftResult();
        Assert.Equal(0, result.ClientId);
        Assert.Equal(0.0, result.DriftScore, Tolerance);
        Assert.Equal(DriftType.None, result.DriftType);
        Assert.Equal(DriftAction.None, result.RecommendedAction);
        Assert.Equal(-1, result.DriftStartRound);
        Assert.Equal(1.0, result.SuggestedWeightMultiplier, Tolerance);
    }

    [Fact]
    public void ClientDriftResult_SetProperties()
    {
        var result = new ClientDriftResult
        {
            ClientId = 3,
            DriftScore = 0.75,
            DriftType = DriftType.Sudden,
            RecommendedAction = DriftAction.ReduceWeight,
            DriftStartRound = 15,
            SuggestedWeightMultiplier = 0.3
        };
        Assert.Equal(3, result.ClientId);
        Assert.Equal(0.75, result.DriftScore, Tolerance);
        Assert.Equal(DriftType.Sudden, result.DriftType);
        Assert.Equal(DriftAction.ReduceWeight, result.RecommendedAction);
        Assert.Equal(15, result.DriftStartRound);
        Assert.Equal(0.3, result.SuggestedWeightMultiplier, Tolerance);
    }

    #endregion

    #region DriftReport

    [Fact]
    public void DriftReport_DefaultValues()
    {
        var report = new DriftReport();
        Assert.Equal(0, report.Round);
        Assert.NotNull(report.ClientResults);
        Assert.Empty(report.ClientResults);
        Assert.False(report.GlobalDriftDetected);
        Assert.Equal(0.0, report.DriftingClientFraction, Tolerance);
        Assert.Equal(0.0, report.AverageDriftScore, Tolerance);
        Assert.Equal(string.Empty, report.Summary);
    }

    [Fact]
    public void DriftReport_WithClients()
    {
        var report = new DriftReport
        {
            Round = 25,
            GlobalDriftDetected = true,
            DriftingClientFraction = 0.4,
            AverageDriftScore = 0.35,
            Method = FederatedDriftMethod.PageHinkley,
            Summary = "40% of clients drifting",
            ClientResults =
            [
                new ClientDriftResult { ClientId = 0, DriftScore = 0.1, DriftType = DriftType.None },
                new ClientDriftResult { ClientId = 1, DriftScore = 0.8, DriftType = DriftType.Sudden },
                new ClientDriftResult { ClientId = 2, DriftScore = 0.15, DriftType = DriftType.Warning }
            ]
        };
        Assert.Equal(25, report.Round);
        Assert.True(report.GlobalDriftDetected);
        Assert.Equal(3, report.ClientResults.Count);
        Assert.Equal(0.4, report.DriftingClientFraction, Tolerance);
        Assert.Equal(FederatedDriftMethod.PageHinkley, report.Method);
    }

    #endregion

    #region UnlearningCertificate

    [Fact]
    public void UnlearningCertificate_DefaultValues()
    {
        var cert = new UnlearningCertificate();
        Assert.Equal(0, cert.TargetClientId);
        Assert.False(cert.Verified);
        Assert.Equal(0.0, cert.MembershipInferenceScore, Tolerance);
        Assert.Equal(0.0, cert.ModelDivergence, Tolerance);
        Assert.Equal(1.0, cert.RetainedAccuracy, Tolerance);
        Assert.Equal(0, cert.ClientRoundsParticipated);
        Assert.Equal(0L, cert.UnlearningTimeMs);
        Assert.Equal(string.Empty, cert.PreUnlearningModelHash);
        Assert.Equal(string.Empty, cert.PostUnlearningModelHash);
        Assert.Equal(string.Empty, cert.Summary);
    }

    [Fact]
    public void UnlearningCertificate_SetProperties()
    {
        var cert = new UnlearningCertificate
        {
            TargetClientId = 7,
            MethodUsed = UnlearningMethod.GradientAscent,
            Verified = true,
            MembershipInferenceScore = 0.52,
            ModelDivergence = 0.03,
            RetainedAccuracy = 0.97,
            ClientRoundsParticipated = 50,
            UnlearningTimeMs = 5000,
            PreUnlearningModelHash = "abc123",
            PostUnlearningModelHash = "def456",
            Summary = "Successfully unlearned client 7"
        };
        Assert.Equal(7, cert.TargetClientId);
        Assert.Equal(UnlearningMethod.GradientAscent, cert.MethodUsed);
        Assert.True(cert.Verified);
        Assert.Equal(0.52, cert.MembershipInferenceScore, Tolerance);
        Assert.Equal(0.97, cert.RetainedAccuracy, Tolerance);
        Assert.Equal(50, cert.ClientRoundsParticipated);
        Assert.Equal("abc123", cert.PreUnlearningModelHash);
    }

    [Fact]
    public void UnlearningCertificate_TimestampDefaultsToUtc()
    {
        var before = DateTime.UtcNow;
        var cert = new UnlearningCertificate();
        var after = DateTime.UtcNow;
        Assert.InRange(cert.Timestamp, before, after);
    }

    #endregion

    #region RemoteAttestationResult

    [Fact]
    public void RemoteAttestationResult_DefaultValues()
    {
        var result = new RemoteAttestationResult();
        Assert.False(result.IsValid);
        Assert.Equal(string.Empty, result.MeasurementHash);
        Assert.Equal(string.Empty, result.SignerIdentity);
        Assert.Equal(string.Empty, result.FailureReason);
        Assert.NotNull(result.RawQuote);
        Assert.Empty(result.RawQuote);
        Assert.False(result.FirmwareVerified);
        Assert.Equal(string.Empty, result.FirmwareVersion);
    }

    [Fact]
    public void RemoteAttestationResult_Valid()
    {
        var result = new RemoteAttestationResult
        {
            IsValid = true,
            MeasurementHash = "a1b2c3d4e5f6",
            SignerIdentity = "1234567890abcdef",
            PolicyApplied = AttestationPolicy.Strict,
            QuoteTimestamp = DateTime.UtcNow,
            FirmwareVerified = true,
            FirmwareVersion = "2.1.0",
            RawQuote = new byte[] { 0x01, 0x02, 0x03 },
            ProviderType = TeeProviderType.Sgx
        };
        Assert.True(result.IsValid);
        Assert.Equal("a1b2c3d4e5f6", result.MeasurementHash);
        Assert.Equal("1234567890abcdef", result.SignerIdentity);
        Assert.True(result.FirmwareVerified);
        Assert.Equal(3, result.RawQuote.Length);
    }

    [Fact]
    public void RemoteAttestationResult_Failed()
    {
        var result = new RemoteAttestationResult
        {
            IsValid = false,
            FailureReason = "Measurement mismatch"
        };
        Assert.False(result.IsValid);
        Assert.Equal("Measurement mismatch", result.FailureReason);
    }

    #endregion

    #region VerificationProof

    [Fact]
    public void VerificationProof_DefaultValues()
    {
        var proof = new VerificationProof();
        Assert.NotNull(proof.ProofData);
        Assert.Empty(proof.ProofData);
        Assert.NotNull(proof.Commitment);
        Assert.Empty(proof.Commitment);
        Assert.Equal(0, proof.ClientId);
        Assert.Equal(0, proof.Round);
        Assert.Equal(string.Empty, proof.ProofSystem);
    }

    [Fact]
    public void VerificationProof_SetProperties()
    {
        var proof = new VerificationProof
        {
            ProofData = new byte[] { 1, 2, 3, 4 },
            Commitment = new byte[] { 5, 6, 7, 8 },
            ClientId = 42,
            Round = 10,
            ProofSystem = "HashCommitment"
        };
        Assert.Equal(4, proof.ProofData.Length);
        Assert.Equal(4, proof.Commitment.Length);
        Assert.Equal(42, proof.ClientId);
        Assert.Equal(10, proof.Round);
        Assert.Equal("HashCommitment", proof.ProofSystem);
    }

    #endregion

    #region VerificationConstraint

    [Fact]
    public void VerificationConstraint_DefaultValues()
    {
        var constraint = new AiDotNet.FederatedLearning.Verification.VerificationConstraint();
        Assert.Equal(0.0, constraint.Bound, Tolerance);
        Assert.Equal(0, constraint.Dimension);
    }

    [Fact]
    public void VerificationConstraint_SetProperties()
    {
        var constraint = new AiDotNet.FederatedLearning.Verification.VerificationConstraint
        {
            Type = AiDotNet.FederatedLearning.Verification.ConstraintType.NormBound,
            Bound = 10.0,
            Dimension = 256
        };
        Assert.Equal(AiDotNet.FederatedLearning.Verification.ConstraintType.NormBound, constraint.Type);
        Assert.Equal(10.0, constraint.Bound, Tolerance);
        Assert.Equal(256, constraint.Dimension);
    }

    #endregion

    #region ConstraintType Enum (Verification)

    [Fact]
    public void VerificationConstraintType_AllValues()
    {
        var values = Enum.GetValues<AiDotNet.FederatedLearning.Verification.ConstraintType>();
        Assert.Equal(4, values.Length);
        Assert.Contains(AiDotNet.FederatedLearning.Verification.ConstraintType.NormBound, values);
        Assert.Contains(AiDotNet.FederatedLearning.Verification.ConstraintType.ElementBound, values);
        Assert.Contains(AiDotNet.FederatedLearning.Verification.ConstraintType.ScalarBound, values);
        Assert.Contains(AiDotNet.FederatedLearning.Verification.ConstraintType.CommitmentOpening, values);
    }

    #endregion

    #region TrainingStepLog and TrainingStep

    [Fact]
    public void TrainingStepLog_DefaultEmpty()
    {
        var log = new TrainingStepLog();
        Assert.NotNull(log.Steps);
        Assert.Empty(log.Steps);
    }

    [Fact]
    public void TrainingStepLog_AddSteps()
    {
        var log = new TrainingStepLog();
        log.Steps.Add(new TrainingStep { Epoch = 0, Loss = 1.5 });
        log.Steps.Add(new TrainingStep { Epoch = 1, Loss = 0.8 });
        Assert.Equal(2, log.Steps.Count);
        Assert.Equal(0, log.Steps[0].Epoch);
        Assert.Equal(1.5, log.Steps[0].Loss, Tolerance);
    }

    [Fact]
    public void TrainingStep_DefaultValues()
    {
        var step = new TrainingStep();
        Assert.Equal(0, step.Epoch);
        Assert.Equal(0.0, step.Loss, Tolerance);
        Assert.NotNull(step.StepHash);
        Assert.Empty(step.StepHash);
        Assert.NotNull(step.ModelStateHash);
        Assert.Empty(step.ModelStateHash);
    }

    [Fact]
    public void TrainingStep_SetProperties()
    {
        var step = new TrainingStep
        {
            Epoch = 5,
            Loss = 0.42,
            StepHash = new byte[] { 0xAB, 0xCD },
            ModelStateHash = new byte[] { 0x01, 0x02, 0x03 }
        };
        Assert.Equal(5, step.Epoch);
        Assert.Equal(0.42, step.Loss, Tolerance);
        Assert.Equal(2, step.StepHash.Length);
        Assert.Equal(3, step.ModelStateHash.Length);
    }

    #endregion

    #region VflAlignmentSummary

    [Fact]
    public void VflAlignmentSummary_DefaultValues()
    {
        var summary = new VflAlignmentSummary();
        Assert.Equal(0, summary.AlignedEntityCount);
        Assert.NotNull(summary.PartyEntityCounts);
        Assert.Empty(summary.PartyEntityCounts);
        Assert.NotNull(summary.PartyOverlapRatios);
        Assert.Empty(summary.PartyOverlapRatios);
        Assert.Null(summary.AlignmentResult);
        Assert.False(summary.MeetsMinimumOverlap);
        Assert.Equal(TimeSpan.Zero, summary.AlignmentTime);
    }

    [Fact]
    public void VflAlignmentSummary_SetProperties()
    {
        var summary = new VflAlignmentSummary
        {
            AlignedEntityCount = 500,
            PartyEntityCounts = new Dictionary<string, int> { ["PartyA"] = 1000, ["PartyB"] = 800 },
            PartyOverlapRatios = new Dictionary<string, double> { ["PartyA"] = 0.5, ["PartyB"] = 0.625 },
            MeetsMinimumOverlap = true,
            AlignmentTime = TimeSpan.FromSeconds(2.5)
        };
        Assert.Equal(500, summary.AlignedEntityCount);
        Assert.Equal(2, summary.PartyEntityCounts.Count);
        Assert.Equal(1000, summary.PartyEntityCounts["PartyA"]);
        Assert.Equal(0.625, summary.PartyOverlapRatios["PartyB"], Tolerance);
        Assert.True(summary.MeetsMinimumOverlap);
    }

    #endregion

    #region VflEpochResult

    [Fact]
    public void VflEpochResult_DefaultValues()
    {
        var result = new VflEpochResult<double>();
        Assert.Equal(0, result.Epoch);
        Assert.Equal(0.0, result.AverageLoss, Tolerance);
        Assert.Equal(0, result.SamplesProcessed);
        Assert.Equal(0, result.BatchesProcessed);
        Assert.Equal(TimeSpan.Zero, result.EpochTime);
        Assert.Null(result.PrivacyBudgetSpent);
    }

    [Fact]
    public void VflEpochResult_SetProperties()
    {
        var result = new VflEpochResult<double>
        {
            Epoch = 3,
            AverageLoss = 0.125,
            SamplesProcessed = 5000,
            BatchesProcessed = 100,
            EpochTime = TimeSpan.FromSeconds(30),
            PrivacyBudgetSpent = (1.0, 1e-5)
        };
        Assert.Equal(3, result.Epoch);
        Assert.Equal(0.125, result.AverageLoss, Tolerance);
        Assert.Equal(5000, result.SamplesProcessed);
        Assert.Equal(100, result.BatchesProcessed);
        Assert.NotNull(result.PrivacyBudgetSpent);
        Assert.Equal(1.0, result.PrivacyBudgetSpent.Value.Epsilon, Tolerance);
        Assert.Equal(1e-5, result.PrivacyBudgetSpent.Value.Delta, Tolerance);
    }

    #endregion

    #region VflTrainingResult

    [Fact]
    public void VflTrainingResult_DefaultValues()
    {
        var result = new VflTrainingResult<double>();
        Assert.NotNull(result.EpochHistory);
        Assert.Empty(result.EpochHistory);
        Assert.Equal(0.0, result.FinalLoss, Tolerance);
        Assert.Equal(TimeSpan.Zero, result.TotalTrainingTime);
        Assert.Equal(0, result.EpochsCompleted);
        Assert.Null(result.AlignmentSummary);
        Assert.False(result.TrainingCompleted);
        Assert.Equal(0, result.NumberOfParties);
    }

    [Fact]
    public void VflTrainingResult_SetProperties()
    {
        var result = new VflTrainingResult<double>
        {
            FinalLoss = 0.05,
            TotalTrainingTime = TimeSpan.FromMinutes(10),
            EpochsCompleted = 20,
            TrainingCompleted = true,
            NumberOfParties = 3,
            AlignmentSummary = new VflAlignmentSummary { AlignedEntityCount = 1000 },
            EpochHistory = new List<VflEpochResult<double>>
            {
                new() { Epoch = 0, AverageLoss = 1.0 },
                new() { Epoch = 1, AverageLoss = 0.5 }
            }
        };
        Assert.Equal(0.05, result.FinalLoss, Tolerance);
        Assert.Equal(20, result.EpochsCompleted);
        Assert.True(result.TrainingCompleted);
        Assert.Equal(3, result.NumberOfParties);
        Assert.NotNull(result.AlignmentSummary);
        Assert.Equal(1000, result.AlignmentSummary.AlignedEntityCount);
        Assert.Equal(2, result.EpochHistory.Count);
    }

    [Fact]
    public void VflTrainingResult_EpochHistory_LossDecreases()
    {
        var result = new VflTrainingResult<double>
        {
            EpochHistory = new List<VflEpochResult<double>>
            {
                new() { Epoch = 0, AverageLoss = 2.0 },
                new() { Epoch = 1, AverageLoss = 1.5 },
                new() { Epoch = 2, AverageLoss = 1.0 },
                new() { Epoch = 3, AverageLoss = 0.7 }
            }
        };
        // Verify loss is monotonically decreasing
        for (int i = 1; i < result.EpochHistory.Count; i++)
        {
            Assert.True(result.EpochHistory[i].AverageLoss < result.EpochHistory[i - 1].AverageLoss);
        }
    }

    #endregion

    #region Cross-Module: UnlearningMethod Enum

    [Fact]
    public void UnlearningMethod_HasExpectedValues()
    {
        Assert.Equal(0, (int)UnlearningMethod.ExactRetraining);
        Assert.Equal(1, (int)UnlearningMethod.GradientAscent);
        Assert.Equal(2, (int)UnlearningMethod.InfluenceFunction);
    }

    #endregion

    #region Cross-Module: PSI with Alignment Mappings

    [Fact]
    public void PsiResult_AlignmentConsistency()
    {
        var result = new PsiResult
        {
            IntersectionIds = new List<string> { "user_1", "user_2", "user_3" },
            IntersectionSize = 3,
            LocalToSharedIndexMap = new Dictionary<int, int>
            {
                [0] = 0,  // local row 0 -> shared index 0
                [3] = 1,  // local row 3 -> shared index 1
                [7] = 2   // local row 7 -> shared index 2
            },
            RemoteToSharedIndexMap = new Dictionary<int, int>
            {
                [2] = 0,
                [5] = 1,
                [8] = 2
            }
        };

        // Both maps should have same number of entries as intersection
        Assert.Equal(result.IntersectionSize, result.LocalToSharedIndexMap.Count);
        Assert.Equal(result.IntersectionSize, result.RemoteToSharedIndexMap.Count);

        // Shared indices should cover 0..N-1
        var localSharedIndices = result.LocalToSharedIndexMap.Values.Order().ToList();
        var remoteSharedIndices = result.RemoteToSharedIndexMap.Values.Order().ToList();
        Assert.Equal(new[] { 0, 1, 2 }, localSharedIndices);
        Assert.Equal(new[] { 0, 1, 2 }, remoteSharedIndices);
    }

    #endregion

    #region Cross-Module: Drift Report Analysis

    [Fact]
    public void DriftReport_MultipleClientDriftTypes()
    {
        var report = new DriftReport
        {
            Round = 50,
            ClientResults =
            [
                new ClientDriftResult { ClientId = 0, DriftType = DriftType.None, DriftScore = 0.05 },
                new ClientDriftResult { ClientId = 1, DriftType = DriftType.Gradual, DriftScore = 0.45 },
                new ClientDriftResult { ClientId = 2, DriftType = DriftType.Sudden, DriftScore = 0.9 },
                new ClientDriftResult { ClientId = 3, DriftType = DriftType.Warning, DriftScore = 0.3 },
                new ClientDriftResult { ClientId = 4, DriftType = DriftType.Recurring, DriftScore = 0.6 }
            ]
        };

        // Count clients by drift type
        var driftTypeCount = report.ClientResults.GroupBy(r => r.DriftType).ToDictionary(g => g.Key, g => g.Count());
        Assert.Equal(1, driftTypeCount[DriftType.None]);
        Assert.Equal(1, driftTypeCount[DriftType.Gradual]);
        Assert.Equal(1, driftTypeCount[DriftType.Sudden]);
        Assert.Equal(1, driftTypeCount[DriftType.Warning]);
        Assert.Equal(1, driftTypeCount[DriftType.Recurring]);

        // Highest drift score should be the sudden drift client
        var maxDrift = report.ClientResults.OrderByDescending(r => r.DriftScore).First();
        Assert.Equal(DriftType.Sudden, maxDrift.DriftType);
    }

    #endregion

    #region Cross-Module: Attestation and Verification

    [Fact]
    public void RemoteAttestationResult_QuoteTimestamp_NotStale()
    {
        var result = new RemoteAttestationResult
        {
            IsValid = true,
            QuoteTimestamp = DateTime.UtcNow
        };
        // Quote should not be older than 1 minute
        Assert.True((DateTime.UtcNow - result.QuoteTimestamp).TotalMinutes < 1);
    }

    [Fact]
    public void VerificationProof_DifferentProofSystems()
    {
        var hashProof = new VerificationProof { ProofSystem = "HashCommitment" };
        var pedersenProof = new VerificationProof { ProofSystem = "PedersenCommitment" };

        Assert.NotEqual(hashProof.ProofSystem, pedersenProof.ProofSystem);
    }

    #endregion

    #region Cross-Module: Certificate Verification Semantics

    [Fact]
    public void UnlearningCertificate_IdealMembershipScore()
    {
        // A well-unlearned model should have membership inference score near 0.5
        // (attacker can't distinguish member from non-member)
        var cert = new UnlearningCertificate
        {
            MembershipInferenceScore = 0.51,
            Verified = true
        };
        Assert.InRange(cert.MembershipInferenceScore, 0.45, 0.55);
    }

    [Fact]
    public void UnlearningCertificate_HashesChange()
    {
        var cert = new UnlearningCertificate
        {
            PreUnlearningModelHash = "aabbccdd",
            PostUnlearningModelHash = "eeff0011"
        };
        // Model hashes should differ after unlearning
        Assert.NotEqual(cert.PreUnlearningModelHash, cert.PostUnlearningModelHash);
    }

    #endregion
}
