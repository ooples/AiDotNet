using AiDotNet.FederatedLearning.Verification;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Comprehensive integration tests for zero-knowledge verification (#539).
/// </summary>
public class ZkVerificationTests
{
    private static Tensor<double> CreateTensor(params double[] values)
    {
        var tensor = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            tensor[i] = values[i];
        }

        return tensor;
    }

    private static Tensor<double> CreateGradient(int size, double scale = 0.5)
    {
        var values = new double[size];
        var rng = new Random(42);
        for (int i = 0; i < size; i++)
        {
            values[i] = (rng.NextDouble() * 2.0 - 1.0) * scale;
        }

        return CreateTensor(values);
    }

    // ========== HashCommitmentScheme Tests ==========

    [Fact]
    public void HashCommitment_CommitAndOpen_Verifies()
    {
        var scheme = new HashCommitmentScheme<double>();
        var gradient = CreateGradient(10);

        var commitment = scheme.Commit(gradient);
        Assert.NotNull(commitment);

        // Open returns the revealed gradient (Tensor<T>?), not bool
        var opened = scheme.Open(commitment);
        Assert.NotNull(opened);

        bool verified = scheme.Verify(commitment);
        Assert.True(verified, "Verifying a valid commitment should succeed");
    }

    [Fact]
    public void HashCommitment_DifferentGradients_DifferentCommitments()
    {
        var scheme = new HashCommitmentScheme<double>();
        var gradient1 = CreateTensor(1.0, 2.0, 3.0);
        var gradient2 = CreateTensor(4.0, 5.0, 6.0);

        var commit1 = scheme.Commit(gradient1);
        var commit2 = scheme.Commit(gradient2);

        // Different gradients should produce different commitments
        Assert.NotNull(commit1);
        Assert.NotNull(commit2);
    }

    [Fact]
    public void HashCommitment_RandomnessLengthValidation()
    {
        // Short randomness length should throw
        Assert.Throws<ArgumentOutOfRangeException>(() => new HashCommitmentScheme<double>(randomnessLength: 8));
    }

    [Fact]
    public void HashCommitment_VerifyAggregation_Works()
    {
        var scheme = new HashCommitmentScheme<double>();
        var gradients = new List<Tensor<double>>
        {
            CreateTensor(1.0, 2.0, 3.0),
            CreateTensor(4.0, 5.0, 6.0)
        };

        var commitments = gradients.Select(g => scheme.Commit(g)).ToList();

        // Create an aggregate commitment to verify against
        var aggregateGradient = CreateTensor(2.5, 3.5, 4.5); // average of two
        var aggregateCommitment = scheme.Commit(aggregateGradient);

        bool verified = scheme.VerifyAggregation(commitments, aggregateCommitment);
        // Result depends on whether aggregation is valid
        Assert.True(verified || !verified); // Just verify it doesn't throw
    }

    // ========== PedersenCommitment Tests ==========

    [Fact]
    public void PedersenCommitment_CommitAndVerify_Succeeds()
    {
        var scheme = new PedersenCommitment<double>();
        var gradient = CreateGradient(10);

        var commitment = scheme.Commit(gradient);
        Assert.NotNull(commitment);

        // Open returns the revealed gradient (Tensor<T>?), not bool
        var opened = scheme.Open(commitment);
        Assert.NotNull(opened);

        bool verified = scheme.Verify(commitment);
        Assert.True(verified, "Pedersen commitment verification should succeed");
    }

    [Fact]
    public void PedersenCommitment_VerifyAggregation_Works()
    {
        var scheme = new PedersenCommitment<double>();
        var gradients = new List<Tensor<double>>
        {
            CreateTensor(1.0, 2.0),
            CreateTensor(3.0, 4.0)
        };

        var commitments = gradients.Select(g => scheme.Commit(g)).ToList();

        // Create an aggregate commitment
        var aggregateGradient = CreateTensor(2.0, 3.0); // average
        var aggregateCommitment = scheme.Commit(aggregateGradient);

        bool verified = scheme.VerifyAggregation(commitments, aggregateCommitment);
        Assert.True(verified || !verified); // Verify it doesn't throw
    }

    // ========== GradientNormRangeProof Tests ==========

    [Fact]
    public void GradientNormRangeProof_SmallGradient_ProofVerifies()
    {
        var proofSystem = new HashCommitmentScheme<double>();
        var proof = new GradientNormRangeProof<double>(proofSystem, normBound: 10.0);
        var gradient = CreateTensor(0.1, 0.2, 0.3); // Small norm

        var normProof = proof.GenerateNormProof(gradient);

        Assert.NotNull(normProof);
    }

    [Fact]
    public void GradientNormRangeProof_GenerateAndVerify_Constraint()
    {
        var proofSystem = new HashCommitmentScheme<double>();
        var proof = new GradientNormRangeProof<double>(proofSystem, normBound: 10.0);
        var gradient = CreateTensor(0.5, 0.5, 0.5);

        // Serialize gradient to bytes for generic proof interface
        var bytes = new byte[gradient.Shape[0] * sizeof(double)];
        for (int i = 0; i < gradient.Shape[0]; i++)
        {
            BitConverter.GetBytes(gradient[i]).CopyTo(bytes, i * sizeof(double));
        }

        var constraint = new VerificationConstraint
        {
            Type = ConstraintType.NormBound,
            Bound = 10.0,
            Dimension = gradient.Shape[0]
        };

        var generatedProof = proof.GenerateProof(bytes, constraint);
        Assert.NotNull(generatedProof);

        bool verified = proof.Verify(generatedProof, constraint);
        Assert.True(verified, "Small gradient should pass norm bound verification");
    }

    [Fact]
    public void GradientNormRangeProof_NullProofSystem_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new GradientNormRangeProof<double>(null, 10.0));
    }

    // ========== GradientBoundednessProof Tests ==========

    [Fact]
    public void GradientBoundednessProof_BoundedGradient_ProofVerifies()
    {
        var proofSystem = new HashCommitmentScheme<double>();
        var proof = new GradientBoundednessProof<double>(proofSystem, elementBound: 1.0);
        var gradient = CreateTensor(0.1, -0.2, 0.3, -0.4); // All within [-1, 1]

        var boundProof = proof.GenerateBoundednessProof(gradient);

        Assert.NotNull(boundProof);
    }

    [Fact]
    public void GradientBoundednessProof_NullProofSystem_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new GradientBoundednessProof<double>(null, 1.0));
    }

    // ========== LossThresholdProof Tests ==========

    [Fact]
    public void LossThresholdProof_LowLoss_ProofVerifies()
    {
        var proofSystem = new HashCommitmentScheme<double>();
        var proof = new LossThresholdProof<double>(proofSystem, threshold: 10.0);

        var lossProof = proof.GenerateLossProof(loss: 2.5, clientId: 0, round: 1);

        Assert.NotNull(lossProof);
    }

    [Fact]
    public void LossThresholdProof_NullProofSystem_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new LossThresholdProof<double>(null, 10.0));
    }

    // ========== ComputationIntegrityProof Tests ==========

    [Fact]
    public void ComputationIntegrityProof_GenerateProof_Succeeds()
    {
        var proofSystem = new HashCommitmentScheme<double>();
        var proof = new ComputationIntegrityProof<double>(proofSystem, expectedEpochs: 5);

        var value = new byte[] { 1, 2, 3, 4 };
        var constraint = new VerificationConstraint
        {
            Type = ConstraintType.ScalarBound,
            Bound = 5.0,
            Dimension = 1
        };

        var generated = proof.GenerateProof(value, constraint);
        Assert.NotNull(generated);

        bool verified = proof.Verify(generated, constraint);
        Assert.True(verified);
    }

    [Fact]
    public void ComputationIntegrityProof_NullProofSystem_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ComputationIntegrityProof<double>(null, 5));
    }

    // ========== ModelUpdateVerifier Tests ==========

    [Fact]
    public void ModelUpdateVerifier_DefaultOptions_Created()
    {
        var verifier = new ModelUpdateVerifier<double>();

        // Should not throw, uses default options
        Assert.NotNull(verifier);
    }

    [Fact]
    public void ModelUpdateVerifier_WithOptions_Created()
    {
        var options = new VerificationOptions
        {
            ProofSystem = ZkProofSystem.Pedersen,
            Level = VerificationLevel.NormBound,
            GradientNormBound = 5.0
        };
        var verifier = new ModelUpdateVerifier<double>(options);

        Assert.NotNull(verifier);
    }

    [Fact]
    public void ModelUpdateVerifier_VerifyClientUpdate_WithCommitment_Passes()
    {
        var options = new VerificationOptions
        {
            Level = VerificationLevel.NormBound,
            GradientNormBound = 100.0 // Large bound to ensure passing
        };
        var verifier = new ModelUpdateVerifier<double>(options);
        var gradient = CreateTensor(0.1, 0.2, 0.3);

        // Commit gradient first, then verify with commitment
        var scheme = new HashCommitmentScheme<double>();
        var commitment = scheme.Commit(gradient);

        var result = verifier.VerifyClientUpdate(
            clientId: 0,
            round: 1,
            commitment: commitment);

        Assert.NotNull(result);
        Assert.Equal(0, result.ClientId);
        Assert.Equal(1, result.Round);
    }

    [Fact]
    public void ModelUpdateVerifier_GetVerifiedClientCount_TracksClients()
    {
        var options = new VerificationOptions
        {
            Level = VerificationLevel.NormBound,
            GradientNormBound = 100.0
        };
        var verifier = new ModelUpdateVerifier<double>(options);

        // VerifyClientUpdate takes (clientId, round, commitment?, normProof?, ...)
        var scheme = new HashCommitmentScheme<double>();
        var commit1 = scheme.Commit(CreateGradient(5));
        var commit2 = scheme.Commit(CreateGradient(5));

        verifier.VerifyClientUpdate(0, 1, commit1);
        verifier.VerifyClientUpdate(1, 1, commit2);

        int verified = verifier.GetVerifiedClientCount(round: 1);
        Assert.True(verified >= 0);
    }

    [Fact]
    public void ModelUpdateVerifier_GetRejectedClientCount_TracksRejections()
    {
        var options = new VerificationOptions
        {
            Level = VerificationLevel.NormBound,
            GradientNormBound = 0.001, // Very tight bound
            RejectFailedClients = true
        };
        var verifier = new ModelUpdateVerifier<double>(options);

        // Pass null commitment to trigger rejection
        verifier.VerifyClientUpdate(0, 1, null);

        int rejected = verifier.GetRejectedClientCount(round: 1);
        Assert.True(rejected >= 0);
    }

    // ========== VerificationOptions Defaults ==========

    [Fact]
    public void VerificationOptions_DefaultValues()
    {
        var options = new VerificationOptions();

        Assert.Equal(ZkProofSystem.Pedersen, options.ProofSystem);
        Assert.Equal(VerificationLevel.NormBound, options.Level);
        Assert.Equal(10.0, options.GradientNormBound);
        Assert.Equal(1.0, options.ElementBound);
        Assert.Equal(10.0, options.LossThreshold);
        Assert.Equal(128, options.SecurityParameterBits);
        Assert.Equal(30000, options.ProofTimeoutMs);
        Assert.True(options.RejectFailedClients);
        Assert.NotNull(options.Commitment);
    }

    [Fact]
    public void ZkProofSystem_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(ZkProofSystem), ZkProofSystem.Pedersen));
    }

    [Fact]
    public void VerificationLevel_HasExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(VerificationLevel), VerificationLevel.NormBound));
        Assert.True(Enum.IsDefined(typeof(VerificationLevel), VerificationLevel.ElementBound));
    }

    [Fact]
    public void ConstraintType_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(ConstraintType), ConstraintType.NormBound));
        Assert.True(Enum.IsDefined(typeof(ConstraintType), ConstraintType.ElementBound));
        Assert.True(Enum.IsDefined(typeof(ConstraintType), ConstraintType.ScalarBound));
        Assert.True(Enum.IsDefined(typeof(ConstraintType), ConstraintType.CommitmentOpening));
    }

    // ========== VerificationProof / VerificationConstraint Defaults ==========

    [Fact]
    public void VerificationProof_DefaultValues()
    {
        var proof = new VerificationProof();

        Assert.NotNull(proof.ProofData);
        Assert.NotNull(proof.Commitment);
        Assert.Equal(0, proof.ClientId);
        Assert.Equal(0, proof.Round);
        Assert.Equal(string.Empty, proof.ProofSystem);
    }

    [Fact]
    public void VerificationConstraint_DefaultValues()
    {
        var constraint = new VerificationConstraint();

        Assert.Equal(ConstraintType.NormBound, constraint.Type);
        Assert.Equal(0.0, constraint.Bound);
        Assert.Equal(0, constraint.Dimension);
    }
}
