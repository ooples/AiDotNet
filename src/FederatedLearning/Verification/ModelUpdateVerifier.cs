using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Server-side verification engine that checks all proofs from clients before aggregation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The model update verifier is the server-side component that examines
/// cryptographic proofs from each client before their updates are included in the global model.
/// Think of it as a "bouncer" that checks credentials before letting clients contribute.</para>
///
/// <para><b>Verification workflow:</b></para>
/// <list type="bullet">
/// <item><description>Each client sends its update along with proofs (commitment, norm proof, etc.).</description></item>
/// <item><description>The verifier checks each proof against the configured verification level.</description></item>
/// <item><description>Clients that fail verification are rejected (or flagged, depending on config).</description></item>
/// <item><description>Only verified updates proceed to aggregation.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ModelUpdateVerifier<T> : FederatedLearningComponentBase<T>
{
    private readonly VerificationOptions _options;
    private readonly IGradientCommitment<T> _commitmentScheme;
    private readonly IVerifiableComputation? _normProver;
    private readonly IVerifiableComputation? _boundednessProver;
    private readonly IVerifiableComputation? _lossProver;
    private readonly List<VerificationRecord> _verificationHistory;

    /// <summary>
    /// Gets the verification history for auditing.
    /// </summary>
    public IReadOnlyList<VerificationRecord> VerificationHistory => _verificationHistory;

    /// <summary>
    /// Initializes a new instance of <see cref="ModelUpdateVerifier{T}"/>.
    /// </summary>
    /// <param name="options">Verification configuration options.</param>
    public ModelUpdateVerifier(VerificationOptions? options = null)
    {
        _options = options ?? new VerificationOptions();
        _verificationHistory = new List<VerificationRecord>();

        // Initialize proof systems based on configured proof system type
        IZkProofSystem proofSystem = _options.ProofSystem switch
        {
            ZkProofSystem.Pedersen => new PedersenCommitment<T>(_options.Commitment.PedersenGroupBitLength),
            _ => new HashCommitmentScheme<T>(_options.Commitment.RandomnessLength)
        };

        // Set up commitment scheme
        if (_options.ProofSystem == ZkProofSystem.Pedersen)
        {
            _commitmentScheme = new PedersenCommitment<T>(_options.Commitment.PedersenGroupBitLength);
        }
        else
        {
            _commitmentScheme = new HashCommitmentScheme<T>(_options.Commitment.RandomnessLength);
        }

        // Set up proof generators based on verification level
        if (_options.Level >= VerificationLevel.NormBound)
        {
            _normProver = new GradientNormRangeProof<T>(proofSystem, _options.GradientNormBound);
        }

        if (_options.Level >= VerificationLevel.ElementBound)
        {
            _boundednessProver = new GradientBoundednessProof<T>(proofSystem, _options.ElementBound);
        }

        if (_options.Level >= VerificationLevel.LossThreshold)
        {
            _lossProver = new LossThresholdProof<T>(proofSystem, _options.LossThreshold);
        }
    }

    /// <summary>
    /// Verifies a client's update including commitment and any required proofs.
    /// </summary>
    /// <param name="clientId">The client identifier.</param>
    /// <param name="round">The current training round.</param>
    /// <param name="commitment">The client's gradient commitment.</param>
    /// <param name="normProof">Optional norm bound proof.</param>
    /// <param name="boundednessProof">Optional element boundedness proof.</param>
    /// <param name="lossProof">Optional loss threshold proof.</param>
    /// <returns>A verification result indicating pass/fail and details.</returns>
    public ClientVerificationResult VerifyClientUpdate(
        int clientId,
        int round,
        GradientCommitmentData<T>? commitment,
        VerificationProof? normProof = null,
        VerificationProof? boundednessProof = null,
        VerificationProof? lossProof = null)
    {
        var result = new ClientVerificationResult
        {
            ClientId = clientId,
            Round = round,
            Level = _options.Level
        };

        if (_options.Level == VerificationLevel.None)
        {
            result.Passed = true;
            RecordVerification(result);
            return result;
        }

        // Level 1: Commitment verification
        if (commitment is null)
        {
            result.Passed = false;
            result.FailureReason = "Missing gradient commitment.";
            RecordVerification(result);
            return result;
        }

        if (!_commitmentScheme.Verify(commitment))
        {
            result.Passed = false;
            result.FailureReason = "Gradient commitment verification failed.";
            RecordVerification(result);
            return result;
        }

        result.CommitmentVerified = true;

        // Level 2: Norm bound verification
        if (_options.Level >= VerificationLevel.NormBound && _normProver is not null)
        {
            if (normProof is null)
            {
                result.Passed = false;
                result.FailureReason = "Missing gradient norm proof.";
                RecordVerification(result);
                return result;
            }

            var normConstraint = new VerificationConstraint
            {
                Type = ConstraintType.NormBound,
                Bound = _options.GradientNormBound
            };

            if (!_normProver.Verify(normProof, normConstraint))
            {
                result.Passed = false;
                result.FailureReason = $"Gradient norm proof failed (bound={_options.GradientNormBound}).";
                RecordVerification(result);
                return result;
            }

            result.NormBoundVerified = true;
        }

        // Level 3: Element boundedness verification
        if (_options.Level >= VerificationLevel.ElementBound && _boundednessProver is not null)
        {
            if (boundednessProof is null)
            {
                result.Passed = false;
                result.FailureReason = "Missing element boundedness proof.";
                RecordVerification(result);
                return result;
            }

            var boundConstraint = new VerificationConstraint
            {
                Type = ConstraintType.ElementBound,
                Bound = _options.ElementBound
            };

            if (!_boundednessProver.Verify(boundednessProof, boundConstraint))
            {
                result.Passed = false;
                result.FailureReason = $"Element boundedness proof failed (bound={_options.ElementBound}).";
                RecordVerification(result);
                return result;
            }

            result.ElementBoundVerified = true;
        }

        // Level 4: Loss threshold verification
        if (_options.Level >= VerificationLevel.LossThreshold && _lossProver is not null)
        {
            if (lossProof is null)
            {
                result.Passed = false;
                result.FailureReason = "Missing loss threshold proof.";
                RecordVerification(result);
                return result;
            }

            var lossConstraint = new VerificationConstraint
            {
                Type = ConstraintType.ScalarBound,
                Bound = _options.LossThreshold
            };

            if (!_lossProver.Verify(lossProof, lossConstraint))
            {
                result.Passed = false;
                result.FailureReason = $"Loss threshold proof failed (threshold={_options.LossThreshold}).";
                RecordVerification(result);
                return result;
            }

            result.LossThresholdVerified = true;
        }

        result.Passed = true;
        RecordVerification(result);
        return result;
    }

    /// <summary>
    /// Gets the commitment scheme for clients to use.
    /// </summary>
    public IGradientCommitment<T> CommitmentScheme => _commitmentScheme;

    /// <summary>
    /// Gets the count of clients that passed verification in a given round.
    /// </summary>
    /// <param name="round">The round number.</param>
    /// <returns>Count of verified clients.</returns>
    public int GetVerifiedClientCount(int round)
    {
        int count = 0;
        for (int i = 0; i < _verificationHistory.Count; i++)
        {
            if (_verificationHistory[i].Round == round && _verificationHistory[i].Passed)
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Gets the count of clients that failed verification in a given round.
    /// </summary>
    /// <param name="round">The round number.</param>
    /// <returns>Count of rejected clients.</returns>
    public int GetRejectedClientCount(int round)
    {
        int count = 0;
        for (int i = 0; i < _verificationHistory.Count; i++)
        {
            if (_verificationHistory[i].Round == round && !_verificationHistory[i].Passed)
            {
                count++;
            }
        }

        return count;
    }

    private void RecordVerification(ClientVerificationResult result)
    {
        _verificationHistory.Add(new VerificationRecord
        {
            ClientId = result.ClientId,
            Round = result.Round,
            Passed = result.Passed,
            FailureReason = result.FailureReason
        });
    }
}

/// <summary>
/// Contains the result of verifying a single client's update.
/// </summary>
public class ClientVerificationResult
{
    /// <summary>Gets or sets the client ID.</summary>
    public int ClientId { get; set; }

    /// <summary>Gets or sets the round number.</summary>
    public int Round { get; set; }

    /// <summary>Gets or sets whether the client passed all required verifications.</summary>
    public bool Passed { get; set; }

    /// <summary>Gets or sets the verification level applied.</summary>
    public VerificationLevel Level { get; set; }

    /// <summary>Gets or sets whether the commitment was verified.</summary>
    public bool CommitmentVerified { get; set; }

    /// <summary>Gets or sets whether the norm bound was verified.</summary>
    public bool NormBoundVerified { get; set; }

    /// <summary>Gets or sets whether element boundedness was verified.</summary>
    public bool ElementBoundVerified { get; set; }

    /// <summary>Gets or sets whether the loss threshold was verified.</summary>
    public bool LossThresholdVerified { get; set; }

    /// <summary>Gets or sets the reason for failure (if any).</summary>
    public string FailureReason { get; set; } = string.Empty;
}

/// <summary>
/// Records a verification event for auditing.
/// </summary>
public class VerificationRecord
{
    /// <summary>Gets or sets the client ID.</summary>
    public int ClientId { get; set; }

    /// <summary>Gets or sets the round number.</summary>
    public int Round { get; set; }

    /// <summary>Gets or sets whether verification passed.</summary>
    public bool Passed { get; set; }

    /// <summary>Gets or sets the failure reason.</summary>
    public string FailureReason { get; set; } = string.Empty;
}
