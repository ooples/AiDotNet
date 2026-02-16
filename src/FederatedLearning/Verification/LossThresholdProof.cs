using AiDotNet.FederatedLearning.Infrastructure;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Proves that a client's local training loss is below a threshold without revealing the actual value.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After local training, each client has a training loss value. If the
/// loss is too high, the client's model may be poorly trained or even poisoned. This proof lets
/// the server reject clients with high loss without seeing the actual loss value (which could
/// leak information about the client's private dataset).</para>
///
/// <para><b>How it works:</b></para>
/// <list type="bullet">
/// <item><description>Client commits to its loss value.</description></item>
/// <item><description>Client generates a range proof that loss is in [0, threshold].</description></item>
/// <item><description>Server verifies the proof and either accepts or rejects the client.</description></item>
/// </list>
///
/// <para><b>Reference:</b> ZKP-FedEval (2025) — privacy-preserving FL evaluation using ZKPs.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LossThresholdProof<T> : FederatedLearningComponentBase<T>, IVerifiableComputation
{
    private readonly IZkProofSystem _proofSystem;
    private readonly double _threshold;

    /// <inheritdoc/>
    public string ProofSystemName => $"LossThreshold({_proofSystem.Name})";

    /// <summary>
    /// Initializes a new instance of <see cref="LossThresholdProof{T}"/>.
    /// </summary>
    /// <param name="proofSystem">The underlying ZK proof system.</param>
    /// <param name="threshold">The maximum acceptable loss value (default 10.0).</param>
    public LossThresholdProof(IZkProofSystem proofSystem, double threshold = 10.0)
    {
        _proofSystem = proofSystem ?? throw new ArgumentNullException(nameof(proofSystem));
        _threshold = threshold;
    }

    /// <summary>
    /// Generates a proof that the given loss value is below the threshold.
    /// </summary>
    /// <param name="loss">The actual loss value.</param>
    /// <param name="clientId">The client's identifier.</param>
    /// <param name="round">The current training round.</param>
    /// <returns>A verification proof, or null if the loss exceeds the threshold.</returns>
    public VerificationProof? GenerateLossProof(double loss, int clientId, int round)
    {
        if (loss < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(loss), "Loss must be non-negative.");
        }

        if (loss > _threshold)
        {
            return null; // Cannot prove — loss exceeds threshold
        }

        byte[] lossBytes = BitConverter.GetBytes(loss);
        byte[] thresholdBytes = BitConverter.GetBytes(_threshold);

        var (commitment, randomness) = _proofSystem.Commit(lossBytes);
        byte[] rangeProof = _proofSystem.GenerateRangeProof(lossBytes, thresholdBytes, commitment, randomness);

        // Pack proof data
        var proofData = new byte[rangeProof.Length + randomness.Length + lossBytes.Length + 12];
        int offset = 0;
        WriteSegment(proofData, ref offset, rangeProof);
        WriteSegment(proofData, ref offset, randomness);
        WriteSegment(proofData, ref offset, lossBytes);

        return new VerificationProof
        {
            Commitment = commitment,
            ProofData = proofData,
            ClientId = clientId,
            Round = round,
            ProofSystem = ProofSystemName
        };
    }

    /// <inheritdoc/>
    public VerificationProof GenerateProof(byte[] value, VerificationConstraint constraint)
    {
        if (value is null || value.Length < 8)
        {
            throw new ArgumentException("Value must be at least 8 bytes (a double).", nameof(value));
        }

        if (constraint is null)
        {
            throw new ArgumentNullException(nameof(constraint));
        }

        double loss = BitConverter.ToDouble(value, 0);
        byte[] thresholdBytes = BitConverter.GetBytes(constraint.Bound);

        var (commitment, randomness) = _proofSystem.Commit(value);
        byte[] rangeProof = _proofSystem.GenerateRangeProof(value, thresholdBytes, commitment, randomness);

        var proofData = new byte[rangeProof.Length + randomness.Length + value.Length + 12];
        int offset = 0;
        WriteSegment(proofData, ref offset, rangeProof);
        WriteSegment(proofData, ref offset, randomness);
        WriteSegment(proofData, ref offset, value);

        return new VerificationProof
        {
            Commitment = commitment,
            ProofData = proofData,
            ProofSystem = ProofSystemName
        };
    }

    /// <inheritdoc/>
    public bool Verify(VerificationProof proof, VerificationConstraint constraint)
    {
        if (proof is null || constraint is null || proof.ProofData.Length < 12)
        {
            return false;
        }

        int offset = 0;
        if (!TryReadSegment(proof.ProofData, ref offset, out byte[] rangeProof))
        {
            return false;
        }

        if (!TryReadSegment(proof.ProofData, ref offset, out byte[] randomness))
        {
            return false;
        }

        if (!TryReadSegment(proof.ProofData, ref offset, out byte[] lossBytes))
        {
            return false;
        }

        // Verify commitment opens correctly
        if (!_proofSystem.VerifyOpening(proof.Commitment, lossBytes, randomness))
        {
            return false;
        }

        // Verify range proof
        byte[] thresholdBytes = BitConverter.GetBytes(constraint.Bound);
        if (!_proofSystem.VerifyRangeProof(rangeProof, thresholdBytes, proof.Commitment))
        {
            return false;
        }

        // Verify the committed loss is non-negative and within threshold
        if (lossBytes.Length >= 8)
        {
            double committedLoss = BitConverter.ToDouble(lossBytes, 0);
            if (committedLoss < 0.0 || committedLoss > constraint.Bound)
            {
                return false;
            }
        }

        return true;
    }

    private static void WriteSegment(byte[] buffer, ref int offset, byte[] data)
    {
        var lenBytes = BitConverter.GetBytes(data.Length);
        Buffer.BlockCopy(lenBytes, 0, buffer, offset, 4);
        offset += 4;
        Buffer.BlockCopy(data, 0, buffer, offset, data.Length);
        offset += data.Length;
    }

    private static bool TryReadSegment(byte[] buffer, ref int offset, out byte[] data)
    {
        data = Array.Empty<byte>();
        if (offset + 4 > buffer.Length)
        {
            return false;
        }

        int len = BitConverter.ToInt32(buffer, offset);
        offset += 4;

        if (offset + len > buffer.Length || len < 0)
        {
            return false;
        }

        data = new byte[len];
        Buffer.BlockCopy(buffer, offset, data, 0, len);
        offset += len;
        return true;
    }
}
