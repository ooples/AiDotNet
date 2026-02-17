using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Generates proofs of local training computation integrity (research-stage).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the "holy grail" of verifiable FL: proving that a client
/// actually ran the correct training algorithm (e.g., SGD for N epochs on its dataset) without
/// revealing the data or the model weights.</para>
///
/// <para><b>How it works (conceptually):</b></para>
/// <list type="bullet">
/// <item><description>The training computation is expressed as an arithmetic circuit.</description></item>
/// <item><description>The circuit is compiled into a constraint system (R1CS or PLONK).</description></item>
/// <item><description>A ZK-SNARK proof is generated, proving all constraints are satisfied.</description></item>
/// <item><description>The verifier checks the proof in milliseconds.</description></item>
/// </list>
///
/// <para><b>Current limitations:</b> Full computation proofs for deep neural networks are extremely
/// expensive. Current research (zkRNN 2026, ZKML Survey 2025) shows:</para>
/// <list type="bullet">
/// <item><description>Linear layers (matrix multiply): 10-100x overhead.</description></item>
/// <item><description>Non-linear activations (ReLU, sigmoid): 100-1000x overhead.</description></item>
/// <item><description>Full training loop: may take hours per proof for a small model.</description></item>
/// </list>
///
/// <para>This implementation provides a simplified proof-of-concept using hash chains to verify
/// training step ordering, not the full SNARK-based approach.</para>
///
/// <para><b>Reference:</b></para>
/// <list type="bullet">
/// <item><description>zkRNN (2026) — ZK proofs for RNN inference</description></item>
/// <item><description>ZK Proofs of Training (2024) — proving training correctness for deep NNs</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ComputationIntegrityProof<T> : FederatedLearningComponentBase<T>, IVerifiableComputation
{
    private readonly IZkProofSystem _proofSystem;
    private readonly int _expectedEpochs;

    /// <inheritdoc/>
    public string ProofSystemName => $"ComputationIntegrity({_proofSystem.Name})";

    /// <summary>
    /// Initializes a new instance of <see cref="ComputationIntegrityProof{T}"/>.
    /// </summary>
    /// <param name="proofSystem">The underlying ZK proof system.</param>
    /// <param name="expectedEpochs">The expected number of training epochs.</param>
    public ComputationIntegrityProof(IZkProofSystem proofSystem, int expectedEpochs = 5)
    {
        _proofSystem = proofSystem ?? throw new ArgumentNullException(nameof(proofSystem));
        _expectedEpochs = expectedEpochs;
    }

    /// <summary>
    /// Records a training step by hashing the current state (model hash + loss + epoch).
    /// </summary>
    /// <param name="log">The training log to append to.</param>
    /// <param name="modelStateHash">Hash of current model parameters.</param>
    /// <param name="loss">The loss after this step.</param>
    /// <param name="epoch">The current epoch number.</param>
    public void RecordTrainingStep(TrainingStepLog log, byte[] modelStateHash, double loss, int epoch)
    {
        if (log is null)
        {
            throw new ArgumentNullException(nameof(log));
        }

        if (modelStateHash is null)
        {
            throw new ArgumentNullException(nameof(modelStateHash));
        }

        // Compute step hash: H(previousHash || modelStateHash || loss || epoch)
        using var sha256 = SHA256.Create();

        var lossBytes = BitConverter.GetBytes(loss);
        var epochBytes = BitConverter.GetBytes(epoch);

        byte[] previousHash = log.Steps.Count > 0
            ? log.Steps[log.Steps.Count - 1].StepHash
            : new byte[32];

        var input = new byte[previousHash.Length + modelStateHash.Length + 8 + 4];
        int offset = 0;
        Buffer.BlockCopy(previousHash, 0, input, offset, previousHash.Length); offset += previousHash.Length;
        Buffer.BlockCopy(modelStateHash, 0, input, offset, modelStateHash.Length); offset += modelStateHash.Length;
        Buffer.BlockCopy(lossBytes, 0, input, offset, 8); offset += 8;
        Buffer.BlockCopy(epochBytes, 0, input, offset, 4);

        var stepHash = sha256.ComputeHash(input);

        log.Steps.Add(new TrainingStep
        {
            Epoch = epoch,
            Loss = loss,
            StepHash = stepHash,
            ModelStateHash = modelStateHash
        });
    }

    /// <summary>
    /// Generates a proof of training integrity from a training log.
    /// </summary>
    /// <param name="log">The complete training log.</param>
    /// <param name="clientId">The client identifier.</param>
    /// <returns>A verification proof.</returns>
    public VerificationProof GenerateTrainingProof(TrainingStepLog log, int clientId)
    {
        if (log is null || log.Steps.Count == 0)
        {
            throw new ArgumentException("Training log must not be null or empty.", nameof(log));
        }

        // The proof contains:
        // 1. The final step hash (chain digest)
        // 2. The number of steps
        // 3. A commitment to the training log
        var finalHash = log.Steps[log.Steps.Count - 1].StepHash;
        var stepCountBytes = BitConverter.GetBytes(log.Steps.Count);

        var proofInput = new byte[finalHash.Length + 4];
        Buffer.BlockCopy(finalHash, 0, proofInput, 0, finalHash.Length);
        Buffer.BlockCopy(stepCountBytes, 0, proofInput, finalHash.Length, 4);

        var (commitment, randomness) = _proofSystem.Commit(proofInput);

        // Build proof data
        var proofData = new byte[randomness.Length + proofInput.Length + 8];
        int offset = 0;
        var lenR = BitConverter.GetBytes(randomness.Length);
        Buffer.BlockCopy(lenR, 0, proofData, offset, 4); offset += 4;
        Buffer.BlockCopy(randomness, 0, proofData, offset, randomness.Length); offset += randomness.Length;
        var lenP = BitConverter.GetBytes(proofInput.Length);
        Buffer.BlockCopy(lenP, 0, proofData, offset, 4); offset += 4;
        Buffer.BlockCopy(proofInput, 0, proofData, offset, proofInput.Length);

        return new VerificationProof
        {
            Commitment = commitment,
            ProofData = proofData,
            ClientId = clientId,
            ProofSystem = ProofSystemName
        };
    }

    /// <inheritdoc/>
    public VerificationProof GenerateProof(byte[] value, VerificationConstraint constraint)
    {
        if (value is null)
        {
            throw new ArgumentNullException(nameof(value));
        }

        var (commitment, randomness) = _proofSystem.Commit(value);

        var proofData = new byte[randomness.Length + value.Length + 8];
        int offset = 0;
        var lenR = BitConverter.GetBytes(randomness.Length);
        Buffer.BlockCopy(lenR, 0, proofData, offset, 4); offset += 4;
        Buffer.BlockCopy(randomness, 0, proofData, offset, randomness.Length); offset += randomness.Length;
        var lenV = BitConverter.GetBytes(value.Length);
        Buffer.BlockCopy(lenV, 0, proofData, offset, 4); offset += 4;
        Buffer.BlockCopy(value, 0, proofData, offset, value.Length);

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
        if (proof is null || proof.ProofData.Length < 8)
        {
            return false;
        }

        // Parse proof data
        int offset = 0;
        if (offset + 4 > proof.ProofData.Length)
        {
            return false;
        }

        int lenR = BitConverter.ToInt32(proof.ProofData, offset); offset += 4;
        if (offset + lenR > proof.ProofData.Length)
        {
            return false;
        }

        var randomness = new byte[lenR];
        Buffer.BlockCopy(proof.ProofData, offset, randomness, 0, lenR); offset += lenR;

        if (offset + 4 > proof.ProofData.Length)
        {
            return false;
        }

        int lenV = BitConverter.ToInt32(proof.ProofData, offset); offset += 4;
        if (offset + lenV > proof.ProofData.Length)
        {
            return false;
        }

        var value = new byte[lenV];
        Buffer.BlockCopy(proof.ProofData, offset, value, 0, lenV);

        // Verify commitment
        if (!_proofSystem.VerifyOpening(proof.Commitment, value, randomness))
        {
            return false;
        }

        // Check that the training log has the expected number of steps
        if (value.Length >= 36)
        {
            int stepCount = BitConverter.ToInt32(value, 32);
            if (stepCount < _expectedEpochs)
            {
                return false;
            }
        }

        return true;
    }
}

/// <summary>
/// Records training steps for computation integrity verification.
/// </summary>
public class TrainingStepLog
{
    /// <summary>Gets the list of recorded training steps.</summary>
    public List<TrainingStep> Steps { get; } = new List<TrainingStep>();
}

/// <summary>
/// Represents a single training step in the integrity log.
/// </summary>
public class TrainingStep
{
    /// <summary>Gets or sets the epoch number.</summary>
    public int Epoch { get; set; }

    /// <summary>Gets or sets the loss after this step.</summary>
    public double Loss { get; set; }

    /// <summary>Gets or sets the hash of this step (chained with previous).</summary>
    public byte[] StepHash { get; set; } = Array.Empty<byte>();

    /// <summary>Gets or sets the hash of the model state at this step.</summary>
    public byte[] ModelStateHash { get; set; } = Array.Empty<byte>();
}
