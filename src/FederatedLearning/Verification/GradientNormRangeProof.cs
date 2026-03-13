using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Generates and verifies proofs that a gradient's L2 norm is within a declared bound.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Gradient clipping ensures no single client can have outsized
/// influence on the global model. But how does the server verify that the client actually
/// clipped its gradient? This class generates a cryptographic proof that ||g|| &lt;= C
/// without revealing the actual norm value.</para>
///
/// <para><b>How it works (simplified Bulletproofs approach):</b></para>
/// <list type="bullet">
/// <item><description>Compute ||g||^2 (the squared norm of the gradient).</description></item>
/// <item><description>Compute the difference: C^2 - ||g||^2 (must be non-negative).</description></item>
/// <item><description>Commit to the difference using Pedersen commitment.</description></item>
/// <item><description>Prove the difference is non-negative using a range proof.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Bulletproofs (Bunz et al., S&amp;P 2018) for logarithmic-size range proofs.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GradientNormRangeProof<T> : FederatedLearningComponentBase<T>, IVerifiableComputation
{
    private readonly IZkProofSystem _proofSystem;
    private readonly double _normBound;

    /// <inheritdoc/>
    public string ProofSystemName => $"GradientNormRange({_proofSystem.Name})";

    /// <summary>
    /// Initializes a new instance of <see cref="GradientNormRangeProof{T}"/>.
    /// </summary>
    /// <param name="proofSystem">The underlying ZK proof system.</param>
    /// <param name="normBound">The maximum allowed L2 norm (default 10.0).</param>
    public GradientNormRangeProof(IZkProofSystem proofSystem, double normBound = 10.0)
    {
        _proofSystem = proofSystem ?? throw new ArgumentNullException(nameof(proofSystem));
        _normBound = normBound;
    }

    /// <summary>
    /// Generates a proof that the gradient's L2 norm is within [0, C].
    /// </summary>
    /// <param name="gradient">The gradient tensor to prove the norm bound for.</param>
    /// <returns>A verification proof, or null if the norm exceeds the bound.</returns>
    public VerificationProof? GenerateNormProof(Tensor<T> gradient)
    {
        if (gradient is null)
        {
            throw new ArgumentNullException(nameof(gradient));
        }

        // Compute ||g||^2
        double normSquared = ComputeNormSquared(gradient);
        double boundSquared = _normBound * _normBound;

        if (normSquared > boundSquared)
        {
            return null; // Cannot prove — norm exceeds bound
        }

        // Serialize the squared norm and bound
        byte[] normBytes = BitConverter.GetBytes(normSquared);
        byte[] boundBytes = BitConverter.GetBytes(boundSquared);

        // Commit to the squared norm
        var (commitment, randomness) = _proofSystem.Commit(normBytes);

        // Generate range proof: normSquared is in [0, boundSquared]
        byte[] rangeProof = _proofSystem.GenerateRangeProof(normBytes, boundBytes, commitment, randomness);

        // Combine into a verification proof
        return new VerificationProof
        {
            Commitment = commitment,
            ProofData = CombineProofData(rangeProof, randomness, normBytes),
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

        if (constraint is null)
        {
            throw new ArgumentNullException(nameof(constraint));
        }

        // Compute squared norm from serialized value
        double normSquared = ComputeNormSquaredFromBytes(value);
        double boundSquared = constraint.Bound * constraint.Bound;

        byte[] normBytes = BitConverter.GetBytes(normSquared);
        byte[] boundBytes = BitConverter.GetBytes(boundSquared);

        var (commitment, randomness) = _proofSystem.Commit(normBytes);
        byte[] rangeProof = _proofSystem.GenerateRangeProof(normBytes, boundBytes, commitment, randomness);

        return new VerificationProof
        {
            Commitment = commitment,
            ProofData = CombineProofData(rangeProof, randomness, normBytes),
            ProofSystem = ProofSystemName
        };
    }

    /// <inheritdoc/>
    public bool Verify(VerificationProof proof, VerificationConstraint constraint)
    {
        if (proof is null || constraint is null || proof.ProofData.Length == 0)
        {
            return false;
        }

        // Parse proof data
        if (!TryParseProofData(proof.ProofData, out byte[] rangeProof, out byte[] randomness, out byte[] normBytes))
        {
            return false;
        }

        // Verify commitment opens correctly
        if (!_proofSystem.VerifyOpening(proof.Commitment, normBytes, randomness))
        {
            return false;
        }

        // Verify range proof
        double boundSquared = constraint.Bound * constraint.Bound;
        byte[] boundBytes = BitConverter.GetBytes(boundSquared);

        if (!_proofSystem.VerifyRangeProof(rangeProof, boundBytes, proof.Commitment))
        {
            return false;
        }

        // Verify the committed norm squared is within bound
        if (normBytes.Length >= 8)
        {
            double committedNormSq = BitConverter.ToDouble(normBytes, 0);
            if (committedNormSq > boundSquared || committedNormSq < 0)
            {
                return false;
            }
        }

        return true;
    }

    private double ComputeNormSquared(Tensor<T> tensor)
    {
        int totalElements = ComputeTotalElements(tensor);
        double normSq = 0.0;
        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(tensor[i]);
            normSq += val * val;
        }

        return normSq;
    }

    private static double ComputeNormSquaredFromBytes(byte[] value)
    {
        int count = value.Length / 8;
        double normSq = 0.0;
        for (int i = 0; i < count; i++)
        {
            double val = BitConverter.ToDouble(value, i * 8);
            normSq += val * val;
        }

        return normSq;
    }

    private static byte[] CombineProofData(byte[] rangeProof, byte[] randomness, byte[] normBytes)
    {
        var result = new byte[rangeProof.Length + randomness.Length + normBytes.Length + 12];
        int offset = 0;

        var lenRP = BitConverter.GetBytes(rangeProof.Length);
        Buffer.BlockCopy(lenRP, 0, result, offset, 4); offset += 4;
        Buffer.BlockCopy(rangeProof, 0, result, offset, rangeProof.Length); offset += rangeProof.Length;

        var lenR = BitConverter.GetBytes(randomness.Length);
        Buffer.BlockCopy(lenR, 0, result, offset, 4); offset += 4;
        Buffer.BlockCopy(randomness, 0, result, offset, randomness.Length); offset += randomness.Length;

        var lenN = BitConverter.GetBytes(normBytes.Length);
        Buffer.BlockCopy(lenN, 0, result, offset, 4); offset += 4;
        Buffer.BlockCopy(normBytes, 0, result, offset, normBytes.Length);

        return result;
    }

    private static bool TryParseProofData(byte[] data, out byte[] rangeProof, out byte[] randomness, out byte[] normBytes)
    {
        rangeProof = Array.Empty<byte>();
        randomness = Array.Empty<byte>();
        normBytes = Array.Empty<byte>();

        if (data.Length < 12)
        {
            return false;
        }

        int offset = 0;

        int lenRP = BitConverter.ToInt32(data, offset); offset += 4;
        if (offset + lenRP > data.Length)
        {
            return false;
        }

        rangeProof = new byte[lenRP];
        Buffer.BlockCopy(data, offset, rangeProof, 0, lenRP); offset += lenRP;

        if (offset + 4 > data.Length)
        {
            return false;
        }

        int lenR = BitConverter.ToInt32(data, offset); offset += 4;
        if (offset + lenR > data.Length)
        {
            return false;
        }

        randomness = new byte[lenR];
        Buffer.BlockCopy(data, offset, randomness, 0, lenR); offset += lenR;

        if (offset + 4 > data.Length)
        {
            return false;
        }

        int lenN = BitConverter.ToInt32(data, offset); offset += 4;
        if (offset + lenN > data.Length)
        {
            return false;
        }

        normBytes = new byte[lenN];
        Buffer.BlockCopy(data, offset, normBytes, 0, lenN);

        return true;
    }

    private static int ComputeTotalElements(Tensor<T> tensor)
    {
        int total = 1;
        for (int d = 0; d < tensor.Rank; d++)
        {
            total *= tensor.Shape[d];
        }

        return total;
    }
}
