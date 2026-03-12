using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Proves that each gradient component is within [-B, B] (element-wise range proof).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> While norm-bound proofs constrain the overall magnitude of a gradient,
/// element-wise bound proofs ensure no single gradient component is abnormally large. This is
/// important because a gradient with a small norm could still have one extreme component
/// (a "needle" attack).</para>
///
/// <para><b>How it works:</b></para>
/// <list type="bullet">
/// <item><description>For each element g_i, prove that g_i + B &gt;= 0 (lower bound).</description></item>
/// <item><description>For each element g_i, prove that B - g_i &gt;= 0 (upper bound).</description></item>
/// <item><description>Both proofs are combined into a single element-wise boundedness proof.</description></item>
/// </list>
///
/// <para><b>Optimization:</b> Rather than proving each element individually (expensive), we use
/// a batched proof that commits to all elements at once and proves the bound in aggregate.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GradientBoundednessProof<T> : FederatedLearningComponentBase<T>, IVerifiableComputation
{
    private readonly IZkProofSystem _proofSystem;
    private readonly double _elementBound;

    /// <inheritdoc/>
    public string ProofSystemName => $"GradientBoundedness({_proofSystem.Name})";

    /// <summary>
    /// Initializes a new instance of <see cref="GradientBoundednessProof{T}"/>.
    /// </summary>
    /// <param name="proofSystem">The underlying ZK proof system.</param>
    /// <param name="elementBound">The per-element bound B (each g_i must be in [-B, B]).</param>
    public GradientBoundednessProof(IZkProofSystem proofSystem, double elementBound = 1.0)
    {
        _proofSystem = proofSystem ?? throw new ArgumentNullException(nameof(proofSystem));
        _elementBound = elementBound;
    }

    /// <summary>
    /// Generates a proof that all gradient elements are within [-B, B].
    /// </summary>
    /// <param name="gradient">The gradient tensor to prove boundedness for.</param>
    /// <returns>A verification proof, or null if any element exceeds the bound.</returns>
    public VerificationProof? GenerateBoundednessProof(Tensor<T> gradient)
    {
        if (gradient is null)
        {
            throw new ArgumentNullException(nameof(gradient));
        }

        int totalElements = ComputeTotalElements(gradient);

        // Check that all elements are within bounds
        double maxViolation = 0.0;
        for (int i = 0; i < totalElements; i++)
        {
            double val = Math.Abs(NumOps.ToDouble(gradient[i]));
            if (val > _elementBound)
            {
                maxViolation = Math.Max(maxViolation, val);
            }
        }

        if (maxViolation > 0)
        {
            return null; // Cannot prove — elements exceed bound
        }

        // Commit to the gradient values
        byte[] serialized = SerializeTensor(gradient);
        var (commitment, randomness) = _proofSystem.Commit(serialized);

        // Generate per-element range proofs (batched)
        // For each element, shift to [0, 2B] by adding B
        var shiftedBytes = new byte[totalElements * 8];
        double upperBound = 2.0 * _elementBound;
        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(gradient[i]);
            double shifted = val + _elementBound; // Now in [0, 2B]
            var shiftedDoubleBytes = BitConverter.GetBytes(shifted);
            Buffer.BlockCopy(shiftedDoubleBytes, 0, shiftedBytes, i * 8, 8);
        }

        byte[] boundBytes = BitConverter.GetBytes(upperBound);
        byte[] rangeProof = _proofSystem.GenerateRangeProof(shiftedBytes, boundBytes, commitment, randomness);

        return new VerificationProof
        {
            Commitment = commitment,
            ProofData = CombineProofData(rangeProof, randomness, shiftedBytes, totalElements),
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

        var (commitment, randomness) = _proofSystem.Commit(value);

        // Shift all values by bound to make them non-negative
        int count = value.Length / 8;
        var shiftedBytes = new byte[count * 8];
        double bound = constraint.Bound;
        for (int i = 0; i < count; i++)
        {
            double val = BitConverter.ToDouble(value, i * 8);
            double shifted = val + bound;
            var shiftedDoubleBytes = BitConverter.GetBytes(shifted);
            Buffer.BlockCopy(shiftedDoubleBytes, 0, shiftedBytes, i * 8, 8);
        }

        byte[] boundBytes = BitConverter.GetBytes(2.0 * bound);
        byte[] rangeProof = _proofSystem.GenerateRangeProof(shiftedBytes, boundBytes, commitment, randomness);

        return new VerificationProof
        {
            Commitment = commitment,
            ProofData = CombineProofData(rangeProof, randomness, shiftedBytes, count),
            ProofSystem = ProofSystemName
        };
    }

    /// <inheritdoc/>
    public bool Verify(VerificationProof proof, VerificationConstraint constraint)
    {
        if (proof is null || constraint is null || proof.ProofData.Length < 16)
        {
            return false;
        }

        if (!TryParseProofData(proof.ProofData, out byte[] rangeProof, out byte[] randomness, out byte[] shiftedBytes, out int elementCount))
        {
            return false;
        }

        // Verify range proof
        double upperBound = 2.0 * constraint.Bound;
        byte[] boundBytes = BitConverter.GetBytes(upperBound);

        if (!_proofSystem.VerifyRangeProof(rangeProof, boundBytes, proof.Commitment))
        {
            return false;
        }

        // Verify each shifted element is in [0, 2B]
        for (int i = 0; i < elementCount && i * 8 + 8 <= shiftedBytes.Length; i++)
        {
            double shifted = BitConverter.ToDouble(shiftedBytes, i * 8);
            if (shifted < 0.0 || shifted > upperBound)
            {
                return false;
            }
        }

        return true;
    }

    private byte[] SerializeTensor(Tensor<T> tensor)
    {
        int totalElements = ComputeTotalElements(tensor);
        var bytes = new byte[totalElements * 8];

        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(tensor[i]);
            var doubleBytes = BitConverter.GetBytes(val);
            Buffer.BlockCopy(doubleBytes, 0, bytes, i * 8, 8);
        }

        return bytes;
    }

    private static byte[] CombineProofData(byte[] rangeProof, byte[] randomness, byte[] shiftedBytes, int elementCount)
    {
        var result = new byte[rangeProof.Length + randomness.Length + shiftedBytes.Length + 16];
        int offset = 0;

        var lenRP = BitConverter.GetBytes(rangeProof.Length);
        Buffer.BlockCopy(lenRP, 0, result, offset, 4); offset += 4;
        Buffer.BlockCopy(rangeProof, 0, result, offset, rangeProof.Length); offset += rangeProof.Length;

        var lenR = BitConverter.GetBytes(randomness.Length);
        Buffer.BlockCopy(lenR, 0, result, offset, 4); offset += 4;
        Buffer.BlockCopy(randomness, 0, result, offset, randomness.Length); offset += randomness.Length;

        var lenS = BitConverter.GetBytes(shiftedBytes.Length);
        Buffer.BlockCopy(lenS, 0, result, offset, 4); offset += 4;
        Buffer.BlockCopy(shiftedBytes, 0, result, offset, shiftedBytes.Length); offset += shiftedBytes.Length;

        var ec = BitConverter.GetBytes(elementCount);
        Buffer.BlockCopy(ec, 0, result, offset, 4);

        return result;
    }

    private static bool TryParseProofData(byte[] data, out byte[] rangeProof, out byte[] randomness, out byte[] shiftedBytes, out int elementCount)
    {
        rangeProof = Array.Empty<byte>();
        randomness = Array.Empty<byte>();
        shiftedBytes = Array.Empty<byte>();
        elementCount = 0;

        if (data.Length < 16)
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

        int lenS = BitConverter.ToInt32(data, offset); offset += 4;
        if (offset + lenS > data.Length)
        {
            return false;
        }

        shiftedBytes = new byte[lenS];
        Buffer.BlockCopy(data, offset, shiftedBytes, 0, lenS); offset += lenS;

        if (offset + 4 > data.Length)
        {
            return false;
        }

        elementCount = BitConverter.ToInt32(data, offset);
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
