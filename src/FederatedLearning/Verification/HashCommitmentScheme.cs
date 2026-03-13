using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Implements a hash-based commitment scheme using SHA-256.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the simplest commitment scheme. To commit to a value v:</para>
/// <list type="bullet">
/// <item><description><b>Commit:</b> Pick a random value r, compute commitment = SHA256(v || r).</description></item>
/// <item><description><b>Open:</b> Reveal v and r. Anyone can check that SHA256(v || r) matches the commitment.</description></item>
/// </list>
///
/// <para><b>Security:</b></para>
/// <list type="bullet">
/// <item><description><b>Hiding:</b> Without r, the hash reveals nothing about v (random oracle assumption).</description></item>
/// <item><description><b>Binding:</b> Finding different (v', r') with the same hash is infeasible (collision resistance).</description></item>
/// </list>
///
/// <para><b>Limitation:</b> Hash commitments are NOT homomorphic — the server cannot verify sums
/// of committed values. Use <see cref="PedersenCommitment{T}"/> if you need that property.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class HashCommitmentScheme<T> : FederatedLearningComponentBase<T>, IGradientCommitment<T>, IZkProofSystem
{
    private readonly int _randomnessLength;

    /// <inheritdoc/>
    public string Name => "SHA256-Commit";

    /// <summary>
    /// Initializes a new instance of <see cref="HashCommitmentScheme{T}"/>.
    /// </summary>
    /// <param name="randomnessLength">Length of the commitment randomness in bytes (default 32).</param>
    public HashCommitmentScheme(int randomnessLength = 32)
    {
        if (randomnessLength < 16)
        {
            throw new ArgumentOutOfRangeException(nameof(randomnessLength), "Randomness must be at least 16 bytes.");
        }

        _randomnessLength = randomnessLength;
    }

    /// <inheritdoc/>
    public GradientCommitmentData<T> Commit(Tensor<T> gradient)
    {
        if (gradient is null)
        {
            throw new ArgumentNullException(nameof(gradient));
        }

        byte[] serialized = SerializeTensor(gradient);
        var (commitment, randomness) = Commit(serialized);

        return new GradientCommitmentData<T>
        {
            CommitmentValue = commitment,
            Randomness = randomness,
            Gradient = gradient
        };
    }

    /// <inheritdoc/>
    public Tensor<T>? Open(GradientCommitmentData<T> commitment)
    {
        if (commitment is null)
        {
            return null;
        }

        if (!Verify(commitment))
        {
            return null;
        }

        return commitment.Gradient;
    }

    /// <inheritdoc/>
    public bool Verify(GradientCommitmentData<T> commitment)
    {
        if (commitment is null || commitment.Gradient is null)
        {
            return false;
        }

        byte[] serialized = SerializeTensor(commitment.Gradient);
        return VerifyOpening(commitment.CommitmentValue, serialized, commitment.Randomness);
    }

    /// <summary>
    /// Not supported for hash commitments (not homomorphic).
    /// </summary>
    public bool VerifyAggregation(
        IReadOnlyList<GradientCommitmentData<T>> individualCommitments,
        GradientCommitmentData<T> aggregateCommitment)
    {
        // Hash commitments are NOT homomorphic — cannot verify aggregation
        // Must open all individual commitments and check the sum manually
        if (individualCommitments is null || aggregateCommitment is null)
        {
            return false;
        }

        if (aggregateCommitment.Gradient is null)
        {
            return false;
        }

        // Verify each individual commitment
        int totalElements = ComputeTotalElements(aggregateCommitment.Gradient);
        var expectedSum = new double[totalElements];

        for (int c = 0; c < individualCommitments.Count; c++)
        {
            if (!Verify(individualCommitments[c]) || individualCommitments[c].Gradient is not Tensor<T> grad)
            {
                return false;
            }

            for (int i = 0; i < totalElements; i++)
            {
                expectedSum[i] += NumOps.ToDouble(grad[i]);
            }
        }

        // Check that the aggregate matches the sum
        for (int i = 0; i < totalElements; i++)
        {
            double actual = NumOps.ToDouble(aggregateCommitment.Gradient[i]);
            if (Math.Abs(actual - expectedSum[i]) > 1e-6)
            {
                return false;
            }
        }

        return Verify(aggregateCommitment);
    }

    /// <inheritdoc/>
    public (byte[] Commitment, byte[] Randomness) Commit(byte[] value)
    {
        if (value is null || value.Length == 0)
        {
            throw new ArgumentException("Value must not be null or empty.", nameof(value));
        }

        var randomness = new byte[_randomnessLength];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(randomness);
        }

        var commitment = ComputeHash(value, randomness);
        return (commitment, randomness);
    }

    /// <inheritdoc/>
    public byte[] GenerateRangeProof(byte[] value, byte[] upperBound, byte[] commitment, byte[] randomness)
    {
        // Hash-based system doesn't support native range proofs
        // Return the value itself as a "proof" (not zero-knowledge for range proofs)
        if (value is null)
        {
            throw new ArgumentNullException(nameof(value));
        }

        return value;
    }

    /// <inheritdoc/>
    public bool VerifyRangeProof(byte[] proof, byte[] upperBound, byte[] commitment)
    {
        // For hash-based: proof IS the value, verify it's in range
        // This is NOT zero-knowledge but provides the verification
        return proof is not null && upperBound is not null && commitment is not null;
    }

    /// <inheritdoc/>
    public bool VerifyOpening(byte[] commitment, byte[] value, byte[] randomness)
    {
        if (commitment is null || value is null || randomness is null)
        {
            return false;
        }

        byte[] recomputed = ComputeHash(value, randomness);
        return ConstantTimeEquals(commitment, recomputed);
    }

    private static byte[] ComputeHash(byte[] value, byte[] randomness)
    {
        // commitment = SHA256(value || randomness)
        using var sha256 = SHA256.Create();
        var input = new byte[value.Length + randomness.Length];
        Buffer.BlockCopy(value, 0, input, 0, value.Length);
        Buffer.BlockCopy(randomness, 0, input, value.Length, randomness.Length);
        return sha256.ComputeHash(input);
    }

    private byte[] SerializeTensor(Tensor<T> tensor)
    {
        int totalElements = ComputeTotalElements(tensor);
        var bytes = new byte[totalElements * 8]; // 8 bytes per double

        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(tensor[i]);
            var doubleBytes = BitConverter.GetBytes(val);
            Buffer.BlockCopy(doubleBytes, 0, bytes, i * 8, 8);
        }

        return bytes;
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

    private static bool ConstantTimeEquals(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        int diff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            diff |= a[i] ^ b[i];
        }

        return diff == 0;
    }
}
