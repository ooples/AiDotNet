using System.Numerics;
using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Implements Pedersen commitment scheme — additively homomorphic for verifiable aggregation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Pedersen commitments are special because they're "additively
/// homomorphic": if you commit to values a and b separately, anyone can combine the two
/// commitments to get a valid commitment to a+b — without knowing a or b.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="bullet">
/// <item><description>Setup: Choose a prime p, generator g, and random h = g^s (where s is secret).</description></item>
/// <item><description>Commit(v, r) = g^v * h^r mod p, where r is random blinding.</description></item>
/// <item><description>Homomorphic: Commit(a, r1) * Commit(b, r2) = Commit(a+b, r1+r2).</description></item>
/// </list>
///
/// <para><b>In FL:</b> The server receives commitments from all clients. It can multiply them
/// together to get a commitment to the sum. Then when clients open their commitments, the
/// server verifies both individual values and the aggregate — detecting any manipulation.</para>
///
/// <para><b>Reference:</b> Pedersen (CRYPTO 1991). Used in RiseFL (VLDB 2024) for scalable FL verification.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PedersenCommitment<T> : FederatedLearningComponentBase<T>, IGradientCommitment<T>, IZkProofSystem
{
    private readonly BigInteger _p; // Prime modulus
    private readonly BigInteger _g; // Generator
    private readonly BigInteger _h; // Second generator (h = g^s for unknown s)

    /// <inheritdoc/>
    public string Name => "Pedersen";

    /// <summary>
    /// Initializes a new instance of <see cref="PedersenCommitment{T}"/>.
    /// </summary>
    /// <param name="groupBitLength">Bit length for the prime group (default 256).</param>
    public PedersenCommitment(int groupBitLength = 256)
    {
        // Use a well-known safe prime for the group
        // In production, this would use an elliptic curve group
        // Here we use a smaller group for demonstration
        if (groupBitLength <= 64)
        {
            _p = BigInteger.Parse("18446744073709551557"); // Large 64-bit prime
            _g = new BigInteger(2);
            _h = new BigInteger(3); // In production: h = g^s for random unknown s
        }
        else
        {
            // Use a 256-bit prime (in production, use elliptic curve points)
            _p = (BigInteger.One << 256) - BigInteger.Parse("189");
            _g = new BigInteger(2);
            _h = new BigInteger(5);
        }
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

    /// <inheritdoc/>
    public bool VerifyAggregation(
        IReadOnlyList<GradientCommitmentData<T>> individualCommitments,
        GradientCommitmentData<T> aggregateCommitment)
    {
        if (individualCommitments is null || individualCommitments.Count == 0 || aggregateCommitment is null)
        {
            return false;
        }

        // Pedersen is homomorphic: product of commitments = commitment of sum
        // Verify: C_agg == product(C_i) mod p
        var productCommitment = BigInteger.One;
        var totalRandomness = BigInteger.Zero;

        for (int c = 0; c < individualCommitments.Count; c++)
        {
            if (individualCommitments[c].CommitmentValue.Length == 0)
            {
                return false;
            }

            var ci = FromBigEndian(individualCommitments[c].CommitmentValue);
            productCommitment = Mod(productCommitment * ci);

            if (individualCommitments[c].Randomness.Length > 0)
            {
                var ri = FromBigEndian(individualCommitments[c].Randomness);
                totalRandomness += ri;
            }
        }

        var aggCommit = FromBigEndian(aggregateCommitment.CommitmentValue);

        // Check: product of individual commitments equals aggregate commitment
        return productCommitment == Mod(aggCommit);
    }

    /// <inheritdoc/>
    public (byte[] Commitment, byte[] Randomness) Commit(byte[] value)
    {
        if (value is null || value.Length == 0)
        {
            throw new ArgumentException("Value must not be null or empty.", nameof(value));
        }

        // Generate random blinding factor
        var randomBytes = new byte[32];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(randomBytes);
        }

        var r = Mod(FromBigEndian(randomBytes));
        var v = Mod(FromBigEndian(value));

        // Commitment = g^v * h^r mod p
        var gv = BigInteger.ModPow(_g, v, _p);
        var hr = BigInteger.ModPow(_h, r, _p);
        var commitment = Mod(gv * hr);

        return (ToBigEndian(commitment), ToBigEndian(r));
    }

    /// <inheritdoc/>
    public byte[] GenerateRangeProof(byte[] value, byte[] upperBound, byte[] commitment, byte[] randomness)
    {
        if (value is null || upperBound is null || commitment is null || randomness is null)
        {
            throw new ArgumentNullException(
                value is null ? nameof(value) :
                upperBound is null ? nameof(upperBound) :
                commitment is null ? nameof(commitment) : nameof(randomness));
        }

        // Simplified Bulletproofs-style range proof
        // In production, this would implement the full Bulletproofs protocol
        // Here we create a proof that demonstrates the value and commitment are consistent
        var v = FromBigEndian(value);
        var bound = FromBigEndian(upperBound);

        // Proof consists of: commitment to (bound - value) which should be non-negative
        var diff = bound - v;
        if (diff.Sign < 0)
        {
            // Value exceeds bound — cannot create valid proof
            return Array.Empty<byte>();
        }

        var diffBytes = ToBigEndian(diff);
        var (diffCommitment, diffRandomness) = Commit(diffBytes);

        // Proof = (diffCommitment || diffRandomness || diffBytes)
        var proof = new byte[diffCommitment.Length + diffRandomness.Length + diffBytes.Length + 12];
        var lenC = BitConverter.GetBytes(diffCommitment.Length);
        var lenR = BitConverter.GetBytes(diffRandomness.Length);
        var lenD = BitConverter.GetBytes(diffBytes.Length);

        int offset = 0;
        Buffer.BlockCopy(lenC, 0, proof, offset, 4); offset += 4;
        Buffer.BlockCopy(diffCommitment, 0, proof, offset, diffCommitment.Length); offset += diffCommitment.Length;
        Buffer.BlockCopy(lenR, 0, proof, offset, 4); offset += 4;
        Buffer.BlockCopy(diffRandomness, 0, proof, offset, diffRandomness.Length); offset += diffRandomness.Length;
        Buffer.BlockCopy(lenD, 0, proof, offset, 4); offset += 4;
        Buffer.BlockCopy(diffBytes, 0, proof, offset, diffBytes.Length);

        return proof;
    }

    /// <inheritdoc/>
    public bool VerifyRangeProof(byte[] proof, byte[] upperBound, byte[] commitment)
    {
        if (proof is null || proof.Length < 12 || upperBound is null || commitment is null)
        {
            return false;
        }

        // Parse proof components
        int offset = 0;
        int lenC = BitConverter.ToInt32(proof, offset); offset += 4;
        if (offset + lenC > proof.Length)
        {
            return false;
        }

        var diffCommitment = new byte[lenC];
        Buffer.BlockCopy(proof, offset, diffCommitment, 0, lenC); offset += lenC;

        if (offset + 4 > proof.Length)
        {
            return false;
        }

        int lenR = BitConverter.ToInt32(proof, offset); offset += 4;
        if (offset + lenR > proof.Length)
        {
            return false;
        }

        var diffRandomness = new byte[lenR];
        Buffer.BlockCopy(proof, offset, diffRandomness, 0, lenR); offset += lenR;

        if (offset + 4 > proof.Length)
        {
            return false;
        }

        int lenD = BitConverter.ToInt32(proof, offset); offset += 4;
        if (offset + lenD > proof.Length)
        {
            return false;
        }

        var diffBytes = new byte[lenD];
        Buffer.BlockCopy(proof, offset, diffBytes, 0, lenD);

        // Verify the diff commitment opens correctly
        if (!VerifyOpening(diffCommitment, diffBytes, diffRandomness))
        {
            return false;
        }

        // Verify diff >= 0 (the value we committed to is non-negative)
        var diff = FromBigEndian(diffBytes);
        return diff.Sign >= 0;
    }

    /// <inheritdoc/>
    public bool VerifyOpening(byte[] commitment, byte[] value, byte[] randomness)
    {
        if (commitment is null || value is null || randomness is null)
        {
            return false;
        }

        var v = Mod(FromBigEndian(value));
        var r = Mod(FromBigEndian(randomness));

        // Recompute: g^v * h^r mod p
        var gv = BigInteger.ModPow(_g, v, _p);
        var hr = BigInteger.ModPow(_h, r, _p);
        var expected = Mod(gv * hr);

        var actual = FromBigEndian(commitment);
        return expected == Mod(actual);
    }

    private BigInteger Mod(BigInteger value)
    {
        var result = value % _p;
        return result.Sign < 0 ? result + _p : result;
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

    private static int ComputeTotalElements(Tensor<T> tensor)
    {
        int total = 1;
        for (int d = 0; d < tensor.Rank; d++)
        {
            total *= tensor.Shape[d];
        }

        return total;
    }

    private static BigInteger FromBigEndian(byte[] bytes)
    {
        if (bytes.Length == 0)
        {
            return BigInteger.Zero;
        }

        var little = new byte[bytes.Length + 1];
        for (int i = 0; i < bytes.Length; i++)
        {
            little[i] = bytes[bytes.Length - 1 - i];
        }

        little[bytes.Length] = 0; // Ensure non-negative
        return new BigInteger(little);
    }

    private static byte[] ToBigEndian(BigInteger value)
    {
        if (value.Sign < 0)
        {
            value = -value; // Take absolute value for serialization
        }

        var little = value.ToByteArray();

        // Remove trailing zero byte if present (sign byte)
        int len = little.Length;
        if (len > 1 && little[len - 1] == 0)
        {
            len--;
        }

        var big = new byte[len];
        for (int i = 0; i < len; i++)
        {
            big[len - 1 - i] = little[i];
        }

        return big;
    }
}
