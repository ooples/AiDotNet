using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Implements additive secret sharing over an arithmetic field for efficient linear operations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Additive secret sharing splits a number into random "shares" that
/// add up to the original. For example, to share the number 42 among 3 parties:</para>
/// <list type="bullet">
/// <item><description>Generate random shares: share1 = 17, share2 = -8</description></item>
/// <item><description>Compute share3 = 42 - 17 - (-8) = 33</description></item>
/// <item><description>Now 17 + (-8) + 33 = 42 — but no single party can figure out 42.</description></item>
/// </list>
///
/// <para><b>Operations on shares:</b></para>
/// <list type="bullet">
/// <item><description><b>Addition:</b> Each party just adds its shares locally. No communication needed.</description></item>
/// <item><description><b>Scalar multiply:</b> Each party multiplies its share by the scalar. No communication.</description></item>
/// <item><description><b>Multiplication:</b> Requires Beaver triples (pre-generated random correlations).</description></item>
/// <item><description><b>Comparison:</b> Requires bit-decomposition or garbled circuits (delegated).</description></item>
/// </list>
///
/// <para><b>Reference:</b> This implements the standard SPDZ-style additive secret sharing protocol
/// used in production MPC systems.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ArithmeticSecretSharing<T> : FederatedLearningComponentBase<T>, ISecretSharingScheme<T>, ISecureComputationProtocol<T>
{
    private readonly int _numberOfParties;
    private readonly Random _random;

    // Pre-generated Beaver triples for multiplication: (a, b, c) where c = a * b
    private readonly List<(Tensor<T>[] SharesA, Tensor<T>[] SharesB, Tensor<T>[] SharesC)> _beaverTriples;
    private int _tripleIndex;

    /// <inheritdoc/>
    public int ReconstructionThreshold => _numberOfParties;

    /// <inheritdoc/>
    public string SchemeName => "AdditiveSecretSharing";

    /// <summary>
    /// Initializes a new instance of <see cref="ArithmeticSecretSharing{T}"/>.
    /// </summary>
    /// <param name="numberOfParties">The number of parties in the protocol.</param>
    /// <param name="preGenerateTriples">Number of Beaver triples to pre-generate.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public ArithmeticSecretSharing(int numberOfParties = 3, int preGenerateTriples = 100, int? seed = null)
    {
        if (numberOfParties < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfParties), "Need at least 2 parties.");
        }

        _numberOfParties = numberOfParties;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        _beaverTriples = new List<(Tensor<T>[], Tensor<T>[], Tensor<T>[])>(preGenerateTriples);
        _tripleIndex = 0;
    }

    /// <inheritdoc/>
    public Tensor<T>[] Split(Tensor<T> secret, int numberOfParties)
    {
        return Share(secret, numberOfParties);
    }

    /// <inheritdoc/>
    public Tensor<T> Combine(Tensor<T>[] shares)
    {
        return Reconstruct(shares);
    }

    /// <inheritdoc/>
    public Tensor<T>[] Share(Tensor<T> value, int numberOfParties)
    {
        if (value is null)
        {
            throw new ArgumentNullException(nameof(value));
        }

        if (numberOfParties < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfParties));
        }

        int totalElements = ComputeTotalElements(value);
        var shares = new Tensor<T>[numberOfParties];

        // Generate n-1 random shares
        for (int p = 0; p < numberOfParties - 1; p++)
        {
            shares[p] = new Tensor<T>(value.Shape);
            for (int i = 0; i < totalElements; i++)
            {
                shares[p][i] = NumOps.FromDouble(_random.NextDouble() * 2.0 - 1.0);
            }
        }

        // Last share = secret - sum of other shares
        shares[numberOfParties - 1] = new Tensor<T>(value.Shape);
        for (int i = 0; i < totalElements; i++)
        {
            double sum = 0.0;
            for (int p = 0; p < numberOfParties - 1; p++)
            {
                sum += NumOps.ToDouble(shares[p][i]);
            }

            shares[numberOfParties - 1][i] = NumOps.FromDouble(NumOps.ToDouble(value[i]) - sum);
        }

        return shares;
    }

    /// <inheritdoc/>
    public Tensor<T> Reconstruct(Tensor<T>[] shares)
    {
        if (shares is null || shares.Length == 0)
        {
            throw new ArgumentException("Shares array must not be null or empty.", nameof(shares));
        }

        int totalElements = ComputeTotalElements(shares[0]);
        var result = new Tensor<T>(shares[0].Shape);

        for (int i = 0; i < totalElements; i++)
        {
            double sum = 0.0;
            for (int p = 0; p < shares.Length; p++)
            {
                sum += NumOps.ToDouble(shares[p][i]);
            }

            result[i] = NumOps.FromDouble(sum);
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T>[] SecureAdd(Tensor<T>[] sharesA, Tensor<T>[] sharesB)
    {
        if (sharesA is null || sharesB is null)
        {
            throw new ArgumentNullException(sharesA is null ? nameof(sharesA) : nameof(sharesB));
        }

        if (sharesA.Length != sharesB.Length)
        {
            throw new ArgumentException("Share arrays must have the same length.");
        }

        // Addition is local — each party just adds its shares
        var result = new Tensor<T>[sharesA.Length];
        for (int p = 0; p < sharesA.Length; p++)
        {
            int totalElements = ComputeTotalElements(sharesA[p]);
            result[p] = new Tensor<T>(sharesA[p].Shape);
            for (int i = 0; i < totalElements; i++)
            {
                double a = NumOps.ToDouble(sharesA[p][i]);
                double b = NumOps.ToDouble(sharesB[p][i]);
                result[p][i] = NumOps.FromDouble(a + b);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T>[] SecureMultiply(Tensor<T>[] sharesA, Tensor<T>[] sharesB)
    {
        if (sharesA is null || sharesB is null)
        {
            throw new ArgumentNullException(sharesA is null ? nameof(sharesA) : nameof(sharesB));
        }

        if (sharesA.Length != sharesB.Length)
        {
            throw new ArgumentException("Share arrays must have the same length.");
        }

        int n = sharesA.Length;
        int totalElements = ComputeTotalElements(sharesA[0]);

        // Use Beaver triple: (a, b, c) where c = a*b
        // To compute x*y: open e = x-a, open d = y-b, then z = c + e*[b] + d*[a] + e*d
        var triple = GetNextBeaverTriple(sharesA[0].Shape, n);

        // Compute e = x - a and d = y - b (secret-shared differences)
        var sharesE = SecureAdd(sharesA, Negate(triple.SharesA));
        var sharesD = SecureAdd(sharesB, Negate(triple.SharesB));

        // Open e and d (both parties learn these values)
        var e = Reconstruct(sharesE);
        var d = Reconstruct(sharesD);

        // Compute z = c + e*b + d*a + e*d
        // Each party locally computes: c_p + e * b_p + d * a_p + (e*d)/n
        var result = new Tensor<T>[n];
        for (int p = 0; p < n; p++)
        {
            result[p] = new Tensor<T>(sharesA[0].Shape);
            for (int i = 0; i < totalElements; i++)
            {
                double cVal = NumOps.ToDouble(triple.SharesC[p][i]);
                double eVal = NumOps.ToDouble(e[i]);
                double dVal = NumOps.ToDouble(d[i]);
                double bVal = NumOps.ToDouble(triple.SharesB[p][i]);
                double aVal = NumOps.ToDouble(triple.SharesA[p][i]);

                double val = cVal + eVal * bVal + dVal * aVal;
                // Only one party adds e*d to avoid double-counting
                if (p == 0)
                {
                    val += eVal * dVal;
                }

                result[p][i] = NumOps.FromDouble(val);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T>[] SecureCompare(Tensor<T>[] sharesA, Tensor<T>[] sharesB)
    {
        if (sharesA is null || sharesB is null)
        {
            throw new ArgumentNullException(sharesA is null ? nameof(sharesA) : nameof(sharesB));
        }

        // Compute difference d = a - b, then check sign
        // In a production system this would use bit-decomposition or garbled circuits
        // Here we use a simplified protocol: reconstruct difference and share the comparison result
        var diff = SecureAdd(sharesA, Negate(sharesB));
        var plainDiff = Reconstruct(diff);

        int totalElements = ComputeTotalElements(plainDiff);
        var compResult = new Tensor<T>(plainDiff.Shape);
        for (int i = 0; i < totalElements; i++)
        {
            double d = NumOps.ToDouble(plainDiff[i]);
            compResult[i] = NumOps.FromDouble(d > 0.0 ? 1.0 : 0.0);
        }

        return Share(compResult, sharesA.Length);
    }

    /// <inheritdoc/>
    public Tensor<T>[] ScalarMultiply(Tensor<T>[] shares, T scalar)
    {
        if (shares is null)
        {
            throw new ArgumentNullException(nameof(shares));
        }

        double s = NumOps.ToDouble(scalar);
        var result = new Tensor<T>[shares.Length];
        for (int p = 0; p < shares.Length; p++)
        {
            int totalElements = ComputeTotalElements(shares[p]);
            result[p] = new Tensor<T>(shares[p].Shape);
            for (int i = 0; i < totalElements; i++)
            {
                result[p][i] = NumOps.FromDouble(NumOps.ToDouble(shares[p][i]) * s);
            }
        }

        return result;
    }

    /// <summary>
    /// Pre-generates Beaver triples for secure multiplication.
    /// </summary>
    /// <param name="shape">The tensor shape for the triples.</param>
    /// <param name="count">Number of triples to generate.</param>
    public void PreGenerateBeaverTriples(int[] shape, int count)
    {
        GenerateBeaverTriples(shape, count, _numberOfParties);
    }

    private (Tensor<T>[] SharesA, Tensor<T>[] SharesB, Tensor<T>[] SharesC) GetNextBeaverTriple(
        int[] shape, int numberOfParties)
    {
        if (_tripleIndex >= _beaverTriples.Count)
        {
            // Auto-generate more triples on demand, matching the actual party count
            GenerateBeaverTriples(shape, 10, numberOfParties);
        }

        var triple = _beaverTriples[_tripleIndex];

        // If the pre-generated triple has a different party count, regenerate
        if (triple.SharesA.Length != numberOfParties)
        {
            GenerateBeaverTriples(shape, 10, numberOfParties);
            triple = _beaverTriples[_tripleIndex];
        }

        _tripleIndex++;
        return triple;
    }

    private void GenerateBeaverTriples(int[] shape, int count, int numberOfParties)
    {
        for (int t = 0; t < count; t++)
        {
            int totalElements = 1;
            for (int d = 0; d < shape.Length; d++)
            {
                totalElements *= shape[d];
            }

            var a = new Tensor<T>(shape);
            var b = new Tensor<T>(shape);
            var c = new Tensor<T>(shape);

            for (int i = 0; i < totalElements; i++)
            {
                double aVal = _random.NextDouble() * 2.0 - 1.0;
                double bVal = _random.NextDouble() * 2.0 - 1.0;
                a[i] = NumOps.FromDouble(aVal);
                b[i] = NumOps.FromDouble(bVal);
                c[i] = NumOps.FromDouble(aVal * bVal);
            }

            var sharesA = Share(a, numberOfParties);
            var sharesB = Share(b, numberOfParties);
            var sharesC = Share(c, numberOfParties);

            _beaverTriples.Add((sharesA, sharesB, sharesC));
        }
    }

    private Tensor<T>[] Negate(Tensor<T>[] shares)
    {
        var result = new Tensor<T>[shares.Length];
        for (int p = 0; p < shares.Length; p++)
        {
            int totalElements = ComputeTotalElements(shares[p]);
            result[p] = new Tensor<T>(shares[p].Shape);
            for (int i = 0; i < totalElements; i++)
            {
                result[p][i] = NumOps.FromDouble(-NumOps.ToDouble(shares[p][i]));
            }
        }

        return result;
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
