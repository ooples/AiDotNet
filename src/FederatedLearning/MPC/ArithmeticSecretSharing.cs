using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

        int[] shape = value.Shape.ToArray();
        var shares = new Tensor<T>[numberOfParties];

        // Generate n-1 random shares (draws in party-major order)
        for (int p = 0; p < numberOfParties - 1; p++)
        {
            shares[p] = UniformTensor(shape);
        }

        // Last share = secret - sum of other shares
        var others = new Tensor<T>[numberOfParties - 1];
        Array.Copy(shares, others, numberOfParties - 1);
        shares[numberOfParties - 1] = Engine.TensorSubtract(value, SumShares(others));

        return shares;
    }

    /// <inheritdoc/>
    public Tensor<T> Reconstruct(Tensor<T>[] shares)
    {
        if (shares is null || shares.Length == 0)
        {
            throw new ArgumentException("Shares array must not be null or empty.", nameof(shares));
        }

        return SumShares(shares);
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
            result[p] = Engine.TensorAdd(sharesA[p], sharesB[p]);
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

        // Use Beaver triple: (a, b, c) where c = a*b
        // To compute x*y: open e = x-a, open d = y-b, then z = c + e*[b] + d*[a] + e*d
        var triple = GetNextBeaverTriple(sharesA[0].Shape.ToArray(), n);

        // Compute e = x - a and d = y - b (secret-shared differences)
        var sharesE = SecureAdd(sharesA, Negate(triple.SharesA));
        var sharesD = SecureAdd(sharesB, Negate(triple.SharesB));

        // Open e and d (both parties learn these values)
        var e = Reconstruct(sharesE);
        var d = Reconstruct(sharesD);

        // Compute z = c + e*b + d*a + e*d
        // Each party locally computes: c_p + e * b_p + d * a_p, and only party 0 adds e*d to avoid double-counting
        var result = new Tensor<T>[n];
        for (int p = 0; p < n; p++)
        {
            var local = Engine.TensorAdd(triple.SharesC[p], Engine.TensorAdd(
                Engine.TensorMultiply(e, triple.SharesB[p]),
                Engine.TensorMultiply(d, triple.SharesA[p])));
            result[p] = p == 0 ? Engine.TensorAdd(local, Engine.TensorMultiply(e, d)) : local;
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

        return Share(Engine.TensorGreaterThan(plainDiff, NumOps.Zero), sharesA.Length);
    }

    /// <inheritdoc/>
    public Tensor<T>[] ScalarMultiply(Tensor<T>[] shares, T scalar)
    {
        if (shares is null)
        {
            throw new ArgumentNullException(nameof(shares));
        }

        var result = new Tensor<T>[shares.Length];
        for (int p = 0; p < shares.Length; p++)
        {
            result[p] = Engine.TensorMultiplyScalar(shares[p], scalar);
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

        // A pre-generated triple for a different party count or shape cannot mask this product; put a matching
        // one in its place (the unmatched triple stays queued for a later call that fits it).
        if (triple.SharesA.Length != numberOfParties || !triple.SharesA[0].Shape.ToArray().SequenceEqual(shape))
        {
            triple = CreateBeaverTriple(shape, numberOfParties);
            _beaverTriples.Insert(_tripleIndex, triple);
        }

        _tripleIndex++;
        return triple;
    }

    private void GenerateBeaverTriples(int[] shape, int count, int numberOfParties)
    {
        for (int t = 0; t < count; t++)
        {
            _beaverTriples.Add(CreateBeaverTriple(shape, numberOfParties));
        }
    }

    private (Tensor<T>[] SharesA, Tensor<T>[] SharesB, Tensor<T>[] SharesC) CreateBeaverTriple(int[] shape, int numberOfParties)
    {
        int totalElements = 1;
        for (int d = 0; d < shape.Length; d++)
        {
            totalElements *= shape[d];
        }

        // a and b are drawn element by element (a then b), and c = a * b is formed in double precision.
        var aValues = new double[totalElements];
        var bValues = new double[totalElements];
        var cValues = new double[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            aValues[i] = _random.NextDouble() * 2.0 - 1.0;
            bValues[i] = _random.NextDouble() * 2.0 - 1.0;
            cValues[i] = aValues[i] * bValues[i];
        }

        var sharesA = Share(new Tensor<double>(aValues, shape).Cast<T>(), numberOfParties);
        var sharesB = Share(new Tensor<double>(bValues, shape).Cast<T>(), numberOfParties);
        var sharesC = Share(new Tensor<double>(cValues, shape).Cast<T>(), numberOfParties);

        return (sharesA, sharesB, sharesC);
    }

    private Tensor<T>[] Negate(Tensor<T>[] shares)
    {
        var result = new Tensor<T>[shares.Length];
        for (int p = 0; p < shares.Length; p++)
        {
            result[p] = Engine.TensorNegate(shares[p]);
        }

        return result;
    }

    private Tensor<T> SumShares(Tensor<T>[] shares)
    {
        var sum = Engine.TensorAdd(shares[0], Engine.TensorFullLike(shares[0], NumOps.Zero));
        for (int p = 1; p < shares.Length; p++)
        {
            sum = Engine.TensorAdd(sum, shares[p]);
        }

        return sum;
    }

    /// <summary>
    /// A share drawn uniformly from [-1, 1) element by element from the protocol generator.
    /// </summary>
    private Tensor<T> UniformTensor(int[] shape)
    {
        int totalElements = 1;
        for (int d = 0; d < shape.Length; d++)
        {
            totalElements *= shape[d];
        }

        var values = new double[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            values[i] = _random.NextDouble() * 2.0 - 1.0;
        }

        return new Tensor<double>(values, shape).Cast<T>();
    }
}
