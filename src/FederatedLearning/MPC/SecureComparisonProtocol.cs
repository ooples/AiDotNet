using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Implements secure greater-than comparison on secret-shared values.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In federated learning, operations like gradient clipping
/// and top-k sparsification require comparing values (e.g., "is this gradient's norm greater
/// than the threshold?"). But the actual gradient values are secret — no single party should
/// see them. Secure comparison lets parties answer "is A &gt; B?" without revealing A or B.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="bullet">
/// <item><description>Convert the comparison to a bit-level operation using Goldreich-Micali-Wigderson style.</description></item>
/// <item><description>Use boolean secret sharing and garbled circuits for the bit comparison.</description></item>
/// <item><description>The result is a secret-shared bit (1 if A &gt; B, 0 otherwise).</description></item>
/// </list>
///
/// <para><b>Applications in FL:</b></para>
/// <list type="bullet">
/// <item><description><b>Gradient clipping:</b> Clip if ||g|| &gt; C, without revealing ||g||.</description></item>
/// <item><description><b>Top-k selection:</b> Find the k largest gradient components secretly.</description></item>
/// <item><description><b>Median aggregation:</b> Requires sorting, which uses comparisons.</description></item>
/// <item><description><b>Krum/Bulyan:</b> Requires distance comparisons between gradient vectors.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SecureComparisonProtocol<T> : FederatedLearningComponentBase<T>
{
    private readonly ISecureComputationProtocol<T> _protocol;
    private readonly int _bitPrecision;

    /// <summary>
    /// Initializes a new instance of <see cref="SecureComparisonProtocol{T}"/>.
    /// </summary>
    /// <param name="protocol">The underlying MPC protocol for arithmetic operations.</param>
    /// <param name="bitPrecision">Number of bits for fixed-point representation (default 32).</param>
    public SecureComparisonProtocol(ISecureComputationProtocol<T> protocol, int bitPrecision = 32)
    {
        _protocol = protocol ?? throw new ArgumentNullException(nameof(protocol));
        _bitPrecision = bitPrecision;
    }

    /// <summary>
    /// Compares two secret-shared tensors element-wise: returns shares of (A &gt; B).
    /// </summary>
    /// <param name="sharesA">Secret shares of tensor A.</param>
    /// <param name="sharesB">Secret shares of tensor B.</param>
    /// <returns>Secret shares of a binary tensor (1 where A &gt; B, 0 otherwise).</returns>
    public Tensor<T>[] Compare(Tensor<T>[] sharesA, Tensor<T>[] sharesB)
    {
        if (sharesA is null || sharesB is null)
        {
            throw new ArgumentNullException(sharesA is null ? nameof(sharesA) : nameof(sharesB));
        }

        return _protocol.SecureCompare(sharesA, sharesB);
    }

    /// <summary>
    /// Computes the element-wise maximum of two secret-shared tensors.
    /// max(A, B) = (A &gt; B) * A + (1 - (A &gt; B)) * B
    /// </summary>
    /// <param name="sharesA">Secret shares of tensor A.</param>
    /// <param name="sharesB">Secret shares of tensor B.</param>
    /// <returns>Secret shares of element-wise max(A, B).</returns>
    public Tensor<T>[] SecureMax(Tensor<T>[] sharesA, Tensor<T>[] sharesB)
    {
        if (sharesA is null || sharesB is null)
        {
            throw new ArgumentNullException(sharesA is null ? nameof(sharesA) : nameof(sharesB));
        }

        // comp = (A > B) as shares of 0/1
        var comp = _protocol.SecureCompare(sharesA, sharesB);

        // max = comp * A + (1 - comp) * B
        var compTimesA = _protocol.SecureMultiply(comp, sharesA);
        var oneMinusComp = ComputeOneMinus(comp);
        var oneMinusCompTimesB = _protocol.SecureMultiply(oneMinusComp, sharesB);

        return _protocol.SecureAdd(compTimesA, oneMinusCompTimesB);
    }

    /// <summary>
    /// Computes the element-wise minimum of two secret-shared tensors.
    /// </summary>
    /// <param name="sharesA">Secret shares of tensor A.</param>
    /// <param name="sharesB">Secret shares of tensor B.</param>
    /// <returns>Secret shares of element-wise min(A, B).</returns>
    public Tensor<T>[] SecureMin(Tensor<T>[] sharesA, Tensor<T>[] sharesB)
    {
        if (sharesA is null || sharesB is null)
        {
            throw new ArgumentNullException(sharesA is null ? nameof(sharesA) : nameof(sharesB));
        }

        // min(A, B) = A + B - max(A, B)
        var sum = _protocol.SecureAdd(sharesA, sharesB);
        var max = SecureMax(sharesA, sharesB);

        // Negate max
        var negMax = new Tensor<T>[max.Length];
        for (int p = 0; p < max.Length; p++)
        {
            int total = ComputeTotalElements(max[p]);
            negMax[p] = new Tensor<T>(max[p].Shape);
            for (int i = 0; i < total; i++)
            {
                negMax[p][i] = NumOps.FromDouble(-NumOps.ToDouble(max[p][i]));
            }
        }

        return _protocol.SecureAdd(sum, negMax);
    }

    /// <summary>
    /// Computes the L2 norm squared of a secret-shared tensor: sum(x_i^2).
    /// </summary>
    /// <param name="shares">Secret shares of the tensor.</param>
    /// <returns>Secret shares of a scalar tensor containing the squared norm.</returns>
    public Tensor<T>[] SecureNormSquared(Tensor<T>[] shares)
    {
        if (shares is null)
        {
            throw new ArgumentNullException(nameof(shares));
        }

        // norm^2 = sum(x_i * x_i)
        var squared = _protocol.SecureMultiply(shares, shares);

        // Sum all elements into a scalar
        int n = squared.Length;
        int totalElements = ComputeTotalElements(squared[0]);

        var result = new Tensor<T>[n];
        for (int p = 0; p < n; p++)
        {
            result[p] = new Tensor<T>(new[] { 1 });
            double sum = 0.0;
            for (int i = 0; i < totalElements; i++)
            {
                sum += NumOps.ToDouble(squared[p][i]);
            }

            result[p][0] = NumOps.FromDouble(sum);
        }

        return result;
    }

    private Tensor<T>[] ComputeOneMinus(Tensor<T>[] shares)
    {
        // Create shares of "1" and subtract
        var result = new Tensor<T>[shares.Length];
        for (int p = 0; p < shares.Length; p++)
        {
            int total = ComputeTotalElements(shares[p]);
            result[p] = new Tensor<T>(shares[p].Shape);
            for (int i = 0; i < total; i++)
            {
                double val = NumOps.ToDouble(shares[p][i]);
                if (p == 0)
                {
                    // Only party 0 subtracts from 1
                    result[p][i] = NumOps.FromDouble(1.0 - val);
                }
                else
                {
                    // Other parties negate their share
                    result[p][i] = NumOps.FromDouble(-val);
                }
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
