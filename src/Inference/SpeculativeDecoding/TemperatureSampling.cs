using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Temperature scaling and sampling shared by the speculative decoders and draft models, so every stage
/// of draft-and-verify samples from the SAME distribution.
/// </summary>
/// <remarks>
/// Temperature 0 (or below) means greedy decoding, as in PyTorch/Hugging Face <c>generate</c>: the
/// distribution collapses to a one-hot at the argmax. Scaling by <c>1/temperature</c> instead divides by
/// zero; every probability below 1 then raises to 0, the distribution sums to 0, and sampling falls through
/// to the last vocabulary id, emitting that token on every step.
/// </remarks>
internal static class TemperatureSampling<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>True when <paramref name="temperature"/> requests greedy (argmax) decoding.</summary>
    public static bool IsGreedy(T temperature) => NumOps.LessThanOrEquals(temperature, NumOps.Zero);

    /// <summary>
    /// Rescales a probability distribution to <paramref name="temperature"/>: <c>p_i^(1/t)</c> renormalized,
    /// a one-hot at the argmax for <c>t &lt;= 0</c>, and the input unchanged for <c>t == 1</c>.
    /// </summary>
    public static Vector<T> Apply(Vector<T> distribution, T temperature)
    {
        if (NumOps.Equals(temperature, NumOps.One))
        {
            return distribution;
        }

        var result = new Vector<T>(distribution.Length);
        if (IsGreedy(temperature))
        {
            if (distribution.Length > 0)
            {
                result[ArgMax(distribution)] = NumOps.One;
            }
            return result;
        }

        T invTemp = NumOps.Divide(NumOps.One, temperature);
        T sum = NumOps.Zero;
        for (int i = 0; i < distribution.Length; i++)
        {
            result[i] = NumOps.Power(distribution[i], invTemp);
            sum = NumOps.Add(sum, result[i]);
        }

        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.Divide(result[i], sum);
            }
            return result;
        }

        // Every entry underflowed (a very low temperature): the limit is the greedy one-hot.
        var oneHot = new Vector<T>(distribution.Length);
        if (distribution.Length > 0)
        {
            oneHot[ArgMax(distribution)] = NumOps.One;
        }
        return oneHot;
    }

    /// <summary>
    /// Draws an index from <paramref name="distribution"/>. Zero-probability entries are never chosen, so a
    /// one-hot distribution always returns its hot index.
    /// </summary>
    public static int Sample(Vector<T> distribution, Random random)
    {
        T r = NumOps.FromDouble(random.NextDouble());
        T cumulative = NumOps.Zero;
        int lastNonZero = -1;

        for (int i = 0; i < distribution.Length; i++)
        {
            if (!NumOps.GreaterThan(distribution[i], NumOps.Zero))
            {
                continue;
            }
            lastNonZero = i;
            cumulative = NumOps.Add(cumulative, distribution[i]);
            if (NumOps.LessThan(r, cumulative))
            {
                return i;
            }
        }

        // Rounding left the cumulative sum just below r: the last index with mass is the right answer.
        return lastNonZero >= 0 ? lastNonZero : ArgMax(distribution);
    }

    /// <summary>Index of the largest entry (the first one on ties).</summary>
    public static int ArgMax(Vector<T> distribution)
    {
        int best = 0;
        for (int i = 1; i < distribution.Length; i++)
        {
            if (NumOps.GreaterThan(distribution[i], distribution[best]))
            {
                best = i;
            }
        }
        return best;
    }
}
