using AiDotNet.Tensors.Engines;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// Sampling building blocks over logits carried in T, for the speech language models: an order with LINQ's tie rule
/// and the draw of a multinomial over weights in a given order.
/// </summary>
/// <remarks>
/// The engine's argsort is not stable on ties (its CPU path is an introsort), so <see cref="StableOrder"/> follows it
/// with an int pass that puts each run of equal values back in ascending index order, which is what
/// <c>OrderBy(v).ThenBy(i)</c> and the stable <c>OrderBy</c> of the reference samplers give.
/// </remarks>
internal static class TtsSampling<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>The indices of <paramref name="values"/> (flattened) by value, descending or ascending, equal values
    /// in ascending index order.</summary>
    public static int[] StableOrder(Tensor<T> values, bool descending)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        var engine = AiDotNetEngine.Current;
        int n = values.Length;
        if (n == 0) return Array.Empty<int>();
        var flat = engine.Reshape(values, new[] { n });
        var order = engine.TensorArgsort(flat, 0, descending).ToArray();
        if (n == 1) return order;
        var sorted = engine.TensorIndexSelect(flat, new Tensor<int>((int[])order.Clone(), new[] { n }), 0);
        var same = engine.TensorEquals(
            engine.TensorSlice(sorted, new[] { 1 }, new[] { n - 1 }),
            engine.TensorSlice(sorted, new[] { 0 }, new[] { n - 1 })).Cast<int>().ToArray();
        int start = 0;
        for (int j = 1; j <= n; j++)
        {
            if (j < n && same[j - 1] != 0) continue;
            if (j - start > 1) Array.Sort(order, start, j - start);
            start = j;
        }
        return order;
    }

    /// <summary>
    /// The position a multinomial draw lands on over non-negative <paramref name="weights"/> (in visiting order):
    /// with <c>d = u · Σw</c>, the first position whose running sum exceeds d (<paramref name="inclusive"/>: reaches d,
    /// skipping zero weights); the last position (inclusive: the last positive weight) when rounding leaves none.
    /// </summary>
    public static int DrawPosition(Tensor<T> weights, double u, bool inclusive)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        var engine = AiDotNetEngine.Current;
        int n = weights.Length;
        if (n == 0) throw new ArgumentException("There is nothing to draw from.", nameof(weights));
        var flat = engine.Reshape(weights, new[] { n });
        var running = engine.TensorCumSum(flat, 0);
        T draw = Ops.Multiply(Ops.FromDouble(u), engine.TensorSum(flat));
        if (!inclusive)
        {
            int below = n - engine.TensorGreaterThan(running, draw).Cast<int>().ToArray().Sum();
            return Math.Min(below, n - 1);
        }
        int before = engine.TensorLessThan(running, draw).Cast<int>().ToArray().Sum();
        var positive = engine.TensorGreaterThan(flat, Ops.Zero).Cast<int>().ToArray();
        for (int j = before; j < n; j++)
            if (positive[j] != 0) return j;
        int last = Array.LastIndexOf(positive, 1);
        return last >= 0 ? last : n - 1;
    }

    /// <summary>A <c>[n]</c> tensor of <paramref name="values"/> with <paramref name="fill"/> where <paramref name="mask"/> is set.</summary>
    public static Tensor<T> Fill(Tensor<T> values, bool[] mask, T fill)
    {
        var bits = new int[mask.Length];
        for (int i = 0; i < bits.Length; i++) bits[i] = mask[i] ? 1 : 0;
        return AiDotNetEngine.Current.TensorWhere(new Tensor<int>(bits, new[] { bits.Length }).Cast<T>(),
            Tensor<T>.CreateDefault(new[] { bits.Length }, fill), values);
    }

    /// <summary>exp(v − max) with removed (−∞) scores at exactly 0.</summary>
    public static Tensor<T> UnnormalizedProbabilities(Tensor<T> scores)
    {
        var engine = AiDotNetEngine.Current;
        T negativeInfinity = Ops.FromDouble(double.NegativeInfinity);
        var removed = engine.TensorEquals(scores, negativeInfinity);
        var shifted = engine.TensorExp(engine.TensorAddScalar(scores, Ops.Negate(engine.TensorMaxValue(scores))));
        return engine.TensorWhere(removed, Tensor<T>.CreateDefault(scores._shape, Ops.Zero), shifted);
    }
}
