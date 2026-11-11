namespace AiDotNet.Interpolation;

/// <summary>
/// Locates query points among sorted knots for the interpolation classes, using the engine's sorted search.
/// </summary>
/// <remarks>
/// <para>
/// Every one-dimensional interpolator has to answer the same question before it can evaluate anything: which
/// segment of the data contains the query point? These helpers answer it with
/// the engine's <c>TensorSearchSorted</c> (torch.searchsorted) instead of a
/// per-element scan, and translate the insertion index into the segment conventions the interpolators use.
/// </para>
/// <para><b>For Beginners:</b> If your x-values are [0, 10, 20, 30] and you ask about 15, the search reports
/// that 15 belongs between 10 and 20, so the interpolator uses the segment that starts at index 1.</para>
/// </remarks>
internal static class InterpolationSearch
{
    /// <summary>
    /// Returns the first segment whose right-hand knot is at or beyond <paramref name="value"/>, clamped to
    /// [<paramref name="minIndex"/>, <paramref name="maxIndex"/>].
    /// </summary>
    /// <remarks>
    /// Equivalent to scanning i upward from <paramref name="minIndex"/> and stopping at the first
    /// <c>value &lt;= knots[i + 1]</c>, falling back to <paramref name="maxIndex"/> when none qualifies.
    /// </remarks>
    /// <param name="sortedKnots">The knots, in ascending order.</param>
    /// <param name="value">The query point.</param>
    /// <param name="minIndex">The smallest segment index to return.</param>
    /// <param name="maxIndex">The largest segment index to return.</param>
    internal static int FirstSegmentReaching<T>(Vector<T> sortedKnots, T value, int minIndex, int maxIndex)
        => Clamp(SearchSorted(sortedKnots, value, right: false) - 1, minIndex, maxIndex);

    /// <summary>
    /// Returns the index of the last knot at or below <paramref name="value"/>, clamped to
    /// [<paramref name="minIndex"/>, <paramref name="maxIndex"/>].
    /// </summary>
    /// <remarks>
    /// Equivalent to the bisection that keeps <c>knots[low] &lt;= value</c> while narrowing [low, high].
    /// </remarks>
    /// <param name="sortedKnots">The knots, in ascending order.</param>
    /// <param name="value">The query point.</param>
    /// <param name="minIndex">The smallest index to return.</param>
    /// <param name="maxIndex">The largest index to return.</param>
    internal static int LastKnotAtOrBelow<T>(Vector<T> sortedKnots, T value, int minIndex, int maxIndex)
        => Clamp(SearchSorted(sortedKnots, value, right: true) - 1, minIndex, maxIndex);

    /// <summary>
    /// Returns the index of a knot exactly equal to <paramref name="value"/>, or -1 when there is none.
    /// </summary>
    /// <param name="sortedKnots">The knots, in ascending order.</param>
    /// <param name="value">The value to find.</param>
    internal static int IndexOfExact<T>(Vector<T> sortedKnots, T value)
    {
        int index = SearchSorted(sortedKnots, value, right: false);
        return index < sortedKnots.Length && MathHelper.GetNumericOperations<T>().Equals(sortedKnots[index], value)
            ? index
            : -1;
    }

    /// <summary>
    /// Returns the index of the first element equal to <paramref name="value"/> in an unordered vector, or -1.
    /// </summary>
    /// <param name="values">The values to search, in any order.</param>
    /// <param name="value">The value to find.</param>
    internal static int IndexOfFirstEqual<T>(Vector<T> values, T value)
    {
        if (values.Length == 0)
        {
            return -1;
        }

        // Nonzero lists matching positions in ascending order on every engine, so the first match wins
        Tensor<int> matches = AiDotNetEngine.Current.TensorNonzero(
            AiDotNetEngine.Current.TensorEquals(values.AsTensor(), value));
        return matches.Length == 0 ? -1 : matches.GetFlat(0);
    }

    /// <summary>
    /// The insertion index of <paramref name="value"/> in <paramref name="sortedKnots"/> (torch.searchsorted).
    /// </summary>
    private static int SearchSorted<T>(Vector<T> sortedKnots, T value, bool right)
    {
        Tensor<int> position = AiDotNetEngine.Current.TensorSearchSorted(
            sortedKnots.AsTensor(), Tensor<T>.FromScalar(value), right);
        return position[0];
    }

    /// <summary>
    /// The [n, n] matrix values[i] - values[j], with both operands broadcast explicitly to [n, n] so the
    /// subtraction is valid under a strict engine shape policy.
    /// </summary>
    /// <param name="values">A rank-1 tensor of n values.</param>
    internal static Tensor<T> PairwiseDifferences<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        int n = values.Length;
        int[] shape = { n, n };
        return engine.TensorSubtract(
            engine.TensorBroadcastTo(values.Reshape(n, 1), shape),
            engine.TensorBroadcastTo(values.Reshape(1, n), shape));
    }

    private static int Clamp(int value, int min, int max) => Math.Max(min, Math.Min(value, max));
}
