using AiDotNet.Tensors.Engines;
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
    /// a one-hot at the argmax for <c>t &lt;= 0</c>, and a copy of the input for <c>t == 1</c>.
    /// </summary>
    public static Vector<T> Apply(Vector<T> distribution, T temperature)
    {
        // A copy on every path, so a caller that edits the result never edits its input.
        if (NumOps.Equals(temperature, NumOps.One)) return (Vector<T>)distribution.Clone();
        if (distribution.Length == 0) return new Vector<T>(0);
        if (IsGreedy(temperature)) return Vector<T>.CreateStandardBasis(distribution.Length, ArgMax(distribution));

        // Entries at or below zero carry no mass; the rest are scaled by the largest first, so raising to
        // 1/temperature at a low positive temperature sets the maximum to 1 instead of underflowing them all to 0.
        var engine = AiDotNetEngine.Current;
        var mass = (Vector<T>)engine.ReLU(distribution);
        T maxProb = TensorPrimitivesHelper<T>.Max(mass);
        if (NumOps.GreaterThan(maxProb, NumOps.Zero))
        {
            var scaled = (Vector<T>)engine.Power(engine.Divide(mass, maxProb), NumOps.Divide(NumOps.One, temperature));
            T sum = engine.Sum(scaled);
            if (NumOps.GreaterThan(sum, NumOps.Zero)) return (Vector<T>)engine.Divide(scaled, sum);
        }

        // Every entry underflowed (a very low temperature): the limit is the greedy one-hot.
        return Vector<T>.CreateStandardBasis(distribution.Length, ArgMax(distribution));
    }

    /// <summary>
    /// Draws an index from <paramref name="distribution"/>. Zero-probability entries are never chosen, so a
    /// one-hot distribution always returns its hot index.
    /// </summary>
    public static int Sample(Vector<T> distribution, Random random)
    {
        if (random is null)
        {
            throw new ArgumentNullException(nameof(random));
        }

        if (distribution.Length == 0)
        {
            throw new ArgumentException("The distribution must not be empty.", nameof(distribution));
        }

        // Inverse CDF: the first index whose cumulative mass exceeds r. A zero entry repeats the previous
        // cumulative value, so the search never lands on it.
        var engine = AiDotNetEngine.Current;
        var mass = (Vector<T>)engine.ReLU(distribution);
        var cumulative = engine.TensorCumSum(mass.AsTensor(), 0);
        var r = Vector<T>.FromArray(new[] { NumOps.FromDouble(random.NextDouble()) }).AsTensor();
        int index = engine.TensorSearchSorted(cumulative, r, true).GetFlat(0);
        if (index < distribution.Length) return index;

        // Rounding left the total mass just below r: the last index with mass is the right answer.
        int lastWithMass = -1;
        foreach (int i in mass.NonZeroIndices()) lastWithMass = i;
        return lastWithMass >= 0 ? lastWithMass : ArgMax(distribution);
    }

    /// <summary>
    /// Speculative acceptance over a run of draft tokens: token <c>i</c> is accepted with probability
    /// <c>min(1, p_target[i] / p_draft[i])</c>, and a draft probability at or below zero accepts exactly when the
    /// target gives the token mass, without a random draw. Stops at the first rejection, and after an accepted
    /// <paramref name="stopToken"/>.
    /// </summary>
    /// <returns>The number of leading draft tokens accepted.</returns>
    public static int CountAccepted(Vector<T> targetProbabilities, Vector<T> draftProbabilities, Random random,
        int[]? tokens = null, int? stopToken = null)
    {
        double[] target = targetProbabilities.Transform(NumOps.ToDouble).ToArray();
        double[] draft = draftProbabilities.Transform(NumOps.ToDouble).ToArray();
        for (int i = 0; i < target.Length; i++)
        {
            double ratio = draft[i] > 0 ? target[i] / draft[i] : 0;
            bool accept = draft[i] <= 0 ? target[i] > 0 : random.NextDouble() < (ratio < 1 ? ratio : 1);
            if (!accept) return i;
            if (stopToken.HasValue && tokens is not null && tokens[i] == stopToken.Value) return i + 1;
        }

        return target.Length;
    }

    /// <summary>Index of the largest entry (the first one on ties).</summary>
    public static int ArgMax(Vector<T> distribution) => distribution.Length == 0 ? 0 : distribution.IndexOfMax();
}
