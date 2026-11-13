using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Transformers' sampling for the speech language models: temperature, then top-p, then a draw.</summary>
internal static class NucleusSampling<T>
{
    /// <summary>
    /// Transformers' sampling: logits ÷ temperature, then top-p (the smallest set of the most likely tokens whose
    /// probability reaches p, at least one), then a draw; <paramref name="allowed"/> keeps only those ids.
    /// </summary>
    public static int Sample(Tensor<T> logits, double temperature, double topP, Func<int, bool>? allowed, Random random) =>
        Sample(logits, temperature, 0, topP, allowed, random);

    /// <summary>
    /// Transformers' sampling with top-k too, in the warpers' order: temperature, top-k (the <paramref name="topK"/> most
    /// likely, ties at the k-th kept; 0 for all), top-p, a draw. Equal probabilities are visited in id order, as the
    /// stable ascending sort of the release does.
    /// </summary>
    public static int Sample(Tensor<T> logits, double temperature, int topK, double topP, Func<int, bool>? allowed, Random random)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        var engine = AiDotNetEngine.Current;
        var ids = Allowed(logits.Length, allowed);
        if (ids.Length == 0) throw new InvalidOperationException("No token may be sampled.");
        // The warpers and the draw run in double: in half precision a large vocabulary's weight total overflows
        // 65504, and the normalized weights the top-p cutoff reads become 0.
        var candidates = engine.TensorDivideScalar(
            TensorCopyHelper.CastLogical<T, double>(Select(engine.Reshape(logits, new[] { logits.Length }), ids)), temperature);
        if (topK > 0 && topK < ids.Length)
        {
            // TopKLogitsWarper: scores below the k-th largest are removed.
            var largest = engine.TensorTopK(candidates, topK, 0, out _);
            double kth = engine.TensorSlice(largest, new[] { topK - 1 }, new[] { 1 }).ToArray()[0];
            var below = engine.TensorLessThan(candidates, kth).Cast<int>().ToArray();
            var keep = Enumerable.Range(0, ids.Length).Where(i => below[i] == 0).ToArray();
            ids = keep.Select(i => ids[i]).ToArray();
            candidates = Select(candidates, keep);
        }
        var probabilities = engine.TensorExp(engine.TensorAddScalar(candidates, -engine.TensorMaxValue(candidates)));
        var order = TtsSampling<double>.StableOrder(probabilities, descending: false);
        var sorted = Select(probabilities, order);
        // TopPLogitsWarper: sorted ascending, drop the tokens whose cumulative probability is at most 1 − p; keep the last.
        var cumulative = engine.TensorCumSum(engine.TensorDivideScalar(sorted, engine.TensorSum(sorted)), 0);
        var above = engine.TensorGreaterThan(cumulative, 1 - topP).Cast<int>().ToArray();
        var kept = Enumerable.Range(0, order.Length).Where(i => above[i] != 0 || i == order.Length - 1).ToArray();
        int position = TtsSampling<double>.DrawPosition(Select(sorted, kept), random.NextDouble(), inclusive: false);
        return ids[order[kept[position]]];
    }

    /// <summary>The first id of the largest logit among the allowed ids (greedy decoding).</summary>
    public static int Greedy(Tensor<T> logits, Func<int, bool>? allowed = null)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        var engine = AiDotNetEngine.Current;
        var ids = Allowed(logits.Length, allowed);
        if (ids.Length == 0) throw new InvalidOperationException("No token may be chosen.");
        var values = Select(engine.Reshape(logits, new[] { logits.Length }), ids);
        var best = engine.TensorEquals(values, engine.TensorMaxValue(values)).Cast<int>().ToArray();
        int first = Array.IndexOf(best, 1);
        return ids[first >= 0 ? first : 0];
    }

    private static int[] Allowed(int count, Func<int, bool>? allowed) =>
        allowed is null ? Enumerable.Range(0, count).ToArray() : Enumerable.Range(0, count).Where(allowed).ToArray();

    private static Tensor<TValue> Select<TValue>(Tensor<TValue> values, int[] positions) =>
        AiDotNetEngine.Current.TensorIndexSelect(values, new Tensor<int>((int[])positions.Clone(), new[] { positions.Length }), 0);
}
