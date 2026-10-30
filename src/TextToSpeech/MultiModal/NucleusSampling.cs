namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Transformers' sampling for the speech language models: temperature, then top-p, then a draw.</summary>
internal static class NucleusSampling
{
    /// <summary>
    /// Transformers' sampling: logits ÷ temperature, then top-p (the smallest set of the most likely tokens whose
    /// probability reaches p, at least one), then a draw; <paramref name="allowed"/> keeps only those ids.
    /// </summary>
    public static int Sample(double[] logits, double temperature, double topP, Func<int, bool>? allowed, Random random) =>
        Sample(logits, temperature, 0, topP, allowed, random);

    /// <summary>
    /// Transformers' sampling with top-k too, in the warpers' order: temperature, top-k (the <paramref name="topK"/> most
    /// likely, ties at the k-th kept; 0 for all), top-p, a draw.
    /// </summary>
    public static int Sample(double[] logits, double temperature, int topK, double topP, Func<int, bool>? allowed, Random random)
    {
        var candidates = new List<(int Id, double Logit)>();
        for (int i = 0; i < logits.Length; i++)
            if (allowed is null || allowed(i)) candidates.Add((i, logits[i] / temperature));
        if (candidates.Count == 0) throw new InvalidOperationException("No token may be sampled.");
        if (topK > 0 && topK < candidates.Count)
        {
            // TopKLogitsWarper: scores below the k-th largest are removed.
            double kth = candidates.Select(c => c.Logit).OrderByDescending(v => v).ElementAt(topK - 1);
            candidates = candidates.Where(c => c.Logit >= kth).ToList();
        }
        double max = candidates.Max(c => c.Logit);
        var sorted = candidates.Select(c => (c.Id, P: Math.Exp(c.Logit - max))).OrderBy(c => c.P).ToList();
        double total = sorted.Sum(c => c.P);
        // TopPLogitsWarper: sorted ascending, drop the tokens whose cumulative probability is at most 1 − p; keep the last.
        double cumulative = 0;
        var kept = new List<(int Id, double P)>();
        for (int i = 0; i < sorted.Count; i++)
        {
            cumulative += sorted[i].P / total;
            if (cumulative > 1 - topP || i == sorted.Count - 1) kept.Add(sorted[i]);
        }
        double mass = kept.Sum(c => c.P), target = random.NextDouble() * mass;
        foreach (var (id, p) in kept)
        {
            target -= p;
            if (target < 0) return id;
        }
        return kept[kept.Count - 1].Id;
    }

    /// <summary>The index of the largest logit among the allowed ids (greedy decoding).</summary>
    public static int Greedy(double[] logits, Func<int, bool>? allowed = null)
    {
        int best = -1;
        for (int i = 0; i < logits.Length; i++)
            if ((allowed is null || allowed(i)) && (best < 0 || logits[i] > logits[best])) best = i;
        return best >= 0 ? best : throw new InvalidOperationException("No token may be chosen.");
    }
}
