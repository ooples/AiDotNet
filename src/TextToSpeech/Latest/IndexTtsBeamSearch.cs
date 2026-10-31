namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// The release's decoding of IndexTTS's mel or semantic codes (<c>inference_speech</c> through Hugging Face
/// <c>generate</c>): beam search over <see cref="IndexTtsDecoding.Beams"/> beams, sampling the candidates when
/// <see cref="IndexTtsDecoding.Sample"/>, with the repetition penalty, temperature, top-k and top-p applied to the
/// log-probabilities, ended by the stop token. Shared by IndexTTS (v1) and IndexTTS2, whose GPT-2 decoders differ only in
/// what precedes the codes.
/// </summary>
internal static class IndexTtsBeamSearch
{
    /// <summary>
    /// Decodes: <paramref name="logitsOfLast"/> gives each beam's logits at its last position; <paramref name="advance"/>
    /// reorders the decoder's cache to the chosen source beams and feeds each beam its new token (the argument is how many
    /// tokens each beam has generated before this one). Returns the best hypothesis's codes, before the stop token.
    /// </summary>
    public static int[] Generate(int startToken, int stopToken, IndexTtsDecoding decoding, Random random,
        Func<double[][]> logitsOfLast, Action<int[], int[], int> advance)
    {
        int beams = decoding.Beams;
        // The ids the repetition penalty sees: generate's placeholder prompt (all 1) and the start token, then the codes.
        var sequences = new List<List<int>>();
        for (int b = 0; b < beams; b++) sequences.Add(new List<int>());
        var beamScores = new double[beams];
        for (int b = 1; b < beams; b++) beamScores[b] = -1e9;
        var hypotheses = new List<(double Score, int[] Codes)>();
        double worst = 1e9;
        bool done = false;
        int keep = 2 * beams;                                           // max(2, 1 + one end token) per beam
        for (int step = 0; step < decoding.MaxTokens && !done; step++)
        {
            var logits = logitsOfLast();
            int vocabulary = logits[0].Length;
            var flat = new double[beams * vocabulary];
            for (int b = 0; b < beams; b++)
            {
                var scores = Warp(LogSoftmax(logits[b]), sequences[b], decoding, startToken);
                for (int t = 0; t < vocabulary; t++) flat[b * vocabulary + t] = scores[t] + beamScores[b];
            }
            int[] picked = decoding.Sample ? SampleWithoutReplacement(flat, keep, random) : TopIndices(flat, keep);
            var ordered = picked.Select(i => (Score: flat[i], Index: i)).OrderByDescending(p => p.Score).ToList();

            int cur = step + 1;                                             // tokens generated after this step
            var nextScores = new double[beams];
            var nextTokens = new int[beams];
            var nextBeams = new int[beams];
            int filled = 0;
            for (int rank = 0; rank < ordered.Count && filled < beams; rank++)
            {
                int token = ordered[rank].Index % vocabulary, beam = ordered[rank].Index / vocabulary;
                if (token == stopToken)
                {
                    if (rank >= beams) continue;
                    AddHypothesis(hypotheses, ref worst, beams, ordered[rank].Score, sequences[beam].ToArray());
                }
                else
                {
                    nextScores[filled] = ordered[rank].Score;
                    nextTokens[filled] = token;
                    nextBeams[filled] = beam;
                    filled++;
                }
            }
            if (filled < beams) throw new InvalidOperationException("Too many end tokens among the beam candidates.");
            // BeamHypotheses.is_done with early_stopping False and length_penalty 0.
            done = hypotheses.Count >= beams && worst >= ordered.Max(p => p.Score);

            sequences = nextBeams.Select((source, i) => new List<int>(sequences[source]) { nextTokens[i] }).ToList();
            beamScores = nextScores;
            if (done || step + 1 >= decoding.MaxTokens) break;
            advance(nextBeams, nextTokens, cur);
        }
        if (!done)
            for (int b = 0; b < beams; b++) AddHypothesis(hypotheses, ref worst, beams, beamScores[b], sequences[b].ToArray());
        // finalize: the best hypothesis; Python's sorted() is stable, so among equal scores the last added wins.
        var best = hypotheses.Select((h, i) => (h, i)).OrderBy(p => p.h.Score).ThenBy(p => p.i).Last().h;
        return best.Codes;
    }

    private static void AddHypothesis(List<(double Score, int[] Codes)> hypotheses, ref double worst, int beams, double score, int[] codes)
    {
        if (hypotheses.Count < beams || score > worst)
        {
            hypotheses.Add((score, codes));
            if (hypotheses.Count > beams)
            {
                var sorted = hypotheses.Select((h, i) => (h.Score, i)).OrderBy(p => p.Score).ThenBy(p => p.i).ToList();
                hypotheses.RemoveAt(sorted[0].i);
                worst = sorted[1].Score;
            }
            else
            {
                worst = Math.Min(score, worst);
            }
        }
    }

    private static double[] LogSoftmax(double[] row)
    {
        double max = row.Max(), sum = 0;
        foreach (var v in row) sum += Math.Exp(v - max);
        double log = max + Math.Log(sum);
        return row.Select(v => v - log).ToArray();
    }

    /// <summary>The logits processors in <c>generate</c>'s order: repetition penalty, then (sampling) temperature, top-k
    /// and top-p, each keeping at least two tokens.</summary>
    private static double[] Warp(double[] scores, IReadOnlyList<int> generated, IndexTtsDecoding decoding, int startToken)
    {
        if (decoding.RepetitionPenalty != 1.0)
        {
            var seen = new HashSet<int>(generated) { 1, startToken };
            foreach (int t in seen)
            {
                if (t >= scores.Length) continue;
                scores[t] = scores[t] < 0 ? scores[t] * decoding.RepetitionPenalty : scores[t] / decoding.RepetitionPenalty;
            }
        }
        if (!decoding.Sample) return scores;
        if (decoding.Temperature != 1.0)
            for (int t = 0; t < scores.Length; t++) scores[t] /= decoding.Temperature;
        if (decoding.TopK > 0)
        {
            int k = Math.Min(Math.Max(decoding.TopK, 2), scores.Length);
            double threshold = scores.OrderByDescending(v => v).ElementAt(k - 1);
            for (int t = 0; t < scores.Length; t++)
                if (scores[t] < threshold) scores[t] = double.NegativeInfinity;
        }
        if (decoding.TopP < 1.0)
        {
            var order = Enumerable.Range(0, scores.Length).OrderBy(t => scores[t]).ToArray();
            double max = scores.Max();
            var probabilities = order.Select(t => double.IsNegativeInfinity(scores[t]) ? 0 : Math.Exp(scores[t] - max)).ToArray();
            double total = probabilities.Sum();
            double cumulative = 0;
            var remove = new bool[order.Length];
            for (int i = 0; i < order.Length; i++)
            {
                cumulative += probabilities[i] / total;
                remove[i] = cumulative <= 1 - decoding.TopP;
            }
            for (int i = Math.Max(0, order.Length - 2); i < order.Length; i++) remove[i] = false;
            for (int i = 0; i < order.Length; i++)
                if (remove[i]) scores[order[i]] = double.NegativeInfinity;
        }
        return scores;
    }

    private static int[] TopIndices(double[] scores, int count) =>
        Enumerable.Range(0, scores.Length).OrderByDescending(i => scores[i]).ThenBy(i => i).Take(count).ToArray();

    /// <summary><c>torch.multinomial(softmax(scores), count)</c> without replacement: successive draws from the
    /// renormalized remainder.</summary>
    private static int[] SampleWithoutReplacement(double[] scores, int count, Random random)
    {
        double max = scores.Where(v => !double.IsNegativeInfinity(v)).DefaultIfEmpty(0).Max();
        var weights = scores.Select(v => double.IsNegativeInfinity(v) ? 0 : Math.Exp(v - max)).ToArray();
        var picked = new List<int>(count);
        for (int n = 0; n < count; n++)
        {
            double total = weights.Sum();
            if (total <= 0) break;
            double u = random.NextDouble() * total, acc = 0;
            int chosen = -1;
            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] <= 0) continue;
                acc += weights[i];
                chosen = i;
                if (acc > u) break;
            }
            picked.Add(chosen);
            weights[chosen] = 0;
        }
        return picked.ToArray();
    }
}
