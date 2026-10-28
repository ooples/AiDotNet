namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// The random draws SpeechT5's pre-training noise makes, one method per call the official code makes (torch's
/// <c>Categorical.sample</c>, <c>randperm</c>, <c>uniform_</c> and <c>randint</c>; numpy's <c>rand</c> and
/// <c>choice</c> without replacement), so the noise can be replayed draw for draw.
/// </summary>
internal interface ISpeechT5Draws
{
    /// <summary><paramref name="count"/> samples of a categorical distribution with the given (unnormalized) weights.</summary>
    int[] Categorical(double[] weights, int count);

    /// <summary>A random permutation of 0 … n − 1.</summary>
    int[] Permutation(int n);

    /// <summary><paramref name="count"/> draws of U[0, 1).</summary>
    double[] Uniforms(int count);

    /// <summary><paramref name="count"/> integers drawn uniformly from [low, high).</summary>
    int[] Integers(int low, int high, int count);

    /// <summary><paramref name="count"/> distinct integers from [0, n), in draw order.</summary>
    int[] Choice(int n, int count);
}

/// <summary>The draws from a <see cref="Random"/>.</summary>
internal sealed class SpeechT5RandomDraws : ISpeechT5Draws
{
    private readonly Random _random;

    public SpeechT5RandomDraws(Random random) => _random = random;

    public int[] Categorical(double[] weights, int count)
    {
        double total = weights.Sum();
        var result = new int[count];
        for (int i = 0; i < count; i++)
        {
            double u = _random.NextDouble() * total, sum = 0;
            int k = 0;
            for (; k < weights.Length - 1; k++)
            {
                sum += weights[k];
                if (u < sum) break;
            }
            result[i] = k;
        }
        return result;
    }

    public int[] Permutation(int n)
    {
        var p = Enumerable.Range(0, n).ToArray();
        for (int i = n - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (p[i], p[j]) = (p[j], p[i]);
        }
        return p;
    }

    public double[] Uniforms(int count) => Enumerable.Range(0, count).Select(_ => _random.NextDouble()).ToArray();

    public int[] Integers(int low, int high, int count) => Enumerable.Range(0, count).Select(_ => _random.Next(low, high)).ToArray();

    public int[] Choice(int n, int count) => Permutation(n).Take(count).ToArray();
}

/// <summary>
/// The noise of SpeechT5's joint pre-training as the official code draws it: BART's text infilling on the text side
/// (<c>TextPretrainDataset.add_whole_word_mask</c> with Poisson span lengths and <c>add_insertion_noise</c>) and HuBERT's
/// span masking on the speech side (fairseq's <c>compute_mask_indices</c>, static spans).
/// </summary>
internal static class SpeechT5Noising
{
    /// <summary>The span-length weights of a Poisson(λ) truncated where a probability falls below 1e-7 (fairseq builds
    /// them in float32).</summary>
    public static double[] PoissonWeights(double lambda)
    {
        var weights = new List<double>();
        double power = 1, factorial = 1, e = Math.Exp(-lambda);
        for (int k = 0; k < 128; k++)
        {
            weights.Add(e * power / factorial);
            power *= lambda;
            factorial *= k + 1;
            if (weights[weights.Count - 1] < 0.0000001) break;
        }
        return weights.Select(w => (double)(float)w).ToArray();
    }

    /// <summary>
    /// BART's text infilling of <paramref name="source"/> (<c>&lt;s&gt; … &lt;/s&gt;</c>; every inner token a word start,
    /// as the official recipe sets no BPE): ⌈p · words⌉ tokens are masked in spans of Poisson(λ) length, each span replaced
    /// by one mask token (or, with probability <paramref name="randomRatio"/>, a random token in [1, vocabulary)); a
    /// zero-length span inserts a mask instead.
    /// </summary>
    public static int[] TextInfilling(IReadOnlyList<int> source, double ratio, double randomRatio, double lambda, int maskId, int vocabulary,
        ISpeechT5Draws draws)
    {
        int n = source.Count;
        if (n < 3) throw new ArgumentException("A noised sequence holds <s>, at least one token and </s>.", nameof(source));
        var tokens = source.ToArray();
        var isStart = new int[n];
        for (int i = 1; i < n - 1; i++) isStart[i] = 1;
        // float32, as is_word_start.float().sum() * p computes it (50 × 0.3 is 15.000001 there, so 16 spans).
        int toMask = (int)Math.Ceiling((float)(n - 2) * (float)ratio);
        if (toMask == 0) return tokens;

        var weights = PoissonWeights(lambda);
        var lengths = draws.Categorical(weights, toMask).ToList();
        var cumulative = Cumulative(lengths);
        while (cumulative[cumulative.Count - 1] < toMask)
        {
            lengths.AddRange(draws.Categorical(weights, toMask));
            cumulative = Cumulative(lengths);
        }
        int cut = 0;
        while (cumulative[cut] < toMask) cut++;
        lengths[cut] = toMask - (cut == 0 ? 0 : cumulative[cut - 1]);
        toMask = cut + 1;
        lengths = lengths.Take(toMask).Where(l => l > 0).ToList();
        int inserts = toMask - lengths.Count;
        toMask -= inserts;
        if (toMask == 0) return InsertionNoise(tokens, (double)inserts / n, randomRatio, maskId, vocabulary, draws);

        var starts = Enumerable.Range(0, n).Where(i => isStart[i] != 0).ToArray();
        var order = draws.Permutation(starts.Length);
        var indices = order.Take(toMask).Select(k => starts[k]).ToList();
        var random = draws.Uniforms(toMask).Select(u => u < randomRatio).ToList();
        var keep = Enumerable.Repeat(true, n).ToArray();
        isStart[n - 1] = 255;   // a long length, so spans stop at the end
        Replace(tokens, indices, random, maskId, vocabulary, draws);

        var remaining = lengths.Select(l => l - 1).ToList();
        while (indices.Count > 0)
        {
            var nextIndices = new List<int>();
            var nextRandom = new List<bool>();
            var nextRemaining = new List<int>();
            for (int k = 0; k < indices.Count; k++)
            {
                int left = remaining[k] - isStart[indices[k] + 1];
                if (left < 0) continue;
                nextIndices.Add(indices[k] + 1);
                nextRandom.Add(random[k]);
                nextRemaining.Add(left);
            }
            indices = nextIndices;
            random = nextRandom;
            remaining = nextRemaining;
            foreach (int i in indices) keep[i] = false;   // replace_length 1: the span's other tokens are deleted
        }
        var result = tokens.Where((_, i) => keep[i]).ToArray();
        return inserts > 0 ? InsertionNoise(result, (double)inserts / result.Length, randomRatio, maskId, vocabulary, draws) : result;
    }

    private static List<int> Cumulative(List<int> lengths)
    {
        var sums = new List<int>(lengths.Count);
        int s = 0;
        foreach (int l in lengths) sums.Add(s += l);
        return sums;
    }

    private static void Replace(int[] tokens, List<int> indices, List<bool> random, int maskId, int vocabulary, ISpeechT5Draws draws)
    {
        foreach (int i in indices) tokens[i] = maskId;
        var randomIndices = indices.Where((_, k) => random[k]).ToList();
        var values = draws.Integers(1, vocabulary, randomIndices.Count);
        for (int k = 0; k < randomIndices.Count; k++) tokens[randomIndices[k]] = values[k];
    }

    /// <summary>fairseq's <c>add_insertion_noise</c>: ⌈p · n⌉ new tokens at random inner positions, the first
    /// ⌈· randomRatio⌉ of them random tokens and the rest masks.</summary>
    public static int[] InsertionNoise(IReadOnlyList<int> tokens, double p, double randomRatio, int maskId, int vocabulary, ISpeechT5Draws draws)
    {
        if (p == 0.0) return tokens.ToArray();
        int count = tokens.Count, added = (int)Math.Ceiling(count * p);
        var positions = draws.Permutation(count + added - 2).Take(added).Select(i => i + 1).ToArray();
        var result = Enumerable.Repeat(-1, count + added).ToArray();
        int randoms = (int)Math.Ceiling(added * randomRatio);
        for (int k = randoms; k < added; k++) result[positions[k]] = maskId;
        var values = draws.Integers(1, vocabulary, randoms);
        for (int k = 0; k < randoms; k++) result[positions[k]] = values[k];
        var inserted = new HashSet<int>(positions);
        int next = 0;
        for (int i = 0; i < result.Length; i++)
            if (!inserted.Contains(i)) result[i] = tokens[next++];
        return result;
    }

    /// <summary>
    /// fairseq's <c>compute_mask_indices</c> for one unpadded sequence with static spans and overlaps allowed: max(min
    /// masks, ⌊p · T / L + u⌋) starts chosen without replacement from the first T − L frames, each masking L frames.
    /// </summary>
    public static bool[] SpanMask(int frames, double probability, int length, int minMasks, ISpeechT5Draws draws)
    {
        var mask = new bool[frames];
        int count = Math.Max(minMasks, (int)(probability * frames / length + draws.Uniforms(1)[0]));
        int minLength = length;
        if (frames - minLength <= count) minLength = frames - count - 1;
        if (frames - minLength < count)
            throw new ArgumentException($"{frames} frames are too few for {count} masked spans.", nameof(frames));
        foreach (int start in draws.Choice(frames - minLength, count))
            for (int k = 0; k < length; k++)
                if (start + k < frames) mask[start + k] = true;
        return mask;
    }
}
