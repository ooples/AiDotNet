namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Moshi's sampling (the release's <c>LMGen</c> defaults: audio temperature 0.8 with top-k 250, text 0.7 with
/// top-k 25; greedy when <see cref="UseSampling"/> is false).</summary>
internal sealed record MoshiSampling
{
    public bool UseSampling { get; init; } = true;
    public double AudioTemperature { get; init; } = 0.8;
    public int AudioTopK { get; init; } = 250;
    public double TextTemperature { get; init; } = 0.7;
    public int TextTopK { get; init; } = 25;
}

/// <summary>
/// Moshi's streaming inference (the release's <c>LMGen._step</c>): one 80 ms frame per step. A ring cache holds each
/// stream's delayed tokens; the Temporal Transformer reads the previous step's column (the start token while a stream's
/// delay has not passed), the text token is sampled from its logits, then each predicted audio stream in turn from the
/// Depth Transformer. Streams the model does not predict (the user's, in the release's layout) are written into the cache
/// at their delay as they arrive. Once the largest delay has passed, each step returns the undelayed column of a frame.
/// </summary>
/// <remarks>Tokens may be forced instead of sampled: the text (TTS, App. C), any predicted audio stream (ASR, or the user's
/// streams in the paper's layout, where the Depth Transformer predicts them too). A forced token is fed onward exactly
/// as a sampled one.</remarks>
internal sealed class MoshiStreamer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly MoshiLanguageModel<T> _lm;
    private readonly MoshiSampling _sampling;
    private readonly Random _random;
    private readonly int[,] _cache;
    private readonly int _maxDelay;
    private readonly MoshiTransformer<T>.Cache _temporal = new();

    public const int Ungenerated = -2;

    public MoshiStreamer(MoshiLanguageModel<T> lm, MoshiSampling sampling, Random random)
    {
        _lm = lm;
        _sampling = sampling;
        _random = random;
        var c = lm.Configuration;
        _maxDelay = c.Delays.Max();
        _cache = new int[c.Streams, _maxDelay + 2];
        for (int k = 0; k < c.Streams; k++) for (int i = 0; i < _maxDelay + 2; i++) _cache[k, i] = Ungenerated;
    }

    /// <summary>Steps taken.</summary>
    public int Offset { get; private set; }

    /// <summary>The largest delay: how many steps pass before the first frame is complete.</summary>
    public int MaxDelay => _maxDelay;

    /// <summary>The last text logits (for TTS pad-rate control and inspection).</summary>
    public double[]? LastTextLogits { get; private set; }

    /// <summary>
    /// One step. <paramref name="input"/> holds the streams the model does not predict (their raw values for this frame,
    /// written at their delays); <paramref name="forcedText"/> and <paramref name="forcedAudio"/> (per predicted stream,
    /// <see cref="Ungenerated"/> to sample) replace sampling; <paramref name="textLogitsBias"/> adjusts the text logits
    /// before sampling and <paramref name="textFilter"/> may replace the sampled text token (App. C's TTS puts the next word
    /// in place of a token that is not padding). Returns the undelayed frame (text, then the predicted audio streams), or
    /// null until the delays have passed.
    /// </summary>
    public int[]? Step(int[]? input = null, int? forcedText = null, int[]? forcedAudio = null, Action<double[]>? textLogitsBias = null,
        Func<int, int>? textFilter = null)
    {
        var c = _lm.Configuration;
        int ct = _maxDelay + 2, dep = c.DepthStreams, needed = c.Streams - dep - 1;
        input ??= Array.Empty<int>();
        if (input.Length < needed) throw new ArgumentException($"The step needs the {needed} input streams' tokens.", nameof(input));
        for (int k = 0; k < needed; k++)
            _cache[dep + 1 + k, (Offset + c.Delays[dep + 1 + k]) % ct] = input[k];
        var column = new int[c.Streams, 1];
        for (int k = 0; k < c.Streams; k++)
        {
            bool init = Offset <= c.Delays[k];
            int start = k == 0 ? c.TextCardinality : c.AudioCardinality;
            column[k, 0] = init ? start : _cache[k, Offset % ct];
            if (column[k, 0] == Ungenerated) throw new InvalidOperationException($"Stream {k} has no token at step {Offset}.");
        }
        var (z, textLogits) = _lm.TemporalForward(column, _temporal);
        var logits = Row(textLogits);
        textLogitsBias?.Invoke(logits);
        LastTextLogits = logits;
        int text = forcedText ?? Sample(logits, _sampling.TextTemperature, _sampling.TextTopK);
        if (forcedText is null && textFilter is not null) text = textFilter(text);
        var audio = new int[dep];
        bool allForced = forcedAudio is not null && forcedAudio.Length >= dep && forcedAudio.Take(dep).All(v => v != Ungenerated);
        if (allForced) Array.Copy(forcedAudio!, audio, dep);
        else
        {
            var depthCache = new MoshiTransformer<T>.Cache();
            int previous = text;
            for (int s = 0; s < dep; s++)
            {
                var stepLogits = DepthStep(s, previous, z, depthCache);
                int token = forcedAudio is not null && s < forcedAudio.Length && forcedAudio[s] != Ungenerated
                    ? forcedAudio[s]
                    : Sample(stepLogits, _sampling.AudioTemperature, _sampling.AudioTopK);
                audio[s] = token;
                previous = token;
            }
        }
        Offset++;
        _cache[0, Offset % ct] = text;
        for (int s = 0; s < dep; s++) _cache[1 + s, Offset % ct] = audio[s];
        if (Offset <= _maxDelay) return null;
        var frame = new int[dep + 1];
        for (int k = 0; k <= dep; k++) frame[k] = _cache[k, (Offset - _maxDelay + c.Delays[k]) % ct];
        return frame;
    }

    // The Depth Transformer's logits for predicted stream s, after the previous stream's token, at context z.
    private double[] DepthStep(int s, int previous, Tensor<T> z, MoshiTransformer<T>.Cache cache)
    {
        var c = _lm.Configuration;
        var input = _lm.DepthInput(s, previous, z);
        var h = _lm.Depth.Forward(input, cache);
        return Row(_lm.Heads[s].Forward(h));
    }

    private static double[] Row(Tensor<T> logits)
    {
        int n = logits.Shape[^1];
        var row = new double[n];
        int offset = logits.Length - n;
        for (int i = 0; i < n; i++) row[i] = NumOps.ToDouble(logits.GetFlat(offset + i));
        return row;
    }

    /// <summary>The release's <c>sample_token</c>: argmax without sampling; otherwise softmax(logits / τ) restricted to
    /// the top k, drawn by the exponential race (probability ÷ Exp(1), the largest wins), the first index on a tie.</summary>
    internal int Sample(double[] logits, double temperature, int topK)
    {
        int best = 0;
        if (!_sampling.UseSampling || temperature <= 0)
        {
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            return best;
        }
        int k = topK > 0 ? Math.Min(topK, logits.Length) : logits.Length;
        var order = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).ThenBy(i => i).Take(k).ToArray();
        double max = logits[order[0]] / temperature;
        var weights = order.Select(i => Math.Exp(logits[i] / temperature - max)).ToArray();
        double total = weights.Sum();
        double bestScore = double.NegativeInfinity;
        for (int j = 0; j < k; j++)
        {
            double exponential = -Math.Log(1.0 - _random.NextDouble());
            double score = weights[j] / total / exponential;
            if (score > bestScore) { bestScore = score; best = order[j]; }
        }
        return best;
    }
}
