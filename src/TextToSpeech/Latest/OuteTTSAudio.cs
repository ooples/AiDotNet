using AiDotNet.Audio.Effects;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// OuteTTS 1.0's audio handling (edwko/OuteTTS <c>dac/interface.py</c> and <c>version/v3/audio_processor.py</c>):
/// loudness normalization to −18 LUFS with a −1 dBFS peak limit around the codec, the codec's 5-second encoding windows
/// and 2048-frame decoding chunks with 15 ms fades, and each word's energy, spectral centroid and pitch features.
/// </summary>
internal static class OuteTTSAudio
{
    /// <summary><c>process_audio_tensor(audio, 24000, -18.0, -1, 0.4)</c>: integrated loudness to the target (short
    /// audio padded to one block for the measurement), then the peak brought down to the limit when it is above it.</summary>
    public static double[] ProcessAudio(double[] audio, int sampleRate, double targetLoudness = -18.0, double peakLimit = -1, double blockSize = 0.4)
    {
        int original = audio.Length;
        int minimum = (int)(blockSize * sampleRate);
        var padded = audio;
        if (original < minimum)
        {
            padded = new double[minimum];
            Array.Copy(audio, padded, original);
        }
        double measured = LoudnessMeter.IntegratedLoudness(padded, sampleRate, blockSize);
        var normalized = LoudnessMeter.NormalizeLoudness(padded, measured, targetLoudness);
        double peak = normalized.Max(Math.Abs);
        if (peak > Math.Pow(10, peakLimit / 20)) normalized = LoudnessMeter.NormalizePeak(normalized, peakLimit);
        if (original < minimum) Array.Resize(ref normalized, original);
        return normalized;
    }

    /// <summary><c>DacInterface.encode</c>: the loudness-normalized recording in 5-second windows, each padded to a
    /// multiple of the hop and encoded, the codes concatenated: <c>[codebooks, frames]</c>.</summary>
    public static int[,] Encode<T>(Func<Tensor<T>, int[,]> encode, int hop, double[] audio, int sampleRate, double windowSeconds = 5.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var x = ProcessAudio(audio, sampleRate);
        int window = (int)(Math.Ceiling((int)(windowSeconds * sampleRate) / (double)hop) * hop);
        var pieces = new List<int[,]>();
        for (int start = 0; start < x.Length; start += window)
        {
            int length = Math.Min(window, x.Length - start);
            int padded = (int)Math.Ceiling(length / (double)hop) * hop;
            var chunk = new Tensor<T>(new[] { 1, 1, padded });
            for (int i = 0; i < length; i++) chunk[0, 0, i] = ops.FromDouble(x[start + i]);
            pieces.Add(encode(chunk));
        }
        int codebooks = pieces[0].GetLength(0), frames = pieces.Sum(p => p.GetLength(1));
        var codes = new int[codebooks, frames];
        int offset = 0;
        foreach (var piece in pieces)
        {
            for (int q = 0; q < codebooks; q++)
                for (int t = 0; t < piece.GetLength(1); t++) codes[q, offset + t] = piece[q, t];
            offset += piece.GetLength(1);
        }
        return codes;
    }

    /// <summary><c>DacInterface.decode</c>: the codes in 2048-frame chunks, each decoded and faded in and out, then the
    /// whole loudness-normalized.</summary>
    public static double[] Decode<T>(Func<int[,], Tensor<T>> decode, int[,] codes, int sampleRate, int chunkFrames = 2048)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int codebooks = codes.GetLength(0), frames = codes.GetLength(1);
        var audio = new List<double>();
        for (int start = 0; start < frames; start += chunkFrames)
        {
            int length = Math.Min(chunkFrames, frames - start);
            var chunk = new int[codebooks, length];
            for (int q = 0; q < codebooks; q++)
                for (int t = 0; t < length; t++) chunk[q, t] = codes[q, start + t];
            var decoded = decode(chunk);
            var samples = decoded.ToArray().Select(v => ops.ToDouble(v)).ToArray();
            ApplyFade(samples, sampleRate);
            audio.AddRange(samples);
        }
        return ProcessAudio(audio.ToArray(), sampleRate);
    }

    // apply_fade: 15 ms linear fades at each end (at most half the chunk each).
    public static void ApplyFade(double[] audio, int sampleRate)
    {
        int total = audio.Length;
        int fadeIn = Math.Min((int)(sampleRate * 0.015), total / 2), fadeOut = Math.Min((int)(sampleRate * 0.015), total / 2);
        var rampIn = Linspace(0, 1, fadeIn);
        var rampOut = Linspace(1, 0, fadeOut);
        for (int i = 0; i < fadeIn; i++) audio[i] *= rampIn[i];
        for (int i = 0; i < fadeOut; i++) audio[total - fadeOut + i] *= rampOut[i];
    }

    // torch.linspace in double: the first half counted up from the start, the second half down from the end.
    internal static double[] Linspace(double start, double end, int steps)
    {
        var values = new double[steps];
        if (steps == 1) values[0] = start;
        if (steps <= 1) return values;
        double step = (end - start) / (steps - 1);
        int halfway = steps / 2;
        for (int i = 0; i < steps; i++) values[i] = i < halfway ? start + step * i : end - step * (steps - 1 - i);
        return values;
    }

    /// <summary><c>Features.extract_audio_features</c>: RMS energy, spectral centroid over the Nyquist frequency and the
    /// normalized mean pitch, each × 100 and rounded (empty or non-finite audio has all three 0).</summary>
    public static OuteTTSFeatures Features(double[] audio, int sampleRate)
    {
        if (audio.Length == 0 || audio.Any(v => double.IsNaN(v) || double.IsInfinity(v))) return new OuteTTSFeatures();
        double energy = Math.Sqrt(audio.Sum(v => v * v) / audio.Length);
        var spectrum = new AiDotNet.LinearAlgebra.FastFourierTransform<double>().Forward(new Vector<double>(audio));
        int bins = audio.Length / 2 + 1;
        var frequencies = Linspace(0, sampleRate / 2.0, bins);
        double total = 0, weighted = 0;
        for (int k = 0; k < bins; k++)
        {
            var c = spectrum[k];
            double magnitude = Math.Sqrt(c.Real * c.Real + c.Imaginary * c.Imaginary);
            total += magnitude;
            weighted += frequencies[k] * magnitude;
        }
        double centroid = weighted / (total + 1e-10) / (sampleRate / 2.0);
        double pitch = SinglePitchValue(audio, sampleRate);
        int Scale(double value)
        {
            if (double.IsNaN(value)) throw new InvalidOperationException("cannot convert float NaN to integer (the segment is too short for one pitch frame).");
            return (int)Math.Round(value * 100, MidpointRounding.ToEven);
        }
        return new OuteTTSFeatures { Energy = Scale(energy), SpectralCentroid = Scale(centroid), Pitch = Scale(pitch) };
    }

    /// <summary><c>extract_single_pitch_value</c>: the mean of <see cref="CalculatePitch"/> mapped from 75–600 Hz to 0–1.</summary>
    public static double SinglePitchValue(double[] audio, int sampleRate, double minFrequency = 75.0, double maxFrequency = 600.0)
    {
        var pitch = CalculatePitch(audio, sampleRate, minFrequency, maxFrequency);
        double mean = pitch.Length == 0 ? double.NaN : pitch.Average();
        double normalized = (mean - minFrequency) / (maxFrequency - minFrequency);
        return double.IsNaN(normalized) ? normalized : Math.Min(1.0, Math.Max(0.0, normalized));
    }

    /// <summary><c>calculate_pitch</c>: per 400-sample frame (hop 160, Hann window), the autocorrelation's peak between
    /// sample rate / 600 and sample rate / 75, refined by a parabola; 0 when the peak is below 0.3 of the frame's energy;
    /// clamped to 75–600 Hz (so unvoiced frames count as 75 Hz).</summary>
    public static double[] CalculatePitch(double[] audio, int sampleRate, double minFrequency = 75.0, double maxFrequency = 600.0,
        int frameLength = 400, int hopLength = 160, double threshold = 0.3)
    {
        int padding = ((frameLength - (audio.Length % hopLength)) % hopLength + hopLength) % hopLength;
        var x = new double[audio.Length + padding];
        Array.Copy(audio, x, audio.Length);
        int frames = x.Length >= frameLength ? (x.Length - frameLength) / hopLength + 1 : 0;
        var window = new double[frameLength];
        for (int n = 0; n < frameLength; n++) window[n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / frameLength);
        int minIndex = Math.Max(1, (int)(sampleRate / maxFrequency)), maxIndex = Math.Min(frameLength, (int)(sampleRate / minFrequency));
        var pitch = new double[frames];
        var frame = new double[frameLength];
        var autocorrelation = new double[frameLength];
        for (int f = 0; f < frames; f++)
        {
            for (int n = 0; n < frameLength; n++) frame[n] = x[f * hopLength + n] * window[n];
            // irfft(|rfft(frame, 2L)|²)[:L]: the linear autocorrelation at lags 0 … L−1.
            for (int lag = 0; lag < frameLength; lag++)
            {
                double sum = 0;
                for (int n = 0; n + lag < frameLength; n++) sum += frame[n] * frame[n + lag];
                autocorrelation[lag] = sum;
            }
            int peak = minIndex;
            double peakValue = autocorrelation[minIndex];
            for (int lag = minIndex + 1; lag < maxIndex; lag++)
                if (autocorrelation[lag] > peakValue)
                {
                    peakValue = autocorrelation[lag];
                    peak = lag;
                }
            int index = Math.Min(Math.Max(peak, 1), frameLength - 2);
            double alpha = autocorrelation[index - 1], beta = autocorrelation[index], gamma = autocorrelation[index + 1];
            double delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-8);
            if (!(peak > 0 && peak < frameLength - 1)) delta = 0.0;
            double period = (peak + delta) / sampleRate;
            double value = period > 0 ? 1.0 / period : 0.0;
            if (!(peakValue / (autocorrelation[0] + 1e-8) > threshold)) value = 0.0;
            pitch[f] = Math.Min(maxFrequency, Math.Max(minFrequency, value));
        }
        return pitch;
    }
}

/// <summary>
/// The sampling OuteTTS 1.0 generates with (Hugging Face <c>generate()</c> with the release's settings): its windowed
/// repetition penalty (the last 64 tokens, ×1/1.1 on positive logits and ×1.1 on the others), then temperature, top-k,
/// top-p and min-p, and a draw from the remaining distribution.
/// </summary>
internal static class OuteTTSSampler<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>The processed scores <c>[n]</c> (−∞ for removed tokens), in the order <c>generate()</c> applies them.</summary>
    public static Tensor<T> Process(Tensor<T> logits, IReadOnlyList<int> ids, double repetitionPenalty, int repetitionRange, double temperature,
        int topK, double topP, double minP)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        int n = logits.Length;
        T negativeInfinity = Ops.FromDouble(double.NegativeInfinity);
        var removedScore = Tensor<T>.CreateDefault(new[] { n }, negativeInfinity);
        var scores = engine.Reshape(logits, new[] { n });
        // RepetitionPenaltyLogitsProcessorPatch
        if (repetitionRange > 0 && repetitionPenalty != 1.0)
        {
            var window = new bool[n];
            for (int i = Math.Max(0, ids.Count - repetitionRange); i < ids.Count; i++)
                if (ids[i] >= 0 && ids[i] < n) window[ids[i]] = true;
            T penalty = Ops.FromDouble(repetitionPenalty);
            var penalized = engine.TensorWhere(engine.TensorGreaterThan(scores, Ops.Zero),
                engine.TensorDivideScalar(scores, penalty), engine.TensorMultiplyScalar(scores, penalty));
            scores = engine.TensorWhere(window, penalized, scores);
        }
        if (temperature != 1.0) scores = engine.TensorDivideScalar(scores, Ops.FromDouble(temperature));
        if (topK > 0)
        {
            int k = Math.Min(topK, n);
            var largest = engine.TensorTopK(scores, k, 0, out _);
            T kth = engine.TensorSlice(largest, new[] { k - 1 }, new[] { 1 }).ToArray()[0];
            scores = engine.TensorWhere(engine.TensorLessThan(scores, kth), removedScore, scores);
        }
        if (topP < 1.0)
        {
            // Ascending order, stable for ties; remove while the cumulative probability stays at most 1 − top_p, keeping
            // the most likely token. Removed tokens (-inf) sort first with probability 0 and stay removed, so only the
            // finite scores are ordered.
            var removedBits = engine.TensorEquals(scores, negativeInfinity).Cast<int>().ToArray();
            var finite = Enumerable.Range(0, n).Where(i => removedBits[i] == 0).ToArray();
            if (finite.Length == 0) throw new InvalidOperationException("Sequence contains no elements");
            var finiteScores = Select(engine, scores, finite);
            var order = TtsSampling<T>.StableOrder(finiteScores, descending: false);
            var cumulative = engine.TensorCumSum(Softmax(Select(engine, finiteScores, order)), 0);
            var above = engine.TensorGreaterThan(cumulative, Ops.FromDouble(1 - topP)).Cast<int>().ToArray();
            var remove = new bool[n];
            for (int j = 0; j < order.Length - 1; j++) remove[finite[order[j]]] = above[j] == 0;
            scores = engine.TensorWhere(remove, removedScore, scores);
        }
        if (minP > 0)
        {
            var probabilities = Softmax(scores);
            T threshold = Ops.Multiply(Ops.FromDouble(minP), engine.TensorMaxValue(probabilities));
            var low = engine.TensorLessThan(probabilities, threshold).Cast<int>().ToArray();
            int best = Greedy(scores);
            var remove = new bool[n];
            for (int i = 0; i < n; i++) remove[i] = low[i] != 0 && i != best;
            scores = engine.TensorWhere(remove, removedScore, scores);
        }
        return scores;
    }

    /// <summary>The softmax of scores, removed (−∞) ones at exactly 0.</summary>
    internal static Tensor<T> Softmax(Tensor<T> scores)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var e = TtsSampling<T>.UnnormalizedProbabilities(scores);
        return engine.TensorDivideScalar(e, engine.TensorSum(e));
    }

    /// <summary>A token drawn from the processed scores' distribution: the first whose running probability exceeds the
    /// draw, the last likely one when rounding leaves none.</summary>
    public static int Sample(Tensor<T> scores, Random random)
    {
        if (scores is null) throw new ArgumentNullException(nameof(scores));
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var probabilities = engine.Reshape(Softmax(scores), new[] { scores.Length });
        T u = Ops.FromDouble(random.NextDouble());
        int n = probabilities.Length;
        int position = n - engine.TensorGreaterThan(engine.TensorCumSum(probabilities, 0), u).Cast<int>().ToArray().Sum();
        if (position < n) return position;
        int last = Array.LastIndexOf(engine.TensorGreaterThan(probabilities, Ops.Zero).Cast<int>().ToArray(), 1);
        return last >= 0 ? last : 0;
    }

    /// <summary>The first index of the largest score (greedy decoding, <c>Array.IndexOf(scores, scores.Max())</c>).</summary>
    public static int Greedy(Tensor<T> scores)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var best = engine.TensorEquals(scores, engine.TensorMaxValue(scores)).Cast<int>().ToArray();
        return Array.IndexOf(best, 1);
    }

    private static Tensor<T> Select(AiDotNet.Tensors.Engines.IEngine engine, Tensor<T> values, int[] positions) =>
        engine.TensorIndexSelect(values, new Tensor<int>((int[])positions.Clone(), new[] { positions.Length }), 0);
}
