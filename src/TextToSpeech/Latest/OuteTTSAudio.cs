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
internal static class OuteTTSSampler
{
    /// <summary>The processed scores (−∞ for removed tokens), in the order <c>generate()</c> applies them.</summary>
    public static double[] Process(double[] logits, IReadOnlyList<int> ids, double repetitionPenalty, int repetitionRange, double temperature,
        int topK, double topP, double minP)
    {
        var scores = (double[])logits.Clone();
        // RepetitionPenaltyLogitsProcessorPatch
        if (repetitionRange > 0 && repetitionPenalty != 1.0)
        {
            var window = new HashSet<int>();
            for (int i = Math.Max(0, ids.Count - repetitionRange); i < ids.Count; i++) window.Add(ids[i]);
            foreach (int token in window)
            {
                if (token >= scores.Length) continue;
                scores[token] = scores[token] <= 0 ? scores[token] * repetitionPenalty : scores[token] / repetitionPenalty;
            }
        }
        if (temperature != 1.0)
            for (int i = 0; i < scores.Length; i++) scores[i] /= temperature;
        if (topK > 0)
        {
            int k = Math.Min(topK, scores.Length);
            double kth = KthLargest(scores, k);
            for (int i = 0; i < scores.Length; i++) if (scores[i] < kth) scores[i] = double.NegativeInfinity;
        }
        if (topP < 1.0)
        {
            // Ascending order, stable for ties; remove while the cumulative probability stays at most 1 − top_p, keeping
            // the most likely token. Removed tokens (-inf) sort first with probability 0 and stay removed, so only the
            // finite scores are ordered.
            var current = scores;
            var order = Enumerable.Range(0, scores.Length).Where(i => !double.IsNegativeInfinity(current[i])).ToArray();
            Array.Sort(order, (a, b) => current[a] != current[b] ? current[a].CompareTo(current[b]) : a.CompareTo(b));
            var probabilities = Softmax(order.Select(i => current[i]).ToArray());
            double cumulative = 0;
            var remove = new bool[order.Length];
            for (int j = 0; j < order.Length; j++)
            {
                cumulative += probabilities[j];
                remove[j] = cumulative <= 1 - topP && j < order.Length - 1;
            }
            for (int j = 0; j < order.Length; j++) if (remove[j]) scores[order[j]] = double.NegativeInfinity;
        }
        if (minP > 0)
        {
            var probabilities = Softmax(scores);
            double threshold = minP * probabilities.Max();
            int best = 0;
            for (int i = 1; i < scores.Length; i++) if (scores[i] > scores[best]) best = i;
            for (int i = 0; i < scores.Length; i++)
                if (probabilities[i] < threshold && i != best) scores[i] = double.NegativeInfinity;
        }
        return scores;
    }

    // The k-th largest score (ties counted), as torch.topk's last value: a min-heap of the k largest so far.
    private static double KthLargest(double[] scores, int k)
    {
        var heap = new double[k];
        int count = 0;
        foreach (double v in scores)
        {
            if (count < k)
            {
                int i = count++;
                heap[i] = v;
                while (i > 0 && heap[(i - 1) / 2] > heap[i])
                {
                    (heap[i], heap[(i - 1) / 2]) = (heap[(i - 1) / 2], heap[i]);
                    i = (i - 1) / 2;
                }
            }
            else if (v > heap[0])
            {
                heap[0] = v;
                for (int i = 0; ;)
                {
                    int l = 2 * i + 1, r = l + 1, m = i;
                    if (l < k && heap[l] < heap[m]) m = l;
                    if (r < k && heap[r] < heap[m]) m = r;
                    if (m == i) break;
                    (heap[i], heap[m]) = (heap[m], heap[i]);
                    i = m;
                }
            }
        }
        return heap[0];
    }

    internal static double[] Softmax(double[] scores)
    {
        double max = scores.Max();
        var e = scores.Select(v => double.IsNegativeInfinity(v) ? 0.0 : Math.Exp(v - max)).ToArray();
        double sum = e.Sum();
        return e.Select(v => v / sum).ToArray();
    }

    /// <summary>A token drawn from the processed scores' distribution.</summary>
    public static int Sample(double[] scores, Random random)
    {
        var probabilities = Softmax(scores);
        double u = random.NextDouble(), cumulative = 0;
        int last = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            if (probabilities[i] <= 0) continue;
            last = i;
            cumulative += probabilities[i];
            if (u < cumulative) return i;
        }
        return last;
    }
}
