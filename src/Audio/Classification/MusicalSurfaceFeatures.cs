using AiDotNet.Diffusion.Audio;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// The 30-dimensional feature vector of Tzanetakis and Cook 2002 (section III): 19 timbral texture features, 6 rhythmic
/// content features from a beat histogram, and 5 pitch content features from pitch histograms, computed for a whole file.
/// </summary>
internal sealed class MusicalSurfaceFeatures
{
    /// <summary>The feature names in vector order.</summary>
    public static readonly string[] Names =
    {
        "CentroidMean", "CentroidVariance", "RolloffMean", "RolloffVariance", "FluxMean", "FluxVariance",
        "ZeroCrossingsMean", "ZeroCrossingsVariance", "LowEnergy",
        "Mfcc1Mean", "Mfcc2Mean", "Mfcc3Mean", "Mfcc4Mean", "Mfcc5Mean",
        "Mfcc1Variance", "Mfcc2Variance", "Mfcc3Variance", "Mfcc4Variance", "Mfcc5Variance",
        "A0", "A1", "RA", "P1", "P2", "BeatSum",
        "FA0", "UP0", "FP0", "IPO1", "PitchSum",
    };

    /// <summary>Length of the vector: 9 FFT + 10 MFCC timbral, 6 rhythmic, 5 pitch.</summary>
    public const int Length = 30;

    private const int MfccCount = 5;
    private const int MelBands = 40;
    private const int DwtLevels = 4;
    private const int EnvelopeDownsampling = 16;
    private const double EnvelopeAlpha = 0.99;

    private readonly int _sampleRate, _window, _hop, _textureWindow, _beatWindow, _beatHop;
    private readonly ShortTimeFourierTransform<double> _stft;
    private readonly MelSpectrogram<double> _mel;
    // cos and sin of 2 pi m / window, the DFT twiddles of the pitch analysis.
    private readonly double[] _cosine, _sine;

    public MusicalSurfaceFeatures(int sampleRate, int analysisWindow, int hop, int textureWindow, int beatWindow, int beatHop)
    {
        _sampleRate = sampleRate;
        _window = analysisWindow;
        _hop = hop;
        if (textureWindow < 1) throw new ArgumentOutOfRangeException(nameof(textureWindow), "The texture window spans at least one analysis frame.");
        // Four DAUB4 levels and the 16x envelope downsampling leave one envelope sample per 256 input samples.
        if (beatWindow < (1 << DwtLevels) * EnvelopeDownsampling)
            throw new ArgumentOutOfRangeException(nameof(beatWindow), $"The beat window needs at least {(1 << DwtLevels) * EnvelopeDownsampling} samples (one envelope sample).");
        if (beatHop < 1) throw new ArgumentOutOfRangeException(nameof(beatHop), "The beat hop advances at least one sample.");
        _textureWindow = textureWindow;
        _beatWindow = beatWindow;
        _beatHop = beatHop;
        _stft = new ShortTimeFourierTransform<double>(nFft: analysisWindow, hopLength: hop, center: false);
        _mel = new MelSpectrogram<double>(sampleRate: sampleRate, nMels: MelBands, nFft: analysisWindow, hopLength: hop,
            logMel: true, logOffset: 1e-10);
        _cosine = Enumerable.Range(0, analysisWindow).Select(m => Math.Cos(2 * Math.PI * m / analysisWindow)).ToArray();
        _sine = Enumerable.Range(0, analysisWindow).Select(m => Math.Sin(2 * Math.PI * m / analysisWindow)).ToArray();
    }

    /// <summary>The 30 features of a recording.</summary>
    public double[] Extract(double[] audio)
    {
        var features = new double[Length];
        Timbral(audio).CopyTo(features, 0);
        Rhythm(audio).CopyTo(features, 19);
        Pitch(audio).CopyTo(features, 25);
        return features;
    }

    private double[] Padded(double[] audio, int length)
    {
        if (audio.Length >= length) return audio;
        var padded = new double[length];
        Array.Copy(audio, padded, audio.Length);
        return padded;
    }

    // Section III-A/B: means and variances over 1 s texture windows of the centroid, 85% rolloff, flux and zero
    // crossings of 23 ms analysis windows, the low-energy fraction, and the first five MFCCs (the DC term excluded),
    // averaged over the file.
    private double[] Timbral(double[] audio)
    {
        var signal = Padded(audio, _window);
        var power = _stft.Power(new Tensor<double>(signal, new[] { signal.Length }));
        var logMel = _mel.FromPowerSpectrogram(power);
        int frames = power.Shape[0], bins = power.Shape[1];
        var perFrame = new double[frames][];
        var rms = new double[frames];
        double[]? previous = null;
        for (int t = 0; t < frames; t++)
        {
            var magnitude = new double[bins];
            double total = 0;
            for (int k = 0; k < bins; k++) { magnitude[k] = Math.Sqrt(Math.Max(0, power[t, k])); total += magnitude[k]; }
            double centroid = 0, cumulative = 0, rolloff = 0, flux = 0;
            for (int k = 0; k < bins; k++) centroid += k * magnitude[k];
            centroid = total > 0 ? centroid / total * _sampleRate / _window : 0;
            for (int k = 0; k < bins; k++)
            {
                cumulative += magnitude[k];
                if (cumulative >= 0.85 * total) { rolloff = k * (double)_sampleRate / _window; break; }
            }
            var normalized = new double[bins];
            for (int k = 0; k < bins; k++) normalized[k] = total > 0 ? magnitude[k] / total : 0;
            if (previous is not null) for (int k = 0; k < bins; k++) flux += (normalized[k] - previous[k]) * (normalized[k] - previous[k]);
            previous = normalized;

            int start = t * _hop, crossings = 0;
            double energy = 0;
            for (int n = start; n < Math.Min(signal.Length, start + _window); n++)
            {
                energy += signal[n] * signal[n];
                if (n > start && (signal[n] >= 0) != (signal[n - 1] >= 0)) crossings++;
            }
            rms[t] = Math.Sqrt(energy / _window);

            var mfcc = Dct(logMel, t);
            perFrame[t] = new[] { centroid, rolloff, flux, crossings, mfcc[0], mfcc[1], mfcc[2], mfcc[3], mfcc[4] };
        }

        // Running means and variances over the texture window (the current frame and up to TextureWindow - 1 before it),
        // then their averages over the file.
        int dims = perFrame.Length > 0 ? perFrame[0].Length : 9;
        var meanOfMeans = new double[dims];
        var meanOfVariances = new double[dims];
        double lowEnergy = 0;
        for (int t = 0; t < frames; t++)
        {
            int first = Math.Max(0, t - _textureWindow + 1), count = t - first + 1;
            double rmsMean = 0;
            for (int s = first; s <= t; s++) rmsMean += rms[s];
            rmsMean /= count;
            int below = 0;
            for (int s = first; s <= t; s++) if (rms[s] < rmsMean) below++;
            lowEnergy += below / (double)count;
            for (int d = 0; d < dims; d++)
            {
                double mean = 0, variance = 0;
                for (int s = first; s <= t; s++) mean += perFrame[s][d];
                mean /= count;
                for (int s = first; s <= t; s++) variance += (perFrame[s][d] - mean) * (perFrame[s][d] - mean);
                meanOfMeans[d] += mean;
                meanOfVariances[d] += variance / count;
            }
        }
        double scale = frames > 0 ? 1.0 / frames : 0;
        var result = new double[19];
        for (int d = 0; d < 4; d++) { result[2 * d] = meanOfMeans[d] * scale; result[2 * d + 1] = meanOfVariances[d] * scale; }
        result[8] = lowEnergy * scale;
        for (int c = 0; c < MfccCount; c++) { result[9 + c] = meanOfMeans[4 + c] * scale; result[14 + c] = meanOfVariances[4 + c] * scale; }
        return result;
    }

    // DCT-II of a log-mel frame: coefficients 1 to 5 (coefficient 0, the DC term, is excluded).
    private static double[] Dct(Tensor<double> logMel, int frame)
    {
        int bands = logMel.Shape[1];
        var coefficients = new double[MfccCount];
        for (int c = 1; c <= MfccCount; c++)
        {
            double sum = 0;
            for (int m = 0; m < bands; m++) sum += logMel[frame, m] * Math.Cos(Math.PI * c * (m + 0.5) / bands);
            coefficients[c - 1] = sum * Math.Sqrt(2.0 / bands);
        }
        return coefficients;
    }

    // Section III-C: a beat histogram from DAUB4 octave bands. Each band's envelope (full-wave rectification, a one-pole
    // low-pass with alpha 0.99, downsampling, mean removal) is summed, the enhanced autocorrelation of the sum is taken,
    // and its first three peaks between 40 and 200 bpm are added to the histogram with their amplitudes.
    private double[] Rhythm(double[] audio)
    {
        var histogram = new double[201];
        int windows = 0;
        var signal = Padded(audio, _beatWindow);
        for (int start = 0; start + _beatWindow <= signal.Length; start += _beatHop)
        {
            var segment = new double[_beatWindow];
            Array.Copy(signal, start, segment, 0, _beatWindow);
            var envelope = BeatEnvelope(segment);
            double envelopeRate = _sampleRate / (double)((1 << DwtLevels) * EnvelopeDownsampling);
            var acf = Enhance(Autocorrelation(envelope), new[] { 2, 3, 4 });
            int minLag = (int)Math.Floor(envelopeRate * 60.0 / 200.0), maxLag = (int)Math.Ceiling(envelopeRate * 60.0 / 40.0);
            foreach (var (lag, amplitude) in Peaks(acf, minLag, Math.Min(maxLag, acf.Length - 2), 3))
            {
                int bpm = (int)Math.Round(60.0 * envelopeRate / lag);
                if (bpm >= 40 && bpm <= 200) histogram[bpm] += amplitude;
            }
            windows++;
        }
        for (int b = 0; b < histogram.Length; b++) histogram[b] /= Math.Max(1, windows);

        var top = Peaks(histogram, 40, 200, 2, includeEdges: true);
        double sum = histogram.Sum();
        double a0 = top.Count > 0 ? top[0].Amplitude : 0, a1 = top.Count > 1 ? top[1].Amplitude : 0;
        return new[]
        {
            sum > 0 ? a0 / sum : 0, sum > 0 ? a1 / sum : 0, a0 > 0 ? a1 / a0 : 0,
            top.Count > 0 ? top[0].Index : 0, top.Count > 1 ? top[1].Index : 0, sum,
        };
    }

    private static readonly double[] Daub4Low =
    {
        (1 + Math.Sqrt(3)) / (4 * Math.Sqrt(2)), (3 + Math.Sqrt(3)) / (4 * Math.Sqrt(2)),
        (3 - Math.Sqrt(3)) / (4 * Math.Sqrt(2)), (1 - Math.Sqrt(3)) / (4 * Math.Sqrt(2)),
    };

    // Equation 6: one DWT level, periodic DAUB4 low-pass (h) and high-pass (g) filtering, each subsampled by two.
    internal static (double[] Approximation, double[] Detail) DwtLevel(double[] x)
    {
        int half = x.Length / 2;
        var low = new double[half];
        var high = new double[half];
        for (int n = 0; n < half; n++)
        {
            for (int k = 0; k < 4; k++)
            {
                double sample = x[(2 * n + k) % x.Length];
                low[n] += Daub4Low[k] * sample;
                high[n] += ((k % 2 == 0) ? 1 : -1) * Daub4Low[3 - k] * sample;
            }
        }
        return (low, high);
    }

    private static double[] BeatEnvelope(double[] segment)
    {
        var bands = new List<(double[] Band, int Level)>();
        var approximation = segment;
        for (int level = 1; level <= DwtLevels; level++)
        {
            var (low, high) = DwtLevel(approximation);
            bands.Add((high, level));
            approximation = low;
        }
        bands.Add((approximation, DwtLevels));
        int length = segment.Length / ((1 << DwtLevels) * EnvelopeDownsampling);
        var sum = new double[length];
        foreach (var (band, level) in bands)
        {
            // Equations 7-10. Every band ends at the same rate: the finer bands are downsampled further.
            int factor = EnvelopeDownsampling << (DwtLevels - level);
            double state = 0;
            var envelope = new double[length];
            for (int n = 0, i = 0; n < band.Length && i < length; n++)
            {
                state = (1 - EnvelopeAlpha) * Math.Abs(band[n]) + EnvelopeAlpha * state;
                if (n % factor == 0) envelope[i++] = state;
            }
            double mean = envelope.Average();
            for (int i = 0; i < length; i++) sum[i] += envelope[i] - mean;
        }
        return sum;
    }

    // Equation 11: the autocorrelation normalized by the signal length.
    private static double[] Autocorrelation(double[] x)
    {
        var r = new double[x.Length];
        for (int lag = 0; lag < x.Length; lag++)
        {
            double s = 0;
            for (int n = lag; n < x.Length; n++) s += x[n] * x[n - lag];
            r[lag] = s / x.Length;
        }
        return r;
    }

    // Enhanced autocorrelation (Tolonen and Karjalainen): clip to positive values, then subtract the clipped function
    // time-scaled by each integer factor, removing the peaks at integer multiples of the basic periodicities.
    internal static double[] Enhance(double[] acf, int[] factors)
    {
        var enhanced = acf.Select(v => Math.Max(0, v)).ToArray();
        foreach (int factor in factors)
        {
            var current = (double[])enhanced.Clone();
            for (int lag = 0; lag < enhanced.Length; lag++)
            {
                // The function stretched by the factor, read at this lag (linear interpolation).
                double position = lag / (double)factor;
                int i = (int)position;
                double fraction = position - i, stretched = i + 1 < current.Length ? (1 - fraction) * current[i] + fraction * current[i + 1] : current[i];
                enhanced[lag] = Math.Max(0, current[lag] - stretched);
            }
        }
        return enhanced;
    }

    // The largest local maxima in [from, to], largest first.
    private static List<(int Index, double Amplitude)> Peaks(double[] f, int from, int to, int count, bool includeEdges = false)
    {
        var peaks = new List<(int, double)>();
        for (int i = Math.Max(includeEdges ? 0 : 1, from); i <= Math.Min(to, f.Length - (includeEdges ? 1 : 2)); i++)
        {
            double left = i > 0 ? f[i - 1] : double.NegativeInfinity, right = i + 1 < f.Length ? f[i + 1] : double.NegativeInfinity;
            if (f[i] > 0 && f[i] >= left && f[i] > right) peaks.Add((i, f[i]));
        }
        return peaks.OrderByDescending(p => p.Item2).Take(count).ToList();
    }

    // Section III-D: Tolonen and Karjalainen's multipitch analysis in 23 ms windows. The signal is split at 1 kHz; the
    // high band's envelope (half-wave rectification and low-pass filtering) and the low band are each autocorrelated
    // (a generalized, |X|^0.67-compressed autocorrelation), summed and enhanced. The three dominant peaks become MIDI
    // notes in the unfolded histogram and pitch classes on the circle of fifths in the folded one.
    private double[] Pitch(double[] audio)
    {
        var unfolded = new double[128];
        var folded = new double[12];
        var low = Biquad(audio, 1000, lowPass: true);
        var high = Biquad(audio, 1000, lowPass: false);
        for (int n = 0; n < high.Length; n++) high[n] = Math.Max(0, high[n]);
        high = Biquad(high, 1000, lowPass: true);
        int windows = 0;
        var hann = Enumerable.Range(0, _window).Select(n => 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / _window)).ToArray();
        var signalLow = Padded(low, _window);
        var signalHigh = Padded(high, _window);
        for (int start = 0; start + _window <= signalLow.Length; start += _hop)
        {
            var sacf = new double[_window];
            foreach (var band in new[] { signalLow, signalHigh })
            {
                var frame = new double[_window];
                for (int n = 0; n < _window; n++) frame[n] = band[start + n] * hann[n];
                var acf = GeneralizedAutocorrelation(frame);
                for (int lag = 0; lag < _window; lag++) sacf[lag] += acf[lag];
            }
            var enhanced = Enhance(sacf, new[] { 2, 3, 4 });
            // Pitches from 50 Hz to 2 kHz.
            int minLag = Math.Max(2, _sampleRate / 2000), maxLag = Math.Min(_window / 2, _sampleRate / 50);
            foreach (var (lag, amplitude) in Peaks(enhanced, minLag, maxLag, 3))
            {
                double frequency = _sampleRate / (double)lag;
                int note = (int)Math.Round(12 * Math.Log(frequency / 440.0, 2) + 69); // Equation 12
                if (note < 0 || note > 127) continue;
                unfolded[note] += amplitude;
                folded[(7 * (note % 12)) % 12] += amplitude; // Equations 13 and 14
            }
            windows++;
        }
        for (int i = 0; i < unfolded.Length; i++) unfolded[i] /= Math.Max(1, windows);
        for (int i = 0; i < folded.Length; i++) folded[i] /= Math.Max(1, windows);

        var foldedTop = Enumerable.Range(0, 12).OrderByDescending(i => folded[i]).Take(2).ToArray();
        int unfoldedTop = Enumerable.Range(0, 128).OrderByDescending(i => unfolded[i]).First();
        return new[]
        {
            folded[foldedTop[0]], unfolded[unfoldedTop] > 0 ? unfoldedTop : 0, folded[foldedTop[0]] > 0 ? foldedTop[0] : 0,
            folded[foldedTop[1]] > 0 ? foldedTop[1] - foldedTop[0] : 0, unfolded.Sum(),
        };
    }

    private double[] GeneralizedAutocorrelation(double[] frame)
    {
        int n = frame.Length;
        var re = new double[n];
        var im = new double[n];
        for (int k = 0; k < n; k++)
            for (int t = 0; t < n; t++)
            {
                int index = (int)((long)k * t % n);
                re[k] += frame[t] * _cosine[index];
                im[k] -= frame[t] * _sine[index];
            }
        var compressed = new double[n];
        for (int k = 0; k < n; k++) compressed[k] = Math.Pow(re[k] * re[k] + im[k] * im[k], 0.67 / 2);
        var acf = new double[n];
        for (int lag = 0; lag < n; lag++)
        {
            double s = 0;
            for (int k = 0; k < n; k++) s += compressed[k] * _cosine[(int)((long)k * lag % n)];
            acf[lag] = s / n;
        }
        return acf;
    }

    // A second-order Butterworth low- or high-pass filter (bilinear transform).
    private double[] Biquad(double[] x, double cutoff, bool lowPass)
    {
        double w = Math.Tan(Math.PI * cutoff / _sampleRate), q = Math.Sqrt(2), norm = 1 / (1 + q * w + w * w);
        double b0 = lowPass ? w * w * norm : norm, b1 = lowPass ? 2 * b0 : -2 * b0, b2 = b0;
        double a1 = 2 * (w * w - 1) * norm, a2 = (1 - q * w + w * w) * norm;
        var y = new double[x.Length];
        double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        for (int n = 0; n < x.Length; n++)
        {
            double v = b0 * x[n] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1; x1 = x[n]; y2 = y1; y1 = v;
            y[n] = v;
        }
        return y;
    }
}
