using System.Collections.Concurrent;
using AiDotNet.Safety.Audio;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Abstract base class for keyed audio watermarkers that mark the block-DCT spectrum of the audio with a pattern
/// derived from a secret key, and detect it by correlation.
/// </summary>
/// <remarks>
/// <para>
/// The audio is cut into non-overlapping frames of about 32 ms (the smallest power of two of at least
/// 0.032 x sample rate samples: 512 at 16 kHz, 2048 at 44.1 or 48 kHz). Each frame is transformed with the
/// orthonormal DCT-II, and the coefficients inside the watermark band (<see cref="BandLowHz"/> to
/// <see cref="BandHighHz"/>, clipped to the Nyquist frequency) are modified by the derived watermarker under the key's
/// +/-1 pattern for that frame (<see cref="WatermarkKey.SignPattern"/>, one stream per frame). The inverse DCT gives the
/// marked frame back; samples after the last full frame are left as they are.
/// </para>
/// <para>
/// Detection computes, per frame, key-independent weights w_k from the received band coefficients
/// (<see cref="DetectionWeights"/>) and the correlation z = sum(w_k x_k) / sqrt(sum(w_k^2)) over all frames. Because
/// the weights do not depend on the key and the pattern bits are fair and independent, z is approximately standard
/// normal for audio without this key's mark, whatever the audio is. The confidence is 0 up to z = 3 (false-positive
/// rate 0.135%) and 1 from z = 6; <see cref="EvaluateAudio"/> reports from <see cref="ReportThreshold"/>.
/// </para>
/// <para>
/// <see cref="EmbedWatermark"/> checks the result: if the marked audio does not reach z = 6 (too short, silent, or too
/// low a strength), it throws instead of returning audio that would not be recognised. Frames are aligned to the
/// first sample, so the mark survives noise, re-quantization and gain changes, but not cropping or time shifts that
/// move the frame grid.
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all audio watermarkers. The key works like a
/// password: the watermark is a faint pattern spread over thousands of frequency values, too weak to hear in any one
/// place, which adds up to strong evidence for whoever knows the exact pattern.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class AudioWatermarkerBase<T> : AudioSafetyModuleBase<T>, IAudioWatermarker<T>, IKeyedWatermarker
{
    /// <summary>The fewest band coefficients a frame must offer to carry the watermark.</summary>
    protected const int MinimumBandCoefficients = 16;

    private static readonly ConcurrentDictionary<int, (double[] Cos, double[] Sin)> Twiddles = new();

    /// <summary>
    /// The watermark strength. Each watermarker documents how it scales its modification with it.
    /// </summary>
    protected readonly double WatermarkStrength;

    /// <summary>
    /// Initializes the audio watermarker base.
    /// </summary>
    /// <param name="watermarkStrength">Watermark embedding strength, positive. Default: 0.5.</param>
    /// <param name="key">The secret key; <c>null</c> uses the published <see cref="WatermarkKey.Public"/> key.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="watermarkStrength"/> is not a positive finite number.</exception>
    /// <exception cref="ArgumentException"><paramref name="key"/> is empty or whitespace.</exception>
    protected AudioWatermarkerBase(double watermarkStrength = 0.5, string? key = null)
    {
        if (double.IsNaN(watermarkStrength) || double.IsInfinity(watermarkStrength) || watermarkStrength <= 0)
            throw new ArgumentOutOfRangeException(nameof(watermarkStrength), "Watermark strength must be a positive, finite number.");
        WatermarkStrength = watermarkStrength;
        Key = WatermarkKey.FromOptional(key);
    }

    /// <summary>The key the watermark is derived from.</summary>
    protected WatermarkKey Key { get; }

    /// <inheritdoc />
    public bool UsesPublicKey => Key.IsPublic;

    /// <summary>The lower edge of the watermark band, in Hz. Default: 1000.</summary>
    protected virtual double BandLowHz => 1000.0;

    /// <summary>The upper edge of the watermark band, in Hz (clipped to the Nyquist frequency). Default: 7000.</summary>
    protected virtual double BandHighHz => 7000.0;

    /// <summary>
    /// The longest audio, in samples, the watermarker accepts: 10 minutes at 48 kHz by default. Embedding longer audio
    /// throws; detection reports no evidence for it, so a safety pipeline checking it still completes.
    /// </summary>
    protected virtual long MaxSamples => 48000L * 600;

    /// <summary>
    /// The detection confidence from which <see cref="EvaluateAudio"/> reports a finding (never at confidence 0).
    /// Default: 0.3 (z = 3.9).
    /// </summary>
    protected virtual double ReportThreshold => 0.3;

    /// <summary>A short description of the watermark, used in findings.</summary>
    protected abstract string WatermarkDescription { get; }

    /// <summary>Separates this watermarker's patterns from those of other watermarkers sharing the key.</summary>
    protected abstract int PatternSalt { get; }

    /// <summary>
    /// Adds the watermark to one frame's band coefficients, in place.
    /// </summary>
    /// <param name="band">The frame's DCT coefficients inside the watermark band.</param>
    /// <param name="pattern">The key's +/-1 pattern for this frame, one entry per band coefficient.</param>
    /// <param name="frameRms">The root-mean-square of the frame's samples before marking.</param>
    protected abstract void EmbedFrame(double[] band, double[] pattern, double frameRms);

    /// <summary>
    /// Fills key-independent detection weights for one frame's received band coefficients.
    /// </summary>
    /// <param name="band">The frame's DCT coefficients inside the watermark band.</param>
    /// <param name="weights">Receives one weight per coefficient.</param>
    /// <returns><c>false</c> if the frame carries no usable evidence (for example digital silence).</returns>
    protected abstract bool DetectionWeights(double[] band, double[] weights);

    /// <inheritdoc />
    /// <exception cref="ArgumentException">
    /// The audio is shorter than one frame, longer than <see cref="MaxSamples"/>, the sample rate leaves too narrow a
    /// band, or the marked audio would not reach full detection confidence (z = 6).
    /// </exception>
    public Vector<T> EmbedWatermark(Vector<T> audioSamples, int sampleRate)
    {
        var layout = Layout(audioSamples, sampleRate);
        if (audioSamples.Length > MaxSamples)
            throw new ArgumentException($"{audioSamples.Length} samples exceed the {MaxSamples}-sample limit of {ModuleName}.", nameof(audioSamples));
        if (layout.Frames == 0)
            throw new ArgumentException($"The audio has {audioSamples.Length} samples; at least one {layout.FrameSize}-sample frame is needed.", nameof(audioSamples));
        if (layout.BandEnd - layout.BandStart < MinimumBandCoefficients)
            throw new ArgumentException($"At {sampleRate} Hz the {BandLowHz}-{BandHighHz} Hz band offers fewer than {MinimumBandCoefficients} coefficients per frame.", nameof(sampleRate));

        var values = ToDoubles(audioSamples);
        double peak = 0;
        foreach (double v in values) peak = Math.Max(peak, Math.Abs(v));
        bool nominal = peak <= 1.0;
        int n = layout.FrameSize, bandLength = layout.BandEnd - layout.BandStart;
        var frame = new double[n];
        var band = new double[bandLength];
        for (int f = 0; f < layout.Frames; f++)
        {
            Array.Copy(values, f * n, frame, 0, n);
            double sum = 0;
            foreach (double v in frame) sum += v * v;
            double rms = Math.Sqrt(sum / n);
            var coefficients = Dct(frame);
            Array.Copy(coefficients, layout.BandStart, band, 0, bandLength);
            EmbedFrame(band, Key.SignPattern(PatternSalt, f, bandLength), rms);
            Array.Copy(band, 0, coefficients, layout.BandStart, bandLength);
            var marked = InverseDct(coefficients);
            for (int i = 0; i < n; i++)
                values[f * n + i] = nominal ? Math.Max(-1.0, Math.Min(1.0, marked[i])) : marked[i];
        }

        var result = FromDoubles(values);
        double z = Detect(result, sampleRate).ZScore;
        if (!(z >= WatermarkStatistics.FullConfidenceZ))
            throw new ArgumentException(
                $"At strength {WatermarkStrength} the mark on this {audioSamples.Length}-sample clip reaches a detection z of {z:F1}, " +
                $"below the {WatermarkStatistics.FullConfidenceZ} needed for full confidence; raise the strength or use longer, non-silent audio.",
                nameof(audioSamples));
        return result;
    }

    /// <summary>
    /// Detects this watermarker's keyed watermark and reports the correlation test behind the confidence.
    /// </summary>
    /// <param name="audioSamples">The audio samples to check.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>The detection result; audio longer than <see cref="MaxSamples"/> reports no evidence.</returns>
    public virtual WatermarkResult Detect(Vector<T> audioSamples, int sampleRate)
    {
        var frames = FrameStatistics(audioSamples, sampleRate);
        int units = 0;
        foreach (var f in frames) units += f.Count;
        return AudioResult(PooledZ(frames), units);
    }

    /// <inheritdoc />
    public double DetectWatermark(Vector<T> audioSamples, int sampleRate) => Detect(audioSamples, sampleRate).Confidence;

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        var findings = new List<SafetyFinding>();
        var result = Detect(audioSamples, sampleRate);
        // A confidence of 0 is no evidence at all, so even a zero threshold does not report it.
        if (result.Confidence > 0 && result.Confidence >= ReportThreshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = result.Confidence,
                Description = $"{WatermarkDescription} detected (confidence: {result.Confidence:F3}, z = {result.ZScore:F2}, p = {result.PValue:E1}).",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>An audio result for a z-score over <paramref name="scoredUnits"/> coefficients.</summary>
    protected WatermarkResult AudioResult(double z, int scoredUnits) =>
        WatermarkStatistics.Result(z, scoredUnits, "audio", WatermarkDescription, ReportThreshold);

    /// <summary>The correlation numerator, weight energy and coefficient count of each frame, in order.</summary>
    /// <remarks>Audio longer than <see cref="MaxSamples"/>, shorter than a frame, or at a sample rate leaving too narrow a band yields no frames.</remarks>
    protected IReadOnlyList<(int StartSample, double Numerator, double Energy, int Count)> FrameStatistics(Vector<T> audioSamples, int sampleRate)
    {
        var layout = Layout(audioSamples, sampleRate);
        var stats = new List<(int, double, double, int)>(layout.Frames);
        int bandLength = layout.BandEnd - layout.BandStart;
        if (audioSamples.Length > MaxSamples || bandLength < MinimumBandCoefficients) return stats;

        var values = ToDoubles(audioSamples);
        int n = layout.FrameSize;
        var frame = new double[n];
        var band = new double[bandLength];
        var weights = new double[bandLength];
        for (int f = 0; f < layout.Frames; f++)
        {
            Array.Copy(values, f * n, frame, 0, n);
            var coefficients = Dct(frame);
            Array.Copy(coefficients, layout.BandStart, band, 0, bandLength);
            bool finite = true;
            foreach (double c in band) finite &= !(double.IsNaN(c) || double.IsInfinity(c));
            if (!finite || !DetectionWeights(band, weights))
            {
                stats.Add((f * n, 0, 0, 0));
                continue;
            }

            var pattern = Key.SignPattern(PatternSalt, f, bandLength);
            double numerator = 0, energy = 0;
            for (int k = 0; k < bandLength; k++)
            {
                numerator += weights[k] * pattern[k];
                energy += weights[k] * weights[k];
            }

            stats.Add((f * n, numerator, energy, bandLength));
        }

        return stats;
    }

    /// <summary>The correlation z-score pooled over frames.</summary>
    protected static double PooledZ(IReadOnlyList<(int StartSample, double Numerator, double Energy, int Count)> frames)
    {
        double numerator = 0, energy = 0;
        foreach (var f in frames)
        {
            numerator += f.Numerator;
            energy += f.Energy;
        }

        return WatermarkStatistics.CorrelationZ(numerator, energy);
    }

    /// <summary>
    /// The additive spread-spectrum rule (Cox et al. 1997; Kirovski and Malvar 2003): adds
    /// a = 0.2 x strength x max(spread, 0.03 x frame RMS) times the pattern, where spread = 1.4826 x median|c| is a
    /// robust estimate of the band's coefficient scale. The mark therefore follows the loudness of each frame and
    /// leaves digital silence untouched.
    /// </summary>
    protected void AdditiveEmbed(double[] band, double[] pattern, double frameRms)
    {
        double amplitude = 0.2 * WatermarkStrength * Math.Max(RobustSpread(band), 0.03 * frameRms);
        for (int k = 0; k < band.Length; k++) band[k] += amplitude * pattern[k];
    }

    /// <summary>
    /// Detection weights for the additive rule: each coefficient over the band's robust spread, clipped to +/-3 so a
    /// few strong tonal coefficients cannot dominate the sum. Frames with no spread carry no evidence.
    /// </summary>
    protected static bool AdditiveWeights(double[] band, double[] weights)
    {
        double spread = RobustSpread(band);
        if (!(spread > 1e-12)) return false;
        for (int k = 0; k < band.Length; k++) weights[k] = Math.Max(-3.0, Math.Min(3.0, band[k] / spread));
        return true;
    }

    /// <summary>1.4826 times the median absolute value: the standard deviation for Gaussian values, robust to outliers.</summary>
    protected static double RobustSpread(double[] values)
    {
        if (values.Length == 0) return 0;
        var magnitudes = new double[values.Length];
        for (int i = 0; i < values.Length; i++) magnitudes[i] = Math.Abs(values[i]);
        Array.Sort(magnitudes);
        int mid = magnitudes.Length / 2;
        double median = magnitudes.Length % 2 == 1 ? magnitudes[mid] : 0.5 * (magnitudes[mid - 1] + magnitudes[mid]);
        return 1.4826 * median;
    }

    /// <summary>The frame size used at <paramref name="sampleRate"/>: the smallest power of two of at least 0.032 x sampleRate samples.</summary>
    protected static int FrameSizeFor(int sampleRate)
    {
        int n = 2;
        while (n < 0.032 * sampleRate) n <<= 1;
        return n;
    }

    private (int FrameSize, int Frames, int BandStart, int BandEnd) Layout(Vector<T> audioSamples, int sampleRate)
    {
        if (audioSamples is null) throw new ArgumentNullException(nameof(audioSamples));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Sample rate must be positive.");
        int n = FrameSizeFor(sampleRate);
        // DCT-II bin k of an n-sample frame sits at k * sampleRate / (2n) Hz.
        int start = Math.Max(1, (int)Math.Ceiling(BandLowHz * 2.0 * n / sampleRate));
        int end = Math.Min(n, (int)Math.Floor(BandHighHz * 2.0 * n / sampleRate));
        return (n, audioSamples.Length / n, start, Math.Max(start, end));
    }

    private double[] ToDoubles(Vector<T> samples)
    {
        var values = new double[samples.Length];
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.ToDouble(samples[i]);
        return values;
    }

    private Vector<T> FromDoubles(double[] values)
    {
        var result = new Vector<T>(values.Length);
        for (int i = 0; i < values.Length; i++) result[i] = NumOps.FromDouble(values[i]);
        return result;
    }

    // Orthonormal DCT-II of a power-of-two frame through one complex FFT (Makhoul 1980).
    private static double[] Dct(double[] x)
    {
        int n = x.Length;
        var re = new double[n];
        var im = new double[n];
        for (int i = 0; i < n / 2; i++)
        {
            re[i] = x[2 * i];
            re[n - 1 - i] = x[2 * i + 1];
        }

        Fft(re, im, inverse: false);
        var result = new double[n];
        double s0 = Math.Sqrt(1.0 / n), s = Math.Sqrt(2.0 / n);
        for (int k = 0; k < n; k++)
        {
            double angle = -Math.PI * k / (2.0 * n);
            result[k] = (re[k] * Math.Cos(angle) - im[k] * Math.Sin(angle)) * (k == 0 ? s0 : s);
        }

        return result;
    }

    // Inverse of Dct: V_k = exp(i pi k / 2n) (C_k - i C_{n-k}), C_n = 0, then an inverse FFT and the even/odd unshuffle.
    private static double[] InverseDct(double[] coefficients)
    {
        int n = coefficients.Length;
        double s0 = Math.Sqrt(1.0 / n), s = Math.Sqrt(2.0 / n);
        var c = new double[n];
        for (int k = 0; k < n; k++) c[k] = coefficients[k] / (k == 0 ? s0 : s);
        var re = new double[n];
        var im = new double[n];
        for (int k = 0; k < n; k++)
        {
            double a = c[k], b = k == 0 ? 0.0 : -c[n - k];
            double angle = Math.PI * k / (2.0 * n);
            double cos = Math.Cos(angle), sin = Math.Sin(angle);
            re[k] = a * cos - b * sin;
            im[k] = a * sin + b * cos;
        }

        Fft(re, im, inverse: true);
        var x = new double[n];
        for (int i = 0; i < n / 2; i++)
        {
            x[2 * i] = re[i];
            x[2 * i + 1] = re[n - 1 - i];
        }

        return x;
    }

    // In-place iterative radix-2 FFT; the inverse is scaled by 1/n.
    private static void Fft(double[] re, double[] im, bool inverse)
    {
        int n = re.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        var (cosTable, sinTable) = Twiddles.GetOrAdd(n, size =>
        {
            var cos = new double[size / 2];
            var sin = new double[size / 2];
            for (int k = 0; k < size / 2; k++)
            {
                cos[k] = Math.Cos(2 * Math.PI * k / size);
                sin[k] = Math.Sin(2 * Math.PI * k / size);
            }

            return (cos, sin);
        });

        for (int length = 2; length <= n; length <<= 1)
        {
            int half = length / 2, step = n / length;
            for (int start = 0; start < n; start += length)
            {
                for (int k = 0; k < half; k++)
                {
                    double wr = cosTable[k * step];
                    double wi = inverse ? sinTable[k * step] : -sinTable[k * step];
                    int a = start + k, b = a + half;
                    double tr = re[b] * wr - im[b] * wi;
                    double ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }

        if (inverse)
        {
            for (int i = 0; i < n; i++)
            {
                re[i] /= n;
                im[i] /= n;
            }
        }
    }
}
