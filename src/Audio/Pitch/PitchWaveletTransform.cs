namespace AiDotNet.Audio.Pitch;

/// <summary>
/// The pitch spectrogram of FastSpeech 2: a 10-scale continuous wavelet transform of the normalized log-F0
/// contour, and its approximate inverse.
/// </summary>
/// <remarks>
/// <para>
/// FastSpeech 2 (Ren et al. 2021, §2.3 and App. C) predicts pitch as a wavelet "pitch spectrogram" rather than a raw
/// contour: unvoiced frames are filled by linear interpolation, the contour is taken to log scale and normalized
/// per utterance, then decomposed with the Mexican hat wavelet into 10 scales; the predicted spectrogram is
/// recomposed with F0 = Σ W_i (i + 2.5)^(-5/2) and denormalized with the predicted utterance mean and deviation.
/// </para>
/// <para>
/// This follows the authors' implementation (NATSpeech <c>utils/audio/cwt.py</c>, which calls pycwt): sampling
/// interval <c>dt = 0.005</c> s regardless of hop, scale spacing <c>dj = 1</c>, smallest scale <c>s0 = 2 dt</c>,
/// <c>J = 9</c> (10 scales), and pycwt's scipy code path, which zero-pads the signal to the next power of two
/// before transforming and trims the result back. The recomposed contour is re-standardized over time before
/// the predicted statistics are applied, as <c>inverse_cwt</c> does.
/// </para>
/// </remarks>
public static class PitchWaveletTransform
{
    /// <summary>Wavelet sampling interval used by the reference implementation, in seconds.</summary>
    public const double SamplingInterval = 0.005;

    /// <summary>Number of wavelet scales in the pitch spectrogram.</summary>
    public const int ScaleCount = 10;

    private const double ScaleSpacing = 1.0;

    /// <summary>
    /// Fills unvoiced frames and takes the log: frames before the first and after the last voiced frame take the
    /// nearest voiced value, interior unvoiced frames are linearly interpolated (NATSpeech <c>get_cont_lf0</c>).
    /// </summary>
    /// <param name="f0">F0 in Hz per frame, 0 where unvoiced.</param>
    /// <returns>The unvoiced mask (1 where the input was 0) and the continuous log-F0 contour. If no frame is
    /// voiced the contour is returned unchanged (all zero), as the reference does.</returns>
    public static (double[] Unvoiced, double[] ContinuousLogF0) ContinuousLogF0(double[] f0)
    {
        if (f0 is null) throw new ArgumentNullException(nameof(f0));
        int n = f0.Length;
        var uv = new double[n];
        for (int i = 0; i < n; i++) uv[i] = f0[i] == 0 ? 1.0 : 0.0;

        int first = Array.FindIndex(f0, v => v != 0);
        if (first < 0) return (uv, (double[])f0.Clone());
        int last = Array.FindLastIndex(f0, v => v != 0);

        var filled = (double[])f0.Clone();
        for (int i = 0; i < first; i++) filled[i] = f0[first];
        for (int i = last; i < n; i++) filled[i] = f0[last];

        int previous = -1;
        for (int i = 0; i < n; i++)
        {
            if (filled[i] == 0) continue;
            if (previous >= 0 && i - previous > 1)
            {
                for (int j = previous + 1; j < i; j++)
                    filled[j] = filled[previous] + (filled[i] - filled[previous]) * (j - previous) / (i - previous);
            }
            previous = i;
        }

        var logF0 = new double[n];
        for (int i = 0; i < n; i++) logF0[i] = Math.Log(filled[i]);
        return (uv, logF0);
    }

    /// <summary>
    /// The 10-scale Mexican hat wavelet transform of <paramref name="signal"/> (pycwt <c>cwt</c> with NATSpeech's
    /// parameters), returned as <c>[time, scale]</c>.
    /// </summary>
    public static double[,] Forward(double[] signal)
    {
        if (signal is null) throw new ArgumentNullException(nameof(signal));
        int n0 = signal.Length;
        if (n0 == 0) return new double[0, ScaleCount];

        // scipy.fftpack path: pad to the next power of two.
        int n = 1;
        while (n < n0) n <<= 1;
        var re = new double[n];
        Array.Copy(signal, re, n0);
        var (sigRe, sigIm) = Fft(re, new double[n], inverse: false);

        // DOG m = 2: psi_ft(f) = f^2 exp(-f^2 / 2) / sqrt(Gamma(2.5)); flambda = 2 pi / sqrt(2.5).
        double gamma25 = 1.329340388179137; // Gamma(2.5) = 3 sqrt(pi) / 4
        double flambda = 2 * Math.PI / Math.Sqrt(2.5);
        double s0 = 2 * SamplingInterval;
        var result = new double[n0, ScaleCount];
        var waveRe = new double[n];
        var waveIm = new double[n];
        for (int j = 0; j < ScaleCount; j++)
        {
            double sj = s0 * Math.Pow(2, j * ScaleSpacing);
            // pycwt recomputes the scale through its Fourier frequency; the round trip is kept for bit-faithfulness.
            double freq = 1 / (flambda * sj);
            sj = 1 / (flambda * freq);
            double norm = Math.Sqrt(sj * (2 * Math.PI / (n * SamplingInterval)) * n);
            for (int k = 0; k < n; k++)
            {
                int signedK = k <= (n - 1) / 2 ? k : k - n; // numpy fftfreq ordering
                double omega = 2 * Math.PI * signedK / (n * SamplingInterval);
                double f = sj * omega;
                double psi = f * f * Math.Exp(-0.5 * f * f) / Math.Sqrt(gamma25);
                double factor = norm * psi;
                waveRe[k] = sigRe[k] * factor;
                waveIm[k] = sigIm[k] * factor;
            }
            var (wRe, _) = Fft(waveRe, waveIm, inverse: true);
            for (int t = 0; t < n0; t++) result[t, j] = wRe[t];
        }
        return result;
    }

    /// <summary>
    /// Recomposes a pitch spectrogram <c>[time, scale]</c> into a standardized log-F0 contour (NATSpeech
    /// <c>inverse_cwt</c>: Σ W_i (i + 2.5)^(-5/2) over i = 1..10, then zero mean and unit deviation over time).
    /// </summary>
    public static double[] Inverse(double[,] spectrogram)
    {
        if (spectrogram is null) throw new ArgumentNullException(nameof(spectrogram));
        int n = spectrogram.GetLength(0), scales = spectrogram.GetLength(1);
        var sum = new double[n];
        for (int t = 0; t < n; t++)
        {
            double acc = 0;
            for (int i = 0; i < scales; i++) acc += spectrogram[t, i] * Math.Pow(i + 1 + 2.5, -2.5);
            sum[t] = acc;
        }
        if (n == 0) return sum;
        double mean = sum.Average();
        double variance = sum.Select(v => (v - mean) * (v - mean)).Average();
        double std = Math.Sqrt(variance);
        // Zero deviation (a constant contour or one frame) standardizes to 0 rather than 0/0 = NaN.
        for (int t = 0; t < n; t++) sum[t] = std > 0 ? (sum[t] - mean) / std : 0.0;
        return sum;
    }

    /// <summary>
    /// Converts a pitch spectrogram back to F0 in Hz with the utterance's log-F0 mean and deviation
    /// (NATSpeech <c>cwt2f0</c>).
    /// </summary>
    public static double[] ToF0(double[,] spectrogram, double logF0Mean, double logF0Std)
    {
        var standardized = Inverse(spectrogram);
        var f0 = new double[standardized.Length];
        for (int t = 0; t < f0.Length; t++) f0[t] = Math.Exp(standardized[t] * logF0Std + logF0Mean);
        return f0;
    }

    /// <summary>
    /// <see cref="ContinuousLogF0(double[])"/> on a tensor contour <c>[frames]</c> with engine operations: each frame
    /// reads the voiced frame at its position clamped to the first and last voiced frame, or interpolates linearly
    /// between the voiced frames around it.
    /// </summary>
    /// <typeparam name="T">The numeric type of the contour.</typeparam>
    /// <remarks>The interpolation product (r − l)(j − i<sub>l</sub>) is scaled by a power of two before it is formed
    /// and back after the division: exact in binary floating point, and a half-precision product cannot overflow.</remarks>
    internal static (Tensor<T> Unvoiced, Tensor<T> ContinuousLogF0) ContinuousLogF0<T>(Tensor<T> f0)
    {
        if (f0 is null) throw new ArgumentNullException(nameof(f0));
        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(f0, new[] { f0.Length });
        var unvoiced = engine.TensorEquals(flat, MathHelper.GetNumericOperations<T>().Zero);
        if (engine.TensorCountNonzero(flat) == 0) return (unvoiced, flat.Clone());
        return (unvoiced, engine.TensorLog(FillUnvoiced(flat)));
    }

    /// <summary>
    /// The F0 contour <c>[frames]</c> with unvoiced (0) frames filled: before the first and after the last voiced frame
    /// by the nearest voiced value, in between by linear interpolation of the voiced frames around them; returned
    /// unchanged when no frame is voiced.
    /// </summary>
    internal static Tensor<T> FillUnvoiced<T>(Tensor<T> f0)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = f0.Length;
        var flat = engine.Reshape(f0, new[] { n });
        var nonzero = engine.TensorNonzero(flat);
        int count = nonzero.Shape[0];
        if (count == 0) return flat.Clone();
        var voiced = engine.Reshape(nonzero, new[] { count });
        int[] voicedFrames = voiced.ToArray();

        var position = engine.TensorClamp(engine.TensorArange<int>(0, n, 1), voicedFrames[0], voicedFrames[count - 1]);
        var atPosition = engine.TensorIndexSelect(flat, position, 0);
        if (count == 1) return atPosition;

        var upper = engine.TensorClamp(engine.TensorSearchSorted(voiced, position, true), 1, count - 1);
        var lower = engine.TensorAddScalar(upper, -1);
        var left = engine.TensorIndexSelect(voiced, lower, 0);
        var right = engine.TensorIndexSelect(voiced, upper, 0);
        var voicedValues = engine.TensorIndexSelect(flat, voiced, 0);
        // The interpolation runs in double and is narrowed once: frame offsets beyond 65504 do not fit a half-precision
        // T, and neither does the power-of-two scale that kept their product in range.
        var leftValue = TensorCopyHelper.CastLogical<T, double>(engine.TensorIndexSelect(voicedValues, lower, 0));
        var rightValue = TensorCopyHelper.CastLogical<T, double>(engine.TensorIndexSelect(voicedValues, upper, 0));
        var step = engine.TensorDivide(
            engine.TensorMultiply(engine.TensorSubtract(rightValue, leftValue), engine.TensorSubtract(position, left).Cast<double>()),
            engine.TensorSubtract(right, left).Cast<double>());
        var interpolated = engine.TensorAdd(leftValue, step).Cast<T>();
        // Voiced positions (every frame outside the voiced span reads one) keep their value exactly.
        return engine.TensorWhere(engine.TensorNotEquals(atPosition, ops.Zero), atPosition, interpolated);
    }

    /// <summary>
    /// <see cref="Forward(double[])"/> on a tensor signal <c>[time]</c> with engine operations, returned as
    /// <c>[time, scale]</c>: one FFT of the zero-padded signal, the ten scales' wavelet spectra as one
    /// <c>[scale, n]</c> product, and one batched inverse FFT.
    /// </summary>
    /// <typeparam name="T">The numeric type of the signal.</typeparam>
    internal static Tensor<T> Forward<T>(Tensor<T> signal)
    {
        if (signal is null) throw new ArgumentNullException(nameof(signal));
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n0 = signal.Length;
        if (n0 == 0) return new Tensor<T>(new[] { 0, ScaleCount });
        int n = 1;
        while (n < n0) n <<= 1;
        var re = engine.Reshape(signal, new[] { n0 });
        if (n > n0) re = engine.PadNd(re, new[] { 0, n - n0 }, AiDotNet.Tensors.Engines.PadMode.Constant, ops.Zero);
        engine.FFT(re, new Tensor<T>(new[] { n }), out var sigRe, out var sigIm);
        var factor = new Tensor<double>(WaveletFactors(n), new[] { ScaleCount, n }).Cast<T>();
        var waveRe = engine.TensorMultiply(engine.TensorBroadcastTo(engine.Reshape(sigRe, new[] { 1, n }), new[] { ScaleCount, n }), factor);
        var waveIm = engine.TensorMultiply(engine.TensorBroadcastTo(engine.Reshape(sigIm, new[] { 1, n }), new[] { ScaleCount, n }), factor);
        engine.IFFT(waveRe, waveIm, out var outRe, out _);
        return engine.TensorTranspose(engine.TensorSlice(outRe, new[] { 0, 0 }, new[] { ScaleCount, n0 }));
    }

    /// <summary>
    /// <see cref="ToF0(double[,], double, double)"/> on a tensor spectrogram <c>[time, scale]</c>: the weighted scale
    /// sum, standardized over time, then <c>exp(z σ + μ)</c>, <c>[time]</c>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the spectrogram.</typeparam>
    internal static Tensor<T> ToF0<T>(Tensor<T> spectrogram, T logF0Mean, T logF0Std)
    {
        if (spectrogram is null) throw new ArgumentNullException(nameof(spectrogram));
        if (spectrogram.Rank != 2) throw new ArgumentException("Expected a [time, scale] spectrogram.", nameof(spectrogram));
        var engine = AiDotNetEngine.Current;
        int n = spectrogram.Shape[0], scales = spectrogram.Shape[1];
        if (n == 0) return new Tensor<T>(new[] { 0 });
        var weights = new double[scales];
        for (int i = 0; i < scales; i++) weights[i] = Math.Pow(i + 1 + 2.5, -2.5);
        var sum = engine.Reshape(engine.TensorMatMul(spectrogram, new Tensor<double>(weights, new[] { scales, 1 }).Cast<T>()), new[] { n });
        var centred = engine.TensorSubtractScalar(sum, engine.TensorMean(sum));
        var ops = MathHelper.GetNumericOperations<T>();
        var std = ops.Sqrt(engine.TensorMean(engine.TensorMultiply(centred, centred)));
        // A constant contour (or a single frame) has zero deviation: its standardized value is 0, so every frame maps
        // to exp(logF0Mean) instead of 0/0 = NaN.
        var standardized = ops.GreaterThan(std, ops.Zero)
            ? engine.TensorDivideScalar(centred, std)
            : engine.TensorZerosLike(centred);
        return engine.TensorExp(engine.TensorAddScalar(engine.TensorMultiplyScalar(standardized, logF0Std), logF0Mean));
    }

    // norm · ψ(s_j ω_k) of every scale j and FFT bin k, [scale, n] (the per-bin factor of Forward).
    private static double[] WaveletFactors(int n)
    {
        double gamma25 = 1.329340388179137; // Gamma(2.5) = 3 sqrt(pi) / 4
        double flambda = 2 * Math.PI / Math.Sqrt(2.5);
        double s0 = 2 * SamplingInterval;
        var factors = new double[ScaleCount * n];
        for (int j = 0; j < ScaleCount; j++)
        {
            double sj = s0 * Math.Pow(2, j * ScaleSpacing);
            double freq = 1 / (flambda * sj);
            sj = 1 / (flambda * freq);
            double norm = Math.Sqrt(sj * (2 * Math.PI / (n * SamplingInterval)) * n);
            for (int k = 0; k < n; k++)
            {
                int signedK = k <= (n - 1) / 2 ? k : k - n;
                double omega = 2 * Math.PI * signedK / (n * SamplingInterval);
                double f = sj * omega;
                factors[j * n + k] = norm * (f * f * Math.Exp(-0.5 * f * f) / Math.Sqrt(gamma25));
            }
        }
        return factors;
    }

    private static (double[] Re, double[] Im) Fft(double[] re, double[] im, bool inverse)
    {
        int n = re.Length;
        var tRe = new Tensor<double>(new[] { n }, new Vector<double>((double[])re.Clone()));
        var tIm = new Tensor<double>(new[] { n }, new Vector<double>((double[])im.Clone()));
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        Tensor<double> outRe, outIm;
        if (inverse) engine.IFFT(tRe, tIm, out outRe, out outIm);
        else engine.FFT(tRe, tIm, out outRe, out outIm);
        return (outRe.AsSpan().ToArray(), outIm.AsSpan().ToArray());
    }
}
