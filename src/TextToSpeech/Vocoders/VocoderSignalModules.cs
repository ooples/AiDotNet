using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// A pseudo-quadrature mirror filter bank (Nguyen 1994, "Near-perfect-reconstruction pseudo-QMF banks"): cosine-modulated
/// analysis filters that split a waveform into critically sampled sub-bands and synthesis filters that merge them back.
/// </summary>
/// <remarks>
/// <para>The prototype is a Kaiser-windowed sinc (Lin and Vaidyanathan 1998) with <c>taps + 1</c> coefficients; band k
/// is <c>2 h[n] cos((2k + 1) π/(2K) (n − taps/2) ± (−1)ᵏ π/4)</c> (+ for analysis, − for synthesis). Analysis
/// zero-pads by taps/2, filters and keeps every K-th sample; synthesis inserts K − 1 zeros between samples, scales by K,
/// zero-pads and filters (reference: kan-bayashi/ParallelWaveGAN <c>layers/pqmf.py</c>, whose 62 taps, cutoff 0.142 and
/// β = 9 are tuned for four bands, the 63-coefficient filters of Multi-band MelGAN, Yang et al. 2021 §3.2).</para>
/// </remarks>
internal sealed class PseudoQmf<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _bands;
    private readonly int _taps;
    private readonly Tensor<T> _analysis;     // [K, 1, 1, taps + 1]
    private readonly Tensor<T> _synthesis;    // [1, K, 1, taps + 1]

    public PseudoQmf(IEngine engine, int bands = 4, int taps = 62, double cutoffRatio = 0.142, double beta = 9.0)
    {
        if (taps % 2 != 0) throw new ArgumentException("The number of taps must be even.", nameof(taps));
        if (cutoffRatio <= 0 || cutoffRatio >= 1) throw new ArgumentOutOfRangeException(nameof(cutoffRatio));
        _engine = engine;
        _bands = bands;
        _taps = taps;
        int n = taps + 1;
        var prototype = new double[n];
        double omega = Math.PI * cutoffRatio, i0Beta = BesselI0(beta);
        for (int i = 0; i < n; i++)
        {
            double t = i - 0.5 * taps;
            double sinc = i == taps / 2 ? cutoffRatio : Math.Sin(omega * t) / (Math.PI * t);
            double ratio = 2.0 * i / taps - 1.0;
            double kaiser = BesselI0(beta * Math.Sqrt(Math.Max(0.0, 1 - ratio * ratio))) / i0Beta;
            prototype[i] = sinc * kaiser;
        }
        _analysis = new Tensor<T>(new[] { bands, 1, 1, n });
        _synthesis = new Tensor<T>(new[] { 1, bands, 1, n });
        for (int k = 0; k < bands; k++)
        {
            double phase = (k % 2 == 0 ? 1 : -1) * Math.PI / 4;
            for (int i = 0; i < n; i++)
            {
                double arg = (2 * k + 1) * (Math.PI / (2 * bands)) * (i - taps / 2.0);
                _analysis[k, 0, 0, i] = NumOps.FromDouble(2 * prototype[i] * Math.Cos(arg + phase));
                _synthesis[0, k, 0, i] = NumOps.FromDouble(2 * prototype[i] * Math.Cos(arg - phase));
            }
        }
    }

    /// <summary>The number of bands.</summary>
    public int Bands => _bands;

    // Modified Bessel function of the first kind, order 0 (the Kaiser window's normalizer).
    private static double BesselI0(double x)
    {
        double sum = 1, term = 1, half = x / 2;
        for (int k = 1; k < 200; k++)
        {
            term *= half / k * (half / k);
            sum += term;
            if (term < 1e-17 * sum) break;
        }
        return sum;
    }

    /// <summary>Sub-bands <c>[1, K, T / K]</c> of a waveform <c>[1, 1, T]</c>.</summary>
    public Tensor<T> Analysis(Tensor<T> x)
    {
        int t = x.Shape[2];
        var padded = _engine.Reshape(VocoderOps.ZeroPad(_engine, x, _taps / 2, _taps / 2), new[] { 1, 1, 1, t + _taps });
        var filtered = _engine.Conv2D(padded, _analysis, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });        // [1, K, 1, t]
        int kept = (t - _bands) / _bands + 1;                                                   // conv1d(·, eye, stride K)
        var index = new Tensor<int>(new[] { kept });
        for (int i = 0; i < kept; i++) index[i] = i * _bands;
        var rows = _engine.TensorIndexSelect(_engine.TensorTranspose(_engine.Reshape(filtered, new[] { _bands, t })), index, 0);   // [kept, K]
        return _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, _bands, kept });
    }

    /// <summary>A waveform <c>[1, 1, T · K]</c> from sub-bands <c>[1, K, T]</c>.</summary>
    public Tensor<T> Synthesis(Tensor<T> bands)
    {
        int t = bands.Shape[2], length = t * _bands;
        // Zero insertion scaled by K: sample j of band k lands at j·K.
        var rows = _engine.TensorTranspose(_engine.Reshape(bands, new[] { _bands, t }));                         // [t, K]
        var zeros = new Tensor<T>(new[] { t, _bands * (_bands - 1) });
        var interleaved = _engine.TensorConcatenate(new[] { _engine.TensorMultiplyScalar(rows, NumOps.FromDouble(_bands)), zeros }, 1); // [t, K·K]
        // Row j holds band k's value at column k followed by zeros; reshape to [t, K, K] then read [K, t·K] per band.
        var spread = _engine.Reshape(interleaved, new[] { t, _bands, _bands });
        var perBand = _engine.Reshape(_engine.TensorTranspose(_engine.Reshape(spread, new[] { t * _bands, _bands })), new[] { 1, _bands, length });
        var padded = _engine.Reshape(VocoderOps.ZeroPad(_engine, perBand, _taps / 2, _taps / 2), new[] { 1, _bands, 1, length + _taps });
        var output = _engine.Conv2D(padded, _synthesis, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });          // [1, 1, 1, length]
        return _engine.Reshape(output, new[] { 1, 1, length });
    }
}

/// <summary>
/// The multi-resolution STFT loss (Yamamoto et al. 2020, Parallel WaveGAN §2.3, Eq. 3–5): for each resolution the
/// spectral convergence <c>‖ |S| − |Ŝ| ‖_F / ‖ |S| ‖_F</c> plus the log-magnitude distance <c>mean |log|S| − log|Ŝ||</c>,
/// averaged over the resolutions.
/// </summary>
/// <remarks>Magnitudes follow the reference (<c>losses/stft_loss.py</c>): <c>torch.stft</c> centred with reflect
/// padding, a periodic Hann window of the window length centred in the FFT frame, <c>√max(re² + im², 1e-7)</c>. The
/// spectral convergence and log-magnitude terms are returned separately (their sums over resolutions divided by the
/// number of resolutions).</remarks>
internal sealed class MultiResolutionStftLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<(int Fft, int Hop, Tensor<T> Cos, Tensor<T> Sin)> _resolutions = new();

    public MultiResolutionStftLoss(IEngine engine, int[] fftSizes, int[] hopSizes, int[] windowSizes)
    {
        if (fftSizes.Length != hopSizes.Length || fftSizes.Length != windowSizes.Length || fftSizes.Length == 0)
            throw new ArgumentException("Every resolution needs an FFT size, a hop and a window.");
        _engine = engine;
        for (int r = 0; r < fftSizes.Length; r++)
        {
            int fft = fftSizes[r], win = windowSizes[r], bins = fft / 2 + 1, offset = (fft - win) / 2;
            var cos = new Tensor<T>(new[] { fft, bins });
            var sin = new Tensor<T>(new[] { fft, bins });
            for (int n = 0; n < win; n++)
            {
                double w = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / win);
                for (int k = 0; k < bins; k++)
                {
                    double a = 2 * Math.PI * (offset + n) * k / fft;
                    cos[offset + n, k] = NumOps.FromDouble(w * Math.Cos(a));
                    sin[offset + n, k] = NumOps.FromDouble(-w * Math.Sin(a));
                }
            }
            _resolutions.Add((fft, hopSizes[r], cos, sin));
        }
    }

    /// <summary>The magnitude <c>[frames, bins]</c> of a waveform <c>[samples]</c> at one resolution.</summary>
    private Tensor<T> Magnitude(Tensor<T> audio, int fft, int hop, Tensor<T> cos, Tensor<T> sin)
    {
        int length = audio.Length, pad = fft / 2, frames = 1 + length / hop;
        var index = new Tensor<int>(new[] { frames * fft });
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < fft; n++)
                index[f * fft + n] = DifferentiableMel<T>.Reflect(f * hop + n - pad, length);
        var framed = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(audio, new[] { length, 1 }), index, 0), new[] { frames, fft });
        var re = _engine.TensorMatMul(framed, cos);
        var im = _engine.TensorMatMul(framed, sin);
        // sqrt(max(p, 1e-7)) = sqrt(1e-7 + relu(p − 1e-7))
        var power = _engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im));
        return _engine.TensorPow(_engine.TensorAddScalar(_engine.ReLU(_engine.TensorAddScalar(power, NumOps.FromDouble(-1e-7))),
            NumOps.FromDouble(1e-7)), NumOps.FromDouble(0.5));
    }

    /// <summary>The spectral convergence and log-magnitude losses of <paramref name="generated"/> against
    /// <paramref name="real"/> (each a set of equal-length waveforms <c>[samples]</c>, averaged over them), averaged over
    /// the resolutions.</summary>
    public (Tensor<T> SpectralConvergence, Tensor<T> LogMagnitude) Forward(IReadOnlyList<Tensor<T>> generated, IReadOnlyList<Tensor<T>> real)
    {
        Tensor<T>? sc = null, mag = null;
        double scale = 1.0 / (_resolutions.Count * generated.Count);
        foreach (var (fft, hop, cos, sin) in _resolutions)
        {
            for (int i = 0; i < generated.Count; i++)
            {
                var g = Magnitude(generated[i], fft, hop, cos, sin);
                Tensor<T> r;
                using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                {
                    var m = Magnitude(real[i], fft, hop, cos, sin);
                    r = new Tensor<T>(m._shape, m.ToVector());
                }
                var difference = _engine.TensorSubtract(r, g);
                var numerator = _engine.TensorPow(_engine.ReduceSum(_engine.TensorMultiply(difference, difference), new[] { 0, 1 }, keepDims: false), NumOps.FromDouble(0.5));
                double denominator = Math.Sqrt(r.ToVector().ToArray().Sum(v => NumOps.ToDouble(v) * NumOps.ToDouble(v)));
                var scTerm = _engine.TensorMultiplyScalar(numerator, NumOps.FromDouble(1.0 / Math.Max(denominator, 1e-12)));
                var magTerm = _engine.ReduceMean(_engine.TensorAbs(_engine.TensorSubtract(_engine.TensorLog(r), _engine.TensorLog(g))), new[] { 0, 1 }, keepDims: false);
                sc = sc is null ? scTerm : _engine.TensorAdd(sc, scTerm);
                mag = mag is null ? magTerm : _engine.TensorAdd(mag, magTerm);
            }
        }
        return (_engine.TensorMultiplyScalar(sc!, NumOps.FromDouble(scale)), _engine.TensorMultiplyScalar(mag!, NumOps.FromDouble(scale)));
    }
}

/// <summary>
/// A librosa-style log-mel spectrogram (<c>librosa.stft</c> centred with reflect padding, a periodic Hann window of the
/// window length centred in the FFT frame, magnitude, Slaney mel filters, <c>log10(max(x, floor))</c>) — the input
/// features of the kan-bayashi/ParallelWaveGAN recipes (<c>logmelfilterbank</c>) that Parallel WaveGAN and Multi-band
/// MelGAN are trained on — optionally standardized per band.
/// </summary>
internal sealed class CenteredLogMel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _fft;
    private readonly int _hop;
    private readonly int _mels;
    private readonly double _floor;
    private readonly bool _naturalLog;
    private readonly Tensor<T> _cos;
    private readonly Tensor<T> _sin;
    private readonly Tensor<T> _mel;

    /// <param name="htkScale">HTK mel filters without normalization (torchaudio's <c>MelSpectrogram</c> default) instead
    /// of librosa's Slaney ones.</param>
    /// <param name="naturalLog">Natural log (Vocos's <c>safe_log</c>) instead of log10.</param>
    /// <param name="normalizedWindow">Divide the STFT by √Σw² (torchaudio <c>Spectrogram(normalized=True)</c>,
    /// DiffWave's features).</param>
    public CenteredLogMel(IEngine engine, int sampleRate, int fftSize, int hopSize, int windowSize, int melChannels, double fMin, double fMax, double floor,
        bool htkScale = false, bool naturalLog = false, bool normalizedWindow = false)
    {
        _naturalLog = naturalLog;
        _engine = engine;
        _fft = fftSize;
        _hop = hopSize;
        _mels = melChannels;
        _floor = floor;
        int bins = fftSize / 2 + 1, offset = (fftSize - windowSize) / 2;
        _cos = new Tensor<T>(new[] { fftSize, bins });
        _sin = new Tensor<T>(new[] { fftSize, bins });
        double energy = 0;
        for (int n = 0; n < windowSize; n++) energy += Math.Pow(0.5 - 0.5 * Math.Cos(2 * Math.PI * n / windowSize), 2);
        double scale = normalizedWindow ? 1 / Math.Sqrt(energy) : 1.0;
        for (int n = 0; n < windowSize; n++)
        {
            double w = scale * (0.5 - 0.5 * Math.Cos(2 * Math.PI * n / windowSize));
            for (int k = 0; k < bins; k++)
            {
                double a = 2 * Math.PI * (offset + n) * k / fftSize;
                _cos[offset + n, k] = NumOps.FromDouble(w * Math.Cos(a));
                _sin[offset + n, k] = NumOps.FromDouble(-w * Math.Sin(a));
            }
        }
        if (htkScale)
        {
            _mel = AiDotNet.TextToSpeech.EndToEnd.HaspSpeakerEncoder<T>.HtkMelFilterbank(bins, melChannels, sampleRate, fMin, fMax);
            return;
        }
        var basis = new TacotronSpectrogram(sampleRate, fftSize, hopSize, windowSize, melChannels, fMin, fMax).MelBasis;
        _mel = new Tensor<T>(new[] { bins, melChannels });
        for (int m = 0; m < melChannels; m++)
            for (int k = 0; k < bins; k++) _mel[k, m] = NumOps.FromDouble(basis[m, k]);
    }

    /// <summary>The log-mel <c>[1, mel, 1 + samples / hop]</c> of <paramref name="audio"/> <c>[samples]</c>, standardized
    /// by <paramref name="mean"/> and <paramref name="scale"/> (per band) when given.</summary>
    public Tensor<T> Forward(Tensor<T> audio, double[]? mean = null, double[]? scale = null)
    {
        int length = audio.Length, pad = _fft / 2, frames = 1 + length / _hop;
        var index = new Tensor<int>(new[] { frames * _fft });
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < _fft; n++)
                index[f * _fft + n] = DifferentiableMel<T>.Reflect(f * _hop + n - pad, length);
        var framed = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(audio, new[] { length, 1 }), index, 0), new[] { frames, _fft });
        var re = _engine.TensorMatMul(framed, _cos);
        var im = _engine.TensorMatMul(framed, _sin);
        var magnitude = _engine.TensorPow(_engine.TensorAddScalar(_engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im)),
            NumOps.FromDouble(1e-20)), NumOps.FromDouble(0.5));
        var mel = _engine.TensorMatMul(magnitude, _mel);
        var floored = _engine.TensorAddScalar(_engine.ReLU(_engine.TensorAddScalar(mel, NumOps.FromDouble(-_floor))), NumOps.FromDouble(_floor));
        var logMel = _naturalLog ? _engine.TensorLog(floored)
            : _engine.TensorMultiplyScalar(_engine.TensorLog(floored), NumOps.FromDouble(1.0 / Math.Log(10.0)));       // [frames, mel]
        if (mean is not null && scale is not null)
        {
            var shift = new Tensor<T>(new[] { 1, _mels });
            var inverse = new Tensor<T>(new[] { 1, _mels });
            for (int m = 0; m < _mels; m++)
            {
                shift[0, m] = NumOps.FromDouble(-mean[m]);
                inverse[0, m] = NumOps.FromDouble(1.0 / scale[m]);
            }
            logMel = _engine.TensorMultiply(_engine.TensorAdd(logMel, _engine.TensorTile(shift, new[] { frames, 1 })),
                _engine.TensorTile(inverse, new[] { frames, 1 }));
        }
        return _engine.Reshape(_engine.TensorTranspose(logMel), new[] { 1, _mels, frames });
    }
}

/// <summary>
/// A differentiable inverse STFT with <c>torch.istft</c>'s semantics (one-sided spectrum, periodic Hann window of the
/// window length centred in the FFT frame, overlap-add normalized by the summed squared window, <c>center=True</c>
/// trimming of n_fft/2 samples at each end).
/// </summary>
internal sealed class InverseStft<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _fft;
    private readonly int _hop;
    private readonly double[] _window;
    private readonly Tensor<T> _cos;       // [bins, fft], windowed and scaled
    private readonly Tensor<T> _sin;       // [bins, fft]
    private readonly Tensor<T> _overlap;   // [fft, 1, 1, fft] identity for overlap-add

    public InverseStft(IEngine engine, int fftSize, int hopSize, int windowSize)
    {
        _engine = engine;
        _fft = fftSize;
        _hop = hopSize;
        int bins = fftSize / 2 + 1, offset = (fftSize - windowSize) / 2;
        _window = new double[fftSize];
        for (int n = 0; n < windowSize; n++) _window[offset + n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / windowSize);
        _cos = new Tensor<T>(new[] { bins, fftSize });
        _sin = new Tensor<T>(new[] { bins, fftSize });
        for (int k = 0; k < bins; k++)
        {
            // irfft: x[n] = (1/N)(X_0 + X_{N/2}(−1)^n + 2 Σ_k Re(X_k e^{2πikn/N})); the DC and Nyquist imaginary parts drop.
            double weight = (k == 0 || (fftSize % 2 == 0 && k == fftSize / 2)) ? 1.0 / fftSize : 2.0 / fftSize;
            bool realOnly = k == 0 || (fftSize % 2 == 0 && k == fftSize / 2);
            for (int n = 0; n < fftSize; n++)
            {
                double a = 2 * Math.PI * k * n / fftSize;
                _cos[k, n] = NumOps.FromDouble(weight * Math.Cos(a) * _window[n]);
                _sin[k, n] = NumOps.FromDouble(realOnly ? 0.0 : -weight * Math.Sin(a) * _window[n]);
            }
        }
        _overlap = new Tensor<T>(new[] { fftSize, 1, 1, fftSize });
        for (int c = 0; c < fftSize; c++) _overlap[c, 0, 0, c] = NumOps.One;
    }

    /// <summary>The waveform <c>[(frames − 1) · hop]</c> of a spectrum given by its magnitude and phase
    /// <c>[1, bins, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> magnitude, Tensor<T> phase)
    {
        int bins = magnitude.Shape[1], frames = magnitude.Shape[2];
        var re = _engine.TensorTranspose(_engine.Reshape(_engine.TensorMultiply(magnitude, _engine.TensorSin(_engine.TensorAddScalar(phase, NumOps.FromDouble(Math.PI / 2)))), new[] { bins, frames }));   // [F, bins]
        var im = _engine.TensorTranspose(_engine.Reshape(_engine.TensorMultiply(magnitude, _engine.TensorSin(phase)), new[] { bins, frames }));
        var framesTime = _engine.TensorAdd(_engine.TensorMatMul(re, _cos), _engine.TensorMatMul(im, _sin));                                       // [F, N]
        var channels = _engine.Reshape(_engine.TensorTranspose(framesTime), new[] { 1, _fft, 1, frames });
        var summed = _engine.ConvTranspose2D(channels, _overlap, new[] { 1, _hop }, new[] { 0, 0 }, new[] { 0, 0 });                              // [1, 1, 1, L]
        int length = (frames - 1) * _hop + _fft;
        var envelope = new Tensor<T>(new[] { 1, 1, 1, length });
        var sums = new double[length];
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < _fft; n++) sums[f * _hop + n] += _window[n] * _window[n];
        int start = _fft / 2, outLength = (frames - 1) * _hop;
        for (int i = 0; i < length; i++)
        {
            if (i >= start && i < start + outLength && sums[i] < 1e-11)
                throw new InvalidOperationException("The window envelope vanishes inside the signal (NOLA fails for this window and hop).");
            envelope[0, 0, 0, i] = NumOps.FromDouble(sums[i] > 1e-11 ? 1.0 / sums[i] : 0.0);
        }
        var normalized = _engine.Reshape(_engine.TensorMultiply(summed, envelope), new[] { length });
        return _engine.TensorSlice(normalized, new[] { start }, new[] { outLength });
    }
}
