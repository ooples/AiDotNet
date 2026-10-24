using AiDotNet.Helpers;

namespace AiDotNet.Audio.Features;

/// <summary>
/// Kaldi's log mel filterbank features, as <c>torchaudio.compliance.kaldi.fbank</c> computes them with no energy term,
/// no dither and snipped edges: frames of <see cref="FrameLengthMilliseconds"/> every <see cref="FrameShiftMilliseconds"/>,
/// each with its DC offset removed, pre-emphasized, windowed, zero-padded to a power of two and transformed; the power
/// spectrum is pooled by triangular filters equally spaced on Kaldi's mel scale (1127·ln(1 + f/700)) and logged.
/// </summary>
/// <remarks>
/// <para>Many audio models train on these features (AST, BEATs, and the HuBERT family's fine-tuning recipes), so a model
/// that reads raw waveforms must reproduce them exactly to use a released checkpoint.</para>
/// <para>Everything between the framed samples and the spectrum (DC removal, pre-emphasis, windowing, padding and the
/// DFT) is linear, so it is folded into one matrix per real and imaginary part, built once.</para>
/// </remarks>
internal sealed class KaldiFbank<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // torchaudio floors with float32's machine epsilon whatever the dtype.
    private const double Epsilon = 1.1920928955078125e-07;

    private readonly IEngine _engine;
    private readonly Tensor<T> _real;     // [window, bins]
    private readonly Tensor<T> _imaginary;
    private readonly Tensor<T> _melBanks; // [bins, mels]

    public KaldiFbank(IEngine engine, int sampleRate, int melBins, WindowType window, double frameLengthMilliseconds = 25.0,
        double frameShiftMilliseconds = 10.0, double lowFrequency = 20.0, double highFrequency = 0.0,
        double preemphasis = 0.97, bool removeDcOffset = true, bool roundToPowerOfTwo = true)
    {
        if (melBins <= 3) throw new ArgumentOutOfRangeException(nameof(melBins), "Kaldi needs more than 3 mel bins.");
        if (preemphasis < 0 || preemphasis > 1) throw new ArgumentOutOfRangeException(nameof(preemphasis));
        _engine = engine;
        SampleRate = sampleRate;
        MelBins = melBins;
        FrameShift = (int)(sampleRate * frameShiftMilliseconds * 0.001);
        WindowSize = (int)(sampleRate * frameLengthMilliseconds * 0.001);
        if (WindowSize < 2 || FrameShift <= 0) throw new ArgumentException("The frame length and shift are too short for the sample rate.");
        int padded = roundToPowerOfTwo ? NextPowerOfTwo(WindowSize) : WindowSize;
        if (padded % 2 != 0) throw new ArgumentException("The padded window must be even; round to a power of two or change the frame length.");
        int bins = padded / 2 + 1, n = WindowSize;

        // The linear map from a frame to its DFT: DFT · pad · window · pre-emphasis · DC removal, all n×n except the last.
        var w = WindowFunction(window, n);
        var linear = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            // Row i of window·pre-emphasis·DC: y_i = w_i·(z_i − p·z_{i−1}) with z = x − mean(x), and z_{−1} = z_0.
            int previous = i == 0 ? 0 : i - 1;
            for (int j = 0; j < n; j++)
            {
                double dc = removeDcOffset ? 1.0 / n : 0.0;
                double zi = (i == j ? 1.0 : 0.0) - dc, zp = (previous == j ? 1.0 : 0.0) - dc;
                linear[i, j] = w[i] * (zi - preemphasis * zp);
            }
        }
        _real = new Tensor<T>(new[] { n, bins });
        _imaginary = new Tensor<T>(new[] { n, bins });
        var re = new double[n, bins];
        var im = new double[n, bins];
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < bins; k++)
            {
                double a = 2 * Math.PI * i * k / padded;
                double c = Math.Cos(a), s = -Math.Sin(a);
                for (int j = 0; j < n; j++)
                {
                    re[j, k] += linear[i, j] * c;
                    im[j, k] += linear[i, j] * s;
                }
            }
        }
        for (int j = 0; j < n; j++)
            for (int k = 0; k < bins; k++)
            {
                _real[j, k] = NumOps.FromDouble(re[j, k]);
                _imaginary[j, k] = NumOps.FromDouble(im[j, k]);
            }

        // Kaldi's triangular filters over the first padded/2 bins (the Nyquist bin gets no weight).
        double nyquist = 0.5 * sampleRate;
        double high = highFrequency <= 0 ? highFrequency + nyquist : highFrequency;
        if (!(lowFrequency >= 0 && lowFrequency < nyquist && high > 0 && high <= nyquist && lowFrequency < high))
            throw new ArgumentException($"Bad frequency range {lowFrequency}..{high} for Nyquist {nyquist}.");
        double Mel(double f) => 1127.0 * Math.Log(1.0 + f / 700.0);
        double binWidth = (double)sampleRate / padded, melLow = Mel(lowFrequency), delta = (Mel(high) - melLow) / (melBins + 1);
        _melBanks = new Tensor<T>(new[] { bins, melBins });
        for (int m = 0; m < melBins; m++)
        {
            double left = melLow + m * delta, center = melLow + (m + 1) * delta, right = melLow + (m + 2) * delta;
            for (int k = 0; k < padded / 2; k++)
            {
                double mel = Mel(binWidth * k);
                double weight = Math.Max(0.0, Math.Min((mel - left) / (center - left), (right - mel) / (right - center)));
                _melBanks[k, m] = NumOps.FromDouble(weight);
            }
        }
    }

    public int SampleRate { get; }

    public int MelBins { get; }

    /// <summary>Samples per frame.</summary>
    public int WindowSize { get; }

    /// <summary>Samples between frame starts.</summary>
    public int FrameShift { get; }

    /// <summary>The number of frames <paramref name="samples"/> samples give (none if fewer than one window).</summary>
    public int Frames(int samples) => samples < WindowSize ? 0 : 1 + (samples - WindowSize) / FrameShift;

    /// <summary>The log filterbank energies <c>[frames, MelBins]</c> of a waveform <c>[samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> waveform)
    {
        int length = waveform.Length, frames = Frames(length), n = WindowSize;
        if (frames == 0) throw new ArgumentException($"The waveform has {length} samples, fewer than one {n}-sample frame.", nameof(waveform));
        var index = new Tensor<int>(new[] { frames * n });
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < n; i++)
                index[f * n + i] = f * FrameShift + i;
        var framed = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(waveform, new[] { length, 1 }), index, 0), new[] { frames, n });
        var re = _engine.TensorMatMul(framed, _real);
        var im = _engine.TensorMatMul(framed, _imaginary);
        var power = _engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im));
        var energies = _engine.TensorMatMul(power, _melBanks);
        // max(energy, ε): ε + relu(energy − ε).
        var epsilon = NumOps.FromDouble(Epsilon);
        return _engine.TensorLog(_engine.TensorAddScalar(_engine.ReLU(_engine.TensorAddScalar(energies, NumOps.Negate(epsilon))), epsilon));
    }

    private static int NextPowerOfTwo(int x)
    {
        int p = 1;
        while (p < x) p <<= 1;
        return p;
    }

    /// <summary>Kaldi's symmetric windows (torchaudio's <c>periodic=False</c>).</summary>
    private static double[] WindowFunction(WindowType type, int n)
    {
        var w = new double[n];
        double a = 2 * Math.PI / (n - 1);
        for (int i = 0; i < n; i++)
        {
            w[i] = type switch
            {
                WindowType.Hann => 0.5 - 0.5 * Math.Cos(a * i),
                WindowType.Hamming => 0.54 - 0.46 * Math.Cos(a * i),
                WindowType.Blackman => 0.42 - 0.5 * Math.Cos(a * i) + 0.08 * Math.Cos(2 * a * i),
                WindowType.Rectangular => 1.0,
                WindowType.Povey => Math.Pow(0.5 - 0.5 * Math.Cos(a * i), 0.85),
                _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unsupported window."),
            };
        }
        return w;
    }
}
