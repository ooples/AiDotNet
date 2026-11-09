using AiDotNet.Audio.Resampling;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Audio.Watermarking;

/// <summary>
/// Resemble AI's Perth implicit watermarker (resemble-perth 1.1.0, <c>PerthImplicitWatermarker</c> with its
/// <c>perth_net_250000</c> PerthNet), the watermark Chatterbox puts on everything it says.
/// </summary>
/// <remarks>
/// <para><b>Embedding</b> (<c>apply_watermark</c>): the signal resampled to 32 kHz (librosa's default soxr_hq), a 2048-point
/// STFT (periodic Hann, hop 320, centred, reflect-padded), each bin's magnitude in dB normalized over
/// [20·log10(1e-9), +15 dB]; the encoder — a 1×1 convolution, five 7-wide ones (LeakyReLU after each) and a 1×1 back —
/// reads the lowest 128 bins (up to 2 kHz) and adds its output to them on frames whose summed magnitude exceeds 5 % of the
/// loudest frame's; the spectrum is rebuilt with the original phase, inverted, and resampled back.</para>
/// <para><b>Detection</b> (<c>get_watermark</c>): the signal resampled to 32 kHz polyphase (scipy's resample_poly), the
/// same magnitude, and the decoder's three copies of that stack over the 128 bins at 1.25×, 1× and 0.75× the frame rate
/// (linear stretches), each giving an attention logit and a watermark score as masked means over loud frames; the
/// softmax-weighted score, clipped to [0, 1]. Watermarked audio scores near 1.</para>
/// <para>The released weights are MIT-licensed and ship with the resemble-perth package
/// (<c>perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar</c>); <see cref="LoadCheckpoint"/> reads them. The
/// network is frozen: its layers belong to it, not to the model that applies it.</para>
/// </remarks>
internal sealed class PerthWatermarker<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int SampleRate = 32000, FftSize = 2048, Hop = 320;
    private const double MagnitudeMin = 1e-9, HeadroomDb = 15;
    private static readonly double MinDb = 20 * Math.Log10(MagnitudeMin);
    private static readonly double[] TimeScales = { 1.25, 1.0, 0.75 };
    private static readonly string[] DecoderNames = { "decoder.slow_layers", "decoder.normal_layers", "decoder.fast_layers" };

    private readonly IEngine _engine;
    private readonly int _hidden;
    private readonly KaimingConv1DLayer<T>[] _encoder;
    private readonly KaimingConv1DLayer<T>[][] _decoder;
    private readonly CenteredComplexStft<T> _stft;
    private readonly InverseStft<T> _istft;

    /// <summary>Creates the watermarker with PerthNet's sizes (hidden 256 for the released model), randomly initialized
    /// until <see cref="LoadCheckpoint"/> or <see cref="LoadTorchWeights(TorchTensorSource)"/>.</summary>
    public PerthWatermarker(IEngine engine, int hidden = 256)
    {
        if (hidden < 1) throw new ArgumentOutOfRangeException(nameof(hidden));
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _hidden = hidden;
        Subband = (int)Math.Round((FftSize / 2 + 1) * 2000.0 / (SampleRate / 2.0), MidpointRounding.ToEven);
        KaimingConv1DLayer<T>[] Stack(int last)
        {
            var stack = new[] { new KaimingConv1DLayer<T>(Subband, hidden, 1, 1, 0, false) }
                .Concat(Enumerable.Range(0, 5).Select(_ => new KaimingConv1DLayer<T>(hidden, hidden, 7, 1, 3, false)))
                .Concat(new[] { new KaimingConv1DLayer<T>(hidden, last, 1, 1, 0, false) }).ToArray();
            foreach (var conv in stack) conv.SetTrainingMode(false);
            return stack;
        }
        _encoder = Stack(Subband);
        _decoder = DecoderNames.Select(_ => Stack(2)).ToArray();
        _stft = new CenteredComplexStft<T>(engine, FftSize, Hop, FftSize);
        _istft = new InverseStft<T>(engine, FftSize, Hop, FftSize);
    }

    /// <summary>The bins the watermark lives in (128: up to 2 kHz).</summary>
    public int Subband { get; }

    // The convolutions with LeakyReLU (slope 0.01) after all but the last: [1, C, frames] → [1, out, frames].
    private Tensor<T> Run(KaimingConv1DLayer<T>[] stack, Tensor<T> x)
    {
        for (int i = 0; i < stack.Length; i++)
        {
            x = stack[i].Forward(x);
            if (i < stack.Length - 1) x = _engine.LeakyReLU(x, NumOps.FromDouble(0.01));
        }
        return x;
    }

    /// <summary>torchaudio's <c>Spectrogram(power=None)</c> then <c>cx_to_magphase</c>: the normalized dB magnitude and the
    /// phase of a 32 kHz signal <c>[samples]</c>, each <c>[1, 1025, frames]</c>.</summary>
    internal (Tensor<T> Magnitude, Tensor<T> Phase) MagnitudeAndPhase(Tensor<T> signal)
    {
        if (signal.Length <= FftSize / 2) throw new ArgumentException($"The signal must be longer than {FftSize / 2} samples.", nameof(signal));
        var (re, im) = _stft.Forward(signal);
        var absolute = _engine.TensorSqrt(_engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im)));
        // 20·log10(max(|X|, 1e-9)), mapped from [MinDb, +15 dB] to [0, 1].
        var db = _engine.TensorMultiplyScalar(_engine.TensorLog(_engine.TensorClampMin(absolute, NumOps.FromDouble(MagnitudeMin))),
            NumOps.FromDouble(20 / Math.Log(10)));
        var magnitude = _engine.TensorMultiplyScalar(_engine.TensorAddScalar(db, NumOps.FromDouble(-MinDb)),
            NumOps.FromDouble(1 / (-MinDb + HeadroomDb)));
        return (magnitude, _engine.TensorAtan2(im, re));
    }

    // magmask: 1 on frames whose summed normalized magnitude exceeds 5 % of the largest sum, else 0: [frames].
    private Tensor<T> Mask(Tensor<T> magnitude)
    {
        int frames = magnitude.Shape[2];
        var sums = _engine.Reshape(_engine.ReduceSum(magnitude, new[] { 1 }, keepDims: false), new[] { frames });
        var threshold = _engine.TensorMultiplyScalar(_engine.ReduceMax(sums, new[] { 0 }, keepDims: true), NumOps.FromDouble(0.05));
        return _engine.TensorGreaterThan(sums, _engine.TensorBroadcastTo(threshold, new[] { frames }));
    }

    private Tensor<T> Band(Tensor<T> magnitude, int start, int count) =>
        _engine.TensorSlice(magnitude, new[] { 0, start, 0 }, new[] { 1, count, magnitude.Shape[2] });

    /// <summary>The watermarked signal (<c>apply_watermark</c>) at <paramref name="sampleRate"/>.</summary>
    public double[] Apply(double[] signal, int sampleRate)
    {
        if (signal is null) throw new ArgumentNullException(nameof(signal));
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var resampled = sampleRate == SampleRate ? signal : SoxrResampler.LibrosaResample(signal, sampleRate, SampleRate);
        var (magnitude, phase) = MagnitudeAndPhase(ToTensor(resampled));
        int bins = magnitude.Shape[1], frames = magnitude.Shape[2];
        var sub = Band(magnitude, 0, Subband);
        var residual = Run(_encoder, sub);
        var mask = _engine.TensorBroadcastTo(_engine.Reshape(Mask(magnitude), new[] { 1, 1, frames }), new[] { 1, Subband, frames });
        var marked = _engine.TensorConcatenate(new[] { _engine.TensorAdd(sub, _engine.TensorMultiply(residual, mask)),
            Band(magnitude, Subband, bins - Subband) }, 1);
        // magphase_to_cx: back to dB, then 10^min(dB/20, 10).
        var db = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(marked, NumOps.FromDouble(-MinDb + HeadroomDb)), NumOps.FromDouble(MinDb));
        var exponent = _engine.TensorClamp(_engine.TensorMultiplyScalar(db, NumOps.FromDouble(1.0 / 20)), NumOps.FromDouble(-1e30), NumOps.FromDouble(10));
        var amplitude = _engine.TensorExp(_engine.TensorMultiplyScalar(exponent, NumOps.FromDouble(Math.Log(10))));
        var watermarked = _istft.Forward(amplitude, phase).ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
        return sampleRate == SampleRate ? watermarked : SoxrResampler.LibrosaResample(watermarked, SampleRate, sampleRate);
    }

    /// <summary>The decoder's unclipped score of a 32 kHz magnitude <c>[1, 1025, frames]</c>.</summary>
    internal double Decode(Tensor<T> magnitude)
    {
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int frames = magnitude.Shape[2];
        var mask = Mask(magnitude);
        var sub = _engine.Reshape(Band(magnitude, 0, Subband), new[] { Subband, frames });
        var attention = new Tensor<T>[TimeScales.Length];
        var scores = new Tensor<T>[TimeScales.Length];
        for (int s = 0; s < TimeScales.Length; s++)
        {
            int length = (int)(frames * TimeScales[s]);
            var input = length == frames ? sub : _engine.TensorMatMul(sub, Stretch(frames, length));
            var output = _engine.Reshape(Run(_decoder[s], _engine.Reshape(input, new[] { 1, Subband, length })), new[] { 2, length });
            // F.interpolate(mask, size, mode="nearest"): source frame ⌊f · frames / length⌋.
            var nearest = new Tensor<int>(new[] { length });
            for (int f = 0; f < length; f++) nearest[f] = Math.Min(frames - 1, (int)Math.Floor(f * (double)frames / length));
            var scaled = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(mask, new[] { frames, 1 }), nearest, 0), new[] { length, 1 });
            var means = _engine.TensorDivide(_engine.TensorMatMul(output, scaled),
                _engine.TensorBroadcastTo(_engine.ReduceSum(scaled, new[] { 0 }, keepDims: true), new[] { 2, 1 }));       // [2, 1]
            attention[s] = _engine.TensorSlice(means, new[] { 0, 0 }, new[] { 1, 1 });
            scores[s] = _engine.TensorSlice(means, new[] { 1, 0 }, new[] { 1, 1 });
        }
        var weights = _engine.Softmax(_engine.Reshape(_engine.TensorConcatenate(attention, 0), new[] { TimeScales.Length }), 0);
        var score = _engine.ReduceSum(_engine.TensorMultiply(weights,
            _engine.Reshape(_engine.TensorConcatenate(scores, 0), new[] { TimeScales.Length })), new[] { 0 }, keepDims: false);
        return NumOps.ToDouble(score.GetFlat(0));
    }

    // F.interpolate(x, size, mode="linear", align_corners=True) over frames, as a [frames, size] matrix.
    private static Tensor<T> Stretch(int frames, int size)
    {
        var weights = new Tensor<T>(new[] { frames, size });
        double scale = size > 1 ? (frames - 1) / (double)(size - 1) : 0;
        for (int i = 0; i < size; i++)
        {
            double source = i * scale;
            int low = (int)Math.Floor(source), high = Math.Min(low + 1, frames - 1);
            double t = source - low;
            weights[low, i] = NumOps.Add(weights[low, i], NumOps.FromDouble(1 - t));
            weights[high, i] = NumOps.Add(weights[high, i], NumOps.FromDouble(t));
        }
        return weights;
    }

    /// <summary>The watermark score of a signal (<c>get_watermark</c>, clipped to [0, 1]; near 1 when watermarked).</summary>
    public double Detect(double[] signal, int sampleRate)
    {
        if (signal is null) throw new ArgumentNullException(nameof(signal));
        var resampled = sampleRate == SampleRate ? signal : ResamplePoly(signal, SampleRate, sampleRate);
        if (sampleRate != SampleRate)
        {
            int size = (int)Math.Ceiling(signal.Length * (double)SampleRate / sampleRate);
            Array.Resize(ref resampled, size);
        }
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Math.Max(0, Math.Min(1, Decode(MagnitudeAndPhase(ToTensor(resampled)).Magnitude)));
    }

    private static Tensor<T> ToTensor(double[] values) => new(new[] { values.Length }, new Vector<T>(values.Select(NumOps.FromDouble).ToArray()));

    // ---------------------------------------------------------------- resampling

    /// <summary>scipy's <c>resample_poly(x, up, down)</c> with its default Kaiser (β 5) filter, as librosa's
    /// <c>res_type="polyphase"</c> calls it, fixed to ⌈n · up / down⌉ samples.</summary>
    internal static double[] ResamplePoly(double[] x, int up, int down)
    {
        int g = Gcd(up, down);
        up /= g;
        down /= g;
        if (up == down) return (double[])x.Clone();
        int nIn = x.Length, nOut = nIn * up / down + (nIn * up % down != 0 ? 1 : 0);
        int maxRate = Math.Max(up, down), halfLength = 10 * maxRate;
        var h = Firwin(2 * halfLength + 1, 1.0 / maxRate, 5.0).Select(v => v * up).ToArray();
        int prePad = down - halfLength % down, postPad = 0, preRemove = (halfLength + prePad) / down;
        int OutputLength(int lengthH) => ((nIn - 1) * up + lengthH - 1) / down + 1;
        while (OutputLength(h.Length + prePad + postPad) < nOut + preRemove) postPad++;
        var filter = new double[prePad + h.Length + postPad];
        Array.Copy(h, 0, filter, prePad, h.Length);
        // upfirdn: zero-stuff by up, convolve fully, keep every down-th sample.
        int full = (nIn - 1) * up + filter.Length;
        var y = new double[nOut];
        for (int o = 0; o < nOut; o++)
        {
            int index = (o + preRemove) * down;
            if (index >= full) break;
            double s = 0;
            for (int k = 0; k < filter.Length; k++)
            {
                int u = index - k;
                if (u < 0) break;
                if (u % up == 0 && u / up < nIn) s += filter[k] * x[u / up];
            }
            y[o] = s;
        }
        return y;
    }

    // scipy.signal.firwin(numtaps, cutoff, window=("kaiser", beta)): a scaled low-pass of unit DC gain.
    private static double[] Firwin(int taps, double cutoff, double beta)
    {
        double alpha = (taps - 1) / 2.0;
        var h = new double[taps];
        for (int i = 0; i < taps; i++)
        {
            double m = i - alpha, x = cutoff * m;
            double sinc = x == 0 ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);
            double r = 2.0 * i / (taps - 1) - 1;
            h[i] = cutoff * sinc * BesselI0(beta * Math.Sqrt(Math.Max(0, 1 - r * r))) / BesselI0(beta);
        }
        double total = h.Sum();
        return h.Select(v => v / total).ToArray();
    }

    private static double BesselI0(double x)
    {
        double sum = 1, term = 1, quarter = x * x / 4;
        for (int k = 1; k < 500; k++)
        {
            term *= quarter / ((double)k * k);
            sum += term;
            if (term < sum * 1e-17) break;
        }
        return sum;
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }

    // ---------------------------------------------------------------- weights

    /// <summary>Loads PerthNet's state (<c>encoder.layers.{i}.conv</c>, <c>decoder.{slow,normal,fast}_layers.{i}.conv</c>).</summary>
    public void LoadTorchWeights(TorchTensorSource source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        void Load(KaimingConv1DLayer<T>[] stack, string prefix)
        {
            for (int i = 0; i < stack.Length; i++)
            {
                string name = $"{prefix}.{i}.conv";
                if (!source.Contains(name + ".weight"))
                    throw new InvalidDataException($"The Perth checkpoint has no tensor '{name}.weight'.");
                stack[i].LoadTorch(source, name + ".weight", name + ".bias");
            }
        }
        Load(_encoder, "encoder.layers");
        for (int s = 0; s < DecoderNames.Length; s++) Load(_decoder[s], DecoderNames[s]);
    }

    /// <summary>Loads a released checkpoint (<c>perth_net_250000.pth.tar</c>, whose state dict is under <c>model</c>).</summary>
    public void LoadCheckpoint(string path)
    {
        using var file = TorchCheckpointFile.Open(path);
        LoadTorchWeights(file.Contains("model.encoder.layers.0.conv.weight") ? file.WithPrefix("model.") : file);
    }

    /// <summary>The hidden width (256 for the released model).</summary>
    public int Hidden => _hidden;
}
