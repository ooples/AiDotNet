using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// The H/ASP speaker encoder (Heo et al. 2020, "Clova Baseline System for the VoxCeleb Speaker Recognition Challenge
/// 2020") as YourTTS uses it (Casanova et al. 2022 §3.1): a squeeze-and-excitation ResNet over a 64-band log-mel
/// spectrogram with attentive statistics pooling and a 512-wide projection.
/// </summary>
/// <remarks>
/// <para>
/// Follows Coqui's <c>ResNetSpeakerEncoder</c> with the released model's <c>config_se.json</c>: 16 kHz audio,
/// pre-emphasis 0.97, a 512-point STFT with a 400-sample periodic Hamming window and hop 160 (torchaudio
/// <c>MelSpectrogram</c>: centered with reflect padding, power 2, 64 HTK mel bands from 0 to 8 kHz, no filter
/// normalization), <c>log(x + 1e-6)</c>, instance normalization over time; a 3×3 convolution with ReLU and batch norm;
/// SE basic blocks (3, 4, 6, 3) of 32, 64, 128, 256 filters, the last three stages striding 2 in time and frequency;
/// attention weights from a 1×1 convolution to 128 channels, ReLU, batch norm and a 1×1 convolution back, softmax over
/// time; the weighted mean and standard deviation concatenated and projected to 512.
/// </para>
/// <para>
/// The encoder is pretrained and frozen: YourTTS never trains it, and its batch norms use their running statistics.
/// <see cref="LoadState"/> takes a Coqui checkpoint's state dictionary. <see cref="Embed"/> is the encoder's forward with
/// L2 normalization (the speaker consistency loss); <see cref="DVector"/> averages it over ten evenly spaced
/// 250-frame windows (Coqui <c>compute_embedding</c>, how YourTTS's d-vectors are made).
/// </para>
/// </remarks>
internal sealed class HaspSpeakerEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The sample rate the encoder reads (16 kHz).</summary>
    public const int SampleRate = 16000;

    private const int FftSize = 512;
    private const int WindowSize = 400;
    private const int HopSize = 160;
    private const int MelBands = 64;
    private const double PreEmphasis = 0.97;
    private static readonly int[] StageBlocks = { 3, 4, 6, 3 };

    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly Dictionary<string, Tensor<T>> _state = new(StringComparer.Ordinal);
    private readonly Tensor<T> _cos;
    private readonly Tensor<T> _sin;
    private readonly Tensor<T> _mel;
    private readonly Conv2DLayer<T> _stem;
    private readonly FrozenBatchNormLayer<T> _stemNorm;
    private readonly List<Block> _blocks = new();
    private readonly Conv2DLayer<T> _attention1;
    private readonly FrozenBatchNormLayer<T> _attentionNorm;
    private readonly Conv2DLayer<T> _attention2;
    private readonly Conv2DLayer<T> _projection;
    private readonly int _projectionDim;
    private readonly int[] _filters;

    private sealed record Block(Conv2DLayer<T> Conv1, FrozenBatchNormLayer<T> Norm1, Conv2DLayer<T> Conv2, FrozenBatchNormLayer<T> Norm2,
        Conv2DLayer<T> Squeeze, Conv2DLayer<T> Excite, Conv2DLayer<T>? Downsample, FrozenBatchNormLayer<T>? DownsampleNorm, int Stride);

    /// <param name="engine">The engine.</param>
    /// <param name="projectionDim">The embedding width (512).</param>
    /// <param name="filters">The four stages' filters (32, 64, 128, 256 in the released model).</param>
    public HaspSpeakerEncoder(IEngine engine, int projectionDim, int[] filters)
    {
        if (filters is null || filters.Length != 4 || filters.Any(f => f < 8 || f % 8 != 0))
            throw new ArgumentException("The H/ASP encoder has four stages whose filters are positive multiples of 8.", nameof(filters));
        _engine = engine;
        _projectionDim = projectionDim;
        _filters = (int[])filters.Clone();

        // STFT basis: a periodic Hamming window of 400 samples centred in the 512-point frame (torch.stft pads it).
        int bins = FftSize / 2 + 1, offset = (FftSize - WindowSize) / 2;
        var window = new double[FftSize];
        for (int n = 0; n < WindowSize; n++) window[offset + n] = 0.54 - 0.46 * Math.Cos(2 * Math.PI * n / WindowSize);
        _cos = new Tensor<T>(new[] { FftSize, bins });
        _sin = new Tensor<T>(new[] { FftSize, bins });
        for (int n = 0; n < FftSize; n++)
            for (int k = 0; k < bins; k++)
            {
                double a = 2 * Math.PI * n * k / FftSize;
                _cos[n, k] = NumOps.FromDouble(window[n] * Math.Cos(a));
                _sin[n, k] = NumOps.FromDouble(-window[n] * Math.Sin(a));
            }
        _mel = HtkMelFilterbank(bins, MelBands, SampleRate, 0.0, SampleRate / 2.0);

        _stem = Own("conv1", new Conv2DLayer<T>(1, _filters[0], 3, 3, 1, 1, 1, 1, true));
        _stemNorm = OwnNorm("bn1", new FrozenBatchNormLayer<T>(_filters[0]));
        int inplanes = _filters[0];
        for (int stage = 0; stage < 4; stage++)
        {
            int planes = _filters[stage], stride = stage == 0 ? 1 : 2;
            for (int b = 0; b < StageBlocks[stage]; b++)
            {
                string p = $"layer{stage + 1}.{b}.";
                int s = b == 0 ? stride : 1;
                var conv1 = Own(p + "conv1", new Conv2DLayer<T>(inplanes, planes, 3, 3, s, s, 1, 1, false));
                var norm1 = OwnNorm(p + "bn1", new FrozenBatchNormLayer<T>(planes));
                var conv2 = Own(p + "conv2", new Conv2DLayer<T>(planes, planes, 3, 3, 1, 1, 1, 1, false));
                var norm2 = OwnNorm(p + "bn2", new FrozenBatchNormLayer<T>(planes));
                var squeeze = Own(p + "se.fc.0", new Conv2DLayer<T>(planes, planes / 8, 1, 1, 1, 1, 0, 0, true));
                var excite = Own(p + "se.fc.2", new Conv2DLayer<T>(planes / 8, planes, 1, 1, 1, 1, 0, 0, true));
                Conv2DLayer<T>? down = null;
                FrozenBatchNormLayer<T>? downNorm = null;
                if (b == 0 && (s != 1 || inplanes != planes))
                {
                    down = Own(p + "downsample.0", new Conv2DLayer<T>(inplanes, planes, 1, 1, s, s, 0, 0, false));
                    downNorm = OwnNorm(p + "downsample.1", new FrozenBatchNormLayer<T>(planes));
                }
                _blocks.Add(new Block(conv1, norm1, conv2, norm2, squeeze, excite, down, downNorm, s));
                inplanes = planes;
            }
        }
        int pooled = _filters[3] * (MelBands / 8);
        _attention1 = Own("attention.0", new Conv2DLayer<T>(pooled, 128, 1, 1, 1, 1, 0, 0, true));
        _attentionNorm = OwnNorm("attention.2", new FrozenBatchNormLayer<T>(128));
        _attention2 = Own("attention.3", new Conv2DLayer<T>(128, pooled, 1, 1, 1, 1, 0, 0, true));
        _projection = Own("fc", new Conv2DLayer<T>(2 * pooled, projectionDim, 1, 1, 1, 1, 0, 0, true));
    }

    /// <summary>Every layer the encoder owns.</summary>
    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>The embedding width (512).</summary>
    public int ProjectionDim => _projectionDim;

    private Conv2DLayer<T> Own(string name, Conv2DLayer<T> layer)
    {
        _layers.Add(layer);
        _state[name + ".weight"] = layer.Kernel;
        if (layer.UsesBias) _state[name + ".bias"] = layer.Bias;
        return layer;
    }

    private FrozenBatchNormLayer<T> OwnNorm(string name, FrozenBatchNormLayer<T> layer)
    {
        _layers.Add(layer);
        _state[name + ".weight"] = layer.Scale;
        _state[name + ".bias"] = layer.Shift;
        _state[name + ".running_mean"] = layer.RunningMean;
        _state[name + ".running_var"] = layer.RunningVariance;
        return layer;
    }

    /// <summary>
    /// Copies a Coqui <c>ResNetSpeakerEncoder</c> state dictionary into the encoder. Linear and 1-D convolution weights
    /// (<c>[out, in]</c>, <c>[out, in, 1]</c>) load into the 1×1 convolutions in the same element order. The frontend's
    /// buffers (<c>torch_spec.*</c>) and <c>num_batches_tracked</c> counters are recomputed or unused here and skipped;
    /// every other key must be known and every tensor of the encoder must be present.
    /// </summary>
    public void LoadState(IReadOnlyDictionary<string, Tensor<T>> state)
    {
        Guard.NotNull(state);
        foreach (var key in state.Keys)
        {
            if (key.StartsWith("torch_spec.", StringComparison.Ordinal) || key.EndsWith(".num_batches_tracked", StringComparison.Ordinal))
                continue;
            if (!_state.ContainsKey(key))
                throw new ArgumentException($"Unknown H/ASP parameter '{key}'.", nameof(state));
        }
        foreach (var (name, target) in _state)
        {
            if (!state.TryGetValue(name, out var source))
                throw new ArgumentException($"The H/ASP state has no '{name}'.", nameof(state));
            if (source.Length != target.Length || source.Shape[0] != target.Shape[0])
                throw new ArgumentException(
                    $"H/ASP parameter '{name}' has shape [{string.Join(", ", source.Shape)}]; expected [{string.Join(", ", target.Shape)}].", nameof(state));
            for (int i = 0; i < target.Length; i++) target[i] = source[i];
            _engine.InvalidatePersistentTensor(target);
        }
    }

    /// <summary>The L2-normalized embedding <c>[projectionDim]</c> of a 16 kHz waveform <c>[samples]</c> (Coqui
    /// <c>forward(x, l2_norm=True)</c>); differentiable with respect to the waveform.</summary>
    public Tensor<T> Embed(Tensor<T> audio)
    {
        var x = Embedding(audio);                                                        // [proj]
        var norm = _engine.TensorPow(_engine.TensorAddScalar(_engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { 0 }, keepDims: true),
            NumOps.FromDouble(1e-24)), NumOps.FromDouble(-0.5));
        return _engine.TensorMultiply(x, _engine.TensorTile(norm, new[] { x.Length }));
    }

    /// <summary>
    /// The d-vector of a 16 kHz recording (Coqui <c>compute_embedding(x, num_frames=250, num_eval=10)</c>): the mean
    /// of the normalized embeddings of ten 250-frame windows at evenly spaced offsets (the whole recording when it is
    /// shorter).
    /// </summary>
    public Tensor<T> DVector(Tensor<T> audio)
    {
        int length = audio.Length, window = Math.Min(250 * HopSize, length);
        const int evaluations = 10;
        Tensor<T>? sum = null;
        for (int e = 0; e < evaluations; e++)
        {
            int start = (int)((length - window) * (double)e / (evaluations - 1));
            var embedding = Embed(_engine.TensorSlice(_engine.Reshape(audio, new[] { length }), new[] { start }, new[] { window }));
            sum = sum is null ? embedding : _engine.TensorAdd(sum, embedding);
        }
        return _engine.TensorMultiplyScalar(sum!, NumOps.FromDouble(1.0 / evaluations));
    }

    // The unnormalized projection [proj].
    private Tensor<T> Embedding(Tensor<T> audio)
    {
        var features = LogMel(audio);                                                    // [mel, frames]
        int frames = features.Shape[1];
        var x = _engine.Reshape(InstanceNorm(features), new[] { 1, 1, MelBands, frames });
        x = _stemNorm.Forward(_engine.ReLU(_stem.Forward(x)));
        foreach (var block in _blocks) x = Residual(block, x);
        int channels = x.Shape[1] * x.Shape[2], time = x.Shape[3];
        var h = _engine.Reshape(x, new[] { 1, channels, 1, time });
        var w = _attention2.Forward(_attentionNorm.Forward(_engine.ReLU(_attention1.Forward(h))));
        var weights = _engine.TensorSoftmax(_engine.Reshape(w, new[] { channels, time }), axis: 1);   // softmax over time
        var flat = _engine.Reshape(h, new[] { channels, time });
        var mu = _engine.ReduceSum(_engine.TensorMultiply(flat, weights), new[] { 1 }, keepDims: false);
        var second = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorMultiply(flat, flat), weights), new[] { 1 }, keepDims: false);
        // sqrt(clamp(E[x²] − μ², 1e-5)) = sqrt(1e-5 + relu(E[x²] − μ² − 1e-5))
        var variance = _engine.TensorSubtract(second, _engine.TensorMultiply(mu, mu));
        var sigma = _engine.TensorPow(_engine.TensorAddScalar(_engine.ReLU(_engine.TensorAddScalar(variance, NumOps.FromDouble(-1e-5))),
            NumOps.FromDouble(1e-5)), NumOps.FromDouble(0.5));
        var pooled = _engine.Reshape(_engine.TensorConcatenate(new[] { mu, sigma }, 0), new[] { 1, 2 * channels, 1, 1 });
        return _engine.Reshape(_projection.Forward(pooled), new[] { _projectionDim });
    }

    // SEBasicBlock: conv → ReLU → BN → conv → BN → SE, plus the (downsampled) input, then ReLU.
    private Tensor<T> Residual(Block b, Tensor<T> x)
    {
        var y = b.Norm1.Forward(_engine.ReLU(b.Conv1.Forward(x)));
        y = b.Norm2.Forward(b.Conv2.Forward(y));
        int c = y.Shape[1], height = y.Shape[2], width = y.Shape[3];
        var pooled = _engine.ReduceMean(y, new[] { 2, 3 }, keepDims: true);                            // [1, c, 1, 1]
        var gate = _engine.Sigmoid(b.Excite.Forward(_engine.ReLU(b.Squeeze.Forward(pooled))));
        y = _engine.TensorMultiply(y, _engine.TensorTile(gate, new[] { 1, 1, height, width }));
        var residual = b.Downsample is null ? x : b.DownsampleNorm!.Forward(b.Downsample.Forward(x));
        return _engine.ReLU(_engine.TensorAdd(y, residual));
    }

    // Pre-emphasis (reflect-padded by one sample), centred reflect-padded STFT, power spectrum, HTK mel, log(x + 1e-6):
    // [mel, frames].
    private Tensor<T> LogMel(Tensor<T> audio)
    {
        int length = audio.Length;
        if (length < 2) throw new ArgumentException("The speaker encoder needs at least two samples.", nameof(audio));
        var column = _engine.Reshape(audio, new[] { length, 1 });
        var previous = new Tensor<int>(new[] { length });
        previous[0] = 1;                                                       // F.pad(x, (1, 0), "reflect") puts x[1] first
        for (int i = 1; i < length; i++) previous[i] = i - 1;
        var emphasized = _engine.TensorSubtract(column,
            _engine.TensorMultiplyScalar(_engine.TensorIndexSelect(column, previous, 0), NumOps.FromDouble(PreEmphasis)));

        int pad = FftSize / 2, frames = 1 + length / HopSize;
        var index = new Tensor<int>(new[] { frames * FftSize });
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < FftSize; n++)
                index[f * FftSize + n] = DifferentiableMel<T>.Reflect(f * HopSize + n - pad, length);
        var framed = _engine.Reshape(_engine.TensorIndexSelect(emphasized, index, 0), new[] { frames, FftSize });
        var re = _engine.TensorMatMul(framed, _cos);
        var im = _engine.TensorMatMul(framed, _sin);
        var power = _engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im));
        var mel = _engine.TensorLog(_engine.TensorAddScalar(_engine.TensorMatMul(power, _mel), NumOps.FromDouble(1e-6)));
        return _engine.TensorTranspose(mel);
    }

    // InstanceNorm1d(64): each band normalized over time, biased variance, ε = 1e-5, no affine.
    private Tensor<T> InstanceNorm(Tensor<T> x)
    {
        int frames = x.Shape[1];
        var mean = _engine.ReduceMean(x, new[] { 1 }, keepDims: true);
        var centred = _engine.TensorSubtract(x, _engine.TensorTile(mean, new[] { 1, frames }));
        var variance = _engine.ReduceMean(_engine.TensorMultiply(centred, centred), new[] { 1 }, keepDims: true);
        var inverse = _engine.TensorPow(_engine.TensorAddScalar(variance, NumOps.FromDouble(1e-5)), NumOps.FromDouble(-0.5));
        return _engine.TensorMultiply(centred, _engine.TensorTile(inverse, new[] { 1, frames }));
    }

    // torchaudio.functional.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm=None, mel_scale="htk"):
    // [bins, mels].
    internal static Tensor<T> HtkMelFilterbank(int bins, int mels, int sampleRate, double fMin, double fMax)
    {
        static double ToMel(double f) => 2595.0 * Math.Log10(1.0 + f / 700.0);
        static double ToHz(double m) => 700.0 * (Math.Pow(10.0, m / 2595.0) - 1.0);
        double melMin = ToMel(fMin), melMax = ToMel(fMax);
        var points = new double[mels + 2];
        for (int i = 0; i < mels + 2; i++) points[i] = ToHz(melMin + (melMax - melMin) * i / (mels + 1));
        var bank = new Tensor<T>(new[] { bins, mels });
        for (int k = 0; k < bins; k++)
        {
            double f = (sampleRate / 2.0) * k / (bins - 1);
            for (int m = 0; m < mels; m++)
            {
                double down = (f - points[m]) / (points[m + 1] - points[m]);
                double up = (points[m + 2] - f) / (points[m + 2] - points[m + 1]);
                bank[k, m] = NumOps.FromDouble(Math.Max(0.0, Math.Min(down, up)));
            }
        }
        return bank;
    }

    /// <summary>
    /// torchaudio's sinc resampling (<c>transforms.Resample</c>, Hann-windowed sinc, 6 zero crossings, roll-off 0.99) of
    /// <paramref name="audio"/> <c>[samples]</c> from <paramref name="from"/> Hz to <paramref name="to"/> Hz, as a strided
    /// convolution (differentiable).
    /// </summary>
    public static Tensor<T> Resample(IEngine engine, Tensor<T> audio, int from, int to)
        => AiDotNet.Helpers.AudioHelper<T>.ResampleBandLimited(engine, audio, from, to);

}
