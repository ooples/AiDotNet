using AiDotNet.Audio.Features;
using AiDotNet.Helpers;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// ERes2NetV2 (Chen et al. 2024, "ERes2NetV2: Boosting Short-Duration Speaker Verification Performance with Computational
/// Efficiency"; 3D-Speaker <c>ERes2NetV2.py</c>, Apache-2.0) in evaluation mode, as GPT-SoVITS runs it for v2Pro's speaker
/// embedding (<c>sv.py</c>, <c>forward3</c>): Kaldi fbank features <c>[T, 80]</c> through a 3×3 stem and four stages of
/// Res2Net-style bottleneck blocks (scale 4, base width 24, expansion 4; the last two stages fuse their splits with
/// attentional feature fusion), the third stage downsampled and fused into the fourth (AFF), then the fused map's
/// channels × frequency rows averaged over time: a 20,480-dimensional embedding.
/// </summary>
/// <remarks>
/// <para>The released checkpoint (<c>pretrained_eres2netv2w24s4ep4.ckpt</c>) is frozen in GPT-SoVITS; this encoder only
/// runs it. BatchNorm uses the running statistics, and the activation after each block is Hardtanh(0, 20) (the
/// reference's "ReLU"), the stem's a plain ReLU.</para>
/// </remarks>
internal sealed class ERes2NetV2Encoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _channels, _baseWidth, _scale, _expansion;
    private readonly int[] _blocks;
    private readonly Dictionary<string, Tensor<T>> _weights = new(StringComparer.Ordinal);
    private readonly KaldiFbank<T> _fbank;
    private readonly Random? _initialization;

    /// <param name="engine">The engine.</param>
    /// <param name="read">The checkpoint's tensors by name (null where absent).</param>
    /// <param name="channels">m_channels (64).</param>
    /// <param name="baseWidth">baseWidth (24).</param>
    /// <param name="scale">scale (4).</param>
    /// <param name="expansion">expansion (4).</param>
    /// <param name="blocks">The blocks per stage (3, 4, 6, 3).</param>
    /// <param name="initialization">When given, tensors <paramref name="read"/> lacks are initialized as PyTorch does
    /// (convolutions from U(±1/√fan_in), BatchNorm at γ 1, β 0, mean 0, variance 1) instead of failing.</param>
    public ERes2NetV2Encoder(IEngine engine, Func<string, double[]?> read, int channels = 64, int baseWidth = 24, int scale = 4,
        int expansion = 4, int[]? blocks = null, Random? initialization = null)
    {
        _engine = engine;
        _initialization = initialization;
        _channels = channels;
        _baseWidth = baseWidth;
        _scale = scale;
        _expansion = expansion;
        _blocks = blocks ?? new[] { 3, 4, 6, 3 };
        _fbank = new KaldiFbank<T>(engine, 16000, 80, WindowType.Povey);
        int inPlanes = channels;
        Conv("conv1", channels, 1, 3, read);
        Norm("bn1", channels, read);
        for (int stage = 0; stage < 4; stage++)
        {
            int planes = channels << stage;
            for (int b = 0; b < _blocks[stage]; b++)
            {
                int stride = b == 0 && stage > 0 ? 2 : 1;
                string n = $"layer{stage + 1}.{b}";
                int width = (int)Math.Floor(planes * (baseWidth / 64.0));
                Conv($"{n}.conv1", width * scale, inPlanes, 1, read);
                Norm($"{n}.bn1", width * scale, read);
                for (int i = 0; i < scale; i++)
                {
                    Conv($"{n}.convs.{i}", width, width, 3, read);
                    Norm($"{n}.bns.{i}", width, read);
                }
                if (stage >= 2)
                    for (int i = 0; i < scale - 1; i++) Fusion($"{n}.fuse_models.{i}", width, read);
                Conv($"{n}.conv3", planes * expansion, width * scale, 1, read);
                Norm($"{n}.bn3", planes * expansion, read);
                if (stride != 1 || inPlanes != planes * expansion)
                {
                    Conv($"{n}.shortcut.0", planes * expansion, inPlanes, 1, read);
                    Norm($"{n}.shortcut.1", planes * expansion, read);
                }
                inPlanes = planes * expansion;
            }
        }
        Conv("layer3_ds", channels * 8 * expansion, channels * 4 * expansion, 3, read);
        Fusion("fuse34", channels * 8 * expansion, read);
    }

    /// <summary>The embedding's width: 8·m_channels·expansion channels × 80/8 frequency rows (20,480).</summary>
    public int EmbeddingSize => _channels * 8 * _expansion * 10;

    private void Conv(string name, int outputs, int inputs, int kernel, Func<string, double[]?> read)
    {
        int fanIn = inputs * kernel * kernel;
        var values = read(name + ".weight")
            ?? (_initialization is { } random ? AiDotNet.TextToSpeech.Latest.IndexTtsOps.Uniform(random, outputs * fanIn, fanIn) : null)
            ?? throw new KeyNotFoundException($"The ERes2NetV2 checkpoint has no '{name}.weight'.");
        if (values.Length != outputs * inputs * kernel * kernel) throw new InvalidDataException($"'{name}.weight' has {values.Length} values.");
        _weights[name + ".weight"] = new Tensor<T>(new[] { outputs, inputs, kernel, kernel }, new Vector<T>(values.Select(v => NumOps.FromDouble(v)).ToArray()));
        var bias = read(name + ".bias");
        if (bias is not null) _weights[name + ".bias"] = new Tensor<T>(new[] { 1, outputs, 1, 1 }, new Vector<T>(bias.Select(v => NumOps.FromDouble(v)).ToArray()));
    }

    // BatchNorm2d in evaluation: y = (x − mean)/√(var + 1e-5)·γ + β, kept as one scale and shift per channel.
    private void Norm(string name, int channels, Func<string, double[]?> read)
    {
        double[] Require(string field) => read($"{name}.{field}")
            ?? (_initialization is not null ? Enumerable.Repeat(field is "weight" or "running_var" ? 1.0 : 0.0, channels).ToArray() : null)
            ?? throw new KeyNotFoundException($"The ERes2NetV2 checkpoint has no '{name}.{field}'.");
        var gamma = Require("weight");
        var beta = Require("bias");
        var mean = Require("running_mean");
        var variance = Require("running_var");
        var scale = new T[channels];
        var shift = new T[channels];
        for (int c = 0; c < channels; c++)
        {
            double s = gamma[c] / Math.Sqrt(variance[c] + 1e-5);
            scale[c] = NumOps.FromDouble(s);
            shift[c] = NumOps.FromDouble(beta[c] - mean[c] * s);
        }
        _weights[name + ".scale"] = new Tensor<T>(new[] { 1, channels, 1, 1 }, new Vector<T>(scale));
        _weights[name + ".shift"] = new Tensor<T>(new[] { 1, channels, 1, 1 }, new Vector<T>(shift));
    }

    private void Fusion(string name, int channels, Func<string, double[]?> read)
    {
        int inter = channels / 4;
        Conv($"{name}.local_att.0", inter, 2 * channels, 1, read);
        Norm($"{name}.local_att.1", inter, read);
        Conv($"{name}.local_att.3", channels, inter, 1, read);
        Norm($"{name}.local_att.4", channels, read);
    }

    private Tensor<T> Conv2d(Tensor<T> x, string name, int stride, int padding)
    {
        var y = _engine.Conv2D(x, _weights[name + ".weight"], new[] { stride, stride }, new[] { padding, padding }, new[] { 1, 1 });
        return _weights.TryGetValue(name + ".bias", out var bias) ? _engine.TensorAdd(y, _engine.TensorBroadcastTo(bias, y.Shape.ToArray())) : y;
    }

    private Tensor<T> BatchNorm(Tensor<T> x, string name)
    {
        var shape = x.Shape.ToArray();
        return _engine.TensorAdd(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(_weights[name + ".scale"], shape)),
            _engine.TensorBroadcastTo(_weights[name + ".shift"], shape));
    }

    private Tensor<T> Hardtanh(Tensor<T> x) => _engine.TensorClamp(x, NumOps.Zero, NumOps.FromDouble(20.0));

    // AFF: x·(1 + tanh(a)) + y·(1 − tanh(a)), a = local attention of [x; y].
    private Tensor<T> Fuse(Tensor<T> x, Tensor<T> y, string name)
    {
        var a = BatchNorm(Conv2d(_engine.TensorConcatenate(new[] { x, y }, 1), $"{name}.local_att.0", 1, 0), $"{name}.local_att.1");
        a = _engine.Swish(a);
        a = BatchNorm(Conv2d(a, $"{name}.local_att.3", 1, 0), $"{name}.local_att.4");
        var gate = _engine.TensorAddScalar(_engine.Tanh(a), NumOps.One);
        // y·(2 − gate) = 2y − y·gate
        var yPart = _engine.TensorSubtract(_engine.TensorMultiplyScalar(y, NumOps.FromDouble(2.0)), _engine.TensorMultiply(y, gate));
        return _engine.TensorAdd(_engine.TensorMultiply(x, gate), yPart);
    }

    private Tensor<T> Block(Tensor<T> x, string n, int planes, int stride, bool fused)
    {
        int width = (int)Math.Floor(planes * (_baseWidth / 64.0));
        var output = Hardtanh(BatchNorm(Conv2d(x, $"{n}.conv1", stride, 0), $"{n}.bn1"));
        var parts = new List<Tensor<T>>();
        Tensor<T>? sp = null;
        for (int i = 0; i < _scale; i++)
        {
            var split = _engine.TensorNarrow(output, 1, i * width, width);
            sp = i == 0 ? split : fused ? Fuse(sp!, split, $"{n}.fuse_models.{i - 1}") : _engine.TensorAdd(sp!, split);
            sp = Hardtanh(BatchNorm(Conv2d(sp, $"{n}.convs.{i}", 1, 1), $"{n}.bns.{i}"));
            parts.Add(sp);
        }
        var y = BatchNorm(Conv2d(_engine.TensorConcatenate(parts.ToArray(), 1), $"{n}.conv3", 1, 0), $"{n}.bn3");
        var residual = _weights.ContainsKey($"{n}.shortcut.0.weight") ? BatchNorm(Conv2d(x, $"{n}.shortcut.0", stride, 0), $"{n}.shortcut.1") : x;
        return Hardtanh(_engine.TensorAdd(y, residual));
    }

    /// <summary><c>forward3</c> of fbank features <c>[T, 80]</c>: the embedding <c>[20480]</c>.</summary>
    public Tensor<T> Embed(Tensor<T> features)
    {
        int frames = features.Shape[0], bins = features.Shape[1];
        var x = _engine.Reshape(_engine.TensorTranspose(features), new[] { 1, 1, bins, frames });
        var output = _engine.ReLU(BatchNorm(Conv2d(x, "conv1", 1, 1), "bn1"));
        Tensor<T>? stage3 = null;
        for (int stage = 0; stage < 4; stage++)
        {
            int planes = _channels << stage;
            for (int b = 0; b < _blocks[stage]; b++)
                output = Block(output, $"layer{stage + 1}.{b}", planes, b == 0 && stage > 0 ? 2 : 1, stage >= 2);
            if (stage == 2) stage3 = output;
        }
        var downsampled = Conv2d(stage3!, "layer3_ds", 2, 1);
        var fused = Fuse(output, downsampled, "fuse34");
        int c = fused.Shape[1], f = fused.Shape[2], t = fused.Shape[3];
        var rows = _engine.Reshape(fused, new[] { c * f, t });
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(rows, new[] { 1 }, keepDims: false), NumOps.FromDouble(1.0 / t));
    }

    /// <summary><c>SV.compute_embedding3</c> of 16 kHz audio <c>[samples]</c> in [−1, 1]: Kaldi fbank (80 bins, no
    /// dither), then <see cref="Embed"/>.</summary>
    public Tensor<T> Embedding(Tensor<T> audio16k) => Embed(_fbank.Forward(audio16k));
}
