using System.IO;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// The shape of SAM 2's video modules (Ravi et al. 2024, section 4), as the official configs define them. The defaults
/// are SAM 2's; <see cref="Sam21"/> adds the two SAM 2.1 changes (the temporal encoding of the object pointers and the
/// occlusion embedding of the spatial memories).
/// </summary>
internal sealed record Sam2MemoryConfiguration
{
    /// <summary>The width of the memories (64).</summary>
    public int MemoryDim { get; init; } = 64;

    /// <summary>The memory bank: the prompted frames and up to <c>MaskMemories − 1</c> recent frames (7).</summary>
    public int MaskMemories { get; init; } = 7;

    public int AttentionLayers { get; init; } = 4;

    public int AttentionHeads { get; init; } = 1;

    public int AttentionFeedForward { get; init; } = 2048;

    public double RopeTheta { get; init; } = 10000.0;

    /// <summary>The memory attention's dropout while training (0.1).</summary>
    public double Dropout { get; init; } = 0.1;

    public int FuserLayers { get; init; } = 2;

    public int FuserHidden { get; init; } = 1024;

    public int FuserKernel { get; init; } = 7;

    /// <summary>The fuser blocks' initial layer scale (1e-6).</summary>
    public double LayerScale { get; init; } = 1e-6;

    public int DownsamplerKernel { get; init; } = 3;

    public int DownsamplerStride { get; init; } = 2;

    public int DownsamplerPadding { get; init; } = 1;

    public int DownsamplerTotalStride { get; init; } = 16;

    /// <summary>The scale and bias that bring the mask probabilities to the memory encoder's input (20, −10).</summary>
    public double SigmoidScale { get; init; } = 20.0;

    public double SigmoidBias { get; init; } = -10.0;

    /// <summary>The most object pointers the memory attention sees (16).</summary>
    public int MaxObjectPointers { get; init; } = 16;

    /// <summary>The most prompted frames the memory attention sees, the closest in time first (−1: all).</summary>
    public int MaxConditioningFrames { get; init; } = -1;

    /// <summary>Whether the object pointers carry a projected temporal encoding (SAM 2.1).</summary>
    public bool TemporalPointerEncoding { get; init; }

    /// <summary>Whether a learned embedding marks the memories of frames predicted occluded (SAM 2.1).</summary>
    public bool OcclusionEmbedding { get; init; }

    /// <summary>SAM 2.1's video modules.</summary>
    public static Sam2MemoryConfiguration Sam21 => new() { TemporalPointerEncoding = true, OcclusionEmbedding = true };
}

/// <summary>
/// SAM 2's video modules (facebookresearch/sam2's <c>SAM2Base</c>; checked against Hugging Face's
/// <c>Sam2VideoModel</c>): the memory encoder (the predicted mask downsampled and added to the frame features, then
/// fused by ConvNeXt blocks and projected to the memory width), the memory attention (pre-norm transformer layers of
/// self-attention and cross-attention to the memories, with 2D rotary encodings that skip the object-pointer tokens),
/// and the object pointers (an MLP over the decoder's mask token).
/// </summary>
internal sealed partial class Sam2Network<T>
{
    internal sealed record MemoryLayer(Attention SelfAttention, Attention CrossAttention, LayerNormalizationLayer<T> Norm1,
        LayerNormalizationLayer<T> Norm2, LayerNormalizationLayer<T> Norm3, DenseLayer<T> Linear1, DenseLayer<T> Linear2);

    internal sealed record DownsamplerLayer(DenseLayer<T> Conv, LayerNormalizationLayer<T> Norm, int Inputs, int Outputs);

    internal sealed record FuserBlock(BiasFreeLinearLayer<T> DepthwiseWeight, BiasFreeLinearLayer<T> DepthwiseBias, LayerNormalizationLayer<T> Norm,
        DenseLayer<T> Pointwise1, DenseLayer<T> Pointwise2, BiasFreeLinearLayer<T> Scale);

    private Sam2MemoryConfiguration MemoryConfiguration
        => Configuration.Memory ?? throw new InvalidOperationException("This SAM 2 network has no video modules.");

    public List<MemoryLayer> MemoryLayers { get; } = new();

    public LayerNormalizationLayer<T>? MemoryNorm { get; private set; }

    public List<DownsamplerLayer> MaskDownsampler { get; } = new();

    public DenseLayer<T>? MaskDownsamplerOutput { get; private set; }

    public DenseLayer<T>? FeatureProjection { get; private set; }

    public List<FuserBlock> Fuser { get; } = new();

    public DenseLayer<T>? MemoryProjection { get; private set; }

    /// <summary>The 4×4, stride-4 convolution that turns a mask prompt into the decoder's mask input for its pointer.</summary>
    public DenseLayer<T>? MaskPromptDownsample { get; private set; }

    public DenseLayer<T>[]? PointerProjection { get; private set; }

    public DenseLayer<T>? PointerTemporalProjection { get; private set; }

    public BiasFreeLinearLayer<T>? NoObjectPointer { get; private set; }

    /// <summary>The temporal encodings of the memory slots <c>[MaskMemories, MemoryDim]</c>.</summary>
    public BiasFreeLinearLayer<T>? MemoryTemporal { get; private set; }

    public BiasFreeLinearLayer<T>? Occlusion { get; private set; }

    private readonly Dictionary<(int, int, int), Tensor<T>> _sines = new();

    private void BuildMemory(List<LayerBase<T>> layers, Sam2MemoryConfiguration m)
    {
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm(int dim, double eps) => Add(new LayerNormalizationLayer<T>(dim, eps));
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        int p = Configuration.Dim, md = m.MemoryDim;
        if (p % md != 0) throw new ArgumentException($"The model width {p} must be a multiple of the memory width {md}.");
        Attention Attend() => new(Linear(p), Linear(p), Linear(p), Linear(p), m.AttentionHeads);

        for (int i = 0; i < m.AttentionLayers; i++)
            MemoryLayers.Add(new MemoryLayer(Attend(), Attend(), Norm(p, 1e-5), Norm(p, 1e-5), Norm(p, 1e-5), Linear(m.AttentionFeedForward), Linear(p)));
        MemoryNorm = Norm(p, 1e-5);

        int count = (int)Math.Round(Math.Log(m.DownsamplerTotalStride) / Math.Log(m.DownsamplerStride));
        int channels = 1;
        for (int i = 0; i < count; i++)
        {
            int outputs = channels * m.DownsamplerStride * m.DownsamplerStride;
            MaskDownsampler.Add(new DownsamplerLayer(Linear(outputs), Norm(outputs, 1e-6), channels, outputs));
            channels = outputs;
        }
        MaskDownsamplerOutput = Linear(p);
        FeatureProjection = Linear(p);
        for (int i = 0; i < m.FuserLayers; i++)
            Fuser.Add(new FuserBlock(Values(p * m.FuserKernel * m.FuserKernel), Values(p), Norm(p, 1e-6), Linear(m.FuserHidden), Linear(p), Values(p)));
        MemoryProjection = Linear(md);

        MaskPromptDownsample = Linear(1);
        PointerProjection = new[] { Linear(p), Linear(p), Linear(p) };
        if (m.TemporalPointerEncoding) PointerTemporalProjection = Linear(md);
        NoObjectPointer = Values(p);
        MemoryTemporal = Values(m.MaskMemories * md);
        if (m.OcclusionEmbedding) Occlusion = Values(md);
    }

    /// <summary>The sine position encoding <c>[1, channels, h, w]</c> of DETR, normalized to 2π over the grid: the frame
    /// features' (Dim channels) and the memories' (MemoryDim channels).</summary>
    public Tensor<T> SinePositional(int channels, int h, int w)
    {
        if (_sines.TryGetValue((channels, h, w), out var cached)) return cached;
        int f = channels / 2;
        const double eps = 1e-6, scale = 2 * Math.PI;
        var pe = new Tensor<T>(new[] { 1, channels, h, w });
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double ye = (y + 1) / (h + eps) * scale, xe = (x + 1) / (w + eps) * scale;
                for (int k = 0; k < f; k++)
                {
                    double dim = Math.Pow(10000.0, 2.0 * (k / 2) / f);
                    double vy = ye / dim, vx = xe / dim;
                    pe[0, k, y, x] = NumOps.FromDouble(k % 2 == 0 ? Math.Sin(vy) : Math.Cos(vy));
                    pe[0, f + k, y, x] = NumOps.FromDouble(k % 2 == 0 ? Math.Sin(vx) : Math.Cos(vx));
                }
            }
        _sines[(channels, h, w)] = pe;
        return pe;
    }

    /// <summary>The memory slots' temporal encoding <paramref name="slot"/> <c>[1, 1, MemoryDim]</c>.</summary>
    public Tensor<T> MemorySlotEncoding(int slot)
    {
        var m = MemoryConfiguration;
        var table = _engine.Reshape(Value(MemoryTemporal!), new[] { m.MaskMemories, m.MemoryDim });
        return _engine.Reshape(_engine.TensorSlice(table, new[] { slot, 0 }, new[] { 1, m.MemoryDim }), new[] { 1, 1, m.MemoryDim });
    }

    /// <summary>
    /// The memory of a frame <c>[1, MemoryDim, Grid, Grid]</c>: its features <c>[1, Dim, Grid, Grid]</c> (without the
    /// no-memory embedding) and its mask input <c>[1, 1, ImageSize, ImageSize]</c> (scaled probabilities), plus the
    /// occlusion embedding where <paramref name="occluded"/> (SAM 2.1).
    /// </summary>
    public Tensor<T> EncodeMemory(Tensor<T> features, Tensor<T> maskInput, bool occluded)
    {
        var m = MemoryConfiguration;
        var x = maskInput;
        foreach (var layer in MaskDownsampler)
            x = Gelu(ChannelNorm(layer.Norm, Convolve(layer.Conv, x, m.DownsamplerKernel, m.DownsamplerStride, m.DownsamplerPadding)));
        x = Convolve(MaskDownsamplerOutput!, x, 1, 1, 0);
        var y = _engine.TensorAdd(Convolve(FeatureProjection!, features, 1, 1, 0), x);
        foreach (var block in Fuser) y = Fuse(block, y);
        var memory = Convolve(MemoryProjection!, y, 1, 1, 0);
        if (occluded && Occlusion is not null)
            memory = _engine.TensorAdd(memory, _engine.TensorBroadcastTo(_engine.Reshape(Value(Occlusion), new[] { 1, m.MemoryDim, 1, 1 }), memory.Shape.ToArray()));
        return memory;
    }

    /// <summary>A ConvNeXt block: a depthwise convolution, LayerNorm over the channels, the pointwise MLP (GELU) scaled per
    /// channel, and the residual.</summary>
    private Tensor<T> Fuse(FuserBlock block, Tensor<T> x)
    {
        int b = x.Shape[0], ch = x.Shape[1], h = x.Shape[2], w = x.Shape[3], k = MemoryConfiguration.FuserKernel, pad = (k - 1) / 2;
        // The depthwise convolution as a sum of its k² shifted taps, each a per-channel scale.
        var padded = x;
        if (pad > 0)
        {
            var columns = new Tensor<T>(new[] { b, ch, h, pad });
            padded = _engine.TensorConcatenate(new[] { columns, padded, columns }, 3);
            var rows = new Tensor<T>(new[] { b, ch, pad, w + 2 * pad });
            padded = _engine.TensorConcatenate(new[] { rows, padded, rows }, 2);
        }
        var weights = _engine.Reshape(Value(block.DepthwiseWeight), new[] { ch, k * k });
        Tensor<T>? sum = null;
        for (int dy = 0; dy < k; dy++)
            for (int dx = 0; dx < k; dx++)
            {
                var tap = _engine.Reshape(_engine.TensorSlice(weights, new[] { 0, dy * k + dx }, new[] { ch, 1 }), new[] { 1, ch, 1, 1 });
                var shifted = _engine.TensorSlice(padded, new[] { 0, 0, dy, dx }, new[] { b, ch, h, w });
                var term = _engine.TensorMultiply(shifted, _engine.TensorBroadcastTo(tap, shifted.Shape.ToArray()));
                sum = sum is null ? term : _engine.TensorAdd(sum, term);
            }
        var y = _engine.TensorAdd(sum!, _engine.TensorBroadcastTo(_engine.Reshape(Value(block.DepthwiseBias), new[] { 1, ch, 1, 1 }), new[] { b, ch, h, w }));
        y = ChannelNorm(block.Norm, y);
        var t = Rows(block.Pointwise2, Gelu(Rows(block.Pointwise1, _engine.TensorPermute(y, new[] { 0, 2, 3, 1 }))));
        t = _engine.TensorMultiply(t, _engine.TensorBroadcastTo(_engine.Reshape(Value(block.Scale), new[] { 1, 1, 1, ch }), t.Shape.ToArray()));
        return _engine.TensorAdd(x, _engine.TensorPermute(t, new[] { 0, 3, 1, 2 }));
    }

    /// <summary>
    /// A frame's features <c>[1, Dim, Grid, Grid]</c> conditioned on the memory bank: <paramref name="memory"/>
    /// <c>[1, M, MemoryDim]</c> (the spatial memories of <paramref name="spatialFrames"/> frames, Grid² tokens each, then
    /// <paramref name="pointerTokens"/> object-pointer tokens) with its position encodings <paramref name="memoryPositional"/>.
    /// </summary>
    public Tensor<T> AttendMemory(Tensor<T> features, Tensor<T> memory, Tensor<T> memoryPositional, int spatialFrames, int pointerTokens)
    {
        int p = Configuration.Dim, g = Configuration.Grid, hw = g * g;
        Tensor<T> Sequence(Tensor<T> map) => _engine.TensorPermute(_engine.Reshape(map, new[] { 1, p, hw }), new[] { 0, 2, 1 });
        var x = _engine.TensorAdd(Sequence(features), _engine.TensorMultiplyScalar(Sequence(SinePositional(p, g, g)), NumOps.FromDouble(0.1)));
        var keys = _engine.TensorAdd(memory, memoryPositional);
        foreach (var layer in MemoryLayers)
        {
            var q = Rows(layer.Norm1, x);
            x = _engine.TensorAdd(x, Dropout(RopeAttend(layer.SelfAttention, q, q, q, 1, 0)));
            q = Rows(layer.Norm2, x);
            x = _engine.TensorAdd(x, Dropout(RopeAttend(layer.CrossAttention, q, keys, memory, spatialFrames, pointerTokens)));
            q = Rows(layer.Norm3, x);
            x = _engine.TensorAdd(x, Dropout(Rows(layer.Linear2, Dropout(_engine.ReLU(Rows(layer.Linear1, q))))));
        }
        x = Rows(MemoryNorm!, x);
        return _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1 }), new[] { 1, p, g, g });
    }

    private Tensor<T> Dropout(Tensor<T> x)
    {
        double rate = MemoryConfiguration.Dropout;
        return Training && rate > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, rate, TrainingRandom) : x;
    }

    /// <summary>Attention with 2D axial rotary encodings on the queries and on the keys' first tokens (the grid's encodings
    /// repeated for each of <paramref name="repeat"/> frames), the last <paramref name="exclude"/> keys left unrotated.</summary>
    private Tensor<T> RopeAttend(Attention attention, Tensor<T> q, Tensor<T> k, Tensor<T> v, int repeat, int exclude)
    {
        int n = q.Shape[0], tq = q.Shape[1], tk = k.Shape[1];
        var qp = Rows(attention.Query, q);
        int inner = qp.Shape[2], heads = attention.Heads, hd = inner / heads;
        Tensor<T> Heads(Tensor<T> x, int t) => _engine.Reshape(_engine.TensorPermute(_engine.Reshape(x, new[] { n, t, heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { n * heads, t, hd });
        var (cos, sin) = Rotary(tq, hd);
        var qh = Rotate(Heads(qp, tq), cos, sin);
        var kh = Heads(Rows(attention.Key, k), tk);
        int rotated = tk - exclude;
        if (rotated > 0)
        {
            var head = _engine.TensorSlice(kh, new[] { 0, 0, 0 }, new[] { n * heads, rotated, hd });
            var cosK = repeat == 1 ? cos : _engine.TensorTile(cos, new[] { repeat, 1 });
            var sinK = repeat == 1 ? sin : _engine.TensorTile(sin, new[] { repeat, 1 });
            head = Rotate(head, cosK, sinK);
            kh = exclude > 0 ? _engine.TensorConcatenate(new[] { head, _engine.TensorSlice(kh, new[] { 0, rotated, 0 }, new[] { n * heads, exclude, hd }) }, 1) : head;
        }
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(qh, _engine.TensorPermute(kh, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var weights = Dropout(_engine.Softmax(scores, axis: 2));
        var context = _engine.BatchMatMul(weights, Heads(Rows(attention.Value, v), tk));
        context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context, new[] { n, heads, tq, hd }), new[] { 0, 2, 1, 3 }), new[] { n, tq, inner });
        return Rows(attention.Output, context);
    }

    /// <summary>The cosines and sines <c>[tokens, hd]</c> of the axial rotary encoding of a square grid of
    /// <paramref name="tokens"/> cells (x = t mod side, y = t div side): the first hd/4 frequency pairs rotate by x, the
    /// rest by y, θ_k = RopeTheta^(−4k/hd).</summary>
    private (Tensor<T> Cos, Tensor<T> Sin) Rotary(int tokens, int hd)
    {
        int side = (int)Math.Round(Math.Sqrt(tokens));
        if (side * side != tokens) throw new ArgumentException($"Rotary encodings need a square grid; got {tokens} tokens.");
        double theta = MemoryConfiguration.RopeTheta;
        var cos = new Tensor<T>(new[] { tokens, hd });
        var sin = new Tensor<T>(new[] { tokens, hd });
        int quarter = hd / 4;
        for (int t = 0; t < tokens; t++)
        {
            int x = t % side, y = t / side;
            for (int j = 0; j < hd; j++)
            {
                int pair = j / 2;
                double frequency = 1.0 / Math.Pow(theta, 4.0 * (pair % quarter) / hd);
                double angle = (pair < quarter ? x : y) * frequency;
                cos[t, j] = NumOps.FromDouble(Math.Cos(angle));
                sin[t, j] = NumOps.FromDouble(Math.Sin(angle));
            }
        }
        return (cos, sin);
    }

    /// <summary>x·cos + rotate(x)·sin, rotate turning each pair (a, b) into (−b, a).</summary>
    private Tensor<T> Rotate(Tensor<T> x, Tensor<T> cos, Tensor<T> sin)
    {
        int n = x.Shape[0], t = x.Shape[1], hd = x.Shape[2];
        var pairs = _engine.Reshape(x, new[] { n, t, hd / 2, 2 });
        var first = _engine.TensorSlice(pairs, new[] { 0, 0, 0, 0 }, new[] { n, t, hd / 2, 1 });
        var second = _engine.TensorSlice(pairs, new[] { 0, 0, 0, 1 }, new[] { n, t, hd / 2, 1 });
        var rotated = _engine.Reshape(_engine.TensorConcatenate(new[] { _engine.TensorMultiplyScalar(second, NumOps.FromDouble(-1.0)), first }, 3), new[] { n, t, hd });
        var shape = x.Shape.ToArray();
        return _engine.TensorAdd(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(_engine.Reshape(cos, new[] { 1, t, hd }), shape)),
            _engine.TensorMultiply(rotated, _engine.TensorBroadcastTo(_engine.Reshape(sin, new[] { 1, t, hd }), shape)));
    }

    /// <summary>The object pointer <c>[N, Dim]</c> of decoder mask tokens <c>[N, Dim]</c>.</summary>
    public Tensor<T> ObjectPointer(Tensor<T> token) => Mlp(PointerProjection!, token);

    /// <summary>The learned pointer of a frame without the object <c>[1, Dim]</c>.</summary>
    public Tensor<T> NoObjectPointerValue() => _engine.Reshape(Value(NoObjectPointer!), new[] { 1, Configuration.Dim });

    /// <summary>The object pointers' position encodings <c>[P, MemoryDim]</c> for their temporal offsets divided by the
    /// largest one: the projected 1D sine encoding (SAM 2.1), or zeros.</summary>
    public Tensor<T> PointerPositional(double[] normalizedOffsets)
    {
        var m = MemoryConfiguration;
        int count = normalizedOffsets.Length;
        if (PointerTemporalProjection is null) return new Tensor<T>(new[] { count, m.MemoryDim });
        int dim = Configuration.Dim, half = dim / 2;
        var pe = new Tensor<T>(new[] { count, dim });
        for (int i = 0; i < count; i++)
            for (int k = 0; k < half; k++)
            {
                double v = normalizedOffsets[i] / Math.Pow(10000.0, 2.0 * (k / 2) / half);
                pe[i, k] = NumOps.FromDouble(Math.Sin(v));
                pe[i, half + k] = NumOps.FromDouble(Math.Cos(v));
            }
        return PointerTemporalProjection.Forward(pe);
    }

    /// <summary>The decoder's mask input of a mask prompt <c>[1, 1, ImageSize, ImageSize]</c>: the 4×4, stride-4
    /// convolution.</summary>
    public Tensor<T> DownsampleMaskPrompt(Tensor<T> mask) => Convolve(MaskPromptDownsample!, mask, 4, 4, 0);

    /// <summary>The value layers that hold biases (the transposed convolutions' and the depthwise convolutions'), which the
    /// training recipe exempts from weight decay.</summary>
    public IEnumerable<LayerBase<T>> BiasValueLayers()
    {
        yield return Upscale1.Bias;
        yield return Upscale2.Bias;
        foreach (var block in Fuser) yield return block.DepthwiseBias;
    }

    /// <summary>Loads a released checkpoint in Hugging Face's format (<c>model.safetensors</c> of a <c>Sam2Model</c> or
    /// <c>Sam2VideoModel</c>), checking every tensor's shape against this network's.</summary>
    public void LoadSafetensors(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("The checkpoint path is empty.", nameof(path));
        using var stream = File.OpenRead(path);
        var file = SafetensorsReader.Read(stream);
        LoadTorchWeights((name, shape) =>
        {
            var tensor = file.Get(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var actual = tensor.Shape.Select(d => (int)d).ToArray();
            if (!actual.SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] in " +
                    "this model; build the model with the checkpoint's size.");
            return file.ReadAsDouble(name);
        });
    }

    /// <summary>Loads the video modules from a Hugging Face <c>Sam2VideoModel</c> state dict.</summary>
    private void LoadMemoryWeights(Func<string, int[], double[]> read)
    {
        var m = MemoryConfiguration;
        int p = Configuration.Dim, md = m.MemoryDim;
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }), read($"{name}.bias", new[] { outputs }));
        void Conv(DenseLayer<T> layer, string name, int inputs, int outputs, int kernel)
            => TorchParameters.Linear(_engine, layer, inputs * kernel * kernel, outputs, read($"{name}.weight", new[] { outputs, inputs, kernel, kernel }),
                read($"{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name, int dim)
            => TorchParameters.LayerNorm(_engine, norm, read($"{name}.weight", new[] { dim }), read($"{name}.bias", new[] { dim }));
        void Values(BiasFreeLinearLayer<T> holder, string name, int[] shape)
            => TorchParameters.Linear(holder, 1, shape.Aggregate(1, (a, b) => a * b), read(name, shape));
        void Attention(Attention a, string name, int keyInputs)
        {
            Linear(a.Query, $"{name}.q_proj", p, p);
            Linear(a.Key, $"{name}.k_proj", keyInputs, p);
            Linear(a.Value, $"{name}.v_proj", keyInputs, p);
            Linear(a.Output, $"{name}.o_proj", p, p);
        }

        for (int i = 0; i < MemoryLayers.Count; i++)
        {
            var layer = MemoryLayers[i];
            string n = $"memory_attention.layers.{i}";
            Attention(layer.SelfAttention, $"{n}.self_attn", p);
            Attention(layer.CrossAttention, $"{n}.cross_attn_image", md);
            Linear(layer.Linear1, $"{n}.linear1", p, m.AttentionFeedForward);
            Linear(layer.Linear2, $"{n}.linear2", m.AttentionFeedForward, p);
            Norm(layer.Norm1, $"{n}.layer_norm1", p);
            Norm(layer.Norm2, $"{n}.layer_norm2", p);
            Norm(layer.Norm3, $"{n}.layer_norm3", p);
        }
        Norm(MemoryNorm!, "memory_attention.layer_norm", p);
        for (int i = 0; i < MaskDownsampler.Count; i++)
        {
            var layer = MaskDownsampler[i];
            Conv(layer.Conv, $"memory_encoder.mask_downsampler.layers.{i}.conv", layer.Inputs, layer.Outputs, m.DownsamplerKernel);
            Norm(layer.Norm, $"memory_encoder.mask_downsampler.layers.{i}.layer_norm", layer.Outputs);
        }
        Conv(MaskDownsamplerOutput!, "memory_encoder.mask_downsampler.final_conv", MaskDownsampler[^1].Outputs, p, 1);
        Conv(FeatureProjection!, "memory_encoder.feature_projection", p, p, 1);
        for (int i = 0; i < Fuser.Count; i++)
        {
            var block = Fuser[i];
            string n = $"memory_encoder.memory_fuser.layers.{i}";
            Values(block.DepthwiseWeight, $"{n}.depthwise_conv.weight", new[] { p, 1, m.FuserKernel, m.FuserKernel });
            Values(block.DepthwiseBias, $"{n}.depthwise_conv.bias", new[] { p });
            Norm(block.Norm, $"{n}.layer_norm", p);
            Linear(block.Pointwise1, $"{n}.pointwise_conv1", p, m.FuserHidden);
            Linear(block.Pointwise2, $"{n}.pointwise_conv2", m.FuserHidden, p);
            Values(block.Scale, $"{n}.scale", new[] { p });
        }
        Conv(MemoryProjection!, "memory_encoder.projection", p, md, 1);
        Conv(MaskPromptDownsample!, "mask_downsample", 1, 1, 4);
        Linear(PointerProjection![0], "object_pointer_proj.proj_in", p, p);
        Linear(PointerProjection[1], "object_pointer_proj.layers.0", p, p);
        Linear(PointerProjection[2], "object_pointer_proj.proj_out", p, p);
        if (PointerTemporalProjection is not null) Linear(PointerTemporalProjection, "temporal_positional_encoding_projection_layer", p, md);
        Values(NoObjectPointer!, "no_object_pointer", new[] { 1, p });
        Values(MemoryTemporal!, "memory_temporal_positional_encoding", new[] { m.MaskMemories, 1, 1, md });
        if (Occlusion is not null) Values(Occlusion, "occlusion_spatial_embedding_parameter", new[] { 1, md });
    }
}
