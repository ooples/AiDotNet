using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// fairseq's multi-head attention as NATSpeech builds it without biases (<c>MultiheadAttention(bias=False)</c>): one packed
/// input projection <c>[3h, h]</c>, queries scaled by the head width's inverse square root, padded keys and (for a decoder)
/// future positions masked out, and an output projection.
/// </summary>
internal sealed class MegaTtsAttention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MegaTtsAttention(IEngine engine, List<LayerBase<T>> layers, int width, int heads)
    {
        if (width % heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(heads));
        _engine = engine;
        Width = width;
        Heads = heads;
        InProjection = new BiasFreeLinearLayer<T>(width, 3 * width);
        OutProjection = new BiasFreeLinearLayer<T>(width, width);
        layers.Add(InProjection);
        layers.Add(OutProjection);
    }

    public int Width { get; }
    public int Heads { get; }
    public BiasFreeLinearLayer<T> InProjection { get; }
    public BiasFreeLinearLayer<T> OutProjection { get; }

    /// <summary>Self-attention over <c>[B, T, h]</c>; <paramref name="keyPadding"/> <c>[B, T]</c> true marks padding.</summary>
    public Tensor<T> Forward(Tensor<T> x, bool[,]? keyPadding, bool causal)
    {
        int b = x.Shape[0], t = x.Shape[1], h = Width, hd = h / Heads;
        var qkv = _engine.Reshape(InProjection.Forward(_engine.Reshape(x, new[] { b * t, h })), new[] { b, t, 3 * h });
        Tensor<T> Part(int p) => _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.TensorSlice(qkv, new[] { 0, 0, p * h }, new[] { b, t, h }),
            new[] { b, t, Heads, hd }), new[] { 0, 2, 1, 3 }), new[] { b * Heads, t, hd });
        var q = _engine.TensorMultiplyScalar(Part(0), NumOps.FromDouble(Math.Pow(hd, -0.5)));
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(Part(1), new[] { 0, 2, 1 }));
        if (causal || keyPadding is not null)
        {
            var bias = new Tensor<T>(new[] { b * Heads, t, t });
            var negative = NumOps.FromDouble(double.NegativeInfinity);
            for (int i = 0; i < b; i++)
                for (int n = 0; n < Heads; n++)
                    for (int r = 0; r < t; r++)
                        for (int c = 0; c < t; c++)
                            if ((causal && c > r) || (keyPadding is not null && keyPadding[i, c])) bias[i * Heads + n, r, c] = negative;
            scores = _engine.TensorAdd(scores, bias);
        }
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Part(2));
        var merged = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context, new[] { b, Heads, t, hd }), new[] { 0, 2, 1, 3 }), new[] { b * t, h });
        return _engine.Reshape(OutProjection.Forward(merged), new[] { b, t, h });
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        TorchParameters.Linear(InProjection, Width, 3 * Width, read($"{prefix}.in_proj_weight"));
        TorchParameters.Linear(OutProjection, Width, Width, read($"{prefix}.out_proj.weight"));
    }
}

/// <summary>
/// NATSpeech's pre-norm Transformer layer: layer norm, self-attention and a residual, then layer norm, a feed-forward
/// convolution of kernel k to the filter width (4h unless given; scaled by k^−½, GELU) and a linear layer back, and a residual. An encoder layer
/// (<c>EncSALayer</c>) pads its convolution on both sides and zeroes padded positions; a decoder layer (<c>DecSALayer</c>
/// without encoder attention) attends causally and pads its convolution on the left.
/// </summary>
internal sealed class MegaTtsTransformerLayer<T>
{
    private readonly IEngine _engine;

    public MegaTtsTransformerLayer(IEngine engine, List<LayerBase<T>> layers, int width, int heads, int kernel, bool decoder, int? filter = null)
    {
        _engine = engine;
        Width = width;
        Filter = filter ?? 4 * width;
        Kernel = kernel;
        Decoder = decoder;
        Norm1 = new LayerNormalizationLayer<T>(width, 1e-5);
        layers.Add(Norm1);
        Attention = new MegaTtsAttention<T>(engine, layers, width, heads);
        Norm2 = new LayerNormalizationLayer<T>(width, 1e-5);
        layers.Add(Norm2);
        Ffn1 = MegaTts3Ops.Conv(layers, width, Filter, kernel, padding: decoder ? 0 : kernel / 2);
        Ffn2 = MegaTts3Ops.Linear(layers, width);
    }

    public int Width { get; }
    public int Filter { get; }
    public int Kernel { get; }
    public bool Decoder { get; }
    public LayerNormalizationLayer<T> Norm1 { get; }
    public MegaTtsAttention<T> Attention { get; }
    public LayerNormalizationLayer<T> Norm2 { get; }
    public NormedConv1DLayer<T> Ffn1 { get; }
    public DenseLayer<T> Ffn2 { get; }

    /// <summary>The layer on <c>[B, T, h]</c>; <paramref name="nonPadding"/> <c>[B, T, 1]</c> (encoder only).</summary>
    public Tensor<T> Forward(Tensor<T> x, bool[,]? keyPadding, Tensor<T>? nonPadding)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = x.Shape[0], t = x.Shape[1];
        Tensor<T> Mask(Tensor<T> v) => nonPadding is null ? v : _engine.TensorMultiply(v, _engine.TensorBroadcastTo(nonPadding, v.Shape.ToArray()));
        x = Mask(_engine.TensorAdd(x, Attention.Forward(Norm1.Forward(x), keyPadding, Decoder)));
        var y = Norm2.Forward(x);
        if (Decoder)
            y = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { b, Kernel - 1, Width }), y }, 1);
        y = _engine.TensorMultiplyScalar(MegaTts3Ops.OverTime(_engine, Ffn1, y), ops.FromDouble(Math.Pow(Kernel, -0.5)));
        y = MegaTts3Ops.Rows(_engine, Ffn2, ActivationHelper.ExactGelu(y, _engine));
        return Mask(_engine.TensorAdd(x, y));
    }

    public void LoadTorch(IEngine engine, Func<string, double[]> read, string prefix)
    {
        TorchParameters.LayerNorm(engine, Norm1, read($"{prefix}.layer_norm1.weight"), read($"{prefix}.layer_norm1.bias"));
        Attention.LoadTorch(read, $"{prefix}.self_attn");
        string norm2 = Decoder ? "layer_norm3" : "layer_norm2";
        TorchParameters.LayerNorm(engine, Norm2, read($"{prefix}.{norm2}.weight"), read($"{prefix}.{norm2}.bias"));
        MegaTts3Ops.LoadConv(read, Ffn1, Decoder ? $"{prefix}.ffn.ffn_1.1" : $"{prefix}.ffn.ffn_1");
        MegaTts3Ops.LoadDense(engine, read, Ffn2, $"{prefix}.ffn.ffn_2", Filter, Width);
    }
}

/// <summary>NATSpeech's sinusoidal positions (<c>SinusoidalPositionalEmbedding</c>, padding index 0): position p ≥ 1 of
/// the non-padding tokens (padding takes zeros), sines then cosines.</summary>
internal static class MegaTtsPositions
{
    public static Tensor<T> Of<T>(bool[,] padding, int width)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = padding.GetLength(0), t = padding.GetLength(1);
        var positions = new Tensor<T>(new[] { b, t, width });
        for (int i = 0; i < b; i++)
        {
            int p = 0;
            for (int j = 0; j < t; j++)
            {
                if (padding[i, j]) continue;
                var e = MegaTts3Ops.Sinusoid(++p, width);
                for (int d = 0; d < width; d++) positions[i, j, d] = ops.FromDouble(e[d]);
            }
        }
        return positions;
    }
}

/// <summary>NATSpeech's <c>FastSpeechEncoder</c>: token embeddings scaled by √h plus sinusoidal positions, encoder layers
/// with padding masked, and a final layer norm.</summary>
internal sealed class MegaTtsContentEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MegaTtsContentEncoder(IEngine engine, List<LayerBase<T>> layers, int vocabulary, int width, int layerCount, int heads, int kernel,
        int? filter = null)
    {
        _engine = engine;
        Width = width;
        Embedding = new TiedEmbeddingLayer<T>(vocabulary, width);
        layers.Add(Embedding);
        for (int i = 0; i < layerCount; i++) Layers.Add(new MegaTtsTransformerLayer<T>(engine, layers, width, heads, kernel, decoder: false, filter));
        FinalNorm = new LayerNormalizationLayer<T>(width, 1e-5);
        layers.Add(FinalNorm);
    }

    public int Width { get; }
    public TiedEmbeddingLayer<T> Embedding { get; }
    public List<MegaTtsTransformerLayer<T>> Layers { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }

    /// <summary>The encoding <c>[B, L, h]</c> of tokens <c>[B, L]</c> (0 is padding).</summary>
    public Tensor<T> Forward(int[,] tokens)
    {
        int b = tokens.GetLength(0), l = tokens.GetLength(1);
        var ids = new Tensor<T>(new[] { b * l });
        var padding = new bool[b, l];
        var nonPadding = new Tensor<T>(new[] { b, l, 1 });
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
            {
                ids[i * l + j] = NumOps.FromDouble(tokens[i, j]);
                padding[i, j] = tokens[i, j] == 0;
                if (!padding[i, j]) nonPadding[i, j, 0] = NumOps.One;
            }
        var x = _engine.TensorMultiplyScalar(_engine.Reshape(Embedding.Forward(ids), new[] { b, l, Width }), NumOps.FromDouble(Math.Sqrt(Width)));
        x = _engine.TensorAdd(x, MegaTtsPositions.Of<T>(padding, Width));
        var mask = _engine.TensorBroadcastTo(nonPadding, new[] { b, l, Width });
        x = _engine.TensorMultiply(x, mask);
        foreach (var layer in Layers) x = _engine.TensorMultiply(layer.Forward(x, padding, nonPadding), mask);
        return _engine.TensorMultiply(FinalNorm.Forward(x), mask);
    }

    public void LoadTorch(IEngine engine, Func<string, double[]> read, string prefix)
    {
        Embedding.LoadTable(read($"{prefix}.embed_tokens.weight"));
        for (int i = 0; i < Layers.Count; i++) Layers[i].LoadTorch(engine, read, $"{prefix}.layers.{i}.op");
        TorchParameters.LayerNorm(engine, FinalNorm, read($"{prefix}.layer_norm.weight"), read($"{prefix}.layer_norm.bias"));
    }
}

/// <summary>NATSpeech's <c>DurationPredictor</c>: convolutions with ReLU and layer norm (padding zeroed after each), then a
/// linear layer and softplus: each token's duration in frames.</summary>
internal sealed class MegaTtsDurationPredictor<T>
{
    private readonly IEngine _engine;

    public MegaTtsDurationPredictor(IEngine engine, List<LayerBase<T>> layers, int input, int channels, int layerCount, int kernel)
    {
        _engine = engine;
        for (int i = 0; i < layerCount; i++)
        {
            var conv = MegaTts3Ops.Conv(layers, i == 0 ? input : channels, channels, kernel, padding: kernel / 2);
            var norm = new LayerNormalizationLayer<T>(channels, 1e-5);
            layers.Add(norm);
            Layers.Add((conv, norm));
        }
        Output = MegaTts3Ops.Linear(layers, 1);
        Channels = channels;
    }

    public int Channels { get; }
    public List<(NormedConv1DLayer<T> Conv, LayerNormalizationLayer<T> Norm)> Layers { get; } = new();
    public DenseLayer<T> Output { get; }

    /// <summary>Durations <c>[B, L]</c> of encodings <c>[B, L, input]</c> (padding true where masked).</summary>
    public Tensor<T> Forward(Tensor<T> x, bool[,] padding)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = x.Shape[0], l = x.Shape[1];
        var keep = new Tensor<T>(new[] { b, l, 1 });
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
                if (!padding[i, j]) keep[i, j, 0] = ops.One;
        foreach (var (conv, norm) in Layers)
            x = _engine.TensorMultiply(norm.Forward(_engine.ReLU(MegaTts3Ops.OverTime(_engine, conv, x))), _engine.TensorBroadcastTo(keep, new[] { b, l, Channels }));
        var y = _engine.TensorMultiply(_engine.Softplus(MegaTts3Ops.Rows(_engine, Output, x)), keep);
        return _engine.Reshape(y, new[] { b, l });
    }

    public void LoadTorch(IEngine engine, Func<string, double[]> read, string prefix)
    {
        for (int i = 0; i < Layers.Count; i++)
        {
            MegaTts3Ops.LoadConv(read, Layers[i].Conv, $"{prefix}.conv.{i}.0");
            TorchParameters.LayerNorm(engine, Layers[i].Norm, read($"{prefix}.conv.{i}.2.weight"), read($"{prefix}.conv.{i}.2.bias"));
        }
        MegaTts3Ops.LoadDense(engine, read, Output, $"{prefix}.linear.0", Channels, 1);
    }
}

/// <summary>
/// NATSpeech's <c>ConvBlocks</c> with layer norm: residual blocks of n layers (layer norm, a kernel-k convolution to 2h
/// scaled by k^−½, GELU, a 1×1 convolution back, added; padded frames zeroed), a final layer norm and a post-net
/// convolution to the output width. Padding is where an input frame is all zeros.
/// </summary>
internal sealed class MegaTtsConvBlocks<T>
{
    private readonly IEngine _engine;

    public MegaTtsConvBlocks(IEngine engine, List<LayerBase<T>> layers, int width, int outputs, int kernel, int blocks, int layersInBlock = 2,
        int postKernel = 3)
    {
        _engine = engine;
        Width = width;
        Kernel = kernel;
        for (int i = 0; i < blocks; i++)
        {
            var block = new List<(LayerNormalizationLayer<T>, NormedConv1DLayer<T>, NormedConv1DLayer<T>)>();
            for (int j = 0; j < layersInBlock; j++)
            {
                var norm = new LayerNormalizationLayer<T>(width, 1e-5);
                layers.Add(norm);
                block.Add((norm, MegaTts3Ops.Conv(layers, width, 2 * width, kernel, padding: (kernel - 1) / 2), MegaTts3Ops.Conv(layers, 2 * width, width, 1)));
            }
            Blocks.Add(block);
        }
        LastNorm = new LayerNormalizationLayer<T>(width, 1e-5);
        layers.Add(LastNorm);
        PostNet = MegaTts3Ops.Conv(layers, width, outputs, postKernel, padding: postKernel / 2);
    }

    public int Width { get; }
    public int Kernel { get; }
    public List<List<(LayerNormalizationLayer<T> Norm, NormedConv1DLayer<T> Conv, NormedConv1DLayer<T> Pointwise)>> Blocks { get; } = new();
    public LayerNormalizationLayer<T> LastNorm { get; }
    public NormedConv1DLayer<T> PostNet { get; }

    // 1 where a frame of [B, T, C] has any non-zero value.
    private Tensor<T> NonPadding(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = x.Shape[0], t = x.Shape[1], c = x.Shape[2];
        var mask = new Tensor<T>(new[] { b, t, 1 });
        for (int i = 0; i < b; i++)
            for (int j = 0; j < t; j++)
            {
                double sum = 0;
                for (int k = 0; k < c; k++) sum += Math.Abs(ops.ToDouble(x[i, j, k]));
                if (sum > 0) mask[i, j, 0] = ops.One;
            }
        return mask;
    }

    /// <summary>The stack on <c>[B, T, h]</c>: <c>[B, T, outputs]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? nonPadding = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> Mask(Tensor<T> v, Tensor<T> m) => _engine.TensorMultiply(v, _engine.TensorBroadcastTo(m, v.Shape.ToArray()));
        var outer = nonPadding ?? NonPadding(x);
        foreach (var block in Blocks)
        {
            var inner = NonPadding(x);
            foreach (var (norm, conv, pointwise) in block)
            {
                var y = _engine.TensorMultiplyScalar(MegaTts3Ops.OverTime(_engine, conv, norm.Forward(x)), ops.FromDouble(Math.Pow(Kernel, -0.5)));
                y = MegaTts3Ops.OverTime(_engine, pointwise, ActivationHelper.ExactGelu(y, _engine));
                x = Mask(_engine.TensorAdd(x, y), inner);
            }
        }
        x = Mask(x, outer);
        x = Mask(LastNorm.Forward(x), outer);
        return Mask(MegaTts3Ops.OverTime(_engine, PostNet, x), outer);
    }

    public void LoadTorch(IEngine engine, Func<string, double[]> read, string prefix)
    {
        for (int i = 0; i < Blocks.Count; i++)
            for (int j = 0; j < Blocks[i].Count; j++)
            {
                var (norm, conv, pointwise) = Blocks[i][j];
                string p = $"{prefix}.res_blocks.{i}.blocks.{j}";
                TorchParameters.LayerNorm(engine, norm, read($"{p}.0.weight"), read($"{p}.0.bias"));
                MegaTts3Ops.LoadConv(read, conv, $"{p}.1");
                MegaTts3Ops.LoadConv(read, pointwise, $"{p}.4");
            }
        TorchParameters.LayerNorm(engine, LastNorm, read($"{prefix}.last_norm.weight"), read($"{prefix}.last_norm.bias"));
        MegaTts3Ops.LoadConv(read, PostNet, $"{prefix}.post_net1");
    }
}

/// <summary>
/// SyntaSpeech's multi-window mel discriminator (<c>MultiWindowDiscriminator</c>): for each window length, a window of the
/// mel spectrogram through three stride-2 3×3 convolutions (LeakyReLU 0.2, batch norm after the first two, with its eps
/// of 0.8) and a linear layer to one score; the scores summed.
/// </summary>
internal sealed class MegaTtsMelDiscriminator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed class Window
    {
        public int Length;
        public Conv2DLayer<T>[] Convs = new Conv2DLayer<T>[3];
        public BatchNormalizationLayer<T>[] Norms = new BatchNormalizationLayer<T>[2];
        public DenseLayer<T> Output = null!;
    }

    public MegaTtsMelDiscriminator(IEngine engine, List<LayerBase<T>> layers, int[] windows, int bins, int hidden)
    {
        _engine = engine;
        Bins = bins;
        Hidden = hidden;
        foreach (int length in windows)
        {
            var w = new Window { Length = length };
            for (int i = 0; i < 3; i++)
            {
                w.Convs[i] = new Conv2DLayer<T>(i == 0 ? 1 : hidden, hidden, 3, 3, 2, 2, 1, 1, true);
                layers.Add(w.Convs[i]);
            }
            for (int i = 0; i < 2; i++)
            {
                w.Norms[i] = new BatchNormalizationLayer<T>(hidden, epsilon: 0.8);
                layers.Add(w.Norms[i]);
            }
            w.Output = MegaTts3Ops.Linear(layers, 1);
            Windows.Add(w);
        }
    }

    public int Bins { get; }
    public int Hidden { get; }
    public List<Window> Windows { get; } = new();

    /// <summary>The summed score <c>[B]</c> of mels <c>[B, T, bins]</c> at the given window starts; null when the mel is
    /// shorter than a window (SyntaSpeech then skips the loss).</summary>
    public Tensor<T>? Forward(Tensor<T> mel, IReadOnlyList<int> starts)
    {
        int b = mel.Shape[0], t = mel.Shape[1];
        Tensor<T>? total = null;
        for (int k = 0; k < Windows.Count; k++)
        {
            var w = Windows[k];
            if (t - w.Length < 0) return null;
            var x = _engine.Reshape(_engine.TensorSlice(mel, new[] { 0, starts[k], 0 }, new[] { b, w.Length, Bins }), new[] { b, 1, w.Length, Bins });
            for (int i = 0; i < 3; i++)
            {
                x = w.Convs[i].Forward(x);
                x = _engine.LeakyReLU(x, NumOps.FromDouble(0.2));
                if (i < 2) x = w.Norms[i].Forward(x);
            }
            var score = w.Output.Forward(_engine.Reshape(x, new[] { b, x.Length / b }));
            total = total is null ? score : _engine.TensorAdd(total, score);
        }
        return total is null ? null : _engine.Reshape(total, new[] { b });
    }

    private void Copy(Tensor<T> target, double[] values)
    {
        if (values.Length != target.Length) throw new InvalidDataException($"Expected {target.Length} values, got {values.Length}.");
        for (int i = 0; i < values.Length; i++) target[i] = NumOps.FromDouble(values[i]);
        _engine.InvalidatePersistentTensor(target);
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        for (int k = 0; k < Windows.Count; k++)
        {
            var w = Windows[k];
            string p = $"{prefix}.discriminators.{k}";
            for (int i = 0; i < 3; i++)
            {
                Copy(w.Convs[i].Kernel, read($"{p}.model.{i}.0.weight"));
                Copy(w.Convs[i].Bias, read($"{p}.model.{i}.0.bias"));
            }
            for (int i = 0; i < 2; i++)
                TorchParameters.BatchNorm(w.Norms[i], read($"{p}.model.{i}.3.weight"), read($"{p}.model.{i}.3.bias"),
                    read($"{p}.model.{i}.3.running_mean"), read($"{p}.model.{i}.3.running_var"));
            var weight = read($"{p}.adv_layer.weight");
            TorchParameters.Linear(_engine, w.Output, weight.Length, 1, weight, read($"{p}.adv_layer.bias"));
        }
    }
}
