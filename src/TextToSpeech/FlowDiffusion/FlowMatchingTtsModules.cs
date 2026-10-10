using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// The building blocks E2 TTS and F5-TTS share (reference SWivid/F5-TTS <c>model/modules.py</c>,
/// <c>model/backbones/dit.py</c>, <c>model/backbones/unett.py</c>), for one sequence <c>[frames, channels]</c>.
/// Each block registers its layers with the owning model through the list it is given.
/// </summary>
internal static class FlowMatchingTts
{
    /// <summary>Sinusoidal embedding <c>[sin; cos](scale · x · e^{−i ln 10⁴ / (half − 1)})</c> of a scalar, <c>[1, dim]</c>.</summary>
    public static Tensor<T> Sinusoidal<T>(double x, int dim, double scale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int half = dim / 2;
        double step = Math.Log(10000) / (half - 1);
        var e = new Tensor<T>(new[] { 1, dim });
        for (int i = 0; i < half; i++)
        {
            double a = scale * x * Math.Exp(-step * i);
            e[0, i] = ops.FromDouble(Math.Sin(a));
            e[0, half + i] = ops.FromDouble(Math.Cos(a));
        }
        return e;
    }

    /// <summary>The absolute text position table <c>[cos(t θ_i); sin(t θ_i)]</c>, θ_i = 10⁴^(−2i/dim) (precompute_freqs_cis).</summary>
    public static Tensor<T> TextPositions<T>(int length, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int half = dim / 2;
        var table = new Tensor<T>(new[] { length, dim });
        for (int t = 0; t < length; t++)
            for (int i = 0; i < half; i++)
            {
                double a = t / Math.Pow(10000.0, 2.0 * i / dim);
                table[t, i] = ops.FromDouble(Math.Cos(a));
                table[t, half + i] = ops.FromDouble(Math.Sin(a));
            }
        return table;
    }

    /// <summary>
    /// Rotary embedding as x-transformers' <c>RotaryEmbedding(dim_head)</c> applies it: frequencies
    /// θ_i = 10⁴^(−2i/d) repeated over adjacent pairs, <c>x cos + rotate(x) sin</c> with rotate(a, b) = (−b, a).
    /// </summary>
    public static Tensor<T> Rotary<T>(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int length = x.Shape[0], d = x.Shape[1];
        var cos = new Tensor<T>(new[] { length, d });
        var sin = new Tensor<T>(new[] { length, d });
        var swap = new Tensor<T>(new[] { d, d });                  // x · swap = rotate(x)
        for (int i = 0; i < d / 2; i++)
        {
            swap[2 * i + 1, 2 * i] = ops.FromDouble(-1);
            swap[2 * i, 2 * i + 1] = ops.One;
        }
        for (int t = 0; t < length; t++)
            for (int i = 0; i < d; i++)
            {
                double a = t / Math.Pow(10000.0, 2.0 * (i / 2) / d);
                cos[t, i] = ops.FromDouble(Math.Cos(a));
                sin[t, i] = ops.FromDouble(Math.Sin(a));
            }
        return engine.TensorAdd(engine.TensorMultiply(x, cos), engine.TensorMultiply(engine.TensorMatMul(x, swap), sin));
    }

    /// <summary>LayerNorm without affine parameters over the last axis, ε = 1e-6.</summary>
    public static Tensor<T> PlainLayerNorm<T>(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int dim = x.Shape[1];
        var mean = engine.TensorTile(engine.ReduceMean(x, new[] { 1 }, keepDims: true), new[] { 1, dim });
        var centered = engine.TensorSubtract(x, mean);
        var variance = engine.ReduceMean(engine.TensorMultiply(centered, centered), new[] { 1 }, keepDims: true);
        var inv = engine.TensorTile(engine.TensorPow(engine.TensorAddScalar(variance, ops.FromDouble(1e-6)), ops.FromDouble(-0.5)), new[] { 1, dim });
        return engine.TensorMultiply(centered, inv);
    }

    /// <summary>x(1 + scale) + shift with <c>[1, dim]</c> modulation rows broadcast over the frames.</summary>
    public static Tensor<T> Modulate<T>(IEngine engine, Tensor<T> x, Tensor<T> scale, Tensor<T> shift)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = x.Shape[0];
        var s = engine.TensorTile(engine.TensorAddScalar(scale, ops.One), new[] { frames, 1 });
        return engine.TensorAdd(engine.TensorMultiply(x, s), engine.TensorTile(shift, new[] { frames, 1 }));
    }

    public static Tensor<T> Chunk<T>(IEngine engine, Tensor<T> row, int index, int dim)
        => engine.TensorSlice(row, new[] { 0, index * dim }, new[] { 1, dim });

    public static TLayer Own<T, TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    public static DenseLayer<T> Linear<T>(List<LayerBase<T>> layers, int outputs, bool zero = false)
        => Own(layers, new DenseLayer<T>(outputs, new IdentityActivation<T>() as IActivationFunction<T>,
            zero ? new AiDotNet.Initialization.ZeroInitializationStrategy<T>() : null));
}

/// <summary>Sinusoidal (256) of 1000 t → Linear → SiLU → Linear (TimestepEmbedding).</summary>
internal sealed class FlowTimestepEmbedding<T>
{
    private readonly IEngine _engine;
    private readonly DenseLayer<T> _first;
    private readonly DenseLayer<T> _second;

    public FlowTimestepEmbedding(IEngine engine, List<LayerBase<T>> layers, int dim)
    {
        _engine = engine;
        _first = FlowMatchingTts.Linear(layers, dim);
        _second = FlowMatchingTts.Linear(layers, dim);
    }

    public Tensor<T> Forward(double t) => _second.Forward(_engine.Swish(_first.Forward(FlowMatchingTts.Sinusoidal<T>(t, 256, 1000.0))));
}

/// <summary>
/// The character embedding (TextEmbedding): ids shifted by one so 0 is the filler token, truncated or filler-padded to
/// the frame count; with ConvNeXt layers (F5-TTS) the absolute position table is added and ConvNeXt V2 blocks refine it.
/// </summary>
internal sealed class FlowTextEmbedding<T>
{
    private readonly IEngine _engine;
    private readonly EmbeddingLayer<T> _embedding;
    private readonly List<ConvNeXtV2Block<T>> _blocks = new();
    private readonly int _dim;

    public FlowTextEmbedding(IEngine engine, List<LayerBase<T>> layers, int vocabulary, int dim, int convLayers, int convMultiplier)
    {
        _engine = engine;
        _dim = dim;
        _embedding = FlowMatchingTts.Own(layers, new EmbeddingLayer<T>(vocabulary + 1, dim));
        for (int i = 0; i < convLayers; i++)
            _blocks.Add(FlowMatchingTts.Own(layers, new ConvNeXtV2Block<T>(dim, dim * convMultiplier)));
    }

    /// <summary>The character table (vocabulary + 1 rows, row 0 the filler).</summary>
    public EmbeddingLayer<T> Embedding => _embedding;

    /// <summary>Embeds <paramref name="tokens"/> stretched to <paramref name="frames"/>; all filler when dropped (CFG).</summary>
    public Tensor<T> Forward(Tensor<T> tokens, int frames, bool drop)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var ids = new Tensor<T>(new[] { frames });
        if (!drop)
            for (int i = 0; i < Math.Min(frames, tokens.Length); i++) ids[i] = ops.FromDouble(ops.ToDouble(tokens[i]) + 1);
        var x = _embedding.Forward(ids);                                               // [frames, dim]
        if (_blocks.Count == 0) return x;
        x = _engine.TensorAdd(x, FlowMatchingTts.TextPositions<T>(frames, _dim));
        var batched = _engine.Reshape(x, new[] { 1, frames, _dim });
        foreach (var block in _blocks) batched = block.Forward(batched);
        return _engine.Reshape(batched, new[] { frames, _dim });
    }
}

/// <summary>Linear on [x; cond; text] plus the convolutional position embedding (two k31 grouped convolutions with
/// Mish), added residually (InputEmbedding, ConvPositionEmbedding).</summary>
internal sealed class FlowInputEmbedding<T>
{
    private readonly IEngine _engine;
    private readonly DenseLayer<T> _projection;
    private readonly GroupedConv1DLayer<T> _conv1;
    private readonly GroupedConv1DLayer<T> _conv2;
    private readonly int _dim;

    public FlowInputEmbedding(IEngine engine, List<LayerBase<T>> layers, int dim, int kernelSize = 31, int groups = 16)
    {
        _engine = engine;
        _dim = dim;
        _projection = FlowMatchingTts.Linear(layers, dim);
        _conv1 = FlowMatchingTts.Own(layers, new GroupedConv1DLayer<T>(dim, dim, kernelSize, groups, kernelSize / 2));
        _conv2 = FlowMatchingTts.Own(layers, new GroupedConv1DLayer<T>(dim, dim, kernelSize, groups, kernelSize / 2));
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> cond, Tensor<T> text)
    {
        int frames = x.Shape[0];
        var h = _projection.Forward(_engine.TensorConcatenate(new[] { x, cond, text }, 1));
        var c = _engine.Reshape(_engine.TensorTranspose(h), new[] { 1, _dim, frames });
        c = _engine.Mish(_conv2.Forward(_engine.Mish(_conv1.Forward(c))));
        return _engine.TensorAdd(_engine.TensorTranspose(_engine.Reshape(c, new[] { _dim, frames })), h);
    }
}

/// <summary>Multi-head self-attention with biased q/k/v/out projections and rotary embedding on the first
/// <c>ropeHeads</c> heads (all when null), and dropout on the output (Attention + AttnProcessor).</summary>
internal sealed class FlowSelfAttention<T>
{
    private readonly IEngine _engine;
    private readonly DenseLayer<T> _query;
    private readonly DenseLayer<T> _key;
    private readonly DenseLayer<T> _value;
    private readonly DenseLayer<T> _out;
    private readonly DropoutLayer<T>? _dropout;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _ropeHeads;

    public FlowSelfAttention(IEngine engine, List<LayerBase<T>> layers, int dim, int heads, int headDim, double dropout, int? ropeHeads)
    {
        _engine = engine;
        _heads = heads;
        _headDim = headDim;
        _ropeHeads = ropeHeads ?? heads;
        _query = FlowMatchingTts.Linear(layers, heads * headDim);
        _key = FlowMatchingTts.Linear(layers, heads * headDim);
        _value = FlowMatchingTts.Linear(layers, heads * headDim);
        _out = FlowMatchingTts.Linear(layers, dim);
        if (dropout > 0) _dropout = FlowMatchingTts.Own(layers, new DropoutLayer<T>(dropout));
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = x.Shape[0];
        var q = _query.Forward(x);
        var k = _key.Forward(x);
        var v = _value.Forward(x);
        var outputs = new Tensor<T>[_heads];
        for (int h = 0; h < _heads; h++)
        {
            var qh = _engine.TensorSlice(q, new[] { 0, h * _headDim }, new[] { frames, _headDim });
            var kh = _engine.TensorSlice(k, new[] { 0, h * _headDim }, new[] { frames, _headDim });
            var vh = _engine.TensorSlice(v, new[] { 0, h * _headDim }, new[] { frames, _headDim });
            if (h < _ropeHeads)
            {
                qh = FlowMatchingTts.Rotary(_engine, qh);
                kh = FlowMatchingTts.Rotary(_engine, kh);
            }
            var logits = _engine.TensorMultiplyScalar(_engine.TensorMatMul(qh, _engine.TensorTranspose(kh)), ops.FromDouble(1.0 / Math.Sqrt(_headDim)));
            outputs[h] = _engine.TensorMatMul(_engine.TensorSoftmax(logits, axis: 1), vh);
        }
        var y = _out.Forward(_heads == 1 ? outputs[0] : _engine.TensorConcatenate(outputs, 1));
        return _dropout is null ? y : _dropout.Forward(y);
    }
}

/// <summary>Linear → GELU (tanh) → dropout → Linear (FeedForward).</summary>
internal sealed class FlowFeedForward<T>
{
    private readonly IEngine _engine;
    private readonly DenseLayer<T> _in;
    private readonly DenseLayer<T> _out;
    private readonly DropoutLayer<T>? _dropout;

    public FlowFeedForward(IEngine engine, List<LayerBase<T>> layers, int dim, int inner, double dropout)
    {
        _engine = engine;
        _in = FlowMatchingTts.Linear(layers, inner);
        _out = FlowMatchingTts.Linear(layers, dim);
        if (dropout > 0) _dropout = FlowMatchingTts.Own(layers, new DropoutLayer<T>(dropout));
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var h = _engine.GELU(_in.Forward(x));
        if (_dropout is not null) h = _dropout.Forward(h);
        return _out.Forward(h);
    }
}

/// <summary>
/// The DiT block with adaLN-zero (DiTBlock): SiLU → Linear(dim → 6·dim) (zero-initialized) gives shift, scale and gate for
/// the attention and the feed-forward branches over plain LayerNorms; each branch's output is gated into the residual.
/// </summary>
internal sealed class FlowDiTBlock<T>
{
    private readonly IEngine _engine;
    private readonly DenseLayer<T> _modulation;
    private readonly FlowSelfAttention<T> _attention;
    private readonly FlowFeedForward<T> _feedForward;
    private readonly int _dim;

    public FlowDiTBlock(IEngine engine, List<LayerBase<T>> layers, int dim, int heads, int headDim, int ffInner, double dropout, int? ropeHeads)
    {
        _engine = engine;
        _dim = dim;
        _modulation = FlowMatchingTts.Linear(layers, 6 * dim, zero: true);
        _attention = new FlowSelfAttention<T>(engine, layers, dim, heads, headDim, dropout, ropeHeads);
        _feedForward = new FlowFeedForward<T>(engine, layers, dim, ffInner, dropout);
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> time)
    {
        int frames = x.Shape[0];
        var m = _modulation.Forward(_engine.Swish(time));                               // [1, 6 dim]
        Tensor<T> C(int i) => FlowMatchingTts.Chunk(_engine, m, i, _dim);
        var normed = FlowMatchingTts.Modulate(_engine, FlowMatchingTts.PlainLayerNorm(_engine, x), C(1), C(0));
        x = _engine.TensorAdd(x, _engine.TensorMultiply(_engine.TensorTile(C(2), new[] { frames, 1 }), _attention.Forward(normed)));
        normed = FlowMatchingTts.Modulate(_engine, FlowMatchingTts.PlainLayerNorm(_engine, x), C(4), C(3));
        return _engine.TensorAdd(x, _engine.TensorMultiply(_engine.TensorTile(C(5), new[] { frames, 1 }), _feedForward.Forward(normed)));
    }
}
