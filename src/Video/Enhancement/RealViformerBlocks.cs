using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// Shared pieces of RealViformer (Zhang and Yao 2024; Yuehan717/RealViformer, archs/realviformer_arch.py and
/// archs/arch_util.py), which builds on Restormer's transposed (channel) attention and gated feed-forward network.
/// </summary>
internal static class RealViformerBlocks
{
    /// <summary>nn.Conv2d with PyTorch's default initialization (the release never re-initializes its weights).</summary>
    internal static ConvolutionalLayer<T> Conv<T>(int outChannels, int kernel, bool bias, int groups = 1)
        => new(outChannels, kernel, 1, kernel / 2, null, new TorchDefaultInitializationStrategy<T>(), null, groups,
            bias ? BiasMode.Always : BiasMode.Never);

    /// <summary>A free learnable vector (an nn.Parameter) held as a bias-free linear map of the constant 1.</summary>
    internal static DenseLayer<T> Vector<T>(int count, double initial)
        => new(count, (IActivationFunction<T>?)null, new ConstantInitializationStrategy<T>(initial), BiasMode.Never);

    /// <summary>The vector a <see cref="Vector{T}"/> layer holds, [1, count].</summary>
    internal static Tensor<T> Read<T>(IReadOnlyList<ILayer<T>> layers, int index)
        => layers[index].Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { MathHelper.GetNumericOperations<T>().One })));

    /// <summary>
    /// BiasFree_LayerNorm over the channels of every pixel: x / sqrt(var + 1e-5) * weight, the variance about the mean
    /// but the input left uncentred.
    /// </summary>
    internal static Tensor<T> BiasFreeNorm<T>(IEngine engine, IReadOnlyList<ILayer<T>> layers, int weight, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var last = engine.TensorPermute(x, new[] { 0, 2, 3, 1 });
        var shape = last.Shape.ToArray();
        var mean = engine.TensorBroadcastTo(engine.ReduceMean(last, new[] { 3 }, true), shape);
        var variance = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(last, mean)), new[] { 3 }, true);
        var scale = engine.TensorBroadcastTo(engine.TensorSqrt(engine.TensorAddScalar(variance, ops.FromDouble(1e-5))), shape);
        var w = engine.TensorBroadcastTo(engine.Reshape(Read(layers, weight), new[] { 1, 1, 1, shape[3] }), shape);
        return engine.TensorPermute(engine.TensorMultiply(engine.TensorDivide(last, scale), w), new[] { 0, 3, 1, 2 });
    }

    /// <summary>F.normalize along the last axis: x / max(||x||, 1e-12).</summary>
    internal static Tensor<T> L2Normalize<T>(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        // max(||x||, 1e-12) as sqrt(max(sum x², 1e-24)): the clamp before the root keeps an all-zero row's gradient zero,
        // as PyTorch's norm backward defines it, where sqrt's infinite slope at 0 times the clamp's zero would be NaN.
        var squares = engine.ReduceSum(engine.TensorSquare(x), new[] { shape.Length - 1 }, true);
        var norm = engine.TensorSqrt(engine.TensorClamp(squares, ops.FromDouble(1e-24), ops.FromDouble(1e30)));
        return engine.TensorDivide(x, engine.TensorBroadcastTo(norm, shape));
    }

    /// <summary>
    /// The transposed attention both Attention and CrossChannelAttention compute: q and k L2-normalized over pixels, a
    /// [C, C] map per head scaled by the head's learnable temperature, softmax over keys, applied to v.
    /// </summary>
    /// <returns>The attention logits before softmax [B * heads, c, c] and the output [B, heads * c, H, W].</returns>
    internal static (Tensor<T> Logits, Tensor<T> Output) ChannelAttention<T>(IEngine engine, Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> temperature, int heads)
    {
        int b = q.Shape[0], c = q.Shape[1] / heads, h = q.Shape[2], w = q.Shape[3];
        Tensor<T> Heads(Tensor<T> t) => engine.Reshape(t, new[] { b * heads, c, h * w });
        var qh = L2Normalize(engine, Heads(q));
        var kh = L2Normalize(engine, Heads(k));
        var logits = engine.TensorBatchMatMul(qh, engine.TensorPermute(kh, new[] { 0, 2, 1 }));
        var temp = engine.TensorBroadcastTo(engine.Reshape(temperature, new[] { 1, heads, 1, 1 }), new[] { b, heads, c, c });
        logits = engine.TensorMultiply(logits, engine.Reshape(temp, new[] { b * heads, c, c }));
        var attn = engine.TensorSoftmax(logits, 2);
        return (logits, engine.Reshape(engine.TensorBatchMatMul(attn, Heads(v)), new[] { b, heads * c, h, w }));
    }

    /// <summary>
    /// SSIM loss, 1 - SSIM, with the usual 11x11 Gaussian window (sigma 1.5), zero padding of 5, C1 = 0.01^2 and
    /// C2 = 0.03^2, per channel and averaged over every pixel.
    /// </summary>
    /// <param name="engine">The engine, so the loss is recorded on the caller's tape.</param>
    /// <param name="x">[N, C, H, W] in [0, 1].</param>
    /// <param name="y">[N, C, H, W] in [0, 1].</param>
    internal static Tensor<T> SsimLoss<T>(IEngine engine, Tensor<T> x, Tensor<T> y)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        const int size = 11;
        const double sigma = 1.5;
        int c = x.Shape[1];
        var g = new double[size];
        for (int i = 0; i < size; i++) g[i] = Math.Exp(-Math.Pow(i - (size - 1) / 2.0, 2) / (2 * sigma * sigma));
        double total = g.Sum();
        var kernel = new Tensor<T>(new[] { c, 1, size, size });
        for (int ch = 0; ch < c; ch++)
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++) kernel[(ch * size + i) * size + j] = ops.FromDouble(g[i] * g[j] / (total * total));
        Tensor<T> Blur(Tensor<T> t) => engine.DepthwiseConv2D(t, kernel, new[] { 1, 1 }, new[] { size / 2, size / 2 });

        var muX = Blur(x);
        var muY = Blur(y);
        var muX2 = engine.TensorSquare(muX);
        var muY2 = engine.TensorSquare(muY);
        var muXY = engine.TensorMultiply(muX, muY);
        var sigmaX = engine.TensorSubtract(Blur(engine.TensorSquare(x)), muX2);
        var sigmaY = engine.TensorSubtract(Blur(engine.TensorSquare(y)), muY2);
        var sigmaXY = engine.TensorSubtract(Blur(engine.TensorMultiply(x, y)), muXY);
        var c1 = ops.FromDouble(0.01 * 0.01);
        var c2 = ops.FromDouble(0.03 * 0.03);
        var numerator = engine.TensorMultiply(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(muXY, ops.FromDouble(2)), c1),
            engine.TensorAddScalar(engine.TensorMultiplyScalar(sigmaXY, ops.FromDouble(2)), c2));
        var denominator = engine.TensorMultiply(
            engine.TensorAddScalar(engine.TensorAdd(muX2, muY2), c1),
            engine.TensorAddScalar(engine.TensorAdd(sigmaX, sigmaY), c2));
        var map = engine.TensorDivide(numerator, denominator);
        var ssim = engine.ReduceMean(map, Enumerable.Range(0, map.Rank).ToArray(), false);
        return engine.TensorAddScalar(engine.TensorNegate(ssim), ops.One);
    }
}

/// <summary>
/// RealViformer's Attention (the ICA): Restormer's transposed self-attention, optionally with the channel mask that
/// re-weights each output channel by a sigmoid of the row-wise max and mean of its attention logits.
/// </summary>
internal sealed class RealViformerAttention<T>
{
    private readonly int _temperature, _qkv, _dw, _out, _linear1 = -1, _linear2 = -1, _heads, _dim;

    internal RealViformerAttention(List<ILayer<T>> layers, int dim, int heads, bool bias, bool masked)
    {
        if (dim % heads != 0) throw new ArgumentException($"{dim} channels do not split into {heads} heads.", nameof(heads));
        _dim = dim;
        _heads = heads;
        _temperature = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(heads, 1.0));
        _qkv = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(3 * dim, 1, bias));
        _dw = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(3 * dim, 3, bias, groups: 3 * dim));
        _out = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(dim, 1, bias));
        if (masked)
        {
            _linear1 = layers.Count;
            layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>?)null, new TorchDefaultInitializationStrategy<T>(), bias ? BiasMode.Always : BiasMode.Never));
            _linear2 = layers.Count;
            layers.Add(new DenseLayer<T>(dim / heads, (IActivationFunction<T>?)null, new TorchDefaultInitializationStrategy<T>(), bias ? BiasMode.Always : BiasMode.Never));
        }
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x)
    {
        int b = x.Shape[0], h = x.Shape[2], w = x.Shape[3], c = _dim / _heads;
        var qkv = layers[_dw].Forward(layers[_qkv].Forward(x));
        var (logits, output) = RealViformerBlocks.ChannelAttention(engine,
            engine.TensorNarrow(qkv, 1, 0, _dim), engine.TensorNarrow(qkv, 1, _dim, _dim), engine.TensorNarrow(qkv, 1, 2 * _dim, _dim),
            RealViformerBlocks.Read(layers, _temperature), _heads);
        if (_linear1 >= 0)
        {
            // dsc = [max, mean] over each logit row -> linear1 -> GELU -> linear2 across the head's channels -> sigmoid.
            var max = engine.ReduceMax(logits, new[] { 2 }, true);
            var mean = engine.ReduceMean(logits, new[] { 2 }, true);
            var dsc = engine.TensorConcatenate(new[] { max, mean }, 2);
            var mask = layers[_linear1].Forward(engine.Reshape(dsc, new[] { b * _heads * c, 2 }));
            mask = engine.Reshape(engine.GELU(mask), new[] { b * _heads, c });
            mask = engine.Sigmoid(layers[_linear2].Forward(mask));
            var gate = engine.TensorBroadcastTo(engine.Reshape(mask, new[] { b, _dim, 1 }), new[] { b, _dim, h * w });
            output = engine.Reshape(engine.TensorMultiply(engine.Reshape(output, new[] { b, _dim, h * w }), gate), new[] { b, _dim, h, w });
        }
        return layers[_out].Forward(output);
    }
}

/// <summary>Restormer's GDFN: a 1x1 expansion to two branches, 3x3 depthwise, GELU(x1) * x2, a 1x1 projection.</summary>
internal sealed class RealViformerFeedForward<T>
{
    private readonly int _in, _dw, _out, _hidden;

    internal RealViformerFeedForward(List<ILayer<T>> layers, int dim, double expansion, bool bias)
    {
        _hidden = (int)(dim * expansion);
        _in = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(2 * _hidden, 1, bias));
        _dw = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(2 * _hidden, 3, bias, groups: 2 * _hidden));
        _out = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(dim, 1, bias));
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x)
    {
        var y = layers[_dw].Forward(layers[_in].Forward(x));
        var gated = engine.TensorMultiply(engine.GELU(engine.TensorNarrow(y, 1, 0, _hidden)), engine.TensorNarrow(y, 1, _hidden, _hidden));
        return layers[_out].Forward(gated);
    }
}

/// <summary>
/// RealViformer's TransformerBlock: x + attn(norm1(x)) and x + ffn(norm2(x)); with channel compression the attention
/// runs on a 3x3-squeezed copy (C / r channels) and a 3x3 convolution expands it back.
/// </summary>
internal sealed class RealViformerTransformerBlock<T>
{
    private readonly int _norm1, _norm2, _compress = -1, _expand = -1;
    private readonly RealViformerAttention<T> _attention;
    private readonly RealViformerFeedForward<T> _ffn;

    internal RealViformerTransformerBlock(List<ILayer<T>> layers, int dim, int heads, double expansion, bool bias, bool masked,
        bool channelCompress, int squeeze)
    {
        int attentionDim = channelCompress ? dim / squeeze : dim;
        _norm1 = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(attentionDim, 1.0));
        _attention = new RealViformerAttention<T>(layers, attentionDim, heads, bias, masked);
        _norm2 = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(dim, 1.0));
        _ffn = new RealViformerFeedForward<T>(layers, dim, expansion, bias);
        if (channelCompress && squeeze > 1)
        {
            _compress = layers.Count;
            layers.Add(RealViformerBlocks.Conv<T>(attentionDim, 3, bias));
            _expand = layers.Count;
            layers.Add(RealViformerBlocks.Conv<T>(dim, 3, bias));
        }
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x)
    {
        if (_compress >= 0)
        {
            var squeezed = layers[_compress].Forward(x);
            var attended = _attention.Forward(engine, layers, RealViformerBlocks.BiasFreeNorm(engine, layers, _norm1, squeezed));
            x = engine.TensorAdd(x, layers[_expand].Forward(attended));
        }
        else
        {
            x = engine.TensorAdd(x, _attention.Forward(engine, layers, RealViformerBlocks.BiasFreeNorm(engine, layers, _norm1, x)));
        }
        return engine.TensorAdd(x, _ffn.Forward(engine, layers, RealViformerBlocks.BiasFreeNorm(engine, layers, _norm2, x)));
    }
}

/// <summary>
/// RealViformer's CAF (ChannelAttentionBlock with CrossChannelAttention, no compression): the current features query the
/// warped hidden state with transposed cross-attention, and [norm_q(f), norm_out(attention)] is fused by a 1x1, a 3x3
/// depthwise, GELU and a 1x1 convolution. There is no residual connection.
/// </summary>
internal sealed class RealViformerMerge<T>
{
    private readonly int _normQ, _normKv, _temperature, _kv, _kvDw, _q, _out, _normOut, _ffn1, _ffn2, _ffn3, _heads, _dim;

    internal RealViformerMerge(List<ILayer<T>> layers, int dim, int heads, bool bias)
    {
        if (dim % heads != 0) throw new ArgumentException($"{dim} channels do not split into {heads} heads.", nameof(heads));
        _dim = dim;
        _heads = heads;
        _normQ = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(dim, 1.0));
        _normKv = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(dim, 1.0));
        _temperature = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(heads, 1.0));
        _kv = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(2 * dim, 1, bias));
        _kvDw = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(2 * dim, 3, bias, groups: 2 * dim));
        _q = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(dim, 3, bias));
        _out = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(dim, 1, bias));
        _normOut = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(dim, 1.0));
        _ffn1 = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(dim, 1, bias));
        _ffn2 = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(dim, 3, bias, groups: dim));
        _ffn3 = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(dim, 1, bias));
    }

    /// <param name="current">[B, C, H, W], the frame's shallow features (the query).</param>
    /// <param name="hidden">[B, C, H, W], the warped hidden state (keys and values).</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> current, Tensor<T> hidden)
    {
        var y = RealViformerBlocks.BiasFreeNorm(engine, layers, _normKv, hidden);
        var x = RealViformerBlocks.BiasFreeNorm(engine, layers, _normQ, current);
        var kv = layers[_kvDw].Forward(layers[_kv].Forward(y));
        var (_, attended) = RealViformerBlocks.ChannelAttention(engine, layers[_q].Forward(x),
            engine.TensorNarrow(kv, 1, 0, _dim), engine.TensorNarrow(kv, 1, _dim, _dim), RealViformerBlocks.Read(layers, _temperature), _heads);
        var v = layers[_out].Forward(attended);
        var fused = engine.TensorConcatenate(new[] { x, RealViformerBlocks.BiasFreeNorm(engine, layers, _normOut, v) }, 1);
        return layers[_ffn3].Forward(engine.GELU(layers[_ffn2].Forward(layers[_ffn1].Forward(fused))));
    }
}
