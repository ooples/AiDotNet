using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// Pieces of Stable Video Diffusion's denoiser as diffusers builds it (<c>UNetSpatioTemporalConditionModel</c> and its
/// blocks in unets/unet_3d_blocks.py, resnet.py, transformers/transformer_temporal.py and attention.py). Frames travel
/// as [B·F, C, H, W] with the clip length passed alongside, exactly as the release flattens them.
/// </summary>
internal static class SvdBlocks
{
    internal static GroupNormalizationLayer<T> GroupNorm<T>(int channels, double epsilon) => new(32, channels, epsilon);

    /// <summary>diffusers <c>Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)</c>: [cos, sin] of t·10000^(-i/half).</summary>
    internal static Tensor<T> Sinusoid<T>(IReadOnlyList<double> timesteps, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int half = dim / 2;
        var emb = new Tensor<T>(new[] { timesteps.Count, dim });
        for (int b = 0; b < timesteps.Count; b++)
            for (int i = 0; i < half; i++)
            {
                double angle = timesteps[b] * Math.Exp(-Math.Log(10000.0) * i / half);
                emb[b * dim + i] = ops.FromDouble(Math.Cos(angle));
                emb[b * dim + half + i] = ops.FromDouble(Math.Sin(angle));
            }
        return emb;
    }

    /// <summary>Repeats each row of [B, ...] <paramref name="times"/> times in place: torch.repeat_interleave(dim=0).</summary>
    internal static Tensor<T> RepeatInterleave<T>(IEngine engine, Tensor<T> x, int times)
    {
        var shape = x.Shape.ToArray();
        var expanded = new[] { shape[0], 1 }.Concat(shape.Skip(1)).ToArray();
        var target = new[] { shape[0], times }.Concat(shape.Skip(1)).ToArray();
        var flat = new[] { shape[0] * times }.Concat(shape.Skip(1)).ToArray();
        return engine.Reshape(engine.TensorBroadcastTo(engine.Reshape(x, expanded), target), flat);
    }

    // Adds a [N, C] vector to every position of [N, C, ...].
    internal static Tensor<T> AddPerChannel<T>(IEngine engine, Tensor<T> x, Tensor<T> v)
    {
        var shape = x.Shape.ToArray();
        var view = new int[shape.Length];
        for (int i = 0; i < view.Length; i++) view[i] = i < 2 ? shape[i] : 1;
        return engine.TensorAdd(x, engine.TensorBroadcastTo(engine.Reshape(v, view), shape));
    }
}

/// <summary>diffusers <c>TimestepEmbedding</c>: Linear, SiLU, Linear.</summary>
internal sealed class SvdTimestepEmbedding<T>
{
    private readonly int _linear1, _linear2;

    internal SvdTimestepEmbedding(List<ILayer<T>> layers, int timeEmbedDim, int outDim)
    {
        _linear1 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(timeEmbedDim, new SiLUActivation<T>()));
        _linear2 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(outDim));
    }

    internal Tensor<T> Forward(IReadOnlyList<ILayer<T>> layers, Tensor<T> x) => layers[_linear2].Forward(layers[_linear1].Forward(x));
}

/// <summary>
/// diffusers <c>AlphaBlender(merge_strategy="learned_with_images")</c> on video (image_only_indicator all zero):
/// α = sigmoid(mix_factor), 1 − α when the spatial-to-temporal switch is set, and α·spatial + (1 − α)·temporal.
/// </summary>
internal sealed class SvdAlphaBlender<T>
{
    private readonly int _mix;
    private readonly bool _switch;

    internal SvdAlphaBlender(List<ILayer<T>> layers, double alpha, bool switchSpatialToTemporal)
    {
        _switch = switchSpatialToTemporal;
        _mix = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(1, alpha));
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> spatial, Tensor<T> temporal)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var alpha = engine.Sigmoid(RealViformerBlocks.Read(layers, _mix));
        if (_switch) alpha = engine.TensorAddScalar(engine.TensorNegate(alpha), ops.One);
        var shape = spatial.Shape.ToArray();
        var a = engine.TensorBroadcastTo(engine.Reshape(alpha, Enumerable.Repeat(1, shape.Length).ToArray()), shape);
        return engine.TensorAdd(temporal, engine.TensorMultiply(a, engine.TensorSubtract(spatial, temporal)));
    }
}

/// <summary>
/// diffusers <c>ResnetBlock2D</c> with a time embedding: GroupNorm, SiLU, 3x3 conv, the SiLU-projected embedding added per
/// channel, GroupNorm, SiLU, 3x3 conv, plus the input (through a 1x1 conv when the width changes).
/// </summary>
internal sealed class SvdResnetBlock<T>
{
    private readonly int _norm1, _conv1, _temb, _norm2, _conv2, _shortcut;

    internal SvdResnetBlock(List<ILayer<T>> layers, int inChannels, int outChannels, double epsilon)
    {
        _norm1 = layers.Count;
        layers.Add(SvdBlocks.GroupNorm<T>(inChannels, epsilon));
        _conv1 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(outChannels, 3, null));
        _temb = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(outChannels));
        _norm2 = layers.Count;
        layers.Add(SvdBlocks.GroupNorm<T>(outChannels, epsilon));
        _conv2 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(outChannels, 3, null));
        _shortcut = -1;
        if (inChannels != outChannels)
        {
            _shortcut = layers.Count;
            layers.Add(BasicVsrBlocks.Conv<T>(outChannels, 1, null));
        }
    }

    /// <param name="x">[N, C, H, W].</param>
    /// <param name="temb">[N, D].</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> temb)
    {
        var h = layers[_conv1].Forward(engine.Swish(layers[_norm1].Forward(x)));
        h = SvdBlocks.AddPerChannel(engine, h, layers[_temb].Forward(engine.Swish(temb)));
        h = layers[_conv2].Forward(engine.Swish(layers[_norm2].Forward(h)));
        return engine.TensorAdd(_shortcut < 0 ? x : layers[_shortcut].Forward(x), h);
    }
}

/// <summary>
/// diffusers <c>TemporalResnetBlock</c> on [B, C, F, H, W]: the ResnetBlock2D pattern with (3, 1, 1) convolutions over
/// time and a per-frame embedding. Each time convolution is a Linear over the concatenated previous, current and next
/// frames (zero beyond the clip), which is Conv3d((3, 1, 1), padding (1, 0, 0)) term for term.
/// </summary>
internal sealed class SvdTemporalResnetBlock<T>
{
    private readonly int _norm1, _conv1, _temb, _norm2, _conv2;

    internal SvdTemporalResnetBlock(List<ILayer<T>> layers, int channels, double epsilon)
    {
        _norm1 = layers.Count;
        layers.Add(SvdBlocks.GroupNorm<T>(channels, epsilon));
        _conv1 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(channels));
        _temb = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(channels));
        _norm2 = layers.Count;
        layers.Add(SvdBlocks.GroupNorm<T>(channels, epsilon));
        _conv2 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(channels));
    }

    // GroupNorm over (C/g, F, H, W) of [B, C, F, H, W], run as [B, C, F·H, W].
    private static Tensor<T> Norm(IEngine engine, IReadOnlyList<ILayer<T>> layers, int index, Tensor<T> x)
    {
        var s = x.Shape.ToArray();
        return engine.Reshape(layers[index].Forward(engine.Reshape(x, new[] { s[0], s[1], s[2] * s[3], s[4] })), s);
    }

    private static Tensor<T> TimeConv(IEngine engine, IReadOnlyList<ILayer<T>> layers, int index, Tensor<T> x)
    {
        int b = x.Shape[0], c = x.Shape[1], f = x.Shape[2], h = x.Shape[3], w = x.Shape[4];
        var last = engine.TensorPermute(x, new[] { 0, 3, 4, 2, 1 });
        var zero = new Tensor<T>(new[] { b, h, w, 1, c });
        var previous = f > 1 ? engine.TensorConcatenate(new[] { zero, engine.TensorNarrow(last, 3, 0, f - 1) }, 3) : zero;
        var next = f > 1 ? engine.TensorConcatenate(new[] { engine.TensorNarrow(last, 3, 1, f - 1), zero }, 3) : zero;
        var stacked = engine.TensorConcatenate(new[] { previous, last, next }, 4);
        var y = layers[index].Forward(engine.Reshape(stacked, new[] { b * h * w * f, 3 * c }));
        return engine.TensorPermute(engine.Reshape(y, new[] { b, h, w, f, y.Shape[1] }), new[] { 0, 4, 3, 1, 2 });
    }

    /// <param name="x">[B, C, F, H, W].</param>
    /// <param name="temb">[B, F, D].</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> temb)
    {
        int b = x.Shape[0], c = x.Shape[1], f = x.Shape[2];
        var h = TimeConv(engine, layers, _conv1, engine.Swish(Norm(engine, layers, _norm1, x)));
        var t = engine.Reshape(layers[_temb].Forward(engine.Swish(engine.Reshape(temb, new[] { b * f, temb.Shape[2] }))), new[] { b, f, c });
        h = engine.TensorAdd(h, engine.TensorBroadcastTo(engine.Reshape(engine.TensorPermute(t, new[] { 0, 2, 1 }), new[] { b, c, f, 1, 1 }),
            h.Shape.ToArray()));
        h = TimeConv(engine, layers, _conv2, engine.Swish(Norm(engine, layers, _norm2, h)));
        return engine.TensorAdd(x, h);
    }
}

/// <summary>
/// diffusers <c>SpatioTemporalResBlock</c>: a ResnetBlock2D per frame, a TemporalResnetBlock across frames and their
/// learned blend (switch_spatial_to_temporal_mix).
/// </summary>
internal sealed class SvdSpatioTemporalResBlock<T>
{
    private readonly SvdResnetBlock<T> _spatial;
    private readonly SvdTemporalResnetBlock<T> _temporal;
    private readonly SvdAlphaBlender<T> _mixer;

    internal SvdSpatioTemporalResBlock(List<ILayer<T>> layers, int inChannels, int outChannels, double epsilon)
    {
        _spatial = new SvdResnetBlock<T>(layers, inChannels, outChannels, epsilon);
        _temporal = new SvdTemporalResnetBlock<T>(layers, outChannels, epsilon);
        _mixer = new SvdAlphaBlender<T>(layers, 0.5, true);
    }

    /// <param name="x">[B·F, C, H, W].</param>
    /// <param name="temb">[B·F, D].</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> temb, int frames)
    {
        var h = _spatial.Forward(engine, layers, x, temb);
        int bf = h.Shape[0], c = h.Shape[1], height = h.Shape[2], width = h.Shape[3], b = bf / frames;
        var video = engine.TensorPermute(engine.Reshape(h, new[] { b, frames, c, height, width }), new[] { 0, 2, 1, 3, 4 });
        var mixed = _temporal.Forward(engine, layers, video, engine.Reshape(temb, new[] { b, frames, temb.Shape[1] }));
        var blended = _mixer.Forward(engine, layers, video, mixed);
        return engine.Reshape(engine.TensorPermute(blended, new[] { 0, 2, 1, 3, 4 }), new[] { bf, c, height, width });
    }
}

/// <summary>diffusers <c>Attention</c>: q/k/v projections (bias-free unless asked), scaled dot-product attention per head, a biased output.</summary>
internal sealed class SvdAttention<T>
{
    private readonly int _q, _k, _v, _out, _heads;

    /// <param name="biasedProjections">Whether q, k and v carry biases (CLIP's attention does; diffusers' does not).</param>
    internal SvdAttention(List<ILayer<T>> layers, int dim, int heads, int headDim, bool biasedProjections = false)
    {
        _heads = heads;
        _q = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(heads * headDim, bias: biasedProjections));
        _k = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(heads * headDim, bias: biasedProjections));
        _v = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(heads * headDim, bias: biasedProjections));
        _out = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
    }

    /// <param name="x">[S, L, D] queries.</param>
    /// <param name="context">[S, M, Dc] keys and values, or null for self-attention.</param>
    /// <param name="captureProbabilities">Receives the softmax map [S·heads, L, M], or null.</param>
    /// <param name="probabilities">A softmax map to use in place of this attention's own (q and k then go unused), or null.</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T>? context = null,
        Action<Tensor<T>>? captureProbabilities = null, Tensor<T>? probabilities = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var kv = context ?? x;
        int s = x.Shape[0], l = x.Shape[1], m = kv.Shape[1];
        var v = layers[_v].Forward(kv);
        int inner = v.Shape[2], hd = inner / _heads;
        Tensor<T> Heads(Tensor<T> t, int length) =>
            engine.Reshape(engine.TensorPermute(engine.Reshape(t, new[] { s, length, _heads, hd }), new[] { 0, 2, 1, 3 }), new[] { s * _heads, length, hd });
        var vh = Heads(v, m);
        var attn = probabilities;
        if (attn is null)
        {
            var qh = engine.TensorMultiplyScalar(Heads(layers[_q].Forward(x), l), ops.FromDouble(Math.Pow(hd, -0.5)));
            var kh = Heads(layers[_k].Forward(kv), m);
            attn = engine.TensorSoftmax(engine.TensorBatchMatMul(qh, engine.TensorPermute(kh, new[] { 0, 2, 1 })), 2);
            captureProbabilities?.Invoke(attn);
        }
        var merged = engine.Reshape(engine.TensorPermute(engine.Reshape(engine.TensorBatchMatMul(attn, vh), new[] { s, _heads, l, hd }),
            new[] { 0, 2, 1, 3 }), new[] { s, l, inner });
        return layers[_out].Forward(merged);
    }
}

/// <summary>diffusers <c>FeedForward(activation_fn="geglu")</c>: Linear to twice the hidden width, value·GELU(gate), Linear.</summary>
internal sealed class SvdGegluFeedForward<T>
{
    private readonly int _proj, _out, _hidden;

    internal SvdGegluFeedForward(List<ILayer<T>> layers, int dim, int outDim, int mult = 4)
    {
        _hidden = dim * mult;
        _proj = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(2 * _hidden));
        _out = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(outDim));
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x)
    {
        var projected = layers[_proj].Forward(x);
        int axis = projected.Rank - 1;
        var value = engine.TensorNarrow(projected, axis, 0, _hidden);
        // GEGLU's F.gelu is the erf form; the library's GELU is the tanh approximation (|difference| < 1e-3).
        var gate = engine.GELU(engine.TensorNarrow(projected, axis, _hidden, _hidden));
        return layers[_out].Forward(engine.TensorMultiply(value, gate));
    }
}

/// <summary>diffusers <c>BasicTransformerBlock</c> (layer norms): self-attention, cross-attention, GEGLU feed-forward.</summary>
internal sealed class SvdBasicTransformerBlock<T>
{
    private readonly int _norm1, _norm2, _norm3;
    private readonly SvdAttention<T> _attn1, _attn2;
    private readonly SvdGegluFeedForward<T> _ff;

    internal SvdBasicTransformerBlock(List<ILayer<T>> layers, int dim, int heads, int headDim)
    {
        _norm1 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _attn1 = new SvdAttention<T>(layers, dim, heads, headDim);
        _norm2 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _attn2 = new SvdAttention<T>(layers, dim, heads, headDim);
        _norm3 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _ff = new SvdGegluFeedForward<T>(layers, dim, dim);
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> context)
    {
        x = engine.TensorAdd(x, _attn1.Forward(engine, layers, layers[_norm1].Forward(x)));
        x = engine.TensorAdd(x, _attn2.Forward(engine, layers, layers[_norm2].Forward(x), context));
        return engine.TensorAdd(x, _ff.Forward(engine, layers, layers[_norm3].Forward(x)));
    }
}

/// <summary>
/// diffusers <c>TemporalBasicTransformerBlock</c> over each position's frames: a residual GEGLU input feed-forward, self-
/// attention across frames, cross-attention to the clip's first-frame context, and a residual GEGLU feed-forward.
/// </summary>
internal sealed class SvdTemporalTransformerBlock<T>
{
    private readonly int _normIn, _norm1, _norm2, _norm3;
    private readonly SvdGegluFeedForward<T> _ffIn, _ff;
    private readonly SvdAttention<T> _attn1, _attn2;

    internal SvdTemporalTransformerBlock(List<ILayer<T>> layers, int dim, int heads, int headDim)
    {
        _normIn = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _ffIn = new SvdGegluFeedForward<T>(layers, dim, dim);
        _norm1 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _attn1 = new SvdAttention<T>(layers, dim, heads, headDim);
        _norm2 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _attn2 = new SvdAttention<T>(layers, dim, heads, headDim);
        _norm3 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _ff = new SvdGegluFeedForward<T>(layers, dim, dim);
    }

    /// <summary>The temporal self-attention, which DAM-VSR's backward pass re-weights.</summary>
    internal SvdAttention<T> TemporalSelfAttention => _attn1;

    /// <param name="x">[B·F, L, D].</param>
    /// <param name="timeContext">[B·L, 1, Dc].</param>
    /// <param name="selfAttention">A replacement for the temporal self-attention ([B·L, F, D] to [B·L, F, D]), or null.</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, int frames, Tensor<T> timeContext,
        Func<Tensor<T>, Tensor<T>>? selfAttention = null)
    {
        int bf = x.Shape[0], l = x.Shape[1], d = x.Shape[2], b = bf / frames;
        var h = engine.Reshape(engine.TensorPermute(engine.Reshape(x, new[] { b, frames, l, d }), new[] { 0, 2, 1, 3 }), new[] { b * l, frames, d });
        h = engine.TensorAdd(h, _ffIn.Forward(engine, layers, layers[_normIn].Forward(h)));
        var normed = layers[_norm1].Forward(h);
        h = engine.TensorAdd(h, selfAttention is null ? _attn1.Forward(engine, layers, normed) : selfAttention(normed));
        h = engine.TensorAdd(h, _attn2.Forward(engine, layers, layers[_norm2].Forward(h), timeContext));
        h = engine.TensorAdd(h, _ff.Forward(engine, layers, layers[_norm3].Forward(h)));
        return engine.Reshape(engine.TensorPermute(engine.Reshape(h, new[] { b, l, frames, d }), new[] { 0, 2, 1, 3 }), new[] { bf, l, d });
    }
}
