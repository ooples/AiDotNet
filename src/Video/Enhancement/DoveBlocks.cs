using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// CogVideoX's 3D rotary position embedding as diffusers' <c>get_3d_rotary_pos_embed(grid_type="slice", use_real=True)</c>
/// builds it for CogVideoX 1.5: a quarter of each head for time and three eighths each for height and width, integer
/// positions, θ = 10000, each frequency repeated for the two halves of a rotated pair.
/// </summary>
internal static class CogVideoXRope
{
    /// <summary>The cosines and sines [1, T·H·W, headDim] for a (frames, height, width) token grid in that order.</summary>
    internal static (Tensor<T> Cos, Tensor<T> Sin) Build<T>(int headDim, int frames, int height, int width)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int dt = headDim / 4, dh = headDim / 8 * 3;
        int tokens = frames * height * width;
        var cos = new Tensor<T>(new[] { 1, tokens, headDim });
        var sin = new Tensor<T>(new[] { 1, tokens, headDim });
        for (int t = 0; t < frames; t++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    int row = ((t * height + h) * width + w) * headDim;
                    int offset = 0;
                    foreach (var (position, dim) in new[] { (t, dt), (h, dh), (w, dh) })
                    {
                        for (int j = 0; j < dim / 2; j++)
                        {
                            double angle = position / Math.Pow(10000.0, 2.0 * j / dim);
                            for (int r = 0; r < 2; r++)
                            {
                                cos[row + offset + 2 * j + r] = ops.FromDouble(Math.Cos(angle));
                                sin[row + offset + 2 * j + r] = ops.FromDouble(Math.Sin(angle));
                            }
                        }
                        offset += dim;
                    }
                }
        return (cos, sin);
    }

    /// <summary>diffusers <c>apply_rotary_emb(use_real_unbind_dim=-1)</c> on [S, N, headDim]: adjacent pairs rotate.</summary>
    internal static Tensor<T> Apply<T>(IEngine engine, Tensor<T> x, Tensor<T> cos, Tensor<T> sin)
    {
        var shape = x.Shape.ToArray();
        int hd = shape[2];
        var pairs = engine.Reshape(x, new[] { shape[0], shape[1], hd / 2, 2 });
        var real = engine.TensorNarrow(pairs, 3, 0, 1);
        var imaginary = engine.TensorNarrow(pairs, 3, 1, 1);
        var rotated = engine.Reshape(engine.TensorConcatenate(new[] { engine.TensorNegate(imaginary), real }, 3), shape);
        return engine.TensorAdd(engine.TensorMultiply(x, engine.TensorBroadcastTo(cos, shape)),
            engine.TensorMultiply(rotated, engine.TensorBroadcastTo(sin, shape)));
    }
}

/// <summary>
/// diffusers' <c>Attention</c> with <c>CogVideoXAttnProcessor2_0</c>: joint attention over the text tokens followed by the
/// video tokens, biased projections, a LayerNorm on each head's queries and keys (ε 1e-6), and the rotary embedding on the
/// video tokens' queries and keys only.
/// </summary>
internal sealed class CogVideoXAttention<T>
{
    private readonly int _q, _k, _v, _normQ, _normK, _out, _heads, _headDim;

    internal CogVideoXAttention(List<ILayer<T>> layers, int dim, int heads, int headDim)
    {
        _heads = heads;
        _headDim = headDim;
        _q = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(heads * headDim));
        _k = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(heads * headDim));
        _v = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(heads * headDim));
        _normQ = layers.Count;
        layers.Add(new LayerNormalizationLayer<T>(headDim, 1e-6));
        _normK = layers.Count;
        layers.Add(new LayerNormalizationLayer<T>(headDim, 1e-6));
        _out = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
    }

    /// <param name="x">[B, M + N, D]: the text tokens then the video tokens.</param>
    /// <param name="cos">The rotary cosines [1, N, headDim], with <paramref name="sin"/>.</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, int textLength, Tensor<T> cos, Tensor<T> sin)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = x.Shape[0], l = x.Shape[1], video = l - textLength;
        Tensor<T> Heads(Tensor<T> t) => engine.Reshape(engine.TensorPermute(engine.Reshape(t, new[] { b, l, _heads, _headDim }), new[] { 0, 2, 1, 3 }),
            new[] { b * _heads, l, _headDim });
        Tensor<T> Rotate(Tensor<T> t) => engine.TensorConcatenate(new[]
        {
            engine.TensorNarrow(t, 1, 0, textLength),
            CogVideoXRope.Apply(engine, engine.TensorNarrow(t, 1, textLength, video), cos, sin),
        }, 1);
        var q = Rotate(layers[_normQ].Forward(Heads(layers[_q].Forward(x))));
        var k = Rotate(layers[_normK].Forward(Heads(layers[_k].Forward(x))));
        var v = Heads(layers[_v].Forward(x));
        var scores = engine.TensorMultiplyScalar(engine.TensorBatchMatMul(q, engine.TensorPermute(k, new[] { 0, 2, 1 })),
            ops.FromDouble(1.0 / Math.Sqrt(_headDim)));
        var merged = engine.TensorBatchMatMul(engine.TensorSoftmax(scores, 2), v);
        merged = engine.Reshape(engine.TensorPermute(engine.Reshape(merged, new[] { b, _heads, l, _headDim }), new[] { 0, 2, 1, 3 }),
            new[] { b, l, _heads * _headDim });
        return layers[_out].Forward(merged);
    }
}

/// <summary>
/// diffusers' <c>CogVideoXBlock</c>, the expert transformer block: <c>CogVideoXLayerNormZero</c> (one LayerNorm, separate
/// shift, scale and gate for the video and the text tokens from the time embedding), joint attention, then the same
/// modulation around a tanh-GELU feed-forward of four times the width applied to both token kinds.
/// </summary>
internal sealed class CogVideoXBlock<T>
{
    private readonly int _modulation1, _norm1, _modulation2, _norm2, _ff1, _ff2, _dim;
    private readonly CogVideoXAttention<T> _attention;

    internal CogVideoXBlock(List<ILayer<T>> layers, int dim, int heads, int headDim, double epsilon)
    {
        _dim = dim;
        _modulation1 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(6 * dim));
        _norm1 = layers.Count;
        layers.Add(new LayerNormalizationLayer<T>(dim, epsilon));
        _attention = new CogVideoXAttention<T>(layers, dim, heads, headDim);
        _modulation2 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(6 * dim));
        _norm2 = layers.Count;
        layers.Add(new LayerNormalizationLayer<T>(dim, epsilon));
        _ff1 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(4 * dim, new GELUActivation<T>()));
        _ff2 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
    }

    // The k-th of the six [B, 1, D] chunks of a modulation.
    private Tensor<T> Chunk(IEngine engine, Tensor<T> modulation, int k)
        => engine.Reshape(engine.TensorNarrow(modulation, 1, k * _dim, _dim), new[] { modulation.Shape[0], 1, _dim });

    private Tensor<T> Modulate(IEngine engine, Tensor<T> normed, Tensor<T> shift, Tensor<T> scale)
    {
        var shape = normed.Shape.ToArray();
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorAdd(engine.TensorMultiply(normed, engine.TensorBroadcastTo(engine.TensorAddScalar(scale, ops.One), shape)),
            engine.TensorBroadcastTo(shift, shape));
    }

    private Tensor<T> Gated(IEngine engine, Tensor<T> x, Tensor<T> gate, Tensor<T> update)
        => engine.TensorAdd(x, engine.TensorMultiply(update, engine.TensorBroadcastTo(gate, update.Shape.ToArray())));

    /// <param name="video">[B, N, D] video tokens.</param>
    /// <param name="text">[B, M, D] text tokens.</param>
    /// <param name="activatedTime">SiLU of the time embedding, [B, E].</param>
    internal (Tensor<T> Video, Tensor<T> Text) Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> video, Tensor<T> text,
        Tensor<T> activatedTime, Tensor<T> cos, Tensor<T> sin)
    {
        int m = text.Shape[1], n = video.Shape[1];
        var mod = layers[_modulation1].Forward(activatedTime);
        var normedVideo = Modulate(engine, layers[_norm1].Forward(video), Chunk(engine, mod, 0), Chunk(engine, mod, 1));
        var normedText = Modulate(engine, layers[_norm1].Forward(text), Chunk(engine, mod, 3), Chunk(engine, mod, 4));
        var attended = _attention.Forward(engine, layers, engine.TensorConcatenate(new[] { normedText, normedVideo }, 1), m, cos, sin);
        video = Gated(engine, video, Chunk(engine, mod, 2), engine.TensorNarrow(attended, 1, m, n));
        text = Gated(engine, text, Chunk(engine, mod, 5), engine.TensorNarrow(attended, 1, 0, m));

        mod = layers[_modulation2].Forward(activatedTime);
        normedVideo = Modulate(engine, layers[_norm2].Forward(video), Chunk(engine, mod, 0), Chunk(engine, mod, 1));
        normedText = Modulate(engine, layers[_norm2].Forward(text), Chunk(engine, mod, 3), Chunk(engine, mod, 4));
        var fed = layers[_ff2].Forward(layers[_ff1].Forward(engine.TensorConcatenate(new[] { normedText, normedVideo }, 1)));
        video = Gated(engine, video, Chunk(engine, mod, 2), engine.TensorNarrow(fed, 1, m, n));
        text = Gated(engine, text, Chunk(engine, mod, 5), engine.TensorNarrow(fed, 1, 0, m));
        return (video, text);
    }
}

/// <summary>
/// diffusers' <c>CogVideoXTransformer3DModel</c> in its CogVideoX 1.5 form: the text projected to the model width, the
/// latent cut into 2x2x2 (time, height, width) patches by a bias-free linear projection, no additive position embedding
/// (the rotary one is used), the sinusoidal timestep through <c>TimestepEmbedding</c>, the expert blocks, the final
/// LayerNorm, <c>AdaLayerNorm</c> (shift then scale from the time embedding) and the projection back to patches.
/// </summary>
internal sealed class CogVideoXTransformer<T>
{
    private readonly int _textProj, _patchProj, _normFinal, _normOutModulation, _normOut, _projOut;
    private readonly int _inner, _headDim, _patch, _patchT, _outChannels;
    private readonly SvdTimestepEmbedding<T> _timeEmbedding;
    private readonly CogVideoXBlock<T>[] _blocks;

    internal CogVideoXTransformer(List<ILayer<T>> layers, int outChannels, int heads, int headDim, int depth, int timeEmbedDim,
        int patch, int patchT, double epsilon)
    {
        _inner = heads * headDim;
        _headDim = headDim;
        _patch = patch;
        _patchT = patchT;
        _outChannels = outChannels;
        First = layers.Count;
        _textProj = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(_inner));
        _patchProj = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(_inner, bias: false));
        _timeEmbedding = new SvdTimestepEmbedding<T>(layers, timeEmbedDim, timeEmbedDim);
        _blocks = new CogVideoXBlock<T>[depth];
        for (int i = 0; i < depth; i++) _blocks[i] = new CogVideoXBlock<T>(layers, _inner, heads, headDim, epsilon);
        _normFinal = layers.Count;
        layers.Add(new LayerNormalizationLayer<T>(_inner, epsilon));
        _normOutModulation = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(2 * _inner));
        _normOut = layers.Count;
        layers.Add(new LayerNormalizationLayer<T>(_inner, epsilon));
        _projOut = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(patch * patch * patchT * outChannels));
        End = layers.Count;
    }

    internal int First { get; }

    internal int End { get; }

    /// <param name="latents">[B, F, C, H, W] with F a multiple of the temporal patch and H, W of the spatial one.</param>
    /// <param name="text">[B, M, textEmbedDim] text embeddings.</param>
    /// <returns>The model output [B, F, outChannels, H, W].</returns>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> latents, Tensor<T> text, IReadOnlyList<double> timesteps)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = latents.Shape[0], f = latents.Shape[1], c = latents.Shape[2], h = latents.Shape[3], w = latents.Shape[4];
        int pt = _patchT, p = _patch, gf = f / pt, gh = h / p, gw = w / p;
        var patches = engine.TensorPermute(latents, new[] { 0, 1, 3, 4, 2 });
        patches = engine.Reshape(patches, new[] { b, gf, pt, gh, p, gw, p, c });
        patches = engine.TensorPermute(patches, new[] { 0, 1, 3, 5, 7, 2, 4, 6 });
        var video = layers[_patchProj].Forward(engine.Reshape(patches, new[] { b, gf * gh * gw, c * pt * p * p }));
        var textTokens = layers[_textProj].Forward(text);

        var time = _timeEmbedding.Forward(layers, SvdBlocks.Sinusoid<T>(timesteps, _inner));
        var activated = engine.Swish(time);
        var (cos, sin) = CogVideoXRope.Build<T>(_headDim, gf, gh, gw);
        foreach (var block in _blocks) (video, textTokens) = block.Forward(engine, layers, video, textTokens, activated, cos, sin);
        video = layers[_normFinal].Forward(video);

        var mod = layers[_normOutModulation].Forward(activated);
        var shape = video.Shape.ToArray();
        var shift = engine.TensorBroadcastTo(engine.Reshape(engine.TensorNarrow(mod, 1, 0, _inner), new[] { b, 1, _inner }), shape);
        var scale = engine.TensorBroadcastTo(engine.Reshape(engine.TensorNarrow(mod, 1, _inner, _inner), new[] { b, 1, _inner }), shape);
        video = engine.TensorAdd(engine.TensorMultiply(layers[_normOut].Forward(video), engine.TensorAddScalar(scale, ops.One)), shift);

        var output = engine.Reshape(layers[_projOut].Forward(video), new[] { b, gf, gh, gw, _outChannels, pt, p, p });
        output = engine.TensorPermute(output, new[] { 0, 1, 5, 4, 2, 6, 3, 7 });
        return engine.Reshape(output, new[] { b, f, _outChannels, h, w });
    }
}

/// <summary>
/// DISTS (Ding et al. 2020; dingkeyan93/DISTS_pytorch): VGG16's five stages with L2 pooling in place of max pooling, and a
/// structure and a texture similarity per channel of the image and of each stage, weighted by the learned α and β
/// (1,475 each, normalized by their joint sum). The loss is 1 − the weighted similarity.
/// </summary>
internal sealed class DistsBlock<T>
{
    private static readonly int[] Widths = { 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512 };
    // The conv (0-based among the 13) that ends each of the five stages.
    private static readonly int[] StageEnds = { 1, 3, 6, 9, 12 };
    private static readonly int[] Channels = { 3, 64, 128, 256, 512, 512 };
    private static readonly double[] Mean = { 0.485, 0.456, 0.406 };
    private static readonly double[] Std = { 0.229, 0.224, 0.225 };
    private readonly int[] _convs = new int[13];
    private readonly int _alpha, _beta;

    internal DistsBlock(List<ILayer<T>> layers)
    {
        First = layers.Count;
        for (int i = 0; i < Widths.Length; i++)
        {
            _convs[i] = layers.Count;
            layers.Add(BasicVsrBlocks.Conv<T>(Widths[i], 3, null));
        }
        int total = Channels.Sum();
        // DISTS_pytorch initializes α and β from N(0.1, 0.01) before loading the trained values.
        _alpha = layers.Count;
        layers.Add(new DenseLayer<T>(total, (IActivationFunction<T>?)null, new NormalInitializationStrategy<T>(0.1, 0.01), AiDotNet.Enums.BiasMode.Never));
        _beta = layers.Count;
        layers.Add(new DenseLayer<T>(total, (IActivationFunction<T>?)null, new NormalInitializationStrategy<T>(0.1, 0.01), AiDotNet.Enums.BiasMode.Never));
        Count = layers.Count - First;
    }

    internal int First { get; }

    internal int Count { get; }

    // L2pooling: sqrt of the Hann-weighted (0.5, 1, 0.5)² / 4 average of x² at stride 2, padding 1, per channel.
    private static Tensor<T> L2Pool(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var kernel = new Tensor<T>(new[] { 1, 1, 3, 3 });
        var g = new[] { 0.5, 1.0, 0.5 };
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) kernel[i * 3 + j] = ops.FromDouble(g[i] * g[j] / 4.0);
        var squared = engine.Reshape(engine.TensorSquare(x), new[] { n * c, 1, h, w });
        var pooled = engine.Conv2D(squared, kernel, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 });
        pooled = engine.TensorSqrt(engine.TensorAddScalar(pooled, ops.FromDouble(1e-12)));
        return engine.Reshape(pooled, new[] { n, c, pooled.Shape[2], pooled.Shape[3] });
    }

    private List<Tensor<T>> Features(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> images)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var channels = new Tensor<T>[3];
        for (int c = 0; c < 3; c++)
            channels[c] = engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorNarrow(images, 1, c, 1), ops.FromDouble(1.0 / Std[c])),
                ops.FromDouble(-Mean[c] / Std[c]));
        var x = engine.TensorConcatenate(channels, 1);
        var features = new List<Tensor<T>> { x };
        int stage = 0;
        for (int i = 0; i < Widths.Length; i++)
        {
            if (stage > 0 && i == StageEnds[stage - 1] + 1) x = L2Pool(engine, x);
            x = engine.ReLU(BasicVsrBlocks.Run(layers, _convs[i], x));
            if (i == StageEnds[stage])
            {
                features.Add(x);
                stage++;
            }
        }
        return features;
    }

    /// <summary>The DISTS distance averaged over images [N, 3, H, W] in [0, 1].</summary>
    internal Tensor<T> Loss(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> output, Tensor<T> target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        List<Tensor<T>> targets;
        using (new NoGradScope<T>()) targets = Features(engine, layers, target);
        var outputs = Features(engine, layers, output);
        var alpha = RealViformerBlocks.Read(layers, _alpha);
        var beta = RealViformerBlocks.Read(layers, _beta);
        var weightSum = engine.TensorAdd(engine.ReduceSum(alpha, new[] { 0, 1 }, false), engine.ReduceSum(beta, new[] { 0, 1 }, false));
        var c = ops.FromDouble(1e-6);
        Tensor<T>? similarity = null;
        int offset = 0;
        for (int k = 0; k < outputs.Count; k++)
        {
            var x = outputs[k];
            var y = targets[k];
            int n = x.Shape[0], ch = x.Shape[1];
            var shape = new[] { n, ch };
            var xMean = engine.ReduceMean(x, new[] { 2, 3 }, true);
            var yMean = engine.ReduceMean(y, new[] { 2, 3 }, true);
            var s1 = engine.TensorDivide(engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorMultiply(xMean, yMean), ops.FromDouble(2.0)), c),
                engine.TensorAddScalar(engine.TensorAdd(engine.TensorSquare(xMean), engine.TensorSquare(yMean)), c));
            var xCentered = engine.TensorSubtract(x, engine.TensorBroadcastTo(xMean, x.Shape.ToArray()));
            var yCentered = engine.TensorSubtract(y, engine.TensorBroadcastTo(yMean, y.Shape.ToArray()));
            var xVar = engine.ReduceMean(engine.TensorSquare(xCentered), new[] { 2, 3 }, true);
            var yVar = engine.ReduceMean(engine.TensorSquare(yCentered), new[] { 2, 3 }, true);
            var cov = engine.TensorSubtract(engine.ReduceMean(engine.TensorMultiply(x, y), new[] { 2, 3 }, true), engine.TensorMultiply(xMean, yMean));
            var s2 = engine.TensorDivide(engine.TensorAddScalar(engine.TensorMultiplyScalar(cov, ops.FromDouble(2.0)), c),
                engine.TensorAddScalar(engine.TensorAdd(xVar, yVar), c));
            var a = engine.TensorBroadcastTo(engine.TensorNarrow(alpha, 1, offset, ch), shape);
            var bWeights = engine.TensorBroadcastTo(engine.TensorNarrow(beta, 1, offset, ch), shape);
            var term = engine.TensorAdd(engine.TensorMultiply(a, engine.Reshape(s1, shape)), engine.TensorMultiply(bWeights, engine.Reshape(s2, shape)));
            var summed = engine.ReduceSum(term, new[] { 1 }, false);
            similarity = similarity is null ? summed : engine.TensorAdd(similarity, summed);
            offset += ch;
        }
        var total = similarity ?? throw new InvalidOperationException("DISTS has no stages.");
        var normalized = engine.TensorDivide(total, engine.TensorBroadcastTo(engine.Reshape(weightSum, new[] { 1 }), total.Shape.ToArray()));
        return engine.TensorAddScalar(engine.TensorNegate(engine.ReduceMean(normalized, new[] { 0 }, false)), ops.One);
    }
}
