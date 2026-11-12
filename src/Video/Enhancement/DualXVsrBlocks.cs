using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// Rotary position embedding (Su et al., RoFormer) as DualX-VSR's Eq. 1-2 state it: channel pairs of q and k are rotated
/// by e^{i·u·θ_d}, θ_d = b^(-2d/|D|), b = 10000. A sequence that runs over two axes (height or width with time) gives
/// each axis half of the head's channels, the axial form of the rotation.
/// </summary>
internal static class DualXRope
{
    /// <summary>The cos and sin tables [L, headDim] for tokens whose positions on each axis are given per token.</summary>
    internal static (Tensor<T> Cos, Tensor<T> Sin) Tables<T>(int[][] positions, int headDim, double theta)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int length = positions.Length, axes = positions[0].Length, segment = headDim / axes, half = segment / 2;
        var cos = new Tensor<T>(new[] { length, headDim });
        var sin = new Tensor<T>(new[] { length, headDim });
        for (int l = 0; l < length; l++)
            for (int a = 0; a < axes; a++)
                for (int i = 0; i < half; i++)
                {
                    double angle = positions[l][a] * Math.Pow(theta, -2.0 * i / segment);
                    foreach (int c in new[] { a * segment + i, a * segment + half + i })
                    {
                        cos[l * headDim + c] = ops.FromDouble(Math.Cos(angle));
                        sin[l * headDim + c] = ops.FromDouble(Math.Sin(angle));
                    }
                }
        return (cos, sin);
    }

    /// <summary>x·cos + rotate_half(x)·sin over [S, L, headDim], rotating the halves of each axis's channels.</summary>
    internal static Tensor<T> Apply<T>(IEngine engine, Tensor<T> x, Tensor<T> cos, Tensor<T> sin, int axes)
    {
        int s = x.Shape[0], l = x.Shape[1], d = x.Shape[2], segment = d / axes, half = segment / 2;
        var pieces = new List<Tensor<T>>();
        for (int a = 0; a < axes; a++)
        {
            pieces.Add(engine.TensorNegate(engine.TensorNarrow(x, 2, a * segment + half, half)));
            pieces.Add(engine.TensorNarrow(x, 2, a * segment, half));
        }
        var rotated = engine.TensorConcatenate(pieces.ToArray(), 2);
        var shape = new[] { s, l, d };
        return engine.TensorAdd(
            engine.TensorMultiply(x, engine.TensorBroadcastTo(engine.Reshape(cos, new[] { 1, l, d }), shape)),
            engine.TensorMultiply(rotated, engine.TensorBroadcastTo(engine.Reshape(sin, new[] { 1, l, d }), shape)));
    }
}

/// <summary>
/// The ViViT transformer block DualX-VSR builds on (vit-pytorch's ViViT): pre-norm multi-head self-attention with
/// bias-free q/k/v projections to heads x dimHead channels, rotary positions on q and k, an output projection back to the
/// width, and a pre-norm GELU MLP, each residual.
/// </summary>
internal sealed class DualXTransformerBlock<T>
{
    private readonly int _norm1, _qkv, _out, _norm2, _fc1, _fc2, _heads, _headDim;

    internal DualXTransformerBlock(List<ILayer<T>> layers, int dim, int heads, int headDim, int mlpDim)
    {
        _heads = heads;
        _headDim = headDim;
        _norm1 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _qkv = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(3 * heads * headDim, bias: false));
        _out = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _norm2 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _fc1 = layers.Count;
        // nn.GELU is the erf form; the library's GELU is the tanh approximation (|difference| < 1e-3).
        layers.Add(RvrtBlocks.Linear<T>(mlpDim, new GELUActivation<T>()));
        _fc2 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
    }

    /// <param name="x">[S, L, D] sequences of L tokens.</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> cos, Tensor<T> sin, int axes)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int s = x.Shape[0], l = x.Shape[1], inner = _heads * _headDim;
        var qkv = engine.TensorPermute(engine.Reshape(layers[_qkv].Forward(layers[_norm1].Forward(x)), new[] { s, l, 3, _heads, _headDim }),
            new[] { 2, 0, 3, 1, 4 });
        Tensor<T> Part(int i) => engine.Reshape(engine.TensorNarrow(qkv, 0, i, 1), new[] { s * _heads, l, _headDim });
        var q = engine.TensorMultiplyScalar(DualXRope.Apply(engine, Part(0), cos, sin, axes), ops.FromDouble(Math.Pow(_headDim, -0.5)));
        var k = DualXRope.Apply(engine, Part(1), cos, sin, axes);
        var attn = engine.TensorSoftmax(engine.TensorBatchMatMul(q, engine.TensorPermute(k, new[] { 0, 2, 1 })), 2);
        var merged = engine.Reshape(engine.TensorPermute(engine.Reshape(engine.TensorBatchMatMul(attn, Part(2)), new[] { s, _heads, l, _headDim }),
            new[] { 0, 2, 1, 3 }), new[] { s, l, inner });
        x = engine.TensorAdd(x, layers[_out].Forward(merged));
        return engine.TensorAdd(x, layers[_fc2].Forward(layers[_fc1].Forward(layers[_norm2].Forward(x))));
    }
}

/// <summary>
/// DualX-VSR's network (Cao et al. 2025, Sec. 3-4): input video embedding (a per-frame 3x3 convolution, a temporal
/// transformer over each pixel's frames and 2x2 patch tokens), the dual axial spatial x temporal transformer (vertical-
/// temporal blocks, which attend over every height x time position of a column, then horizontal-temporal blocks over
/// every width x time position of a row), and the spatiotemporal reconstruction (tokens decoded back to pixels, a second
/// per-pixel temporal transformer and per-frame convolution and pixel-shuffle upsampling). There is no flow, alignment
/// or recurrence.
/// </summary>
internal sealed class DualXBackbone<T>
{
    private readonly int _convFirst, _patchNorm, _patchEmbed, _embedNorm, _embedFinal, _axialFinal, _decode, _reconFinal, _convHr, _convLast;
    private readonly int[] _upconv;
    private readonly DualXTransformerBlock<T>[] _embedTemporal, _vertical, _horizontal, _reconTemporal;
    private readonly int _features, _patch, _headDim;
    private readonly double _theta;

    internal DualXBackbone(List<ILayer<T>> layers, int features, int patch, int embedDim, int heads, int headDim, double mlpRatio,
        int embedTemporalDepth, int verticalDepth, int horizontalDepth, int reconTemporalDepth, int scale, double theta)
    {
        _features = features;
        _patch = patch;
        _headDim = headDim;
        _theta = theta;
        int patchDim = features * patch * patch;
        _convFirst = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(features, 3, null));
        _embedTemporal = Stack(layers, embedTemporalDepth, features, heads, headDim, (int)(features * mlpRatio));
        _embedFinal = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(features));
        // vit-pytorch's to_patch_embedding: LayerNorm(patch_dim) -> Linear(patch_dim, dim) -> LayerNorm(dim).
        _patchNorm = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(patchDim));
        _patchEmbed = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(embedDim));
        _embedNorm = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(embedDim));
        _vertical = Stack(layers, verticalDepth, embedDim, heads, headDim, (int)(embedDim * mlpRatio));
        _horizontal = Stack(layers, horizontalDepth, embedDim, heads, headDim, (int)(embedDim * mlpRatio));
        _axialFinal = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(embedDim));
        _decode = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(patchDim));
        _reconTemporal = Stack(layers, reconTemporalDepth, features, heads, headDim, (int)(features * mlpRatio));
        _reconFinal = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(features));
        // BasicVSR's upsampler: a 3x3 convolution to 4x the channels, pixel shuffle x2 and LeakyReLU(0.1) per factor of
        // two, then conv_hr and conv_last. LeakyReLU is elementwise and pixel shuffle a permutation, so the activation
        // sits on the convolution.
        int stages = 0;
        for (int s = scale; s > 1; s /= 2) stages++;
        _upconv = new int[stages];
        for (int i = 0; i < stages; i++)
        {
            _upconv[i] = layers.Count;
            layers.Add(BasicVsrBlocks.Conv<T>(features * 4, 3, new LeakyReLUActivation<T>(0.1)));
        }
        _convHr = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(features, 3, new LeakyReLUActivation<T>(0.1)));
        _convLast = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(3, 3, null));
    }

    private static DualXTransformerBlock<T>[] Stack(List<ILayer<T>> layers, int depth, int dim, int heads, int headDim, int mlpDim)
    {
        var blocks = new DualXTransformerBlock<T>[depth];
        for (int i = 0; i < depth; i++) blocks[i] = new DualXTransformerBlock<T>(layers, dim, heads, headDim, mlpDim);
        return blocks;
    }

    private (Tensor<T> Cos, Tensor<T> Sin) Positions(int outer, int frames)
    {
        var positions = new int[outer * frames][];
        for (int i = 0; i < outer; i++)
            for (int t = 0; t < frames; t++) positions[i * frames + t] = outer == 1 ? new[] { t } : new[] { i, t };
        return DualXRope.Tables<T>(positions, _headDim, _theta);
    }

    private static Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, DualXTransformerBlock<T>[] blocks, Tensor<T> x,
        (Tensor<T> Cos, Tensor<T> Sin) rope, int axes)
    {
        foreach (var block in blocks) x = block.Forward(engine, layers, x, rope.Cos, rope.Sin, axes);
        return x;
    }

    // [B, N, C, H, W] -> each pixel's frames through a temporal transformer and its final norm.
    private Tensor<T> Temporal(IEngine engine, IReadOnlyList<ILayer<T>> layers, DualXTransformerBlock<T>[] blocks, int norm, Tensor<T> x)
    {
        int b = x.Shape[0], n = x.Shape[1], c = x.Shape[2], h = x.Shape[3], w = x.Shape[4];
        var sequences = engine.Reshape(engine.TensorPermute(x, new[] { 0, 3, 4, 1, 2 }), new[] { b * h * w, n, c });
        var y = layers[norm].Forward(Run(engine, layers, blocks, sequences, Positions(1, n), 1));
        return engine.TensorPermute(engine.Reshape(y, new[] { b, h, w, n, c }), new[] { 0, 3, 4, 1, 2 });
    }

    /// <param name="clips">[B, N, 3, H, W] frames scaled to [0, 1], H and W multiples of the patch size.</param>
    /// <returns>[B, N, 3, sH, sW].</returns>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips)
    {
        int b = clips.Shape[0], n = clips.Shape[1], h = clips.Shape[3], w = clips.Shape[4], c = _features, p = _patch;
        if (h % p != 0 || w % p != 0)
            throw new ArgumentException($"Frame height and width must be multiples of the {p}x{p} patch; got {h}x{w}.", nameof(clips));
        int nh = h / p, nw = w / p;

        var shallow = layers[_convFirst].Forward(engine.Reshape(clips, new[] { b * n, 3, h, w }));
        var x = Temporal(engine, layers, _embedTemporal, _embedFinal, engine.Reshape(shallow, new[] { b, n, c, h, w }));

        // 'b n c (nh p1) (nw p2) -> b n nh nw (p1 p2 c)', then the patch embedding.
        var patches = engine.TensorPermute(engine.Reshape(x, new[] { b * n, c, nh, p, nw, p }), new[] { 0, 2, 4, 3, 5, 1 });
        var tokens = layers[_embedNorm].Forward(layers[_patchEmbed].Forward(layers[_patchNorm].Forward(
            engine.Reshape(patches, new[] { b, n, nh, nw, p * p * c }))));
        int d = tokens.Shape[4];

        // VTAB: [B, N, nh, nw, D] -> [(B nw), (nh N), D]; HTAB: -> [(B nh), (nw N), D].
        var vertical = engine.Reshape(engine.TensorPermute(tokens, new[] { 0, 3, 2, 1, 4 }), new[] { b * nw, nh * n, d });
        vertical = Run(engine, layers, _vertical, vertical, Positions(nh, n), 2);
        var horizontal = engine.Reshape(engine.TensorPermute(engine.Reshape(vertical, new[] { b, nw, nh, n, d }), new[] { 0, 2, 1, 3, 4 }),
            new[] { b * nh, nw * n, d });
        horizontal = layers[_axialFinal].Forward(Run(engine, layers, _horizontal, horizontal, Positions(nw, n), 2));

        // Decode and unpatchify: [(B nh), (nw N), p·p·C] -> [B, N, C, H, W].
        var decoded = engine.Reshape(layers[_decode].Forward(horizontal), new[] { b, nh, nw, n, p, p, c });
        var pixels = engine.Reshape(engine.TensorPermute(decoded, new[] { 0, 3, 6, 1, 4, 2, 5 }), new[] { b, n, c, h, w });
        var features = engine.Reshape(Temporal(engine, layers, _reconTemporal, _reconFinal, pixels), new[] { b * n, c, h, w });

        foreach (int up in _upconv) features = engine.PixelShuffle(layers[up].Forward(features), 2);
        var output = layers[_convLast].Forward(layers[_convHr].Forward(features));
        return engine.Reshape(output, new[] { b, n, 3, output.Shape[2], output.Shape[3] });
    }
}
