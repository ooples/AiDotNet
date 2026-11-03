using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// Wan 2.1's 3D rotary embedding (<c>precompute_freqs_cis_3d</c>): of each head, hd − 2·(hd / 3) for time and hd / 3 each for
/// height and width, θ = 10000, integer positions, adjacent pairs rotated as complex numbers.
/// </summary>
internal static class WanRope
{
    /// <summary>The cosines and sines [1, F·H·W, headDim] for a (frames, height, width) token grid in that order.</summary>
    internal static (Tensor<T> Cos, Tensor<T> Sin) Build<T>(int headDim, int frames, int height, int width)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int dh = headDim / 3, dt = headDim - 2 * dh;
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
}

/// <summary>Which self-attention a DiT pass uses.</summary>
internal enum WanAttentionMode
{
    /// <summary>Full attention over the whole clip (the teacher, Stage 1).</summary>
    Dense,

    /// <summary>FlashVSR's locality-constrained block-sparse causal attention (Stage 2 on, and inference).</summary>
    SparseCausal,
}

/// <summary>The sparse attention's settings for one pass.</summary>
internal sealed class WanSparsePlan
{
    internal WanSparsePlan(int frames, int height, int width, double topkRatio, int kvChunks, int localRange)
    {
        Frames = frames;
        Height = height;
        Width = width;
        KvChunks = kvChunks;
        LocalRange = localRange;
        int spatial = (height / 8) * (width / 8);
        // model_fn_wan_video: window_size = 2·h·w / 128 blocks per chunk, topk = int(window_size² · topk_ratio) − 1.
        TopK = (int)(spatial * (double)spatial * topkRatio) - 1;
    }

    internal int Frames { get; }

    internal int Height { get; }

    internal int Width { get; }

    internal int KvChunks { get; }

    internal int LocalRange { get; }

    internal int TopK { get; }
}

/// <summary>
/// The release's block selection and mask (<c>generate_draft_block_mask</c> under the streaming KV cache), evaluated for the
/// whole clip at once. Tokens fall into (2, 8, 8) windows of 128; each chunk of two latent frames attends the first three
/// chunks together (the first streamed step) or itself and the <c>kv</c> chunks before it (the cache). Per head, the
/// windows' mean queries and keys score every visible block pair, the locality window (<c>local_range</c> blocks around
/// the query block) masks the rest, a softmax over each query block's row follows, and of each query chunk's map the
/// entries above its (topk + 1)-th largest are kept.
/// </summary>
internal static class WanSparseMask
{
    /// <summary>The 0/1 token mask [BH, L, L] for queries and keys [BH, L, headDim] after the rotary embedding.</summary>
    internal static Tensor<T> Build<T>(Tensor<T> q, Tensor<T> k, WanSparsePlan plan)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int bh = q.Shape[0], l = q.Shape[1], hd = q.Shape[2];
        int f = plan.Frames, h = plan.Height, w = plan.Width, bw = w / 8, bhgt = h / 8, spatial = bw * bhgt, chunks = f / 2;
        int blocks = chunks * spatial;
        var blockOf = new int[l];
        for (int t = 0; t < f; t++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    blockOf[(t * h + y) * w + x] = (t / 2) * spatial + (y / 8) * bw + x / 8;

        var local = new bool[spatial, spatial];
        int half = plan.LocalRange / 2;
        for (int a = 0; a < spatial; a++)
            for (int b = 0; b < spatial; b++)
            {
                int ra = a / bw, ca = a % bw, rb = b / bw, cb = b % bw;
                local[a, b] = rb >= ra - half && rb <= ra - half + plan.LocalRange - 1 && cb >= ca - half && cb <= ca - half + plan.LocalRange - 1;
            }

        int first = Math.Min(3, chunks);
        var mask = new Tensor<T>(new[] { bh, l, l });
        var keep = new bool[blocks, blocks];
        for (int n = 0; n < bh; n++)
        {
            var qMean = Means(q, n, blockOf, blocks, hd, l);
            var kMean = Means(k, n, blockOf, blocks, hd, l);
            Array.Clear(keep, 0, keep.Length);
            for (int i = 0; i < chunks; i++)
            {
                int lo = i < first ? 0 : Math.Max(0, i - plan.KvChunks), hi = i < first ? first - 1 : i;
                int visible = (hi - lo + 1) * spatial;
                var probabilities = new double[spatial * visible];
                for (int s = 0; s < spatial; s++)
                {
                    int qb = i * spatial + s;
                    double max = double.NegativeInfinity;
                    var row = new double[visible];
                    for (int v = 0; v < visible; v++)
                    {
                        int kb = lo * spatial + v;
                        if (!local[s, kb % spatial]) { row[v] = double.NegativeInfinity; continue; }
                        double dot = 0;
                        for (int d = 0; d < hd; d++) dot += qMean[qb, d] * kMean[kb, d];
                        row[v] = dot / Math.Sqrt(hd);
                        max = Math.Max(max, row[v]);
                    }
                    double sum = 0;
                    for (int v = 0; v < visible; v++) sum += row[v] = double.IsNegativeInfinity(row[v]) ? 0 : Math.Exp(row[v] - max);
                    for (int v = 0; v < visible; v++) probabilities[s * visible + v] = row[v] / sum;
                }
                int count = probabilities.Length, apply = Math.Min(count - 1, plan.TopK);
                if (apply < 0) continue;
                var sorted = (double[])probabilities.Clone();
                Array.Sort(sorted);
                double threshold = sorted[count - 1 - apply];
                for (int s = 0; s < spatial; s++)
                    for (int v = 0; v < visible; v++)
                        if (probabilities[s * visible + v] > threshold) keep[i * spatial + s, lo * spatial + v] = true;
            }
            int baseIndex = n * l * l;
            for (int a = 0; a < l; a++)
                for (int b = 0; b < l; b++)
                    if (keep[blockOf[a], blockOf[b]]) mask[baseIndex + a * l + b] = ops.One;
        }
        return mask;
    }

    private static double[,] Means<T>(Tensor<T> x, int n, int[] blockOf, int blocks, int hd, int l)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var sums = new double[blocks, hd];
        var counts = new int[blocks];
        for (int a = 0; a < l; a++)
        {
            int b = blockOf[a];
            counts[b]++;
            for (int d = 0; d < hd; d++) sums[b, d] += ops.ToDouble(x[(n * l + a) * hd + d]);
        }
        for (int b = 0; b < blocks; b++)
            for (int d = 0; d < hd; d++) sums[b, d] /= Math.Max(1, counts[b]);
        return sums;
    }
}

/// <summary>
/// A linear map with FlashVSR's LoRA (rank 384 in the paper; DiffSynth-Studio's adapter, α = rank so the scale is one):
/// the frozen base plus B·A with A in PyTorch's default initialization and B zero. Three adapter sets share the base: the
/// network being trained, the frozen teacher and DMD's fake-score critic.
/// </summary>
internal sealed class WanLoraLinear<T>
{
    internal const int Sets = 3;
    private readonly int _base;
    private readonly int[] _down = new int[Sets], _up = new int[Sets];

    internal WanLoraLinear(List<ILayer<T>> layers, int outputs, int rank)
    {
        _base = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(outputs));
        for (int s = 0; s < Sets; s++)
        {
            _down[s] = layers.Count;
            layers.Add(RvrtBlocks.Linear<T>(rank, bias: false));
            _up[s] = layers.Count;
            layers.Add(RvrtBlocks.Linear<T>(outputs, bias: false, init: new ZeroInitializationStrategy<T>()));
        }
    }

    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, int set)
        => engine.TensorAdd(layers[_base].Forward(x), layers[_up[set]].Forward(layers[_down[set]].Forward(x)));

    internal IEnumerable<int> Adapter(int set)
    {
        yield return _down[set];
        yield return _up[set];
    }
}

/// <summary>Wan's norms and modulation, shared by the blocks and the head.</summary>
internal static class WanOps
{
    /// <summary><c>RMSNorm</c> over the last axis with its learned scale (a free vector layer).</summary>
    internal static Tensor<T> RmsNorm<T>(IEngine engine, IReadOnlyList<ILayer<T>> layers, int weight, Tensor<T> x, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        int last = shape.Length - 1;
        var meanSquare = engine.ReduceMean(engine.TensorSquare(x), new[] { last }, true);
        var scale = engine.TensorBroadcastTo(engine.TensorSqrt(engine.TensorAddScalar(meanSquare, ops.FromDouble(epsilon))), shape);
        var gammaShape = Enumerable.Repeat(1, shape.Length).ToArray();
        gammaShape[last] = shape[last];
        var gamma = engine.TensorBroadcastTo(engine.Reshape(RealViformerBlocks.Read(layers, weight), gammaShape), shape);
        return engine.TensorMultiply(engine.TensorDivide(x, scale), gamma);
    }

    /// <summary><c>nn.LayerNorm(elementwise_affine=False)</c> over the last axis.</summary>
    internal static Tensor<T> LayerNorm<T>(IEngine engine, Tensor<T> x, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        int last = shape.Length - 1;
        var centred = engine.TensorSubtract(x, engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { last }, true), shape));
        var variance = engine.ReduceMean(engine.TensorSquare(centred), new[] { last }, true);
        return engine.TensorDivide(centred, engine.TensorBroadcastTo(engine.TensorSqrt(engine.TensorAddScalar(variance, ops.FromDouble(epsilon))), shape));
    }

    /// <summary>x·(1 + scale) + shift with [B, 1, D] modulation.</summary>
    internal static Tensor<T> Modulate<T>(IEngine engine, Tensor<T> x, Tensor<T> shift, Tensor<T> scale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        return engine.TensorAdd(engine.TensorMultiply(x, engine.TensorBroadcastTo(engine.TensorAddScalar(scale, ops.One), shape)),
            engine.TensorBroadcastTo(shift, shape));
    }

    /// <summary>A modulation parameter [1, n·D] initialized from N(0, 1/D) as <c>torch.randn(1, n, dim) / dim**0.5</c>.</summary>
    internal static DenseLayer<T> Modulation<T>(int count, int dim)
        => new(count * dim, (IActivationFunction<T>?)null, new NormalInitializationStrategy<T>(0.0, 1.0 / Math.Sqrt(dim)), AiDotNet.Enums.BiasMode.Never);

    /// <summary>Attention of per-head queries [B·H, Lq, hd] on keys and values [B·H, Lk, hd]; an optional 0/1 mask [B·H, Lq, Lk].</summary>
    internal static Tensor<T> Attend<T>(IEngine engine, Tensor<T> q, Tensor<T> k, Tensor<T> v, Tensor<T>? mask)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hd = q.Shape[2];
        var scores = engine.TensorMultiplyScalar(engine.TensorBatchMatMul(q, engine.TensorPermute(k, new[] { 0, 2, 1 })),
            ops.FromDouble(1.0 / Math.Sqrt(hd)));
        Tensor<T> weights;
        if (mask is null) weights = engine.TensorSoftmax(scores, 2);
        else
        {
            // A softmax over the kept keys only; a query block with none kept outputs zero, as the block-sparse kernel does.
            var shape = scores.Shape.ToArray();
            var masked = engine.TensorAdd(scores, engine.TensorMultiplyScalar(engine.TensorAddScalar(mask, ops.FromDouble(-1.0)), ops.FromDouble(1e9)));
            var exp = engine.TensorMultiply(engine.TensorExp(engine.TensorSubtract(masked,
                engine.TensorBroadcastTo(engine.ReduceMax(masked, new[] { 2 }, true), shape))), mask);
            var sum = engine.TensorMax(engine.ReduceSum(exp, new[] { 2 }, keepDims: true), ops.FromDouble(1e-30));
            weights = engine.TensorDivide(exp, engine.TensorBroadcastTo(sum, shape));
        }
        return engine.TensorBatchMatMul(weights, v);
    }

    /// <summary>[B, L, H·hd] to [B·H, L, hd].</summary>
    internal static Tensor<T> Heads<T>(IEngine engine, Tensor<T> x, int heads)
    {
        int b = x.Shape[0], l = x.Shape[1], hd = x.Shape[2] / heads;
        return engine.Reshape(engine.TensorPermute(engine.Reshape(x, new[] { b, l, heads, hd }), new[] { 0, 2, 1, 3 }), new[] { b * heads, l, hd });
    }

    /// <summary>[B·H, L, hd] back to [B, L, H·hd].</summary>
    internal static Tensor<T> Merge<T>(IEngine engine, Tensor<T> x, int heads)
    {
        int b = x.Shape[0] / heads, l = x.Shape[1], hd = x.Shape[2];
        return engine.Reshape(engine.TensorPermute(engine.Reshape(x, new[] { b, heads, l, hd }), new[] { 0, 2, 1, 3 }), new[] { b, l, heads * hd });
    }
}

/// <summary>
/// One Wan 2.1 <c>DiTBlock</c> as FlashVSR runs it: the six modulation chunks (a learned [1, 6, D] plus the projected time),
/// self-attention (RMS-normed queries and keys over the full width, the 3D rotary embedding, dense or block-sparse),
/// cross-attention on the fixed prompt after an affine LayerNorm, and a tanh-GELU feed-forward; LoRA on q, k, v, o and
/// both feed-forward maps.
/// </summary>
internal sealed class WanDiTBlock<T>
{
    private readonly int _modulation, _selfNormQ, _selfNormK, _crossNormQ, _crossNormK, _norm3, _dim, _heads;
    private readonly WanLoraLinear<T> _sq, _sk, _sv, _so, _cq, _ck, _cv, _co, _ffn0, _ffn2;
    private readonly double _epsilon;
    private readonly GELUActivation<T> _gelu = new();

    internal WanDiTBlock(List<ILayer<T>> layers, int dim, int heads, int ffnDim, int rank, double epsilon)
    {
        _dim = dim;
        _heads = heads;
        _epsilon = epsilon;
        _modulation = layers.Count;
        layers.Add(WanOps.Modulation<T>(6, dim));
        _sq = new WanLoraLinear<T>(layers, dim, rank);
        _sk = new WanLoraLinear<T>(layers, dim, rank);
        _sv = new WanLoraLinear<T>(layers, dim, rank);
        _so = new WanLoraLinear<T>(layers, dim, rank);
        _selfNormQ = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(dim, 1.0));
        _selfNormK = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(dim, 1.0));
        _cq = new WanLoraLinear<T>(layers, dim, rank);
        _ck = new WanLoraLinear<T>(layers, dim, rank);
        _cv = new WanLoraLinear<T>(layers, dim, rank);
        _co = new WanLoraLinear<T>(layers, dim, rank);
        _crossNormQ = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(dim, 1.0));
        _crossNormK = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(dim, 1.0));
        _norm3 = layers.Count;
        layers.Add(new LayerNormalizationLayer<T>(dim, epsilon));
        _ffn0 = new WanLoraLinear<T>(layers, ffnDim, rank);
        _ffn2 = new WanLoraLinear<T>(layers, dim, rank);
    }

    internal IEnumerable<int> Adapter(int set)
        => new[] { _sq, _sk, _sv, _so, _cq, _ck, _cv, _co, _ffn0, _ffn2 }.SelectMany(linear => linear.Adapter(set));

    private Tensor<T> Chunk(IEngine engine, Tensor<T> modulation, int k)
        => engine.TensorNarrow(modulation, 1, k, 1);

    /// <param name="x">[B, L, D] tokens.</param>
    /// <param name="context">[B, Lt, D] embedded prompt.</param>
    /// <param name="timeModulation">[B, 6, D] projected time.</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> context, Tensor<T> timeModulation,
        int set, WanSparsePlan? sparse, Tensor<T> cos, Tensor<T> sin)
    {
        int b = x.Shape[0];
        var learned = engine.Reshape(RealViformerBlocks.Read(layers, _modulation), new[] { 1, 6, _dim });
        var mod = engine.TensorAdd(engine.TensorBroadcastTo(learned, new[] { b, 6, _dim }), timeModulation);

        var input = WanOps.Modulate(engine, WanOps.LayerNorm(engine, x, _epsilon), Chunk(engine, mod, 0), Chunk(engine, mod, 1));
        var q = WanOps.Heads(engine, WanOps.RmsNorm(engine, layers, _selfNormQ, _sq.Forward(engine, layers, input, set), _epsilon), _heads);
        var k = WanOps.Heads(engine, WanOps.RmsNorm(engine, layers, _selfNormK, _sk.Forward(engine, layers, input, set), _epsilon), _heads);
        var v = WanOps.Heads(engine, _sv.Forward(engine, layers, input, set), _heads);
        q = CogVideoXRope.Apply(engine, q, cos, sin);
        k = CogVideoXRope.Apply(engine, k, cos, sin);
        Tensor<T>? mask = null;
        if (sparse is not null)
            using (new NoGradScope<T>()) mask = WanSparseMask.Build(q, k, sparse);
        var attended = WanOps.Merge(engine, WanOps.Attend(engine, q, k, v, mask), _heads);
        var update = _so.Forward(engine, layers, attended, set);
        x = engine.TensorAdd(x, engine.TensorMultiply(update, engine.TensorBroadcastTo(Chunk(engine, mod, 2), update.Shape.ToArray())));

        var normed = layers[_norm3].Forward(x);
        var cq = WanOps.Heads(engine, WanOps.RmsNorm(engine, layers, _crossNormQ, _cq.Forward(engine, layers, normed, set), _epsilon), _heads);
        var ck = WanOps.Heads(engine, WanOps.RmsNorm(engine, layers, _crossNormK, _ck.Forward(engine, layers, context, set), _epsilon), _heads);
        var cv = WanOps.Heads(engine, _cv.Forward(engine, layers, context, set), _heads);
        x = engine.TensorAdd(x, _co.Forward(engine, layers, WanOps.Merge(engine, WanOps.Attend(engine, cq, ck, cv, null), _heads), set));

        input = WanOps.Modulate(engine, WanOps.LayerNorm(engine, x, _epsilon), Chunk(engine, mod, 3), Chunk(engine, mod, 4));
        var fed = _ffn2.Forward(engine, layers, _gelu.Activate(_ffn0.Forward(engine, layers, input, set)), set);
        return engine.TensorAdd(x, engine.TensorMultiply(fed, engine.TensorBroadcastTo(Chunk(engine, mod, 5), fed.Shape.ToArray())));
    }
}

/// <summary>
/// Wan 2.1's <c>WanModel</c> as FlashVSR uses it (wan_video_dit.py): a (1, 2, 2) patch embedding, the prompt embedded by
/// Linear-GELU(tanh)-Linear, the frequency-dim sinusoid through Linear-SiLU-Linear and projected by SiLU-Linear to six
/// modulations, the blocks with the LR projection's tokens added before the first <c>layer_num</c>, and the head (a
/// non-affine LayerNorm modulated by a learned [1, 2, D] plus the time embedding, then a linear map to the patches).
/// </summary>
internal sealed class WanDiT<T>
{
    private readonly int _patch, _text1, _text2, _time1, _time2, _timeProjection, _head, _headModulation;
    private readonly int _dim, _heads, _frequencyDim, _channels;
    private readonly double _epsilon;
    private readonly WanDiTBlock<T>[] _blocks;

    internal WanDiT(List<ILayer<T>> layers, int channels, int dim, int ffnDim, int heads, int depth, int frequencyDim, int rank, double epsilon)
    {
        _channels = channels;
        _dim = dim;
        _heads = heads;
        _frequencyDim = frequencyDim;
        _epsilon = epsilon;
        _patch = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _text1 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim, new GELUActivation<T>()));
        _text2 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _time1 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim, new SiLUActivation<T>()));
        _time2 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _timeProjection = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(6 * dim));
        _blocks = new WanDiTBlock<T>[depth];
        for (int i = 0; i < depth; i++) _blocks[i] = new WanDiTBlock<T>(layers, dim, heads, ffnDim, rank, epsilon);
        _head = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(4 * channels));
        _headModulation = layers.Count;
        layers.Add(WanOps.Modulation<T>(2, dim));
    }

    /// <summary>The layer indices of one LoRA set.</summary>
    internal IEnumerable<int> Adapter(int set) => _blocks.SelectMany(block => block.Adapter(set));

    /// <param name="latents">[B, C, F, H, W] with H and W even.</param>
    /// <param name="prompt">[B, Lt, textDim] prompt embeddings.</param>
    /// <param name="lqTokens">Tokens [B, F·H/2·W/2, D] to add before each of the first blocks, or null.</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> latents, IReadOnlyList<double> timesteps,
        Tensor<T> prompt, IReadOnlyList<Tensor<T>>? lqTokens, int set, WanAttentionMode mode, Func<int, int, int, WanSparsePlan>? sparsePlan)
    {
        int b = latents.Shape[0], c = latents.Shape[1], f = latents.Shape[2], h = latents.Shape[3] / 2, w = latents.Shape[4] / 2;
        // Conv3d's (c, 1, 2, 2) receptive field as one patch vector per token, tokens in (f, h, w) order.
        var patches = engine.Reshape(latents, new[] { b, c, f, h, 2, w, 2 });
        patches = engine.TensorPermute(patches, new[] { 0, 2, 3, 5, 1, 4, 6 });
        var x = layers[_patch].Forward(engine.Reshape(patches, new[] { b, f * h * w, c * 4 }));

        var context = layers[_text2].Forward(layers[_text1].Forward(prompt));
        var time = layers[_time2].Forward(layers[_time1].Forward(SvdBlocks.Sinusoid<T>(timesteps, _frequencyDim)));
        var timeModulation = engine.Reshape(layers[_timeProjection].Forward(engine.Swish(time)), new[] { b, 6, _dim });
        var (cos, sin) = WanRope.Build<T>(_dim / _heads, f, h, w);
        var sparse = mode == WanAttentionMode.SparseCausal && sparsePlan is not null ? sparsePlan(f, h, w) : null;

        for (int i = 0; i < _blocks.Length; i++)
        {
            if (lqTokens is not null && i < lqTokens.Count) x = engine.TensorAdd(x, lqTokens[i]);
            x = _blocks[i].Forward(engine, layers, x, context, timeModulation, set, sparse, cos, sin);
        }

        // Head: modulation [1, 2, D] + time [B, D] (the release's broadcast, exact for one sample a batch).
        var learned = engine.Reshape(RealViformerBlocks.Read(layers, _headModulation), new[] { 1, 2, _dim });
        var t = engine.Reshape(time, new[] { b, 1, _dim });
        var shift = engine.TensorAdd(engine.TensorBroadcastTo(engine.TensorNarrow(learned, 1, 0, 1), new[] { b, 1, _dim }), t);
        var scale = engine.TensorAdd(engine.TensorBroadcastTo(engine.TensorNarrow(learned, 1, 1, 1), new[] { b, 1, _dim }), t);
        var output = layers[_head].Forward(WanOps.Modulate(engine, WanOps.LayerNorm(engine, x, _epsilon), shift, scale));
        // 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)' with patch (1, 2, 2).
        output = engine.Reshape(output, new[] { b, f, h, w, 2, 2, _channels });
        output = engine.TensorPermute(output, new[] { 0, 6, 1, 2, 4, 3, 5 });
        return engine.Reshape(output, new[] { b, _channels, f, 2 * h, 2 * w });
    }
}

/// <summary>
/// The released <c>Buffer_LQ4x_Proj</c> (FlashVSR utils.py): the bicubic-upscaled LR clip, after three copies of its first
/// frame, cut into groups of four frames; each group pixel-unshuffled by 16 in space, then a (4, 3, 3) causal convolution
/// with stride 2 in time and replicate padding run on the group's last two frames and itself, RMS norm and SiLU, the same
/// again on the two output frames twice over, and per-layer linear maps to the DiT width. The first group only primes the
/// cache, so latent j reads frames 4j + 1 to 4j + 4.
/// </summary>
internal sealed class FlashVsrLqProjection<T> : LazyWeightModule<T>
{
    internal const int Shuffle = 16;
    private readonly LazyWeight<T> _conv1, _bias1, _gamma1, _conv2, _bias2, _gamma2;
    private readonly LazyWeight<T>[] _linear, _linearBias;
    private readonly int _hidden2;

    public FlashVsrLqProjection(int hidden1, int hidden2, int outDim, int layers)
    {
        _hidden2 = hidden2;
        int inChannels = 3 * Shuffle * Shuffle, fan1 = inChannels * 36, fan2 = hidden1 * 36;
        _conv1 = LazyWeight<T>.Uniform(new[] { hidden1, inChannels, 4, 3, 3 }, fan1);
        _bias1 = LazyWeight<T>.Uniform(new[] { hidden1 }, fan1);
        _gamma1 = LazyWeight<T>.Constant(new[] { hidden1 }, 1.0);
        _conv2 = LazyWeight<T>.Uniform(new[] { hidden2, hidden1, 4, 3, 3 }, fan2);
        _bias2 = LazyWeight<T>.Uniform(new[] { hidden2 }, fan2);
        _gamma2 = LazyWeight<T>.Constant(new[] { hidden2 }, 1.0);
        _linear = new LazyWeight<T>[layers];
        _linearBias = new LazyWeight<T>[layers];
        for (int i = 0; i < layers; i++)
        {
            _linear[i] = LazyWeight<T>.Uniform(new[] { outDim, hidden2 }, hidden2);
            _linearBias[i] = LazyWeight<T>.Uniform(new[] { outDim }, hidden2);
        }
    }

    private static Tensor<T> ChannelBias(Tensor<T> x, Tensor<T> bias)
        => Engine.TensorAdd(x, Engine.TensorBroadcastTo(Engine.Reshape(bias, new[] { 1, bias.Length, 1, 1, 1 }), x.Shape.ToArray()));

    // RMS_norm(images=False): F.normalize over the channels · sqrt(C) · γ, then SiLU.
    private static Tensor<T> NormAct(Tensor<T> x, Tensor<T> gamma)
    {
        var shape = x.Shape.ToArray();
        var norm = Engine.TensorMax(Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(x), new[] { 1 }, keepDims: true)), NumOps.FromDouble(1e-12));
        var unit = Engine.TensorDivide(x, Engine.TensorBroadcastTo(norm, shape));
        var scaled = Engine.TensorMultiply(unit, Engine.TensorBroadcastTo(Engine.Reshape(gamma, new[] { 1, gamma.Length, 1, 1, 1 }), shape));
        return Engine.Swish(Engine.TensorMultiplyScalar(scaled, NumOps.FromDouble(Math.Sqrt(shape[1]))));
    }

    // The buffered causal convolution: the input's last two frames in front of it, replicate padding in space only.
    private static Tensor<T> BufferedConv(Tensor<T> x, Tensor<T> weight, Tensor<T> bias)
    {
        int t = x.Shape[2];
        var cache = Engine.TensorNarrow(x, 2, t - 2, 2);
        var padded = Engine.PadNd(Engine.TensorConcatenate(new[] { cache, x }, 2), new[] { 1, 1, 1, 1, 0, 0 }, PadMode.Replicate, NumOps.Zero);
        return ChannelBias(Engine.Conv3D(padded, weight, new[] { 2, 1, 1 }, new[] { 0, 0, 0 }, new[] { 1, 1, 1 }), bias);
    }

    /// <summary>LR video [B, 3, 4n + 1, H, W] (H, W multiples of 16) to per-layer tokens [B, n·H/16·W/16, D].</summary>
    public IReadOnlyList<Tensor<T>> Forward(Tensor<T> video)
    {
        var s = video.Shape;
        int b = s[0], n = (s[2] - 1) / 4, h = s[3] / Shuffle, w = s[4] / Shuffle;
        // Groups 1..n are frames 4j - 3 .. 4j; each runs alone, so they batch.
        var groups = Engine.Reshape(Engine.TensorNarrow(video, 2, 1, 4 * n), new[] { b, 3, n, 4, h, Shuffle, w, Shuffle });
        groups = Engine.TensorPermute(groups, new[] { 0, 2, 1, 5, 7, 3, 4, 6 });
        var x = Engine.Reshape(groups, new[] { b * n, 3 * Shuffle * Shuffle, 4, h, w });
        x = NormAct(BufferedConv(x, _conv1.Value, _bias1.Value), _gamma1.Value);
        x = NormAct(BufferedConv(x, _conv2.Value, _bias2.Value), _gamma2.Value);
        var tokens = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(x, new[] { b, n, _hidden2, h, w }), new[] { 0, 1, 3, 4, 2 }),
            new[] { b, n * h * w, _hidden2 });
        var outputs = new Tensor<T>[_linear.Length];
        for (int i = 0; i < outputs.Length; i++)
        {
            var y = Engine.TensorMatMul(Engine.Reshape(tokens, new[] { b * n * h * w, _hidden2 }), Engine.TensorPermute(_linear[i].Value, new[] { 1, 0 }));
            y = Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(_linearBias[i].Value, new[] { 1, _linearBias[i].Value.Length }), y.Shape.ToArray()));
            outputs[i] = Engine.Reshape(y, new[] { b, n * h * w, y.Shape[1] });
        }
        return outputs;
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _conv1;
        yield return _bias1;
        yield return _gamma1;
        yield return _conv2;
        yield return _bias2;
        yield return _gamma2;
        for (int i = 0; i < _linear.Length; i++)
        {
            yield return _linear[i];
            yield return _linearBias[i];
        }
    }
}

/// <summary>
/// FlashVSR's Tiny Conditional decoder (TCDecoder.py, a deepened TAEHV decoder): the latents with the LR clip
/// pixel-unshuffled by (4, 8, 8) in front of them, Clamp(tanh(x/3)·3), a stem, three stages of three memory blocks (each
/// reads the previous frame's input) followed by nearest x2, a temporal grow (x1, x2, x2) and a bias-free convolution, and
/// an RGB head; a Dirac-initialized 3x3 convolution and ReLU deepen the stem and the head. The first three frames are
/// trimmed, leaving 4n − 3.
/// </summary>
internal sealed class FlashVsrTcDecoder<T> : LazyWeightModule<T>
{
    private readonly int[] _channels;
    private readonly LazyWeight<T> _stem, _stemBias, _stemIdentity, _headIdentity, _out, _outBias;
    private readonly LazyWeight<T>[][] _memory;
    private readonly LazyWeight<T>[] _grow, _stage;
    private static readonly int[] GrowStride = { 1, 2, 2 };

    public FlashVsrTcDecoder(IReadOnlyList<int> channels, int latentChannels)
    {
        _channels = channels.ToArray();
        int inChannels = latentChannels + 3 * 4 * 8 * 8;
        _stem = LazyWeight<T>.Uniform(new[] { _channels[0], inChannels, 3, 3 }, inChannels * 9);
        _stemBias = LazyWeight<T>.Uniform(new[] { _channels[0] }, inChannels * 9);
        _stemIdentity = Dirac(_channels[0]);
        _memory = new LazyWeight<T>[9][];
        _grow = new LazyWeight<T>[3];
        _stage = new LazyWeight<T>[3];
        for (int stage = 0; stage < 3; stage++)
        {
            int c = _channels[stage];
            for (int m = 0; m < 3; m++)
            {
                _memory[stage * 3 + m] = new[]
                {
                    LazyWeight<T>.Uniform(new[] { c, 2 * c, 3, 3 }, 2 * c * 9), LazyWeight<T>.Uniform(new[] { c }, 2 * c * 9),
                    LazyWeight<T>.Uniform(new[] { c, c, 3, 3 }, c * 9), LazyWeight<T>.Uniform(new[] { c }, c * 9),
                    LazyWeight<T>.Uniform(new[] { c, c, 3, 3 }, c * 9), LazyWeight<T>.Uniform(new[] { c }, c * 9),
                };
            }
            _grow[stage] = LazyWeight<T>.Uniform(new[] { c * GrowStride[stage], c, 1, 1 }, c);
            _stage[stage] = LazyWeight<T>.Uniform(new[] { _channels[stage + 1], c, 3, 3 }, c * 9);
        }
        int last = _channels[3];
        _headIdentity = Dirac(last);
        _out = LazyWeight<T>.Uniform(new[] { 3, last, 3, 3 }, last * 9);
        _outBias = LazyWeight<T>.Uniform(new[] { 3 }, last * 9);
    }

    private static LazyWeight<T> Dirac(int channels) => new(new[] { channels, channels, 3, 3 }, (tensor, _) =>
    {
        for (int c = 0; c < channels; c++) tensor[((c * channels + c) * 3 + 1) * 3 + 1] = NumOps.One;
    });

    private static Tensor<T> Conv(Tensor<T> x, Tensor<T> weight, Tensor<T>? bias)
    {
        int pad = weight.Shape[2] / 2;
        var y = Engine.Conv2D(x, weight, new[] { 1, 1 }, new[] { pad, pad }, new[] { 1, 1 });
        return bias is null ? y : Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(bias, new[] { 1, bias.Length, 1, 1 }), y.Shape.ToArray()));
    }

    // MemBlock: ReLU(conv(ReLU(conv(ReLU(conv([x, past]))))) + x), past being the previous frame's input (zero for the first).
    private static Tensor<T> Memory(Tensor<T> x, int batch, LazyWeight<T>[] w)
    {
        var s = x.Shape;
        int frames = s[0] / batch;
        var clip = Engine.Reshape(x, new[] { batch, frames, s[1], s[2], s[3] });
        var past = Engine.PadNd(Engine.TensorNarrow(clip, 1, 0, frames - 1), new[] { 0, 0, 0, 0, 0, 0, 1, 0 }, PadMode.Constant, NumOps.Zero);
        var y = Engine.ReLU(Conv(Engine.TensorConcatenate(new[] { x, Engine.Reshape(past, s.ToArray()) }, 1), w[0].Value, w[1].Value));
        y = Engine.ReLU(Conv(y, w[2].Value, w[3].Value));
        y = Conv(y, w[4].Value, w[5].Value);
        return Engine.ReLU(Engine.TensorAdd(y, x));
    }

    private static Tensor<T> Nearest2x(Tensor<T> x)
    {
        var s = x.Shape;
        var wide = Engine.TensorBroadcastTo(Engine.Reshape(x, new[] { s[0], s[1], s[2], 1, s[3], 1 }), new[] { s[0], s[1], s[2], 2, s[3], 2 });
        return Engine.Reshape(wide, new[] { s[0], s[1], 2 * s[2], 2 * s[3] });
    }

    /// <summary>Latents [B, C, n, h, w] and the LR clip [B, 3, 4n − 3, 8h, 8w] in [-1, 1] to frames [B, 3, 4n − 3, 8h, 8w] in about [0, 1].</summary>
    public Tensor<T> Forward(Tensor<T> latents, Tensor<T> cond)
    {
        int b = latents.Shape[0], c = latents.Shape[1], n = latents.Shape[2], h = latents.Shape[3], w = latents.Shape[4];
        var first = Engine.TensorNarrow(cond, 2, 0, 1);
        int copies = (4 - cond.Shape[2] % 4) % 4;
        if (copies > 0)
            cond = Engine.TensorConcatenate(new[] { Engine.TensorBroadcastTo(first, new[] { b, 3, copies, cond.Shape[3], cond.Shape[4] }), cond }, 2);
        // 'b c (f 4) (h 8) (w 8) -> b f (c 4 8 8) h w'.
        var shuffled = Engine.Reshape(cond, new[] { b, 3, n, 4, h, 8, w, 8 });
        shuffled = Engine.Reshape(Engine.TensorPermute(shuffled, new[] { 0, 2, 1, 3, 5, 7, 4, 6 }), new[] { b, n, 3 * 4 * 8 * 8, h, w });
        var frames = Engine.TensorPermute(latents, new[] { 0, 2, 1, 3, 4 });
        var x = Engine.Reshape(Engine.TensorConcatenate(new[] { shuffled, frames }, 2), new[] { b * n, 3 * 4 * 8 * 8 + c, h, w });

        var three = NumOps.FromDouble(3.0);
        x = Engine.TensorMultiplyScalar(Engine.Tanh(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(1.0 / 3.0))), three);
        x = Engine.ReLU(Conv(x, _stem.Value, _stemBias.Value));
        x = Engine.ReLU(Conv(x, _stemIdentity.Value, null));
        for (int stage = 0; stage < 3; stage++)
        {
            for (int m = 0; m < 3; m++) x = Memory(x, b, _memory[stage * 3 + m]);
            x = Nearest2x(x);
            var grown = Conv(x, _grow[stage].Value, null);
            int ch = _channels[stage];
            x = Engine.Reshape(grown, new[] { grown.Shape[0] * GrowStride[stage], ch, grown.Shape[2], grown.Shape[3] });
            x = Conv(x, _stage[stage].Value, null);
        }
        x = Engine.ReLU(x);
        x = Engine.ReLU(Conv(x, _headIdentity.Value, null));
        x = Conv(x, _out.Value, _outBias.Value);
        int total = x.Shape[0] / b;
        var video = Engine.Reshape(x, new[] { b, total, 3, x.Shape[2], x.Shape[3] });
        video = Engine.TensorNarrow(video, 1, 3, total - 3);
        return Engine.TensorPermute(video, new[] { 0, 2, 1, 3, 4 });
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _stem;
        yield return _stemBias;
        yield return _stemIdentity;
        for (int stage = 0; stage < 3; stage++)
        {
            for (int m = 0; m < 3; m++)
                foreach (var weight in _memory[stage * 3 + m]) yield return weight;
            yield return _grow[stage];
            yield return _stage[stage];
        }
        yield return _headIdentity;
        yield return _out;
        yield return _outBias;
    }
}

/// <summary>
/// LPIPS (Zhang et al. 2018, richzhang/PerceptualSimilarity) on VGG16: the scaling layer, the five ReLU stages' features
/// unit-normalized over channels, squared differences weighted per channel by the linear layers, averaged over space and
/// summed over stages. The linear weights come from the released checkpoint; untrained they are one, which is LPIPS's
/// uncalibrated baseline.
/// </summary>
internal sealed class LpipsVggBlock<T>
{
    private static readonly int[] Widths = { 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512 };
    private static readonly int[] StageEnds = { 1, 3, 6, 9, 12 };
    private static readonly int[] StageChannels = { 64, 128, 256, 512, 512 };
    private static readonly double[] Shift = { -0.030, -0.088, -0.188 };
    private static readonly double[] Scale = { 0.458, 0.448, 0.450 };
    private readonly int[] _convs = new int[13];
    private readonly int[] _lin = new int[5];

    internal LpipsVggBlock(List<ILayer<T>> layers)
    {
        First = layers.Count;
        for (int i = 0; i < Widths.Length; i++)
        {
            _convs[i] = layers.Count;
            layers.Add(BasicVsrBlocks.Conv<T>(Widths[i], 3, null));
        }
        for (int i = 0; i < 5; i++)
        {
            _lin[i] = layers.Count;
            layers.Add(RealViformerBlocks.Vector<T>(StageChannels[i], 1.0));
        }
        End = layers.Count;
    }

    internal int First { get; }

    internal int End { get; }

    private List<Tensor<T>> Features(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> images)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var channels = new Tensor<T>[3];
        for (int c = 0; c < 3; c++)
            channels[c] = engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorNarrow(images, 1, c, 1), ops.FromDouble(-Shift[c])),
                ops.FromDouble(1.0 / Scale[c]));
        var x = engine.TensorConcatenate(channels, 1);
        var features = new List<Tensor<T>>();
        int stage = 0;
        for (int i = 0; i < Widths.Length; i++)
        {
            if (stage > 0 && i == StageEnds[stage - 1] + 1) x = engine.MaxPool2D(x, 2, 2);
            x = engine.ReLU(BasicVsrBlocks.Run(layers, _convs[i], x));
            if (i == StageEnds[stage])
            {
                features.Add(x);
                stage++;
            }
        }
        return features;
    }

    private static Tensor<T> Unit(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(x), new[] { 1 }, keepDims: true));
        return engine.TensorDivide(x, engine.TensorBroadcastTo(engine.TensorAddScalar(norm, ops.FromDouble(1e-10)), x.Shape.ToArray()));
    }

    /// <summary>The LPIPS distance averaged over images [N, 3, H, W] in [-1, 1].</summary>
    internal Tensor<T> Loss(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> output, Tensor<T> target)
    {
        List<Tensor<T>> targets;
        using (new NoGradScope<T>()) targets = Features(engine, layers, target);
        var outputs = Features(engine, layers, output);
        Tensor<T>? total = null;
        for (int k = 0; k < outputs.Count; k++)
        {
            var diff = engine.TensorSquare(engine.TensorSubtract(Unit(engine, outputs[k]), Unit(engine, targets[k])));
            int ch = diff.Shape[1];
            var lin = engine.TensorBroadcastTo(engine.Reshape(RealViformerBlocks.Read(layers, _lin[k]), new[] { 1, ch, 1, 1 }), diff.Shape.ToArray());
            var weighted = engine.ReduceSum(engine.TensorMultiply(diff, lin), new[] { 1 }, keepDims: false);
            var term = engine.ReduceMean(weighted, new[] { 0, 1, 2 }, false);
            total = total is null ? term : engine.TensorAdd(total, term);
        }
        return total ?? throw new InvalidOperationException("LPIPS has no stages.");
    }
}
