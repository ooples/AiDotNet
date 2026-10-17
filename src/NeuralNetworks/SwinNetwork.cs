using AiDotNet.Helpers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.NeuralNetworks;

/// <summary>The shape of a Swin Transformer; the defaults are Swin-T.</summary>
internal sealed record SwinConfiguration
{
    public int Channels { get; init; } = 3;

    public int PatchSize { get; init; } = 4;

    public int EmbedDim { get; init; } = 96;

    public int[] Depths { get; init; } = { 2, 2, 6, 2 };

    public int[] Heads { get; init; } = { 3, 6, 12, 24 };

    public int WindowSize { get; init; } = 7;

    public double MlpRatio { get; init; } = 4.0;

    public double LayerNormEpsilon { get; init; } = 1e-5;

    /// <summary>Stochastic depth at the last block, rising linearly from 0 (0.1 for Swin-T as a detection backbone).</summary>
    public double DropPathRate { get; init; } = 0.1;

    /// <summary>The stages (1-based, as Hugging Face's <c>out_indices</c>) whose outputs are returned, each through its
    /// own LayerNorm.</summary>
    public int[] OutStages { get; init; } = { 2, 3, 4 };
}

/// <summary>
/// The Swin Transformer (Liu et al. 2021) as a feature backbone, as Hugging Face's <c>SwinBackbone</c> computes it: a
/// patch embedding, stages of window-attention blocks whose windows shift by half every other block, patch merging
/// between stages, and the chosen stages' outputs (before merging) layer-normalized.
/// </summary>
/// <remarks>
/// <para>Windows are always partitioned at the configured size: a feature map smaller than a window, or not a multiple
/// of it, is zero-padded, as the backbone does (<c>always_partition=True</c>).</para>
/// <para>Parameter names follow the Hugging Face <c>SwinBackbone</c> checkpoints (<c>swin.embeddings.*</c>,
/// <c>swin.encoder.layers.{s}.blocks.{b}.*</c>, <c>hidden_states_norms.stage{s}.*</c>).</para>
/// </remarks>
internal sealed class SwinNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private Random _dropPathRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(17);

    internal sealed record Block(LayerNormalizationLayer<T> Norm1, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value,
        DenseLayer<T> Output, BiasFreeLinearLayer<T> RelativeBias, LayerNormalizationLayer<T> Norm2, DenseLayer<T> Fc1, DenseLayer<T> Fc2,
        int Heads, int Shift, double DropPath);

    internal sealed record Merge(LayerNormalizationLayer<T> Norm, BiasFreeLinearLayer<T> Reduction);

    public SwinNetwork(IEngine engine, List<LayerBase<T>> layers, SwinConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Depths.Length != c.Heads.Length) throw new ArgumentException("One head count per stage.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm(int dim) => Add(new LayerNormalizationLayer<T>(dim, c.LayerNormEpsilon));

        PatchProjection = Linear(c.EmbedDim);
        EmbeddingNorm = Norm(c.EmbedDim);
        int total = c.Depths.Sum(), index = 0, ws = c.WindowSize;
        for (int s = 0; s < c.Depths.Length; s++)
        {
            int dim = c.EmbedDim << s;
            var blocks = new List<Block>();
            for (int i = 0; i < c.Depths[s]; i++, index++)
            {
                double rate = c.DropPathRate * index / Math.Max(total - 1, 1);
                blocks.Add(new Block(Norm(dim), Linear(dim), Linear(dim), Linear(dim), Linear(dim),
                    Add(new BiasFreeLinearLayer<T>(1, (2 * ws - 1) * (2 * ws - 1) * c.Heads[s])), Norm(dim),
                    Linear((int)(c.MlpRatio * dim)), Linear(dim), c.Heads[s], i % 2 == 0 ? 0 : ws / 2, rate));
            }
            Stages.Add(blocks);
            if (s < c.Depths.Length - 1) Merges.Add(new Merge(Norm(4 * dim), Add(new BiasFreeLinearLayer<T>(4 * dim, 2 * dim))));
        }
        foreach (int stage in c.OutStages) OutNorms.Add(Norm(c.EmbedDim << (stage - 1)));
        _relativeIndex = RelativeIndex(ws);
    }

    public SwinConfiguration Configuration { get; }

    public DenseLayer<T> PatchProjection { get; }

    public LayerNormalizationLayer<T> EmbeddingNorm { get; }

    public List<List<Block>> Stages { get; } = new();

    public List<Merge> Merges { get; } = new();

    public List<LayerNormalizationLayer<T>> OutNorms { get; } = new();

    /// <summary>Channels of each returned feature map.</summary>
    public int[] OutChannels => Configuration.OutStages.Select(s => Configuration.EmbedDim << (s - 1)).ToArray();

    /// <summary>Strides of each returned feature map.</summary>
    public int[] OutStrides => Configuration.OutStages.Select(s => Configuration.PatchSize << (s - 1)).ToArray();

    private readonly Tensor<int> _relativeIndex;

    /// <summary>Seeds the stochastic-depth draws (training only).</summary>
    public void SeedDropPath(Random random) => _dropPathRandom = random;

    /// <summary>Whether stochastic depth is active.</summary>
    public bool Training { get; set; }

    private static Tensor<int> RelativeIndex(int ws)
    {
        int n = ws * ws;
        var index = new Tensor<int>(new[] { n * n });
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                int dy = i / ws - j / ws + ws - 1, dx = i % ws - j % ws + ws - 1;
                index[i * n + j] = dy * (2 * ws - 1) + dx;
            }
        return index;
    }

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(_engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    /// <summary>The chosen stages' feature maps <c>[B, C_s, H_s, W_s]</c> of images <c>[B, Channels, H, W]</c>.</summary>
    public List<Tensor<T>> Forward(Tensor<T> images)
    {
        var c = Configuration;
        int b = images.Shape[0], p = c.PatchSize;
        int h = images.Shape[2], w = images.Shape[3];
        // maybe_pad to a multiple of the patch, then the strided patch convolution as unfold + linear.
        int ph = (h + p - 1) / p * p, pw = (w + p - 1) / p * p;
        var x4 = images;
        if (ph != h) x4 = _engine.TensorConcatenate(new[] { x4, new Tensor<T>(new[] { b, images.Shape[1], ph - h, w }) }, 2);
        if (pw != w) x4 = _engine.TensorConcatenate(new[] { x4, new Tensor<T>(new[] { b, images.Shape[1], ph, pw - w }) }, 3);
        int gh = ph / p, gw = pw / p;
        var columns = _engine.Unfold(x4, new[] { p, p }, new[] { p, p }, new[] { 0, 0 });              // [B, C·p·p, L]
        var x = Rows(EmbeddingNorm, Rows(PatchProjection, _engine.TensorPermute(columns, new[] { 0, 2, 1 })));  // [B, L, D]
        var outputs = new List<Tensor<T>>();
        for (int s = 0; s < Stages.Count; s++)
        {
            foreach (var block in Stages[s]) x = Run(block, x, gh, gw);
            int outIndex = Array.IndexOf(c.OutStages, s + 1);
            if (outIndex >= 0)
            {
                int dim = c.EmbedDim << s;
                var normed = Rows(OutNorms[outIndex], x);
                outputs.Add(_engine.TensorPermute(_engine.Reshape(normed, new[] { b, gh, gw, dim }), new[] { 0, 3, 1, 2 }));
            }
            if (s < Merges.Count)
            {
                x = MergePatches(Merges[s], x, gh, gw);
                gh = (gh + 1) / 2;
                gw = (gw + 1) / 2;
            }
        }
        return outputs;
    }

    private Tensor<T> MergePatches(Merge merge, Tensor<T> x, int gh, int gw)
    {
        int b = x.Shape[0], d = x.Shape[2];
        var grid = _engine.Reshape(x, new[] { b, gh, gw, d });
        int eh = gh + gh % 2, ew = gw + gw % 2;
        if (eh != gh) grid = _engine.TensorConcatenate(new[] { grid, new Tensor<T>(new[] { b, eh - gh, gw, d }) }, 1);
        if (ew != gw) grid = _engine.TensorConcatenate(new[] { grid, new Tensor<T>(new[] { b, eh, ew - gw, d }) }, 2);
        // [x[0::2, 0::2], x[1::2, 0::2], x[0::2, 1::2], x[1::2, 1::2]] concatenated on channels.
        var split = _engine.Reshape(grid, new[] { b, eh / 2, 2, ew / 2, 2, d });                  // [B, h, r, w, c, D]
        var ordered = _engine.TensorPermute(split, new[] { 0, 1, 3, 4, 2, 5 });                    // [B, h, w, c, r, D]
        var flat = _engine.Reshape(ordered, new[] { b, eh / 2 * (ew / 2), 4 * d });
        return Rows(merge.Reduction, Rows(merge.Norm, flat));
    }

    private Tensor<T> Run(Block block, Tensor<T> x, int gh, int gw)
    {
        int b = x.Shape[0], d = x.Shape[2], ws = Configuration.WindowSize, heads = block.Heads, hd = d / heads, n = ws * ws;
        var h = _engine.Reshape(Rows(block.Norm1, x), new[] { b, gh, gw, d });
        int ph = (gh + ws - 1) / ws * ws, pw = (gw + ws - 1) / ws * ws;
        if (pw != gw) h = _engine.TensorConcatenate(new[] { h, new Tensor<T>(new[] { b, gh, pw - gw, d }) }, 2);
        if (ph != gh) h = _engine.TensorConcatenate(new[] { h, new Tensor<T>(new[] { b, ph - gh, pw, d }) }, 1);
        if (block.Shift > 0) h = Roll(h, -block.Shift, ph, pw);
        int nh = ph / ws, nw = pw / ws, windows = nh * nw;
        var win = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(h, new[] { b, nh, ws, nw, ws, d }), new[] { 0, 1, 3, 2, 4, 5 }),
            new[] { b * windows, n, d });
        Tensor<T> Heads(DenseLayer<T> projection) => _engine.Reshape(
            _engine.TensorPermute(_engine.Reshape(Rows(projection, win), new[] { b * windows, n, heads, hd }), new[] { 0, 2, 1, 3 }),
            new[] { b * windows * heads, n, hd });
        var q = Heads(block.Query);
        var k = Heads(block.Key);
        var v = Heads(block.Value);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        // Relative position bias [heads, n, n], plus the shifted-window mask [windows, n, n].
        var table = _engine.Reshape(block.RelativeBias.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One }))),
            new[] { (2 * ws - 1) * (2 * ws - 1), heads });
        var bias = _engine.TensorPermute(_engine.Reshape(_engine.TensorIndexSelect(table, _relativeIndex, 0), new[] { n, n, heads }), new[] { 2, 0, 1 });
        var full = _engine.Reshape(scores, new[] { b, windows, heads, n, n });
        full = _engine.TensorAdd(full, _engine.TensorBroadcastTo(_engine.Reshape(bias, new[] { 1, 1, heads, n, n }), new[] { b, windows, heads, n, n }));
        if (block.Shift > 0)
            full = _engine.TensorAdd(full, _engine.TensorBroadcastTo(_engine.Reshape(ShiftMask(ph, pw, ws, block.Shift), new[] { 1, windows, 1, n, n }),
                new[] { b, windows, heads, n, n }));
        var probabilities = _engine.Softmax(_engine.Reshape(full, new[] { b * windows * heads, n, n }), axis: 2);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.BatchMatMul(probabilities, v), new[] { b * windows, heads, n, hd }),
            new[] { 0, 2, 1, 3 }), new[] { b * windows, n, d });
        var attended = Rows(block.Output, context);
        var back = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(attended, new[] { b, nh, nw, ws, ws, d }), new[] { 0, 1, 3, 2, 4, 5 }),
            new[] { b, ph, pw, d });
        if (block.Shift > 0) back = Roll(back, block.Shift, ph, pw);
        if (ph != gh || pw != gw) back = _engine.TensorSlice(back, new[] { 0, 0, 0, 0 }, new[] { b, gh, gw, d });
        x = _engine.TensorAdd(x, DropPath(_engine.Reshape(back, new[] { b, gh * gw, d }), block.DropPath));
        var mlp = Rows(block.Fc2, ActivationHelper.ExactGelu(Rows(block.Fc1, Rows(block.Norm2, x)), _engine));
        return _engine.TensorAdd(x, DropPath(mlp, block.DropPath));
    }

    /// <summary>torch.roll over the two spatial axes of <c>[B, H, W, D]</c>.</summary>
    private Tensor<T> Roll(Tensor<T> x, int shift, int h, int w)
    {
        Tensor<T> RollAxis(Tensor<T> t, int axis, int size)
        {
            int k = ((shift % size) + size) % size;
            if (k == 0) return t;
            var shape = t.Shape.ToArray();
            var start = new int[4];
            var tailStart = (int[])start.Clone();
            tailStart[axis] = size - k;
            var tailShape = (int[])shape.Clone();
            tailShape[axis] = k;
            var headShape = (int[])shape.Clone();
            headShape[axis] = size - k;
            return _engine.TensorConcatenate(new[] { _engine.TensorSlice(t, tailStart, tailShape), _engine.TensorSlice(t, start, headShape) }, axis);
        }
        return RollAxis(RollAxis(x, 1, h), 2, w);
    }

    /// <summary>The shifted-window attention mask (−100 between tokens from different regions), <c>[windows, n, n]</c>.</summary>
    private static Tensor<T> ShiftMask(int h, int w, int ws, int shift)
    {
        int nh = h / ws, nw = w / ws, n = ws * ws;
        var region = new int[h, w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int ry = (y >= h - ws ? 1 : 0) + (y >= h - shift ? 1 : 0);
                int rx = (x >= w - ws ? 1 : 0) + (x >= w - shift ? 1 : 0);
                region[y, x] = ry * 3 + rx;
            }
        var mask = new Tensor<T>(new[] { nh * nw, n, n });
        var blocked = NumOps.FromDouble(-100.0);
        for (int wy = 0; wy < nh; wy++)
            for (int wx = 0; wx < nw; wx++)
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        if (region[wy * ws + i / ws, wx * ws + i % ws] != region[wy * ws + j / ws, wx * ws + j % ws])
                            mask[wy * nw + wx, i, j] = blocked;
        return mask;
    }

    /// <summary>Stochastic depth (drop the whole residual branch per sample with probability <paramref name="rate"/>, scale
    /// the kept ones by 1/(1−rate)); the identity outside training.</summary>
    private Tensor<T> DropPath(Tensor<T> x, double rate)
    {
        if (!Training || rate <= 0) return x;
        int b = x.Shape[0];
        var keep = new Tensor<T>(new[] { b, 1, 1 });
        for (int i = 0; i < b; i++) keep[i, 0, 0] = NumOps.FromDouble(_dropPathRandom.NextDouble() < 1 - rate ? 1.0 / (1 - rate) : 0.0);
        return _engine.TensorMultiply(x, _engine.TensorBroadcastTo(keep, x.Shape.ToArray()));
    }

    /// <summary>Loads a Hugging Face <c>SwinBackbone</c> state dict (names after an optional <paramref name="prefix"/>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "")
    {
        var c = Configuration;
        int p = c.PatchSize, ws = c.WindowSize;
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{prefix}{name}.weight", new[] { outputs, inputs }),
                read($"{prefix}{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name, int dim)
            => TorchParameters.LayerNorm(_engine, norm, read($"{prefix}{name}.weight", new[] { dim }), read($"{prefix}{name}.bias", new[] { dim }));
        TorchParameters.Linear(_engine, PatchProjection, c.Channels * p * p, c.EmbedDim,
            read($"{prefix}swin.embeddings.patch_embeddings.projection.weight", new[] { c.EmbedDim, c.Channels, p, p }),
            read($"{prefix}swin.embeddings.patch_embeddings.projection.bias", new[] { c.EmbedDim }));
        Norm(EmbeddingNorm, "swin.embeddings.norm", c.EmbedDim);
        for (int s = 0; s < Stages.Count; s++)
        {
            int dim = c.EmbedDim << s;
            for (int i = 0; i < Stages[s].Count; i++)
            {
                var blk = Stages[s][i];
                string n = $"swin.encoder.layers.{s}.blocks.{i}";
                Norm(blk.Norm1, $"{n}.layernorm_before", dim);
                Linear(blk.Query, $"{n}.attention.q_proj", dim, dim);
                Linear(blk.Key, $"{n}.attention.k_proj", dim, dim);
                Linear(blk.Value, $"{n}.attention.v_proj", dim, dim);
                Linear(blk.Output, $"{n}.attention.o_proj", dim, dim);
                int entries = (2 * ws - 1) * (2 * ws - 1) * blk.Heads;
                TorchParameters.Linear(blk.RelativeBias, 1, entries,
                    read($"{prefix}{n}.attention.relative_position_bias.relative_position_bias_table", new[] { (2 * ws - 1) * (2 * ws - 1), blk.Heads }));
                Norm(blk.Norm2, $"{n}.layernorm_after", dim);
                Linear(blk.Fc1, $"{n}.mlp.fc1", dim, (int)(c.MlpRatio * dim));
                Linear(blk.Fc2, $"{n}.mlp.fc2", (int)(c.MlpRatio * dim), dim);
            }
            if (s < Merges.Count)
            {
                Norm(Merges[s].Norm, $"swin.encoder.layers.{s}.downsample.norm", 4 * dim);
                TorchParameters.Linear(Merges[s].Reduction, 4 * dim, 2 * dim, read($"{prefix}swin.encoder.layers.{s}.downsample.reduction.weight", new[] { 2 * dim, 4 * dim }));
            }
        }
        for (int i = 0; i < c.OutStages.Length; i++)
            Norm(OutNorms[i], $"hidden_states_norms.stage{c.OutStages[i]}", c.EmbedDim << (c.OutStages[i] - 1));
    }
}
