using AiDotNet.Helpers;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// A pre-norm transformer block of the grounding heads: <c>x + SelfAttention(LN(x))</c>, optionally
/// <c>x + CrossAttention(LN(x), memory)</c>, then <c>x + W2·GELU(W1·LN(x))</c>.
/// </summary>
internal sealed record GroundingTransformerBlock<T>(
    LayerNormalizationLayer<T> SelfNorm, MultiHeadAttentionLayer<T> SelfAttention,
    LayerNormalizationLayer<T>? CrossNorm, MultiHeadAttentionLayer<T>? CrossAttention,
    LayerNormalizationLayer<T> FeedForwardNorm, DenseLayer<T> FeedForwardUp, DenseLayer<T> FeedForwardDown);

/// <summary>
/// The layers of one multi-scale deformable attention (Zhu et al. 2021, Deformable DETR, Eq. 3): a value projection,
/// per-head sampling offsets and attention weights for every level and point, and an output projection.
/// </summary>
internal sealed record DeformableAttentionLayers<T>(DenseLayer<T> Value, DenseLayer<T> Offsets, DenseLayer<T> Weights,
    DenseLayer<T> Output, int Heads, int Levels, int Points);

/// <summary>
/// The layers of a simple feature pyramid (Li et al. 2022, ViTDet, section 3) over a plain backbone's last feature map:
/// one 1x1 projection and normalization per level, and a learned embedding per level (Deformable DETR's level embedding).
/// </summary>
internal sealed record FeaturePyramidLayers<T>(DenseLayer<T>[] Projections, LayerNormalizationLayer<T>[] Norms,
    DenseLayer<T> LevelEmbedding, int Levels);

/// <summary>
/// The multi-scale tokens of a feature pyramid, finest level first, flattened row-major level after level.
/// </summary>
/// <param name="Tokens">The <c>[S, d]</c> tokens of every level.</param>
/// <param name="LevelEmbedding">The <c>[S, d]</c> level embedding of every token.</param>
/// <param name="Centres">The <c>[S, 2]</c> normalized <c>(x, y)</c> centre of every token on its level.</param>
/// <param name="Shapes">The <c>(H, W)</c> of every level.</param>
/// <param name="Starts">The index of every level's first token.</param>
internal sealed record FeatureLevels<T>(Tensor<T> Tokens, Tensor<T> LevelEmbedding, Tensor<T> Centres, (int H, int W)[] Shapes, int[] Starts)
{
    public int Count => Shapes.Length;
}

/// <summary>
/// Shared plumbing of the trainable grounding heads: every layer a head creates is added to one stream the model
/// registers as an auxiliary encoder stream, so the head's weights are trained, counted and saved with the model's.
/// Every layer is built lazily (its input width resolves on the first forward), as the models' own layers are.
/// </summary>
/// <typeparam name="T">The numeric type of the head.</typeparam>
internal abstract class GroundingHeadModule<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected GroundingHeadModule(IEngine engine, List<ILayer<T>> stream)
    {
        Engine = engine ?? throw new ArgumentNullException(nameof(engine));
        Stream = stream ?? throw new ArgumentNullException(nameof(stream));
    }

    protected IEngine Engine { get; }

    /// <summary>The layers of the head, in creation order.</summary>
    public List<ILayer<T>> Stream { get; }

    protected TLayer Add<TLayer>(TLayer layer) where TLayer : ILayer<T>
    {
        Stream.Add(layer);
        return layer;
    }

    /// <summary>A linear layer <c>xW + b</c> (or <c>xW</c> without a bias) whose input width resolves on first use.</summary>
    protected DenseLayer<T> Linear(int outputs, bool bias = true)
        => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>?)null, null, bias ? BiasMode.Always : BiasMode.Never));

    /// <summary>A layer normalization over the last axis whose width resolves on first use.</summary>
    protected LayerNormalizationLayer<T> Norm() => Add(new LayerNormalizationLayer<T>(1e-5));

    /// <summary>
    /// Multi-head attention of <paramref name="width"/> features split over <paramref name="heads"/> heads, or over the
    /// most heads below that which divide the width evenly.
    /// </summary>
    protected MultiHeadAttentionLayer<T> Attention(int width, int heads, bool causal = false)
    {
        int h = HeadsFor(width, heads);
        var layer = Add(new MultiHeadAttentionLayer<T>(h, width / h));
        layer.UseCausalMask = causal;
        return layer;
    }

    /// <summary>The most heads, at most <paramref name="heads"/>, that split <paramref name="width"/> features evenly.</summary>
    internal static int HeadsFor(int width, int heads)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "The attention width must be positive.");
        int h = Math.Max(1, Math.Min(heads, width));
        while (width % h != 0) h--;
        return h;
    }

    /// <summary>A pre-norm transformer block of <paramref name="width"/> features with a <paramref name="feedForward"/>-wide MLP.</summary>
    protected GroundingTransformerBlock<T> Block(int width, int heads, int feedForward, bool cross, bool causal = false)
    {
        var selfNorm = Norm();
        var self = Attention(width, heads, causal);
        var crossNorm = cross ? Norm() : null;
        var crossAttention = cross ? Attention(width, heads) : null;
        return new GroundingTransformerBlock<T>(selfNorm, self, crossNorm, crossAttention, Norm(), Linear(feedForward), Linear(width));
    }

    /// <summary>Runs a layer on the rows of a <c>[n, d]</c> matrix.</summary>
    protected Tensor<T> Rows(ILayer<T> layer, Tensor<T> x) => layer.Forward(x);

    /// <summary>Self-attention of the rows of <c>[n, d]</c> over themselves.</summary>
    protected Tensor<T> SelfAttend(MultiHeadAttentionLayer<T> attention, Tensor<T> x)
    {
        int n = x.Shape[0], d = x.Shape[1];
        return Engine.Reshape(attention.Forward(Engine.Reshape(x, new[] { 1, n, d })), new[] { n, d });
    }

    /// <summary>Cross-attention of the rows of <c>[n, d]</c> to the rows of <c>[m, d]</c>.</summary>
    protected Tensor<T> CrossAttend(MultiHeadAttentionLayer<T> attention, Tensor<T> x, Tensor<T> memory)
    {
        int n = x.Shape[0], d = x.Shape[1], m = memory.Shape[0];
        return Engine.Reshape(
            attention.Forward(Engine.Reshape(x, new[] { 1, n, d }), Engine.Reshape(memory, new[] { 1, m, memory.Shape[1] })),
            new[] { n, d });
    }

    /// <summary>Runs a pre-norm transformer block on <c>[n, d]</c> rows, cross-attending to <paramref name="memory"/>
    /// when the block has a cross-attention.</summary>
    protected Tensor<T> Run(GroundingTransformerBlock<T> block, Tensor<T> x, Tensor<T>? memory)
    {
        x = Engine.TensorAdd(x, SelfAttend(block.SelfAttention, Rows(block.SelfNorm, x)));
        if (block.CrossAttention is not null && block.CrossNorm is not null && memory is not null)
            x = Engine.TensorAdd(x, CrossAttend(block.CrossAttention, Rows(block.CrossNorm, x), memory));
        var hidden = Engine.GELU(Rows(block.FeedForwardUp, Rows(block.FeedForwardNorm, x)));
        return Engine.TensorAdd(x, Rows(block.FeedForwardDown, hidden));
    }

    /// <summary>An MLP with <paramref name="activation"/> between its layers (none after the last).</summary>
    protected Tensor<T> Mlp(DenseLayer<T>[] layers, Tensor<T> x, Func<Tensor<T>, Tensor<T>> activation)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            x = Rows(layers[i], x);
            if (i < layers.Length - 1) x = activation(x);
        }
        return x;
    }

    /// <summary>The one-hot rows <c>[ids.Count, depth]</c> of integer ids.</summary>
    protected Tensor<T> OneHot(IReadOnlyList<int> ids, int depth)
    {
        if (ids.Any(id => id < 0 || id >= depth))
            throw new ArgumentOutOfRangeException(nameof(ids), $"Every token id must lie in [0, {depth}).");
        var values = new Tensor<int>(new[] { ids.Count }, Vector<int>.FromList(ids.ToList()));
        return Engine.TensorOneHot<T>(values, depth);
    }

    /// <summary>
    /// <c>torch.special.logit</c> with the reference's clamp: <c>log(p / (1 - p))</c> of <paramref name="p"/> clamped
    /// to <c>[eps, 1 - eps]</c> (DETR's <c>inverse_sigmoid</c>, eps 1e-5 as Grounding DINO's, or the type's precision if
    /// coarser).
    /// </summary>
    internal static Tensor<T> InverseSigmoid(IEngine engine, Tensor<T> p, double eps = 1e-5)
    {
        // A type too narrow to hold 1 - eps (Half) clamps one unit in the last place below 1 instead, so the logit
        // stays finite.
        double guard = Math.Max(eps, Math.Pow(2, -(NumOps.PrecisionBits - 1)));
        var clamped = engine.TensorClamp(p, NumOps.FromDouble(guard), NumOps.FromDouble(1 - guard));
        return engine.TensorLog(engine.TensorDivide(clamped, engine.ScalarMinusTensor(NumOps.One, clamped)));
    }

    /// <summary>Boxes <c>[n, 4]</c> of (cx, cy, w, h) as (x1, y1, x2, y2) clipped to the unit square.</summary>
    internal static Tensor<T> Corners(IEngine engine, Tensor<T> centred)
    {
        int n = centred.Shape[0];
        Tensor<T> Column(int c) => engine.TensorSlice(centred, new[] { 0, c }, new[] { n, 1 });
        return GroundingDetections<T>.ClippedCornerBoxes(engine, Column(0), Column(1), Column(2), Column(3));
    }

    /// <summary>
    /// The centre of every cell of an <c>h x w</c> grid of tokens in row-major order, normalized to the unit square:
    /// <c>[h·w, 2]</c> rows of (x, y) = ((j + 0.5) / w, (i + 0.5) / h).
    /// </summary>
    internal static Tensor<T> GridCentres(IEngine engine, int h, int w)
    {
        var columns = engine.TensorDivideScalar(engine.TensorAddScalar(engine.TensorArange<T>(0, w, 1), NumOps.FromDouble(0.5)), NumOps.FromDouble(w));
        var rows = engine.TensorDivideScalar(engine.TensorAddScalar(engine.TensorArange<T>(0, h, 1), NumOps.FromDouble(0.5)), NumOps.FromDouble(h));
        var x = engine.TensorTile(columns.Reshape(new[] { 1, w }), new[] { h, 1 }).Reshape(new[] { h * w, 1 });
        var y = engine.TensorRepeatInterleave(rows, w, 0).Reshape(new[] { h * w, 1 });
        return engine.TensorConcatenate(new[] { x, y }, 1);
    }

    /// <summary>The token grid of <paramref name="tokens"/> image tokens: square when the count is a square, else one row.</summary>
    internal static (int H, int W) TokenGrid(int tokens)
    {
        int side = (int)Math.Round(Math.Sqrt(tokens));
        return side * side == tokens ? (side, side) : (1, tokens);
    }

    /// <summary>
    /// Greedy decoding through a language model: each step runs <paramref name="languageModel"/> on the prefix and the
    /// embeddings of the tokens chosen so far, scores the last position with <paramref name="head"/>, keeps only the
    /// tokens <paramref name="allowed"/> permits after the tokens so far (a <c>[1, V]</c> 0/1 row), and takes the most
    /// likely one (the first on a tie). Stops after <paramref name="stopToken"/> or <paramref name="maxSteps"/> tokens.
    /// </summary>
    /// <returns>The chosen tokens and each one's probability <c>[1, 1]</c> under the constrained distribution.</returns>
    protected (List<int> Tokens, List<Tensor<T>> Probabilities) DecodeGreedy(Tensor<T> prefix,
        Func<Tensor<T>, Tensor<T>> languageModel, DenseLayer<T> head, Func<int, Tensor<T>> embed,
        Func<IReadOnlyList<int>, Tensor<T>> allowed, int stopToken, int maxSteps)
    {
        var tokens = new List<int>();
        var probabilities = new List<Tensor<T>>();
        var sequence = prefix;
        var blockedValue = NumOps.MinValue;
        for (int step = 0; step < maxSteps; step++)
        {
            int width = sequence.Shape[1];
            var hidden = AsRows(Engine, languageModel(sequence));
            var last = Engine.TensorSlice(hidden, new[] { hidden.Shape[0] - 1, 0 }, new[] { 1, hidden.Shape[1] });
            var logits = Rows(head, last);
            var permitted = allowed(tokens);
            var blocked = Tensor<T>.CreateDefault(logits.Shape.ToArray(), blockedValue);
            var constrained = Engine.TensorWhere(permitted, logits, blocked);
            int token = ArgExtremumSearch<T>.FirstIndex(Engine, constrained, true, out _);
            if (token < 0) break;
            var distribution = Engine.Softmax(constrained, 1);
            probabilities.Add(Engine.TensorSlice(distribution, new[] { 0, token }, new[] { 1, 1 }));
            tokens.Add(token);
            if (token == stopToken) break;
            sequence = Engine.TensorConcatenate(new[] { sequence, Engine.Reshape(embed(token), new[] { 1, width }) }, 0);
        }
        return (tokens, probabilities);
    }

    /// <summary>A <c>[1, depth]</c> row of ones over <c>[from, to)</c> and zeros elsewhere.</summary>
    protected Tensor<T> Span(int depth, int from, int to)
    {
        var inside = Engine.TensorMultiply(
            Engine.TensorGreaterThan(Engine.TensorArange<T>(0, depth, 1), NumOps.FromDouble(from - 0.5)),
            Engine.TensorLessThan(Engine.TensorArange<T>(0, depth, 1), NumOps.FromDouble(to - 0.5)));
        return inside.Reshape(new[] { 1, depth });
    }

    /// <summary>The layers of a <paramref name="levels"/>-level simple feature pyramid of <paramref name="width"/> features.</summary>
    protected FeaturePyramidLayers<T> PyramidLayers(int width, int levels)
    {
        if (levels <= 0) throw new ArgumentOutOfRangeException(nameof(levels), "A feature pyramid needs at least one level.");
        var projections = new DenseLayer<T>[levels];
        var norms = new LayerNormalizationLayer<T>[levels];
        for (int level = 0; level < levels; level++)
        {
            projections[level] = Linear(width);
            norms[level] = Norm();
        }
        return new FeaturePyramidLayers<T>(projections, norms, Linear(width, bias: false), levels);
    }

    /// <summary>
    /// Builds the multi-scale feature maps of a plain (single-scale) backbone, ViTDet's simple feature pyramid: the
    /// backbone map upsampled x2, the map itself, then max-pooled x2 per further level while the map is at least 2x2.
    /// With the backbone at stride 16 these are DINO's strides 8, 16, 32 and 64.
    /// </summary>
    /// <param name="pyramid">The pyramid's layers.</param>
    /// <param name="imageTokens">The backbone's tokens in row-major grid order.</param>
    internal FeatureLevels<T> Pyramid(FeaturePyramidLayers<T> pyramid, Tensor<T> imageTokens)
    {
        if (pyramid is null) throw new ArgumentNullException(nameof(pyramid));
        if (imageTokens is null) throw new ArgumentNullException(nameof(imageTokens));
        var rows = AsRows(Engine, imageTokens);
        int n = rows.Shape[0], d = rows.Shape[1];
        var grid = TokenGrid(n);
        var map = Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, d, grid.H, grid.W });
        var maps = new List<Tensor<T>>(pyramid.Levels);
        if (pyramid.Levels > 1) maps.Add(Engine.Upsample(map, 2, 2));
        maps.Add(map);
        while (maps.Count < pyramid.Levels && maps[maps.Count - 1].Shape[2] >= 2 && maps[maps.Count - 1].Shape[3] >= 2)
            maps.Add(Engine.MaxPool2D(maps[maps.Count - 1], 2, 2, 0));

        int levels = maps.Count;
        var levelEmbedding = Rows(pyramid.LevelEmbedding, OneHot(Enumerable.Range(0, levels).ToList(), pyramid.Levels));
        int width = levelEmbedding.Shape[1];
        var tokens = new Tensor<T>[levels];
        var embeddings = new Tensor<T>[levels];
        var centres = new Tensor<T>[levels];
        var shapes = new (int H, int W)[levels];
        var starts = new int[levels];
        int start = 0;
        for (int level = 0; level < levels; level++)
        {
            int h = maps[level].Shape[2], w = maps[level].Shape[3], count = h * w;
            var levelRows = Engine.TensorTranspose(Engine.Reshape(maps[level], new[] { d, count }));
            tokens[level] = Rows(pyramid.Norms[level], Rows(pyramid.Projections[level], levelRows));
            embeddings[level] = Engine.TensorBroadcastTo(
                Engine.TensorSlice(levelEmbedding, new[] { level, 0 }, new[] { 1, width }), new[] { count, width });
            centres[level] = GridCentres(Engine, h, w);
            shapes[level] = (h, w);
            starts[level] = start;
            start += count;
        }
        return new FeatureLevels<T>(Engine.TensorConcatenate(tokens, 0), Engine.TensorConcatenate(embeddings, 0),
            Engine.TensorConcatenate(centres, 0), shapes, starts);
    }

    /// <summary>The layers of a multi-scale deformable attention of <paramref name="width"/> features.</summary>
    protected DeformableAttentionLayers<T> DeformableAttention(int width, int heads, int levels, int points)
    {
        if (levels <= 0 || points <= 0)
            throw new ArgumentOutOfRangeException(nameof(levels), "Deformable attention needs at least one level and one point.");
        int h = HeadsFor(width, heads);
        return new DeformableAttentionLayers<T>(Linear(width), Linear(h * levels * points * 2), Linear(h * levels * points),
            Linear(width), h, levels, points);
    }

    /// <summary>
    /// Multi-scale deformable attention (Deformable DETR, Eq. 3): every query attends to <c>Points</c> bilinearly
    /// sampled values per head and level around its reference, at offsets and with weights (a softmax over the levels
    /// and points of each head) predicted from the query.
    /// </summary>
    /// <param name="attention">The attention's layers.</param>
    /// <param name="query">The <c>[Lq, d]</c> queries.</param>
    /// <param name="reference">
    /// <c>[Lq, 2]</c> reference points <c>(x, y)</c> whose offsets are in units of each level's token (the encoder), or
    /// <c>[Lq, 4]</c> reference boxes <c>(cx, cy, w, h)</c> whose offsets are in units of half the box per point (the decoder).
    /// </param>
    /// <param name="value">The multi-scale values.</param>
    internal Tensor<T> Deform(DeformableAttentionLayers<T> attention, Tensor<T> query, Tensor<T> reference, FeatureLevels<T> value)
    {
        if (attention is null) throw new ArgumentNullException(nameof(attention));
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (reference is null) throw new ArgumentNullException(nameof(reference));
        if (value is null) throw new ArgumentNullException(nameof(value));
        int lq = query.Shape[0], d = value.Tokens.Shape[1], m = attention.Heads, p = attention.Points, levels = value.Count;
        if (levels > attention.Levels)
            throw new ArgumentException($"The attention samples {attention.Levels} levels; the values have {levels}.", nameof(value));
        if (d % m != 0) throw new ArgumentException($"{m} heads do not split {d} value features evenly.", nameof(value));
        int dh = d / m, refDim = reference.Shape[1];
        if (reference.Shape[0] != lq || (refDim != 2 && refDim != 4))
            throw new ArgumentException("The references must be [queries, 2] points or [queries, 4] boxes.", nameof(reference));

        var projected = Rows(attention.Value, value.Tokens);
        // A pyramid with fewer levels than the attention was built for uses the first levels' offsets and weights.
        var offsets = Engine.TensorSlice(Engine.Reshape(Rows(attention.Offsets, query), new[] { lq, m, attention.Levels, p, 2 }),
            new[] { 0, 0, 0, 0, 0 }, new[] { lq, m, levels, p, 2 });
        var logits = Engine.TensorSlice(Engine.Reshape(Rows(attention.Weights, query), new[] { lq, m, attention.Levels, p }),
            new[] { 0, 0, 0, 0 }, new[] { lq, m, levels, p });
        var weights = Engine.Softmax(Engine.Reshape(logits, new[] { lq, m, levels * p }), 2);

        var shape = new[] { lq, m, levels, p, 2 };
        Tensor<T> locations;
        if (refDim == 2)
        {
            var inverseSizes = new Tensor<T>(new[] { 1, 1, levels, 1, 2 }, Vector<T>.FromArray(
                value.Shapes.SelectMany(s => new[] { NumOps.FromDouble(1.0 / s.W), NumOps.FromDouble(1.0 / s.H) }).ToArray()));
            locations = Engine.TensorAdd(Engine.TensorBroadcastTo(Engine.Reshape(reference, new[] { lq, 1, 1, 1, 2 }), shape),
                Engine.TensorMultiply(offsets, Engine.TensorBroadcastTo(inverseSizes, shape)));
        }
        else
        {
            var centre = Engine.Reshape(Engine.TensorSlice(reference, new[] { 0, 0 }, new[] { lq, 2 }), new[] { lq, 1, 1, 1, 2 });
            var size = Engine.Reshape(Engine.TensorSlice(reference, new[] { 0, 2 }, new[] { lq, 2 }), new[] { lq, 1, 1, 1, 2 });
            var reach = Engine.TensorMultiplyScalar(Engine.TensorBroadcastTo(size, shape), NumOps.FromDouble(0.5 / p));
            locations = Engine.TensorAdd(Engine.TensorBroadcastTo(centre, shape), Engine.TensorMultiply(offsets, reach));
        }

        // [0, 1] to grid_sample's [-1, 1] (align_corners = false, zero padding), as the reference implementation.
        var grids = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(locations, NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));
        var sampled = new Tensor<T>[levels];
        for (int level = 0; level < levels; level++)
        {
            var (h, w) = value.Shapes[level];
            // [h*w, d] -> [d, h*w] -> [m, dh, h, w]: feature f of head k is column k*dh + f.
            var levelValue = Engine.Reshape(Engine.TensorTranspose(
                Engine.TensorSlice(projected, new[] { value.Starts[level], 0 }, new[] { h * w, d })), new[] { m, dh, h, w });
            // [lq, m, 1, p, 2] -> [m, lq, p, 2]
            var levelGrid = Engine.TensorPermute(Engine.Reshape(
                Engine.TensorSlice(grids, new[] { 0, 0, level, 0, 0 }, new[] { lq, m, 1, p, 2 }), new[] { lq, m, p, 2 }), new[] { 1, 0, 2, 3 });
            sampled[level] = Engine.GridSample(levelValue, levelGrid);
        }

        // [m, dh, lq, levels*p] weighted by [m, 1, lq, levels*p], summed over the samples -> [m*dh, lq] -> [lq, d].
        var samples = Engine.Reshape(Engine.TensorStack(sampled, 3), new[] { m, dh, lq, levels * p });
        var sampleWeights = Engine.Reshape(Engine.TensorPermute(weights, new[] { 1, 0, 2 }), new[] { m, 1, lq, levels * p });
        var summed = Engine.ReduceSum(Engine.TensorMultiply(samples, Engine.TensorBroadcastTo(sampleWeights, new[] { m, dh, lq, levels * p })),
            new[] { 3 }, false);
        return Rows(attention.Output, Engine.TensorTranspose(Engine.Reshape(summed, new[] { d, lq })));
    }

    /// <summary>The rows <c>[n, d]</c> of a tensor whose last axis is <c>d</c>.</summary>
    internal static Tensor<T> AsRows(IEngine engine, Tensor<T> x)
    {
        int d = x.Shape[x.Rank - 1];
        return engine.Reshape(x, new[] { x.Length / d, d });
    }
}
