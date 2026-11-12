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
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "The attention width must be positive.");
        int h = Math.Max(1, Math.Min(heads, width));
        while (width % h != 0) h--;
        var layer = Add(new MultiHeadAttentionLayer<T>(h, width / h));
        layer.UseCausalMask = causal;
        return layer;
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
        var values = new Tensor<int>(new[] { ids.Count }, Vector<int>.FromList(ids.ToList()));
        return Engine.TensorOneHot<T>(values, depth);
    }

    /// <summary>
    /// <c>torch.special.logit</c> with the reference's clamp: <c>log(p / (1 - p))</c> of <paramref name="p"/> clamped
    /// to <c>[eps, 1 - eps]</c> (DETR's <c>inverse_sigmoid</c>, eps 1e-5 as Grounding DINO's).
    /// </summary>
    internal static Tensor<T> InverseSigmoid(IEngine engine, Tensor<T> p, double eps = 1e-5)
    {
        var clamped = engine.TensorClamp(p, NumOps.FromDouble(eps), NumOps.FromDouble(1 - eps));
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
        for (int step = 0; step < maxSteps; step++)
        {
            int width = sequence.Shape[1];
            var hidden = AsRows(Engine, languageModel(sequence));
            var last = Engine.TensorSlice(hidden, new[] { hidden.Shape[0] - 1, 0 }, new[] { 1, hidden.Shape[1] });
            var logits = Rows(head, last);
            var permitted = allowed(tokens);
            var blocked = Tensor<T>.CreateDefault(logits.Shape.ToArray(), NumOps.MinValue);
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

    /// <summary>The rows <c>[n, d]</c> of a tensor whose last axis is <c>d</c>.</summary>
    internal static Tensor<T> AsRows(IEngine engine, Tensor<T> x)
    {
        int d = x.Shape[x.Rank - 1];
        return engine.Reshape(x, new[] { x.Length / d, d });
    }
}
