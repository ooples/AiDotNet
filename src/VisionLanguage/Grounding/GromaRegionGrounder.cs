namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>The shape of a <see cref="GromaRegionGrounder{T}"/>.</summary>
/// <param name="Width">The language model's width.</param>
/// <param name="ProposerWidth">Width of the region proposer (Deformable DETR: 256).</param>
/// <param name="ProposerHeads">Attention heads of the proposer (8).</param>
/// <param name="ProposerLayers">Decoder layers of the proposer (6).</param>
/// <param name="Proposals">Region proposals, and so region tokens, per image (Groma: 300 before filtering).</param>
/// <param name="VocabSize">Text vocabulary size.</param>
/// <param name="MaxTextLength">Longest text the grounder embeds.</param>
internal sealed record GromaRegionGrounderConfiguration(int Width, int ProposerWidth, int ProposerHeads, int ProposerLayers,
    int Proposals, int VocabSize, int MaxTextLength);

/// <summary>
/// Groma's localized visual tokenization (Ma et al. 2024, §3): a class-agnostic region proposer (a DETR-style query
/// decoder with an objectness head, §3.1) proposes boxes, a region encoder pools the image features inside each box and
/// projects them into one region token per box (§3.1), and the language model grounds a phrase by writing region
/// tokens (<c>&lt;r_k&gt;</c>, part of its vocabulary, whose input embeddings are the region features) followed by the
/// end of the answer (§3.2). A referred region's box is the proposal its token names; its score is the probability
/// the model gives that token.
/// </summary>
/// <remarks>The region encoder pools by averaging the image tokens whose grid cell centre lies in the box (the token
/// nearest the box centre when none does), in place of the reference's multi-scale RoIAlign on DINOv2 feature maps,
/// which these models' single-scale backbones do not produce.</remarks>
internal sealed class GromaRegionGrounder<T> : GroundingHeadModule<T>
{
    private readonly GroundingQueryDecoder<T> _proposer;
    private readonly DenseLayer<T>[] _regionEncoder;
    private readonly DenseLayer<T> _textEmbedding;
    private readonly DenseLayer<T> _head;

    public GromaRegionGrounder(IEngine engine, List<ILayer<T>> stream, GromaRegionGrounderConfiguration configuration)
        : base(engine, stream)
    {
        var c = configuration ?? throw new ArgumentNullException(nameof(configuration));
        if (c.Width <= 0 || c.Proposals <= 0 || c.VocabSize <= 0 || c.MaxTextLength <= 0)
            throw new ArgumentException("The grounder needs a positive width, proposal count, vocabulary and text length.", nameof(configuration));
        Configuration = c;
        _proposer = new GroundingQueryDecoder<T>(engine, stream, new GroundingQueryDecoderConfiguration(
            c.ProposerWidth, c.ProposerHeads, 8 * c.ProposerWidth, EnhancerLayers: 0, c.ProposerLayers, c.Proposals, VocabSize: 0, MaxTextLength: 1));
        _regionEncoder = new[] { Linear(c.Width), Linear(c.Width) };
        _textEmbedding = Linear(c.Width, bias: false);
        _head = Linear(c.Proposals + 1, bias: false);
    }

    public GromaRegionGrounderConfiguration Configuration { get; }

    /// <summary>The end-of-answer token; region token k is token k.</summary>
    public int End => Configuration.Proposals;

    /// <summary>The proposer's boxes <c>[k, 4]</c> as (cx, cy, w, h) and objectness logits <c>[k]</c>.</summary>
    public (Tensor<T> Boxes, Tensor<T> Objectness) Propose(Tensor<T> imageTokens)
    {
        var prediction = _proposer.Forward(imageTokens, null);
        return (prediction.Boxes, prediction.Logits.Reshape(new[] { prediction.Logits.Shape[0] }));
    }

    /// <summary>
    /// Each box's pooling weights over the image tokens of an <c>h x w</c> grid <c>[k, h·w]</c>: 1 for the tokens whose
    /// cell centre lies in the box (x1 &lt;= x &lt;= x2, y1 &lt;= y &lt;= y2), divided by their count; a box that holds no
    /// centre takes the token nearest its centre.
    /// </summary>
    internal Tensor<T> PoolingWeights(Tensor<T> corners, int h, int w)
    {
        int k = corners.Shape[0], n = h * w;
        var centres = GridCentres(Engine, h, w);
        Tensor<T> Tokens(int axis) => Engine.TensorBroadcastTo(Engine.TensorSlice(centres, new[] { 0, axis }, new[] { n, 1 }).Reshape(new[] { 1, n }), new[] { k, n });
        Tensor<T> Box(int column) => Engine.TensorBroadcastTo(Engine.TensorSlice(corners, new[] { 0, column }, new[] { k, 1 }), new[] { k, n });
        var x = Tokens(0);
        var y = Tokens(1);
        var one = Tensor<T>.CreateDefault(new[] { k, n }, NumOps.One);
        Tensor<T> NotBelow(Tensor<T> a, Tensor<T> b) => Engine.TensorSubtract(one, Engine.TensorLessThan(a, b));
        var inside = Engine.TensorMultiply(Engine.TensorMultiply(NotBelow(x, Box(0)), NotBelow(Box(2), x)),
            Engine.TensorMultiply(NotBelow(y, Box(1)), NotBelow(Box(3), y)));

        var half = NumOps.FromDouble(0.5);
        var dx = Engine.TensorSubtract(x, Engine.TensorMultiplyScalar(Engine.TensorAdd(Box(0), Box(2)), half));
        var dy = Engine.TensorSubtract(y, Engine.TensorMultiplyScalar(Engine.TensorAdd(Box(1), Box(3)), half));
        var nearest = Engine.TensorOneHot<T>(Engine.TensorArgMin(Engine.TensorAdd(Engine.TensorMultiply(dx, dx), Engine.TensorMultiply(dy, dy)), 1), n);
        var count = Engine.ReduceSum(inside, new[] { 1 }, true);
        var empty = Engine.TensorBroadcastTo(Engine.TensorEquals(count, NumOps.Zero), new[] { k, n });
        var weights = Engine.TensorAdd(inside, Engine.TensorMultiply(empty, nearest));
        var total = Engine.TensorBroadcastTo(Engine.ReduceSum(weights, new[] { 1 }, true), new[] { k, n });
        return Engine.TensorDivide(weights, total);
    }

    /// <summary>The region tokens <c>[k, Width]</c> of boxes <c>[k, 4]</c> (x1, y1, x2, y2) over the image tokens.</summary>
    public Tensor<T> RegionTokens(Tensor<T> imageTokens, Tensor<T> corners)
    {
        var rows = AsRows(Engine, imageTokens);
        var grid = TokenGrid(rows.Shape[0]);
        var pooled = Engine.TensorMatMul(PoolingWeights(corners, grid.H, grid.W), rows);
        return Mlp(_regionEncoder, pooled, x => Engine.GELU(x));
    }

    /// <summary>
    /// Grounds a phrase: proposes regions on <paramref name="imageTokens"/> (the image's language-model tokens
    /// <c>[N, Width]</c>), tokenizes them, and lets <paramref name="languageModel"/> (on <c>[n, Width]</c> rows) write region
    /// tokens after the image tokens, the region tokens and the instruction.
    /// </summary>
    /// <returns>The referred boxes <c>[b, 4]</c> as normalized (x1, y1, x2, y2) and their scores <c>[b]</c>.</returns>
    public (Tensor<T> Boxes, Tensor<T> Scores) Ground(Tensor<T> imageTokens, IReadOnlyList<int> tokenIds,
        Func<Tensor<T>, Tensor<T>> languageModel, int maxRegions)
    {
        if (tokenIds is null || tokenIds.Count == 0) throw new ArgumentException("The instruction has no tokens.", nameof(tokenIds));
        var (boxes, _) = Propose(imageTokens);
        int k = boxes.Shape[0];
        var corners = Corners(Engine, boxes);
        var regions = RegionTokens(imageTokens, corners);
        var text = Rows(_textEmbedding, OneHot(tokenIds.Take(Math.Min(tokenIds.Count, Configuration.MaxTextLength)).ToList(), Configuration.VocabSize));
        var prefix = Engine.TensorConcatenate(new[] { AsRows(Engine, imageTokens), regions, text }, 0);
        int vocabulary = Configuration.Proposals + 1;
        // The proposals this image has, plus the end token; a region is referred to once.
        Tensor<T> Allowed(IReadOnlyList<int> written)
        {
            var permitted = Engine.TensorAdd(Span(vocabulary, 0, k), Span(vocabulary, End, End + 1));
            if (written.Count == 0) return permitted;
            var used = Engine.ReduceSum(OneHot(written, vocabulary), new[] { 0 }, true);
            return Engine.TensorMultiply(permitted, Engine.TensorEquals(used, NumOps.Zero));
        }
        var (tokens, probabilities) = DecodeGreedy(prefix, languageModel, _head,
            token => Engine.TensorSlice(regions, new[] { token, 0 }, new[] { 1, regions.Shape[1] }),
            Allowed, End, Math.Min(Math.Max(0, maxRegions), k) + 1);
        var referred = tokens.Where(t => t < k).ToList();
        if (referred.Count == 0) return (new Tensor<T>(new[] { 0, 4 }), new Tensor<T>(new[] { 0 }));
        var index = new Tensor<int>(new[] { referred.Count }, Vector<int>.FromList(referred));
        var scores = Engine.TensorConcatenate(probabilities.Take(referred.Count).ToArray(), 0).Reshape(new[] { referred.Count });
        return (Engine.TensorIndexSelect(corners, index, 0), scores);
    }
}
