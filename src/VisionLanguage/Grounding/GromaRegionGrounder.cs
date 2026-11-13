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
/// <remarks>The proposer is a Deformable DETR (multi-scale deformable encoder and decoder) on the backbone's simple
/// feature pyramid. The region encoder builds its own pyramid of the backbone's map, takes each box's RoIAlign
/// (7x7 bins, 2x2 samples per bin, aligned) on every level, averages the bins and the levels, and projects the result
/// through a two-layer MLP into the language model's width.</remarks>
internal sealed class GromaRegionGrounder<T> : GroundingHeadModule<T>
{
    private readonly GroundingQueryDecoder<T> _proposer;
    private readonly FeaturePyramidLayers<T> _regionPyramid;
    private readonly DenseLayer<T>[] _regionEncoder;

    // torchvision MultiScaleRoIAlign's defaults in Mask R-CNN's box head: 7x7 bins, 2x2 samples per bin.
    private const int RoiSize = 7;
    private const int RoiSamples = 2;
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
            c.ProposerWidth, c.ProposerHeads, 8 * c.ProposerWidth, EnhancerLayers: c.ProposerLayers, c.ProposerLayers, c.Proposals,
            VocabSize: 0, MaxTextLength: 1));
        _regionPyramid = PyramidLayers(c.Width, 4);
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
    /// The proposer's training loss (Deformable DETR's class-agnostic set loss): its queries matched one-to-one to the
    /// image's ground-truth regions <c>[G, 4]</c> (normalized cx, cy, w, h) by the Hungarian algorithm, then the focal
    /// objectness loss, L1 and GIoU.
    /// </summary>
    internal GroundingSetLossTerms<T> ProposalLoss(Tensor<T> imageTokens, Tensor<T> targetBoxes)
    {
        if (targetBoxes is null) throw new ArgumentNullException(nameof(targetBoxes));
        var prediction = _proposer.Forward(imageTokens, null);
        var labels = Tensor<T>.CreateDefault(new[] { targetBoxes.Shape[0], 1 }, NumOps.One);
        return new GroundingSetLoss<T>(Engine).Compute(prediction.Boxes, prediction.Logits, targetBoxes, labels);
    }

    /// <summary>
    /// RoIAlign (He et al. 2017; torchvision's <c>roi_align</c>, aligned) of every box on one feature map: the
    /// <c>[n, d]</c> tokens of an <c>h x w</c> grid, the boxes as normalized corners <c>[K, 4]</c>; each box's
    /// <c>[d]</c> feature is the mean of its 7x7 bins.
    /// </summary>
    internal Tensor<T> AlignedRegions(Tensor<T> tokens, int h, int w, Tensor<T> corners)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (corners is null) throw new ArgumentNullException(nameof(corners));
        int k = corners.Shape[0], d = tokens.Shape[1];
        var map = Engine.Reshape(Engine.TensorTranspose(tokens), new[] { 1, d, h, w });
        var scale = new Tensor<T>(new[] { 1, 4 }, Vector<T>.FromArray(new[]
        {
            NumOps.FromDouble(w), NumOps.FromDouble(h), NumOps.FromDouble(w), NumOps.FromDouble(h),
        }));
        var boxes = Engine.TensorConcatenate(new[]
        {
            new Tensor<T>(new[] { k, 1 }),
            Engine.TensorMultiply(corners, Engine.TensorBroadcastTo(scale, new[] { k, 4 })),
        }, 1);
        var aligned = Engine.RoIAlign(map, boxes, RoiSize, RoiSize, 1f, RoiSamples, true);
        return Engine.ReduceMean(aligned, new[] { 2, 3 }, false);
    }

    /// <summary>
    /// The region token of every box <c>[K, Width]</c>: its RoIAlign feature on every level of the backbone's feature
    /// pyramid, averaged over the levels, through the region encoder's MLP.
    /// </summary>
    public Tensor<T> RegionTokens(Tensor<T> imageTokens, Tensor<T> corners)
    {
        var levels = Pyramid(_regionPyramid, imageTokens);
        int d = levels.Tokens.Shape[1];
        var perLevel = new Tensor<T>[levels.Count];
        for (int level = 0; level < levels.Count; level++)
        {
            var (h, w) = levels.Shapes[level];
            var tokens = Engine.TensorSlice(levels.Tokens, new[] { levels.Starts[level], 0 }, new[] { h * w, d });
            perLevel[level] = AlignedRegions(tokens, h, w, corners);
        }
        var pooled = Engine.ReduceMean(Engine.TensorStack(perLevel, 0), new[] { 0 }, false);
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
