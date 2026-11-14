namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>The shape of an <see cref="OwlDetectionHead{T}"/>.</summary>
/// <param name="QueryWidth">Width of the shared image-text embedding the class head scores in (OWL-ViT B/32: 512).</param>
/// <param name="TextWidth">Width of the text transformer (512).</param>
/// <param name="TextLayers">Layers of the text transformer (12).</param>
/// <param name="TextHeads">Attention heads of the text transformer (8).</param>
/// <param name="VocabSize">Text vocabulary size.</param>
/// <param name="MaxTextLength">Longest query the text transformer reads (16 for OWL-ViT).</param>
/// <param name="Objectness">Whether the head has OWLv2's objectness head.</param>
internal sealed record OwlDetectionHeadConfiguration(int QueryWidth, int TextWidth, int TextLayers, int TextHeads, int VocabSize,
    int MaxTextLength, bool Objectness);

/// <summary>What an <see cref="OwlDetectionHead{T}"/> predicts for every image token.</summary>
/// <param name="Boxes">One box per image token <c>[N, 4]</c> as normalized (cx, cy, w, h).</param>
/// <param name="Logits">Each token's class logit against each query <c>[N, Q]</c>.</param>
/// <param name="Objectness">OWLv2's objectness logit per token <c>[N]</c>, or null for OWL-ViT.</param>
internal sealed record OwlPrediction<T>(Tensor<T> Boxes, Tensor<T> Logits, Tensor<T>? Objectness);

/// <summary>
/// The detection heads of OWL-ViT (Minderer et al. 2022, §3.1) and OWLv2 (Minderer et al. 2023, §3.2): every image
/// token of the vision transformer becomes one detection. A box head (an MLP with GELU) predicts each token's box as
/// an offset from its grid cell (the reference's <c>compute_box_bias</c>); a class head projects the token into the
/// text embedding space and scores it against each text query by cosine similarity with a learned per-token shift
/// and scale (<c>(cos + shift) · (elu(scale) + 1)</c>); the text query is the CLIP text transformer's end-of-text
/// output projected into that space. OWLv2 adds an objectness head (an MLP like the box head, on detached features).
/// </summary>
/// <remarks>The vision transformer reads a learned class token ahead of the patch tokens (<see cref="ClassToken"/>);
/// the head merges the class token's output into every patch token and normalizes the result (the reference's
/// <c>image_embeds[:, 1:] * image_embeds[:, :1]</c> then <c>layer_norm</c>, <see cref="MergeClassToken"/>).</remarks>
internal sealed class OwlDetectionHead<T> : GroundingHeadModule<T>
{
    // compute_box_bias: log(x + 1e-4) - log1p(-x + 1e-4).
    private const double BiasEpsilon = 1e-4;

    // The reference normalizes embeddings as x / (||x|| + 1e-6).
    private const double NormEpsilon = 1e-6;

    private readonly LayerNormalizationLayer<T> _imageNorm;
    private readonly DenseLayer<T>[] _boxHead;
    private readonly DenseLayer<T> _classEmbedding;
    private readonly DenseLayer<T> _logitShift;
    private readonly DenseLayer<T> _logitScale;
    private readonly DenseLayer<T>[]? _objectnessHead;
    private readonly DenseLayer<T> _tokenEmbedding;
    private readonly DenseLayer<T> _positionEmbedding;
    private readonly List<GroundingTransformerBlock<T>> _textBlocks = new();
    private readonly LayerNormalizationLayer<T> _textFinalNorm;
    private readonly DenseLayer<T> _textProjection;
    private readonly DenseLayer<T> _classToken;

    public OwlDetectionHead(IEngine engine, List<ILayer<T>> stream, int imageWidth, OwlDetectionHeadConfiguration configuration)
        : base(engine, stream)
    {
        var c = configuration ?? throw new ArgumentNullException(nameof(configuration));
        if (imageWidth <= 0) throw new ArgumentOutOfRangeException(nameof(imageWidth));
        Configuration = c;
        _imageNorm = Norm();
        _boxHead = new[] { Linear(imageWidth), Linear(imageWidth), Linear(4) };
        _classEmbedding = Linear(c.QueryWidth);
        _logitShift = Linear(1);
        _logitScale = Linear(1);
        if (c.Objectness) _objectnessHead = new[] { Linear(imageWidth), Linear(imageWidth), Linear(1) };
        _tokenEmbedding = Linear(c.TextWidth, bias: false);
        _positionEmbedding = Linear(c.TextWidth, bias: false);
        for (int i = 0; i < c.TextLayers; i++)
            _textBlocks.Add(Block(c.TextWidth, c.TextHeads, 4 * c.TextWidth, cross: false, causal: true));
        _textFinalNorm = Norm();
        _textProjection = Linear(c.QueryWidth, bias: false);
        _classToken = Linear(imageWidth, bias: false);
    }

    public OwlDetectionHeadConfiguration Configuration { get; }

    /// <summary>
    /// The text query embedding <c>[1, QueryWidth]</c>: token and position embeddings through the causal text
    /// transformer, the final layer norm, and the projection of the last (end-of-text) position.
    /// </summary>
    public Tensor<T> EmbedQuery(IReadOnlyList<int> tokenIds)
    {
        if (tokenIds is null || tokenIds.Count == 0) throw new ArgumentException("The query has no tokens.", nameof(tokenIds));
        int length = Math.Min(tokenIds.Count, Configuration.MaxTextLength);
        var ids = tokenIds.Take(length).ToList();
        var x = Engine.TensorAdd(Rows(_tokenEmbedding, OneHot(ids, Configuration.VocabSize)),
            Rows(_positionEmbedding, OneHot(Enumerable.Range(0, length).ToList(), Configuration.MaxTextLength)));
        foreach (var block in _textBlocks) x = Run(block, x, null);
        var last = Engine.TensorSlice(Rows(_textFinalNorm, x), new[] { length - 1, 0 }, new[] { 1, Configuration.TextWidth });
        return Rows(_textProjection, last);
    }

    /// <summary>
    /// The reference's box bias for an <c>h x w</c> token grid <c>[h·w, 4]</c>: the logit of each cell's far corner
    /// ((j + 1) / w, (i + 1) / h) and of the cell size (1 / w, 1 / h).
    /// </summary>
    internal Tensor<T> BoxBias(int h, int w)
    {
        int n = h * w;
        var corners = Engine.TensorAdd(GridCentres(Engine, h, w), Engine.TensorConcatenate(new[]
        {
            Tensor<T>.CreateDefault(new[] { n, 1 }, NumOps.FromDouble(0.5 / w)),
            Tensor<T>.CreateDefault(new[] { n, 1 }, NumOps.FromDouble(0.5 / h)),
        }, 1));
        var sizes = Engine.TensorConcatenate(new[]
        {
            Tensor<T>.CreateDefault(new[] { n, 1 }, NumOps.FromDouble(1.0 / w)),
            Tensor<T>.CreateDefault(new[] { n, 1 }, NumOps.FromDouble(1.0 / h)),
        }, 1);
        return Engine.TensorConcatenate(new[] { BiasLogit(corners), BiasLogit(sizes) }, 1);
    }

    private Tensor<T> BiasLogit(Tensor<T> x)
    {
        var clipped = Engine.TensorClamp(x, NumOps.Zero, NumOps.One);
        var eps = NumOps.FromDouble(BiasEpsilon);
        return Engine.TensorSubtract(Engine.TensorLog(Engine.TensorAddScalar(clipped, eps)),
            Engine.TensorLog(Engine.TensorAddScalar(Engine.ScalarMinusTensor(NumOps.One, clipped), eps)));
    }

    /// <summary>The learned class token <c>[1, imageWidth]</c> the vision transformer reads ahead of the patch tokens.</summary>
    public Tensor<T> ClassToken() => Rows(_classToken, OneHot(new[] { 0 }, 1));

    /// <summary>
    /// OWL-ViT's class-token merge: the backbone's output rows <c>[1 + n, d]</c> (class token first) become the
    /// <c>[n, d]</c> patch tokens times the class token, layer-normalized.
    /// </summary>
    internal Tensor<T> MergeClassToken(Tensor<T> backboneTokens)
    {
        if (backboneTokens is null) throw new ArgumentNullException(nameof(backboneTokens));
        var rows = AsRows(Engine, backboneTokens);
        int n = rows.Shape[0] - 1, d = rows.Shape[1];
        if (n <= 0) throw new ArgumentException("The backbone output needs the class token and at least one patch token.", nameof(backboneTokens));
        var classToken = Engine.TensorBroadcastTo(Engine.TensorSlice(rows, new[] { 0, 0 }, new[] { 1, d }), new[] { n, d });
        var patches = Engine.TensorSlice(rows, new[] { 1, 0 }, new[] { n, d });
        return Rows(_imageNorm, Engine.TensorMultiply(patches, classToken));
    }

    /// <summary>Predicts a box, class logits against <paramref name="queries"/> <c>[Q, QueryWidth]</c> and (OWLv2) an
    /// objectness logit for every image token <c>[N, d]</c> in row-major grid order.</summary>
    public OwlPrediction<T> Forward(Tensor<T> imageTokens, Tensor<T> queries)
    {
        if (imageTokens is null) throw new ArgumentNullException(nameof(imageTokens));
        if (queries is null) throw new ArgumentNullException(nameof(queries));
        var image = MergeClassToken(imageTokens);
        int n = image.Shape[0];
        var grid = TokenGrid(n);
        var boxes = Engine.TensorSigmoid(Engine.TensorAdd(Mlp(_boxHead, image, x => Engine.GELU(x)), BoxBias(grid.H, grid.W)));

        var cosine = Engine.TensorMatMul(Normalized(Rows(_classEmbedding, image)), Engine.TensorTranspose(Normalized(queries)));
        int q = queries.Shape[0];
        var shift = Engine.TensorBroadcastTo(Rows(_logitShift, image), new[] { n, q });
        var scale = Engine.TensorBroadcastTo(Engine.TensorAddScalar(Engine.ELU(Rows(_logitScale, image), 1.0), NumOps.One), new[] { n, q });
        var logits = Engine.TensorMultiply(Engine.TensorAdd(cosine, shift), scale);

        Tensor<T>? objectness = null;
        if (_objectnessHead is not null)
        {
            Tensor<T> detached;
            using (new Tensors.Engines.Autodiff.NoGradScope<T>())
                detached = image.Clone();
            objectness = Mlp(_objectnessHead, detached, x => Engine.GELU(x)).Reshape(new[] { n });
        }
        return new OwlPrediction<T>(boxes, logits, objectness);
    }

    // x / (||x|| + 1e-6) per row, the norm overflow-safe.
    private Tensor<T> Normalized(Tensor<T> rows)
    {
        int n = rows.Shape[0], d = rows.Shape[1];
        var norms = Engine.TensorAddScalar(GroundingDetections<T>.RowNorm(Engine, rows), NumOps.FromDouble(NormEpsilon)).Reshape(new[] { n, 1 });
        return Engine.TensorDivide(rows, Engine.TensorBroadcastTo(norms, new[] { n, d }));
    }
}
