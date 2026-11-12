namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>The shape of a <see cref="GroundingQueryDecoder{T}"/>.</summary>
/// <param name="Width">Feature width of the head (Grounding DINO: 256).</param>
/// <param name="Heads">Attention heads (8).</param>
/// <param name="FeedForward">Feed-forward width of the enhancer and decoder layers (2048).</param>
/// <param name="EnhancerLayers">Feature-enhancer layers (6); 0 for a head without text.</param>
/// <param name="DecoderLayers">Cross-modality decoder layers (6).</param>
/// <param name="Queries">Queries the language-guided query selection keeps (900).</param>
/// <param name="VocabSize">Text vocabulary size; 0 for a class-agnostic head without text.</param>
/// <param name="MaxTextLength">Longest text the head embeds (256).</param>
internal sealed record GroundingQueryDecoderConfiguration(int Width, int Heads, int FeedForward, int EnhancerLayers,
    int DecoderLayers, int Queries, int VocabSize, int MaxTextLength);

/// <summary>What a <see cref="GroundingQueryDecoder{T}"/> predicts for one image.</summary>
/// <param name="Boxes">The last decoder layer's boxes <c>[k, 4]</c> as normalized (cx, cy, w, h).</param>
/// <param name="Logits">Per query, its contrastive logit against every text token <c>[k, L]</c>, or its objectness
/// logit <c>[k, 1]</c> for a class-agnostic head.</param>
/// <param name="Selected">The image-token indices <c>[k]</c> the query selection kept, in selection order.</param>
internal sealed record GroundingQueryPrediction<T>(Tensor<T> Boxes, Tensor<T> Logits, Tensor<int> Selected)
{
    /// <summary>Each query's score: the sigmoid of its largest logit (Grounding DINO's <c>predict</c>).</summary>
    public Tensor<T> Scores(IEngine engine)
        => engine.TensorSigmoid(engine.ReduceMax(Logits, new[] { 1 }, false));
}

/// <summary>
/// The detection head of Grounding DINO (Liu et al. 2023, §3) as DINO-X and Grounding DINO 1.5 build on it: a feature
/// enhancer of bi-directional image-text cross-attention with self-attention on both sides (§3.1), language-guided
/// query selection of the image tokens most similar to the text (§3.2, Eq. 1-2), a cross-modality decoder whose
/// queries self-attend, attend to the text and then to the image (§3.3), shared box-regression and contrastive
/// classification heads with layer-by-layer box refinement (DINO's look-forward-once), and DINO's mixed query
/// selection (positional queries from the selected tokens' boxes, learnable content queries).
/// </summary>
/// <remarks>
/// <para>The selected tokens' first boxes follow DINO's two-stage proposals: each token's grid cell (centre, side
/// 0.05) in logit space plus the encoder box head's offset. The decoder's positional query is the sine embedding of
/// the current box through a two-layer MLP (DINO's <c>ref_point_head</c>). Attention over the image is standard
/// multi-head attention over the given image tokens rather than multi-scale deformable attention, which needs a
/// multi-scale feature pyramid the models' backbones do not produce.</para>
/// <para>With <see cref="GroundingQueryDecoderConfiguration.VocabSize"/> 0 the head is class-agnostic (Groma's
/// region proposer): no text, the query selection ranks tokens by an objectness head, and each query's logit is its
/// objectness.</para>
/// </remarks>
internal sealed class GroundingQueryDecoder<T> : GroundingHeadModule<T>
{
    private sealed record Enhancer(GroundingTransformerBlock<T> Image, GroundingTransformerBlock<T> Text,
        LayerNormalizationLayer<T> ImageToTextNorm, MultiHeadAttentionLayer<T> ImageToText,
        LayerNormalizationLayer<T> TextToImageNorm, MultiHeadAttentionLayer<T> TextToImage);

    private sealed record Decoder(MultiHeadAttentionLayer<T> SelfAttention, LayerNormalizationLayer<T> SelfNorm,
        MultiHeadAttentionLayer<T>? TextAttention, LayerNormalizationLayer<T>? TextNorm,
        MultiHeadAttentionLayer<T> ImageAttention, LayerNormalizationLayer<T> ImageNorm,
        DenseLayer<T> FeedForwardUp, DenseLayer<T> FeedForwardDown, LayerNormalizationLayer<T> FeedForwardNorm);

    // DINO's proposal side for single-level features: 0.05 * 2^0.
    private const double ProposalSide = 0.05;

    private readonly DenseLayer<T> _imageProjection;
    private readonly DenseLayer<T>? _tokenEmbedding;
    private readonly DenseLayer<T>? _positionEmbedding;
    private readonly List<Enhancer> _enhancers = new();
    private readonly DenseLayer<T> _encoderOutput;
    private readonly LayerNormalizationLayer<T> _encoderOutputNorm;
    private readonly DenseLayer<T>[] _encoderBoxHead;
    private readonly DenseLayer<T>? _encoderObjectness;
    private readonly DenseLayer<T> _contentQueries;
    private readonly DenseLayer<T>[] _referenceHead;
    private readonly List<Decoder> _decoders = new();
    private readonly LayerNormalizationLayer<T> _decoderNorm;
    private readonly DenseLayer<T>[] _boxHead;
    private readonly DenseLayer<T>? _objectnessHead;

    public GroundingQueryDecoder(IEngine engine, List<ILayer<T>> stream, GroundingQueryDecoderConfiguration configuration)
        : base(engine, stream)
    {
        var c = configuration ?? throw new ArgumentNullException(nameof(configuration));
        if (c.Width < 4) throw new ArgumentException("The head width must be at least 4.", nameof(configuration));
        if (c.Queries <= 0) throw new ArgumentException("The head needs at least one query.", nameof(configuration));
        Configuration = c;
        _imageProjection = Linear(c.Width);
        if (HasText)
        {
            _tokenEmbedding = Linear(c.Width, bias: false);
            _positionEmbedding = Linear(c.Width, bias: false);
            for (int i = 0; i < c.EnhancerLayers; i++)
                _enhancers.Add(new Enhancer(Block(c.Width, c.Heads, c.FeedForward, cross: false), Block(c.Width, c.Heads, c.FeedForward, cross: false),
                    Norm(), Attention(c.Width, c.Heads), Norm(), Attention(c.Width, c.Heads)));
        }
        _encoderOutput = Linear(c.Width);
        _encoderOutputNorm = Norm();
        _encoderBoxHead = new[] { Linear(c.Width), Linear(c.Width), Linear(4) };
        if (!HasText) _encoderObjectness = Linear(1);
        _contentQueries = Linear(c.Width, bias: false);
        _referenceHead = new[] { Linear(c.Width), Linear(c.Width) };
        for (int i = 0; i < c.DecoderLayers; i++)
            _decoders.Add(new Decoder(Attention(c.Width, c.Heads), Norm(),
                HasText ? Attention(c.Width, c.Heads) : null, HasText ? Norm() : null,
                Attention(c.Width, c.Heads), Norm(), Linear(c.FeedForward), Linear(c.Width), Norm()));
        _decoderNorm = Norm();
        _boxHead = new[] { Linear(c.Width), Linear(c.Width), Linear(4) };
        if (!HasText) _objectnessHead = Linear(1);
    }

    public GroundingQueryDecoderConfiguration Configuration { get; }

    private bool HasText => Configuration.VocabSize > 0;

    /// <summary>The text tokens embedded and enhanced against nothing yet: token plus learned position embeddings <c>[L, D]</c>.</summary>
    internal Tensor<T> EmbedText(IReadOnlyList<int> tokenIds)
    {
        if (_tokenEmbedding is null || _positionEmbedding is null)
            throw new InvalidOperationException("A class-agnostic head reads no text.");
        if (tokenIds is null || tokenIds.Count == 0) throw new ArgumentException("The text has no tokens.", nameof(tokenIds));
        int length = Math.Min(tokenIds.Count, Configuration.MaxTextLength);
        var ids = tokenIds.Take(length).ToList();
        var tokens = Rows(_tokenEmbedding, OneHot(ids, Configuration.VocabSize));
        var positions = Rows(_positionEmbedding, OneHot(Enumerable.Range(0, length).ToList(), Configuration.MaxTextLength));
        return Engine.TensorAdd(tokens, positions);
    }

    /// <summary>
    /// Detects in one image: <paramref name="imageTokens"/> are the backbone's tokens <c>[N, d]</c> in row-major
    /// grid order, <paramref name="tokenIds"/> the text's token ids (ignored by a class-agnostic head).
    /// </summary>
    public GroundingQueryPrediction<T> Forward(Tensor<T> imageTokens, IReadOnlyList<int>? tokenIds)
    {
        if (imageTokens is null) throw new ArgumentNullException(nameof(imageTokens));
        var c = Configuration;
        var image = Rows(_imageProjection, AsRows(Engine, imageTokens));
        int n = image.Shape[0];
        Tensor<T>? text = null;
        if (HasText)
        {
            text = EmbedText(tokenIds ?? throw new ArgumentNullException(nameof(tokenIds)));
            foreach (var e in _enhancers)
            {
                // Bi-directional cross-attention (§3.1): image-to-text and text-to-image, then self-attention and FFN.
                var imageUpdate = CrossAttend(e.ImageToText, Rows(e.ImageToTextNorm, image), text);
                var textUpdate = CrossAttend(e.TextToImage, Rows(e.TextToImageNorm, text), image);
                image = Run(e.Image, Engine.TensorAdd(image, imageUpdate), null);
                text = Run(e.Text, Engine.TensorAdd(text, textUpdate), null);
            }
        }

        // Language-guided query selection (Eq. 1-2): each image token's largest similarity to any text token, top k.
        var memory = Rows(_encoderOutputNorm, Rows(_encoderOutput, image));
        var grid = TokenGrid(n);
        var proposals = Engine.TensorConcatenate(new[]
        {
            GridCentres(Engine, grid.H, grid.W),
            Tensor<T>.CreateDefault(new[] { n, 2 }, NumOps.FromDouble(ProposalSide)),
        }, 1);
        var tokenScores = text is not null
            ? Engine.ReduceMax(Engine.TensorMatMul(memory, Engine.TensorTranspose(text)), new[] { 1 }, false)
            : Rows(_encoderObjectness ?? throw new InvalidOperationException("The objectness head is missing."), memory).Reshape(new[] { n });
        int k = Math.Min(c.Queries, n);
        Engine.TensorTopK(tokenScores, k, 0, out var selected);
        var selectedMemory = Engine.TensorIndexSelect(memory, selected, 0);
        var selectedProposals = Engine.TensorIndexSelect(proposals, selected, 0);
        var reference = Engine.TensorSigmoid(Engine.TensorAdd(
            Mlp(_encoderBoxHead, selectedMemory, x => Engine.ReLU(x)), InverseSigmoid(Engine, selectedProposals)));

        // Mixed query selection: learnable content queries, positional queries from the selected boxes.
        var queries = Rows(_contentQueries, OneHot(Enumerable.Range(0, k).ToList(), c.Queries));
        foreach (var layer in _decoders)
        {
            var position = Mlp(_referenceHead, SineEmbedding(reference), x => Engine.ReLU(x));
            var withPosition = Engine.TensorAdd(queries, position);
            queries = Rows(layer.SelfNorm, Engine.TensorAdd(queries, SelfAttendValues(layer.SelfAttention, withPosition, queries)));
            if (layer.TextAttention is not null && layer.TextNorm is not null && text is not null)
                queries = Rows(layer.TextNorm, Engine.TensorAdd(queries, CrossAttend(layer.TextAttention, queries, text)));
            queries = Rows(layer.ImageNorm, Engine.TensorAdd(queries, CrossAttend(layer.ImageAttention, Engine.TensorAdd(queries, position), image)));
            var hidden = Rows(layer.FeedForwardDown, Engine.ReLU(Rows(layer.FeedForwardUp, queries)));
            queries = Rows(layer.FeedForwardNorm, Engine.TensorAdd(queries, hidden));

            // Layer-by-layer refinement with the shared box head: sigmoid(offset + logit(previous box)).
            reference = Engine.TensorSigmoid(Engine.TensorAdd(
                Mlp(_boxHead, Rows(_decoderNorm, queries), x => Engine.ReLU(x)), InverseSigmoid(Engine, reference)));
        }

        var output = Rows(_decoderNorm, queries);
        var logits = text is not null
            ? Engine.TensorMatMul(output, Engine.TensorTranspose(text))
            : Rows(_objectnessHead ?? throw new InvalidOperationException("The objectness head is missing."), output);
        return new GroundingQueryPrediction<T>(reference, logits, selected);
    }

    // Self-attention whose queries and keys carry the positional query and whose values do not (DETR).
    private Tensor<T> SelfAttendValues(MultiHeadAttentionLayer<T> attention, Tensor<T> withPosition, Tensor<T> values)
    {
        int n = values.Shape[0], d = values.Shape[1];
        var q = Engine.Reshape(withPosition, new[] { 1, n, d });
        return Engine.Reshape(attention.Forward(q, q, Engine.Reshape(values, new[] { 1, n, d })), new[] { n, d });
    }

    /// <summary>
    /// DINO's <c>gen_sineembed_for_position</c> of boxes <c>[k, 4]</c>: per coordinate, D/2 features
    /// <c>sin</c> then <c>cos</c> of <c>2π·v / 10000^(2i/(D/2))</c>, concatenated to <c>[k, 2D]</c>.
    /// </summary>
    internal Tensor<T> SineEmbedding(Tensor<T> boxes)
    {
        int k = boxes.Shape[0], half = Configuration.Width / 2, quarter = half / 2;
        // Frequencies 10000^(2i/half) for i < half/2, as a [1, quarter] row.
        var exponents = Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, quarter, 1), NumOps.FromDouble(2.0 / half));
        var frequencies = Engine.TensorPower(Tensor<T>.CreateDefault(new[] { quarter }, NumOps.FromDouble(10000)), exponents).Reshape(new[] { 1, quarter });
        var parts = new List<Tensor<T>>(8);
        for (int coordinate = 0; coordinate < 4; coordinate++)
        {
            var angle = Engine.TensorDivide(
                Engine.TensorBroadcastTo(Engine.TensorMultiplyScalar(Engine.TensorSlice(boxes, new[] { 0, coordinate }, new[] { k, 1 }), NumOps.FromDouble(2 * Math.PI)), new[] { k, quarter }),
                Engine.TensorBroadcastTo(frequencies, new[] { k, quarter }));
            parts.Add(Engine.TensorSin(angle));
            parts.Add(Engine.TensorCos(angle));
        }
        return Engine.TensorConcatenate(parts.ToArray(), 1);
    }
}
