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
/// <param name="Levels">Feature-pyramid levels the deformable attention samples (4).</param>
/// <param name="Points">Sampling points per head and level of the deformable attention (4).</param>
internal sealed record GroundingQueryDecoderConfiguration(int Width, int Heads, int FeedForward, int EnhancerLayers,
    int DecoderLayers, int Queries, int VocabSize, int MaxTextLength, int Levels = 4, int Points = 4);

/// <summary>What a <see cref="GroundingQueryDecoder{T}"/> predicts for one image.</summary>
/// <param name="Boxes">The last decoder layer's boxes <c>[k, 4]</c> as normalized (cx, cy, w, h).</param>
/// <param name="Logits">Per query, its contrastive logit against every text token <c>[k, L]</c>, or its objectness
/// logit <c>[k, 1]</c> for a class-agnostic head.</param>
/// <param name="Selected">The indices <c>[k]</c> into the multi-scale tokens (finest level first) the query selection
/// kept, in selection order.</param>
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
/// <para>The backbone's single-scale map becomes four levels through ViTDet's simple feature pyramid (strides 8 to 64
/// for a stride-16 backbone), each projected and normalized and tagged with a learned level embedding. Image
/// self-attention in the enhancer and the decoder's attention to the image are multi-scale deformable attention
/// (Deformable DETR): encoder tokens sample around their own centre, decoder queries around their current box. The
/// selected tokens' first boxes follow DINO's two-stage proposals: each token's cell (centre, side 0.05 * 2^level) in
/// logit space plus the encoder box head's offset. The decoder's positional query is the sine embedding of the current
/// box through a two-layer MLP (DINO's <c>ref_point_head</c>).</para>
/// <para>With <see cref="GroundingQueryDecoderConfiguration.VocabSize"/> 0 the head is class-agnostic (Groma's
/// region proposer): no text, the query selection ranks tokens by an objectness head, and each query's logit is its
/// objectness.</para>
/// </remarks>
internal sealed class GroundingQueryDecoder<T> : GroundingHeadModule<T>
{
    private sealed record ImageEncoderLayer(LayerNormalizationLayer<T> SelfNorm, DeformableAttentionLayers<T> SelfAttention,
        LayerNormalizationLayer<T> FeedForwardNorm, DenseLayer<T> FeedForwardUp, DenseLayer<T> FeedForwardDown);

    private sealed record Enhancer(ImageEncoderLayer Image, GroundingTransformerBlock<T>? Text,
        LayerNormalizationLayer<T>? ImageToTextNorm, MultiHeadAttentionLayer<T>? ImageToText,
        LayerNormalizationLayer<T>? TextToImageNorm, MultiHeadAttentionLayer<T>? TextToImage);

    private sealed record Decoder(MultiHeadAttentionLayer<T> SelfAttention, LayerNormalizationLayer<T> SelfNorm,
        MultiHeadAttentionLayer<T>? TextAttention, LayerNormalizationLayer<T>? TextNorm,
        DeformableAttentionLayers<T> ImageAttention, LayerNormalizationLayer<T> ImageNorm,
        DenseLayer<T> FeedForwardUp, DenseLayer<T> FeedForwardDown, LayerNormalizationLayer<T> FeedForwardNorm);

    // DINO's proposal side of the finest level; it doubles per level (0.05 * 2^level).
    private const double ProposalSide = 0.05;

    private readonly FeaturePyramidLayers<T> _pyramid;
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
        if (c.Levels <= 0 || c.Points <= 0)
            throw new ArgumentException("The deformable attention needs at least one level and one point.", nameof(configuration));
        _pyramid = PyramidLayers(c.Width, c.Levels);
        if (HasText)
        {
            _tokenEmbedding = Linear(c.Width, bias: false);
            _positionEmbedding = Linear(c.Width, bias: false);
        }
        for (int i = 0; i < c.EnhancerLayers; i++)
        {
            var image = new ImageEncoderLayer(Norm(), DeformableAttention(c.Width, c.Heads, c.Levels, c.Points), Norm(),
                Linear(c.FeedForward), Linear(c.Width));
            _enhancers.Add(HasText
                ? new Enhancer(image, Block(c.Width, c.Heads, c.FeedForward, cross: false),
                    Norm(), Attention(c.Width, c.Heads), Norm(), Attention(c.Width, c.Heads))
                : new Enhancer(image, null, null, null, null, null));
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
                DeformableAttention(c.Width, c.Heads, c.Levels, c.Points), Norm(), Linear(c.FeedForward), Linear(c.Width), Norm()));
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
        var levels = Pyramid(_pyramid, imageTokens);
        var image = levels.Tokens;
        int n = image.Shape[0];
        // Every token's cell on its level: centre and DINO's side 0.05 * 2^level.
        var proposals = Engine.TensorConcatenate(new[]
        {
            levels.Centres,
            Engine.TensorConcatenate(levels.Shapes.Select((shape, level) => Tensor<T>.CreateDefault(
                new[] { shape.H * shape.W, 2 }, NumOps.FromDouble(ProposalSide * Math.Pow(2, level)))).ToArray(), 0),
        }, 1);
        // Deformable DETR's encoder position: the sine embedding of each token's centre plus its level embedding.
        var tokenPosition = Engine.TensorAdd(CentrePosition(levels.Centres), levels.LevelEmbedding);
        Tensor<T>? text = HasText ? EmbedText(tokenIds ?? throw new ArgumentNullException(nameof(tokenIds))) : null;
        foreach (var e in _enhancers)
        {
            if (text is not null && e.Text is not null && e.ImageToText is not null && e.ImageToTextNorm is not null
                && e.TextToImage is not null && e.TextToImageNorm is not null)
            {
                // Bi-directional cross-attention (section 3.1): image-to-text and text-to-image, then each side's self-attention.
                var imageUpdate = CrossAttend(e.ImageToText, Rows(e.ImageToTextNorm, image), text);
                var textUpdate = CrossAttend(e.TextToImage, Rows(e.TextToImageNorm, text), image);
                image = Engine.TensorAdd(image, imageUpdate);
                text = Run(e.Text, Engine.TensorAdd(text, textUpdate), null);
            }

            // Deformable self-attention of the image tokens: each samples the pyramid around its own centre.
            var normed = Rows(e.Image.SelfNorm, image);
            image = Engine.TensorAdd(image, Deform(e.Image.SelfAttention, Engine.TensorAdd(normed, tokenPosition),
                levels.Centres, levels with { Tokens = normed }));
            var hidden = Rows(e.Image.FeedForwardDown, Engine.ReLU(Rows(e.Image.FeedForwardUp, Rows(e.Image.FeedForwardNorm, image))));
            image = Engine.TensorAdd(image, hidden);
        }
        var imageLevels = levels with { Tokens = image };

        // Language-guided query selection (Eq. 1-2): each image token's largest similarity to any text token, top k.
        var memory = Rows(_encoderOutputNorm, Rows(_encoderOutput, image));
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
            // Deformable cross-attention to the pyramid around each query's current box.
            queries = Rows(layer.ImageNorm, Engine.TensorAdd(queries,
                Deform(layer.ImageAttention, Engine.TensorAdd(queries, position), reference, imageLevels)));
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

    // The sine embedding of [S, 2] centres: D/2 features per coordinate, zero-padded to D when D is not a multiple of 4.
    private Tensor<T> CentrePosition(Tensor<T> centres)
    {
        var embedding = SineEmbedding(centres);
        int s = centres.Shape[0], missing = Configuration.Width - embedding.Shape[1];
        return missing > 0
            ? Engine.TensorConcatenate(new[] { embedding, new Tensor<T>(new[] { s, missing }) }, 1)
            : embedding;
    }

    /// <summary>
    /// DINO's <c>gen_sineembed_for_position</c> of positions <c>[k, c]</c> (boxes: c = 4): per coordinate, D/2 features
    /// <c>sin</c> then <c>cos</c> of <c>2π·v / 10000^(2i/(D/2))</c>, concatenated to <c>[k, c·D/2]</c>.
    /// </summary>
    internal Tensor<T> SineEmbedding(Tensor<T> boxes)
    {
        int k = boxes.Shape[0], coordinates = boxes.Shape[1], half = Configuration.Width / 2, quarter = half / 2;
        // Frequencies 10000^(2i/half) for i < half/2, as a [1, quarter] row.
        var exponents = Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, quarter, 1), NumOps.FromDouble(2.0 / half));
        var frequencies = Engine.TensorPower(Tensor<T>.CreateDefault(new[] { quarter }, NumOps.FromDouble(10000)), exponents).Reshape(new[] { 1, quarter });
        var twoPi = NumOps.FromDouble(2 * Math.PI);
        var parts = new List<Tensor<T>>(2 * coordinates);
        for (int coordinate = 0; coordinate < coordinates; coordinate++)
        {
            var angle = Engine.TensorDivide(
                Engine.TensorBroadcastTo(Engine.TensorMultiplyScalar(Engine.TensorSlice(boxes, new[] { 0, coordinate }, new[] { k, 1 }), twoPi), new[] { k, quarter }),
                Engine.TensorBroadcastTo(frequencies, new[] { k, quarter }));
            parts.Add(Engine.TensorSin(angle));
            parts.Add(Engine.TensorCos(angle));
        }
        return Engine.TensorConcatenate(parts.ToArray(), 1);
    }
}
