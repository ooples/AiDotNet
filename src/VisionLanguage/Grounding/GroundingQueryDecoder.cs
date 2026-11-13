using AiDotNet.Initialization;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>The shape of a <see cref="GroundingQueryDecoder{T}"/>.</summary>
/// <param name="Width">Feature width of the head (Grounding DINO: 256).</param>
/// <param name="Heads">Attention heads (8).</param>
/// <param name="FeedForward">Feed-forward width of the image encoder and decoder layers (2048); the text layers and the
/// image-text fusion use half of it (Grounding DINO's <c>dim_feedforward // 2</c>).</param>
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
/// <param name="LayerBoxes">Every decoder layer's boxes, first to last (the last is <paramref name="Boxes"/>).</param>
/// <param name="LayerLogits">Every decoder layer's logits from the shared class head, first to last.</param>
/// <param name="EncoderBoxes">The two-stage proposals of the selected tokens <c>[k, 4]</c> (the decoder's first boxes).</param>
/// <param name="EncoderLogits">The selected tokens' encoder logits: their similarity to the text, or their objectness.</param>
internal sealed record GroundingQueryPrediction<T>(Tensor<T> Boxes, Tensor<T> Logits, Tensor<int> Selected,
    IReadOnlyList<Tensor<T>> LayerBoxes, IReadOnlyList<Tensor<T>> LayerLogits, Tensor<T> EncoderBoxes, Tensor<T> EncoderLogits)
{
    /// <summary>Each query's score: the sigmoid of its largest logit (Grounding DINO's <c>predict</c>).</summary>
    public Tensor<T> Scores(IEngine engine)
        => engine.TensorSigmoid(engine.ReduceMax(Logits, new[] { 1 }, false));
}

/// <summary>A deformable encoder layer of the image tokens (Deformable DETR): post-norm self-attention, then FFN.</summary>
internal sealed record GroundingImageEncoderLayer<T>(DeformableAttentionLayers<T> SelfAttention, LayerNormalizationLayer<T> SelfNorm,
    DenseLayer<T> FeedForwardUp, DenseLayer<T> FeedForwardDown, LayerNormalizationLayer<T> FeedForwardNorm);

/// <summary>A text enhancer layer (Grounding DINO's <c>TransformerEncoderLayer</c>): post-norm self-attention, then a
/// ReLU FFN.</summary>
internal sealed record GroundingTextEncoderLayer<T>(MultiHeadAttentionLayer<T> SelfAttention, LayerNormalizationLayer<T> SelfNorm,
    DenseLayer<T> FeedForwardUp, DenseLayer<T> FeedForwardDown, LayerNormalizationLayer<T> FeedForwardNorm);

/// <summary>
/// Grounding DINO's bi-directional image-text cross-attention (<c>BiAttentionBlock</c> over <c>BiMultiHeadAttention</c>):
/// both sides are layer-normalized, one attention matrix of image queries against text keys is softmaxed over the text
/// for the image update and, transposed, over the image for the text update, and each update enters its residual
/// through a learned per-channel layer scale initialized to 1e-4.
/// </summary>
internal sealed record GroundingBiAttention<T>(LayerNormalizationLayer<T> ImageNorm, LayerNormalizationLayer<T> TextNorm,
    DenseLayer<T> ImageQuery, DenseLayer<T> TextKey, DenseLayer<T> ImageValue, DenseLayer<T> TextValue,
    DenseLayer<T> ImageOutput, DenseLayer<T> TextOutput, DenseLayer<T> ImageScale, DenseLayer<T> TextScale, int Heads, int Width);

/// <summary>One feature-enhancer layer (Grounding DINO §3.1): fusion, text self-attention, image deformable self-attention.</summary>
internal sealed record GroundingEnhancerLayer<T>(GroundingImageEncoderLayer<T> Image, GroundingTextEncoderLayer<T>? Text,
    GroundingBiAttention<T>? Fusion);

/// <summary>One cross-modality decoder layer (Grounding DINO §3.3), post-norm throughout.</summary>
internal sealed record GroundingDecoderLayer<T>(MultiHeadAttentionLayer<T> SelfAttention, LayerNormalizationLayer<T> SelfNorm,
    MultiHeadAttentionLayer<T>? TextAttention, LayerNormalizationLayer<T>? TextNorm,
    DeformableAttentionLayers<T> ImageAttention, LayerNormalizationLayer<T> ImageNorm,
    DenseLayer<T> FeedForwardUp, DenseLayer<T> FeedForwardDown, LayerNormalizationLayer<T> FeedForwardNorm);

/// <summary>
/// The detection head of Grounding DINO (Liu et al. 2023, §3) as DINO-X and Grounding DINO 1.5 build on it: a feature
/// enhancer of bi-directional image-text cross-attention with self-attention on both sides (§3.1), language-guided
/// query selection of the image tokens most similar to the text (§3.2, Eq. 1-2), a cross-modality decoder whose
/// queries self-attend, attend to the text and then to the image (§3.3), shared box-regression and contrastive
/// classification heads with layer-by-layer box refinement (DINO's look-forward-twice), and DINO's mixed query
/// selection (positional queries from the selected tokens' boxes, learnable content queries).
/// </summary>
/// <remarks>
/// <para>The backbone's single-scale map becomes four levels through ViTDet's simple feature pyramid (strides 8 to 64
/// for a stride-16 backbone), each projected and normalized and tagged with a learned level embedding. Image
/// self-attention in the enhancer and the decoder's attention to the image are multi-scale deformable attention
/// (Deformable DETR): encoder tokens sample around their own centre, decoder queries around their current box. The
/// selected tokens' first boxes follow DINO's two-stage proposals: each token's cell (centre, side 0.05 * 2^level) in
/// logit space plus the encoder box head's offset, a cell outside (0.01, 0.99) being invalid (its memory zeroed and
/// its proposal at +infinity). The decoder's positional query is the sine embedding of the current box through a
/// two-layer MLP (DINO's <c>ref_point_head</c>).</para>
/// <para>As the reference, the boxes a decoder layer reads are detached: the encoder's proposals enter the decoder
/// without gradient, and each layer's refined box reaches the next layer's queries without gradient, while the box
/// the loss sees refines the previous layer's undetached box (look-forward-twice).</para>
/// <para>With <see cref="GroundingQueryDecoderConfiguration.VocabSize"/> 0 the head is class-agnostic (Groma's
/// region proposer): no text, the query selection ranks tokens by an objectness head, and each query's logit is its
/// objectness.</para>
/// </remarks>
internal sealed class GroundingQueryDecoder<T> : GroundingHeadModule<T>
{
    // DINO's proposal side of the finest level; it doubles per level (0.05 * 2^level).
    private const double ProposalSide = 0.05;

    // gen_encoder_output_proposals keeps a proposal only when every coordinate lies in (0.01, 0.99).
    private const double ProposalLow = 0.01;
    private const double ProposalHigh = 0.99;

    // Grounding DINO's image position embedding: PositionEmbeddingSineHW with temperature 20 on both axes.
    private const double ImagePositionTemperature = 20;

    // The sine embeddings of DINO's positional queries and of the text positions use temperature 10000.
    private const double QueryPositionTemperature = 10000;

    // BiAttentionBlock's layer-scale initial value.
    private const double FusionLayerScale = 1e-4;

    // BiMultiHeadAttention clamps its attention logits to +-50000 against overflow.
    private const double FusionLogitClamp = 50000;

    private readonly FeaturePyramidLayers<T> _pyramid;
    private readonly DenseLayer<T>? _tokenEmbedding;
    private readonly DenseLayer<T>? _positionEmbedding;
    private readonly List<GroundingEnhancerLayer<T>> _enhancers = new();
    private readonly DenseLayer<T> _encoderOutput;
    private readonly LayerNormalizationLayer<T> _encoderOutputNorm;
    private readonly DenseLayer<T>[] _encoderBoxHead;
    private readonly DenseLayer<T>? _encoderObjectness;
    private readonly DenseLayer<T> _contentQueries;
    private readonly DenseLayer<T>[] _referenceHead;
    private readonly List<GroundingDecoderLayer<T>> _decoders = new();
    private readonly LayerNormalizationLayer<T> _decoderNorm;
    private readonly DenseLayer<T>[] _boxHead;
    private readonly DenseLayer<T>? _objectnessHead;

    public GroundingQueryDecoder(IEngine engine, List<ILayer<T>> stream, GroundingQueryDecoderConfiguration configuration)
        : base(engine, stream)
    {
        var c = configuration ?? throw new ArgumentNullException(nameof(configuration));
        if (c.Width < 4) throw new ArgumentException("The head width must be at least 4.", nameof(configuration));
        if (c.Queries <= 0) throw new ArgumentException("The head needs at least one query.", nameof(configuration));
        if (c.FeedForward < 2) throw new ArgumentException("The feed-forward width must be at least 2.", nameof(configuration));
        Configuration = c;
        if (c.Levels <= 0 || c.Points <= 0)
            throw new ArgumentException("The deformable attention needs at least one level and one point.", nameof(configuration));
        _pyramid = PyramidLayers(c.Width, c.Levels);
        if (HasText)
        {
            _tokenEmbedding = Linear(c.Width, bias: false);
            _positionEmbedding = Linear(c.Width, bias: false);
        }
        int halfFeedForward = c.FeedForward / 2;
        for (int i = 0; i < c.EnhancerLayers; i++)
        {
            var fusion = HasText ? BiAttention(c.Width, halfFeedForward, Math.Max(1, c.Heads / 2)) : null;
            var text = HasText
                ? new GroundingTextEncoderLayer<T>(Attention(c.Width, c.Heads), Norm(), Linear(halfFeedForward), Linear(c.Width), Norm())
                : null;
            var image = new GroundingImageEncoderLayer<T>(DeformableAttention(c.Width, c.Heads, c.Levels, c.Points), Norm(),
                Linear(c.FeedForward), Linear(c.Width), Norm());
            _enhancers.Add(new GroundingEnhancerLayer<T>(image, text, fusion));
        }
        _encoderOutput = Linear(c.Width);
        _encoderOutputNorm = Norm();
        _encoderBoxHead = new[] { Linear(c.Width), Linear(c.Width), Linear(4) };
        if (!HasText) _encoderObjectness = Linear(1);
        _contentQueries = Linear(c.Width, bias: false);
        _referenceHead = new[] { Linear(c.Width), Linear(c.Width) };
        for (int i = 0; i < c.DecoderLayers; i++)
            _decoders.Add(new GroundingDecoderLayer<T>(Attention(c.Width, c.Heads), Norm(),
                HasText ? Attention(c.Width, c.Heads) : null, HasText ? Norm() : null,
                DeformableAttention(c.Width, c.Heads, c.Levels, c.Points), Norm(), Linear(c.FeedForward), Linear(c.Width), Norm()));
        _decoderNorm = Norm();
        _boxHead = new[] { Linear(c.Width), Linear(c.Width), Linear(4) };
        if (!HasText) _objectnessHead = Linear(1);
    }

    public GroundingQueryDecoderConfiguration Configuration { get; }

    /// <summary>The feature-enhancer layers.</summary>
    internal IReadOnlyList<GroundingEnhancerLayer<T>> EnhancerLayers => _enhancers;

    /// <summary>The cross-modality decoder layers.</summary>
    internal IReadOnlyList<GroundingDecoderLayer<T>> DecoderLayers => _decoders;

    /// <summary>The encoder's (two-stage proposal) box head.</summary>
    internal IReadOnlyList<DenseLayer<T>> EncoderBoxHead => _encoderBoxHead;

    /// <summary>The decoder's shared box head.</summary>
    internal IReadOnlyList<DenseLayer<T>> BoxHead => _boxHead;

    /// <summary>The MLP from a box's sine embedding to the positional query (DINO's <c>ref_point_head</c>).</summary>
    internal IReadOnlyList<DenseLayer<T>> ReferenceHead => _referenceHead;

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
        int n = levels.Tokens.Shape[0];
        Tensor<T>? embeddedText = HasText ? EmbedText(tokenIds ?? throw new ArgumentNullException(nameof(tokenIds))) : null;
        var (image, text) = Enhance(levels, embeddedText);
        var imageLevels = levels with { Tokens = image };

        // gen_encoder_output_proposals: every token's cell (centre, DINO's side 0.05 * 2^level) in logit space; a cell
        // with a coordinate outside (0.01, 0.99) is invalid: its memory is zeroed and its proposal sent to +infinity.
        var (proposals, valid) = Proposals(levels);
        var validRows = Engine.TensorBroadcastTo(valid, new[] { n, image.Shape[1] });
        var maskedImage = Engine.TensorWhere(validRows, image, new Tensor<T>(image.Shape.ToArray()));
        var proposalLogits = Engine.TensorWhere(Engine.TensorBroadcastTo(valid, new[] { n, 4 }), InverseSigmoid(Engine, proposals),
            Tensor<T>.CreateDefault(new[] { n, 4 }, NumOps.MaxValue));

        // Language-guided query selection (Eq. 1-2): each image token's largest similarity to any text token, top k.
        var memory = Rows(_encoderOutputNorm, Rows(_encoderOutput, maskedImage));
        var tokenScores = text is not null
            ? Engine.ReduceMax(Engine.TensorMatMul(memory, Engine.TensorTranspose(text)), new[] { 1 }, false)
            : Rows(_encoderObjectness ?? throw new InvalidOperationException("The objectness head is missing."), memory).Reshape(new[] { n });
        int k = Math.Min(c.Queries, n);
        Engine.TensorTopK(tokenScores, k, 0, out var selected);
        var selectedMemory = Engine.TensorIndexSelect(memory, selected, 0);
        var selectedProposals = Engine.TensorIndexSelect(proposalLogits, selected, 0);
        var encoderBoxes = Engine.TensorSigmoid(Engine.TensorAdd(Mlp(_encoderBoxHead, selectedMemory, x => Engine.ReLU(x)), selectedProposals));
        var encoderLogits = text is not null
            ? Engine.TensorMatMul(selectedMemory, Engine.TensorTranspose(text))
            : Rows(_encoderObjectness ?? throw new InvalidOperationException("The objectness head is missing."), selectedMemory);
        var layerBoxes = new List<Tensor<T>>(_decoders.Count);
        var layerLogits = new List<Tensor<T>>(_decoders.Count);

        // The decoder starts from the proposals without gradient (DINO's refpoint_embed = refpoint_embed_undetach.detach()).
        var reference = Engine.StopGradient(encoderBoxes);
        var previous = reference;

        // Mixed query selection: learnable content queries, positional queries from the selected boxes.
        var queries = Rows(_contentQueries, OneHot(Enumerable.Range(0, k).ToList(), c.Queries));
        foreach (var layer in _decoders)
        {
            var position = Mlp(_referenceHead, SineEmbedding(reference, c.Width / 2, QueryPositionTemperature), x => Engine.ReLU(x));
            var withPosition = Engine.TensorAdd(queries, position);
            queries = Rows(layer.SelfNorm, Engine.TensorAdd(queries, SelfAttendValues(layer.SelfAttention, withPosition, queries)));
            // Text cross-attention: the query carries the positional query (with_pos_embed(tgt, tgt_query_pos)).
            if (layer.TextAttention is not null && layer.TextNorm is not null && text is not null)
                queries = Rows(layer.TextNorm, Engine.TensorAdd(queries,
                    CrossAttend(layer.TextAttention, Engine.TensorAdd(queries, position), text)));
            // Deformable cross-attention to the pyramid around each query's current box.
            queries = Rows(layer.ImageNorm, Engine.TensorAdd(queries,
                Deform(layer.ImageAttention, Engine.TensorAdd(queries, position), reference, imageLevels)));
            var hidden = Rows(layer.FeedForwardDown, Engine.ReLU(Rows(layer.FeedForwardUp, queries)));
            queries = Rows(layer.FeedForwardNorm, Engine.TensorAdd(queries, hidden));

            // Refinement with the shared box head. The box the loss sees refines the previous layer's undetached box
            // (look-forward-twice); the next layer reads it detached.
            var output = Rows(_decoderNorm, queries);
            var delta = Mlp(_boxHead, output, x => Engine.ReLU(x));
            var boxes = Engine.TensorSigmoid(Engine.TensorAdd(delta, InverseSigmoid(Engine, previous)));
            layerBoxes.Add(boxes);
            layerLogits.Add(ClassLogits(output, text));
            previous = boxes;
            reference = Engine.StopGradient(boxes);
        }

        var finalBoxes = layerBoxes.Count > 0 ? layerBoxes[layerBoxes.Count - 1] : encoderBoxes;
        var logits = layerLogits.Count > 0 ? layerLogits[layerLogits.Count - 1] : ClassLogits(Rows(_decoderNorm, queries), text);
        return new GroundingQueryPrediction<T>(finalBoxes, logits, selected, layerBoxes, layerLogits, encoderBoxes, encoderLogits);
    }

    /// <summary>
    /// The feature enhancer (§3.1) on the pyramid's tokens and the embedded text: per layer, the bi-directional fusion,
    /// the text self-attention (positions as their sine embedding), then the deformable self-attention of the image
    /// tokens around their own centres (positions as Grounding DINO's sine embedding plus the level embedding).
    /// </summary>
    internal (Tensor<T> Image, Tensor<T>? Text) Enhance(FeatureLevels<T> levels, Tensor<T>? text)
    {
        if (levels is null) throw new ArgumentNullException(nameof(levels));
        var image = levels.Tokens;
        if (_enhancers.Count == 0) return (image, text);
        var imagePosition = Engine.TensorAdd(ImagePosition(levels), levels.LevelEmbedding);
        Tensor<T>? textPosition = null;
        if (text is not null)
        {
            int length = text.Shape[0];
            // get_sine_pos_embed(position_ids[..., None], num_pos_feats=d): D features of the integer position.
            textPosition = PadToWidth(SineEmbedding(Engine.TensorArange<T>(0, length, 1).Reshape(new[] { length, 1 }), Configuration.Width,
                QueryPositionTemperature));
        }
        foreach (var e in _enhancers)
        {
            if (text is not null && textPosition is not null && e.Fusion is not null && e.Text is not null)
            {
                var (imageUpdate, textUpdate) = BiAttend(e.Fusion, image, text);
                image = imageUpdate;
                text = RunText(e.Text, textUpdate, textPosition);
            }

            // Deformable DETR's encoder layer: src = norm1(src + attn(src + pos, src)); src = norm2(src + ffn(src)).
            var attended = Deform(e.Image.SelfAttention, Engine.TensorAdd(image, imagePosition), levels.Centres, levels with { Tokens = image });
            image = Rows(e.Image.SelfNorm, Engine.TensorAdd(image, attended));
            var hidden = Rows(e.Image.FeedForwardDown, Engine.ReLU(Rows(e.Image.FeedForwardUp, image)));
            image = Rows(e.Image.FeedForwardNorm, Engine.TensorAdd(image, hidden));
        }
        return (image, text);
    }

    /// <summary>
    /// <c>BiAttentionBlock</c>: the image <c>[N, d]</c> and text <c>[L, d]</c> after their layer-scaled bi-directional
    /// updates. The attention logits <c>(W_v v)(W_l l)^T / sqrt(d_h)</c> per head, clamped to +-50000, are softmaxed
    /// over the text for the image (values from the text) and, transposed and shifted by their row maximum, over the
    /// image for the text (values from the image).
    /// </summary>
    internal (Tensor<T> Image, Tensor<T> Text) BiAttend(GroundingBiAttention<T> fusion, Tensor<T> image, Tensor<T> text)
    {
        if (fusion is null) throw new ArgumentNullException(nameof(fusion));
        int nv = image.Shape[0], nl = text.Shape[0], h = fusion.Heads, dh = fusion.Width / fusion.Heads;
        var v = Rows(fusion.ImageNorm, image);
        var l = Rows(fusion.TextNorm, text);
        Tensor<T> Heads(Tensor<T> rows, int count) => Engine.TensorPermute(Engine.Reshape(rows, new[] { count, h, dh }), new[] { 1, 0, 2 });
        var queries = Heads(Engine.TensorMultiplyScalar(Rows(fusion.ImageQuery, v), NumOps.FromDouble(1.0 / Math.Sqrt(dh))), nv);
        var keys = Heads(Rows(fusion.TextKey, l), nl);
        var imageValues = Heads(Rows(fusion.ImageValue, v), nv);
        var textValues = Heads(Rows(fusion.TextValue, l), nl);

        var clamp = NumOps.FromDouble(FusionLogitClamp);
        var negativeClamp = NumOps.FromDouble(-FusionLogitClamp);
        // stable_softmax_2d: the logits shifted by their global maximum, then clamped.
        var raw = Engine.BatchMatMul(queries, Engine.TensorPermute(keys, new[] { 0, 2, 1 }));
        var globalMax = Engine.TensorBroadcastTo(Engine.ReduceMax(raw, new[] { 0, 1, 2 }, true), new[] { h, nv, nl });
        var logits = Engine.TensorClamp(Engine.TensorSubtract(raw, globalMax), negativeClamp, clamp);
        var toText = Engine.Softmax(logits, 2);
        var transposed = Engine.TensorPermute(logits, new[] { 0, 2, 1 });
        var shifted = Engine.TensorSubtract(transposed, Engine.TensorBroadcastTo(Engine.ReduceMax(transposed, new[] { 2 }, true), new[] { h, nl, nv }));
        var toImage = Engine.Softmax(Engine.TensorClamp(shifted, negativeClamp, clamp), 2);

        Tensor<T> Merge(Tensor<T> perHead, int count) => Engine.Reshape(Engine.TensorPermute(perHead, new[] { 1, 0, 2 }), new[] { count, h * dh });
        var deltaImage = Rows(fusion.ImageOutput, Merge(Engine.BatchMatMul(toText, textValues), nv));
        var deltaText = Rows(fusion.TextOutput, Merge(Engine.BatchMatMul(toImage, imageValues), nl));
        var one = Tensor<T>.CreateDefault(new[] { 1, 1 }, NumOps.One);
        var imageScale = Engine.TensorBroadcastTo(Rows(fusion.ImageScale, one), new[] { nv, image.Shape[1] });
        var textScale = Engine.TensorBroadcastTo(Rows(fusion.TextScale, one), new[] { nl, text.Shape[1] });
        return (Engine.TensorAdd(image, Engine.TensorMultiply(imageScale, deltaImage)),
            Engine.TensorAdd(text, Engine.TensorMultiply(textScale, deltaText)));
    }

    // Grounding DINO's text TransformerEncoderLayer: q = k = src + pos, v = src; post-norm; ReLU feed-forward.
    private Tensor<T> RunText(GroundingTextEncoderLayer<T> layer, Tensor<T> text, Tensor<T> position)
    {
        text = Rows(layer.SelfNorm, Engine.TensorAdd(text, SelfAttendValues(layer.SelfAttention, Engine.TensorAdd(text, position), text)));
        var hidden = Rows(layer.FeedForwardDown, Engine.ReLU(Rows(layer.FeedForwardUp, text)));
        return Rows(layer.FeedForwardNorm, Engine.TensorAdd(text, hidden));
    }

    // The layers of a BiAttentionBlock with an embedding of `width` features over `heads` heads.
    private GroundingBiAttention<T> BiAttention(int modelWidth, int width, int heads)
    {
        int h = HeadsFor(width, heads);
        DenseLayer<T> Scale() => Add(new DenseLayer<T>(modelWidth, (IActivationFunction<T>?)null,
            new ConstantInitializationStrategy<T>(FusionLayerScale), BiasMode.Never));
        return new GroundingBiAttention<T>(Norm(), Norm(), Linear(width), Linear(width), Linear(width), Linear(width),
            Linear(modelWidth), Linear(modelWidth), Scale(), Scale(), h, width);
    }

    /// <summary>
    /// The encoder proposals of every multi-scale token <c>[S, 4]</c> (cell centre and DINO's side 0.05 * 2^level) and
    /// the <c>[S, 1]</c> 0/1 mask of those whose every coordinate lies in (0.01, 0.99).
    /// </summary>
    internal (Tensor<T> Proposals, Tensor<T> Valid) Proposals(FeatureLevels<T> levels)
    {
        if (levels is null) throw new ArgumentNullException(nameof(levels));
        int n = levels.Centres.Shape[0];
        var levelOfToken = new Tensor<int>(new[] { n }, Vector<int>.FromArray(
            levels.Shapes.SelectMany((shape, level) => Enumerable.Repeat(level, shape.H * shape.W)).ToArray()));
        var sides = Engine.TensorMultiplyScalar(Engine.TensorPower(
            Tensor<T>.CreateDefault(new[] { levels.Count }, NumOps.FromDouble(2.0)), Engine.TensorArange<T>(0, levels.Count, 1)),
            NumOps.FromDouble(ProposalSide));
        var tokenSides = Engine.TensorIndexSelect(sides, levelOfToken, 0).Reshape(new[] { n, 1 });
        var proposals = Engine.TensorConcatenate(new[] { levels.Centres, Engine.TensorBroadcastTo(tokenSides, new[] { n, 2 }) }, 1);
        var inside = Engine.TensorMultiply(Engine.TensorGreaterThan(proposals, NumOps.FromDouble(ProposalLow)),
            Engine.TensorLessThan(proposals, NumOps.FromDouble(ProposalHigh)));
        Tensor<T> Column(int j) => Engine.TensorSlice(inside, new[] { 0, j }, new[] { n, 1 });
        var valid = Engine.TensorMultiply(Engine.TensorMultiply(Column(0), Column(1)), Engine.TensorMultiply(Column(2), Column(3)));
        return (proposals, valid);
    }

    // The shared class head: contrastive logits against the text tokens, or objectness for a class-agnostic head.
    private Tensor<T> ClassLogits(Tensor<T> output, Tensor<T>? text)
        => text is not null
            ? Engine.TensorMatMul(output, Engine.TensorTranspose(text))
            : Rows(_objectnessHead ?? throw new InvalidOperationException("The objectness head is missing."), output);

    // Self-attention whose queries and keys carry the positional query and whose values do not (DETR).
    private Tensor<T> SelfAttendValues(MultiHeadAttentionLayer<T> attention, Tensor<T> withPosition, Tensor<T> values)
    {
        int n = values.Shape[0], d = values.Shape[1];
        var q = Engine.Reshape(withPosition, new[] { 1, n, d });
        return Engine.Reshape(attention.Forward(q, q, Engine.Reshape(values, new[] { 1, n, d })), new[] { n, d });
    }

    /// <summary>
    /// Grounding DINO's image position embedding (<c>PositionEmbeddingSineHW</c>, normalized, temperature 20): per level,
    /// the cumulative position of each token, <c>((j + 1) / w, (i + 1) / h)</c>, embedded with D/2 features per axis,
    /// y first; zero-padded to D when D is not a multiple of 4.
    /// </summary>
    internal Tensor<T> ImagePosition(FeatureLevels<T> levels)
    {
        if (levels is null) throw new ArgumentNullException(nameof(levels));
        // Every token's (j + 1, i + 1) over its level's (w, h), counted in int and divided in T.
        var cumulative = levels.Shapes.SelectMany(shape => Enumerable.Range(0, shape.H * shape.W)
            .SelectMany(t => new[] { t % shape.W + 1, t / shape.W + 1 })).ToArray();
        var extents = levels.Shapes.SelectMany(shape => Enumerable.Range(0, shape.H * shape.W)
            .SelectMany(_ => new[] { shape.W, shape.H })).ToArray();
        int s = cumulative.Length / 2;
        var cells = Engine.TensorDivide(new Tensor<int>(new[] { s, 2 }, Vector<int>.FromArray(cumulative)).Cast<T>(),
            new Tensor<int>(new[] { s, 2 }, Vector<int>.FromArray(extents)).Cast<T>());
        return PadToWidth(SineEmbedding(cells, Configuration.Width / 2, ImagePositionTemperature));
    }

    // Zero-pads [s, f] features to the head width.
    private Tensor<T> PadToWidth(Tensor<T> features)
    {
        int s = features.Shape[0], missing = Configuration.Width - features.Shape[1];
        return missing > 0 ? Engine.TensorConcatenate(new[] { features, new Tensor<T>(new[] { s, missing }) }, 1) : features;
    }

    /// <summary>
    /// DINO's <c>gen_sineembed_for_position</c> of boxes <c>[k, 4]</c> (D/2 features per coordinate, temperature 10000).
    /// </summary>
    internal Tensor<T> SineEmbedding(Tensor<T> boxes) => SineEmbedding(boxes, Configuration.Width / 2, QueryPositionTemperature);

    /// <summary>
    /// The sine embedding of positions <c>[k, c]</c> as DINO's <c>gen_sineembed_for_position</c> and Grounding DINO's
    /// <c>get_sine_pos_embed</c> build it: per coordinate, <paramref name="features"/> features interleaving
    /// <c>sin</c> and <c>cos</c> of <c>2π·v / temperature^(2i/features)</c> for <c>i &lt; features/2</c>; the coordinates
    /// are emitted with the first two swapped (y before x, then w and h), as the reference does.
    /// </summary>
    internal Tensor<T> SineEmbedding(Tensor<T> positions, int features, double temperature)
    {
        int k = positions.Shape[0], coordinates = positions.Shape[1], pairs = features / 2;
        if (pairs <= 0) throw new ArgumentOutOfRangeException(nameof(features), "The embedding needs at least two features per coordinate.");
        // Frequencies temperature^(2i/features) for i < features/2, as a [1, pairs] row.
        var exponents = Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, pairs, 1), NumOps.FromDouble(2.0 / features));
        var frequencies = Engine.TensorPower(Tensor<T>.CreateDefault(new[] { pairs }, NumOps.FromDouble(temperature)), exponents)
            .Reshape(new[] { 1, pairs });
        var twoPi = NumOps.FromDouble(2 * Math.PI);
        var parts = new List<Tensor<T>>(coordinates);
        for (int index = 0; index < coordinates; index++)
        {
            int coordinate = coordinates >= 2 && index < 2 ? 1 - index : index;
            var angle = Engine.TensorDivide(
                Engine.TensorBroadcastTo(Engine.TensorMultiplyScalar(Engine.TensorSlice(positions, new[] { 0, coordinate }, new[] { k, 1 }), twoPi), new[] { k, pairs }),
                Engine.TensorBroadcastTo(frequencies, new[] { k, pairs }));
            // [k, pairs, 2] (sin, cos) flattened: sin f0, cos f0, sin f1, cos f1, ...
            parts.Add(Engine.Reshape(Engine.TensorStack(new[] { Engine.TensorSin(angle), Engine.TensorCos(angle) }, 2), new[] { k, 2 * pairs }));
        }
        return Engine.TensorConcatenate(parts.ToArray(), 1);
    }
}
