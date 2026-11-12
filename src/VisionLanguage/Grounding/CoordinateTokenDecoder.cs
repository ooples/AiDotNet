namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>The shape of a <see cref="CoordinateTokenDecoder{T}"/>.</summary>
/// <param name="Width">The language model's width.</param>
/// <param name="VocabSize">Text vocabulary size.</param>
/// <param name="MaxTextLength">Longest text the decoder embeds.</param>
/// <param name="Bins">How many coordinate values the decoder can write (Shikra: 1001 for 0.000-1.000; Ferret: 1000 for
/// 0-999).</param>
/// <param name="Divisor">The value a coordinate token is divided by to give a normalized coordinate (1000 for both).</param>
internal sealed record CoordinateTokenDecoderConfiguration(int Width, int VocabSize, int MaxTextLength, int Bins, double Divisor);

/// <summary>
/// Grounding by writing box coordinates as language-model tokens, as Shikra (Chen et al. 2023, §3: "the coordinates
/// are expressed in natural language as numbers", [x1, y1, x2, y2] normalized with three decimals) and Ferret / Ferret-v2
/// (You et al. 2023, Zhang et al. 2024: discrete coordinates in [0, 999] written as text) do. The instruction's tokens
/// follow the image tokens; the language model then generates, greedily, four coordinate tokens per box, each box
/// followed by a separator or the end of the answer. A box's score is the probability the model gives its four
/// coordinates.
/// </summary>
/// <remarks>The coordinate values are their own tokens (one per value, the bins), with a separator and an end token:
/// the reference models spell them with the digit and punctuation tokens of the LLaMA vocabulary, which carry the same
/// values. Decoding is constrained to well-formed boxes, as structured-output decoding of the references does.</remarks>
internal sealed class CoordinateTokenDecoder<T> : GroundingHeadModule<T>
{
    private readonly DenseLayer<T> _textEmbedding;
    private readonly DenseLayer<T> _coordinateEmbedding;
    private readonly DenseLayer<T> _head;

    public CoordinateTokenDecoder(IEngine engine, List<ILayer<T>> stream, CoordinateTokenDecoderConfiguration configuration)
        : base(engine, stream)
    {
        var c = configuration ?? throw new ArgumentNullException(nameof(configuration));
        if (c.Width <= 0 || c.VocabSize <= 0 || c.MaxTextLength <= 0 || c.Bins <= 1 || !(c.Divisor > 0))
            throw new ArgumentException("The coordinate decoder needs a positive width, vocabulary, text length, bins and divisor.", nameof(configuration));
        Configuration = c;
        _textEmbedding = Linear(c.Width, bias: false);
        _coordinateEmbedding = Linear(c.Width, bias: false);
        _head = Linear(Vocabulary, bias: false);
    }

    public CoordinateTokenDecoderConfiguration Configuration { get; }

    /// <summary>The token that separates two boxes.</summary>
    public int Separator => Configuration.Bins;

    /// <summary>The token that ends the answer.</summary>
    public int End => Configuration.Bins + 1;

    private int Vocabulary => Configuration.Bins + 2;

    /// <summary>The instruction's tokens embedded at the language model's width <c>[L, Width]</c>.</summary>
    public Tensor<T> EmbedText(IReadOnlyList<int> tokenIds)
    {
        if (tokenIds is null || tokenIds.Count == 0) throw new ArgumentException("The instruction has no tokens.", nameof(tokenIds));
        var ids = tokenIds.Take(Math.Min(tokenIds.Count, Configuration.MaxTextLength)).ToList();
        return Rows(_textEmbedding, OneHot(ids, Configuration.VocabSize));
    }

    /// <summary>The tokens that may follow <paramref name="tokens"/> in a well-formed answer of at most
    /// <paramref name="maxBoxes"/> boxes, as a <c>[1, Bins + 2]</c> 0/1 row.</summary>
    internal Tensor<T> Allowed(IReadOnlyList<int> tokens, int maxBoxes)
    {
        int position = tokens.Count % 5, boxes = tokens.Count / 5;
        if (position == 4) return Span(Vocabulary, Separator, End + 1);
        if (position > 0) return Span(Vocabulary, 0, Configuration.Bins);
        // A box may start (unless the answer already holds the most boxes), or the answer may end.
        return boxes >= maxBoxes ? Span(Vocabulary, End, End + 1) : Engine.TensorAdd(Span(Vocabulary, 0, Configuration.Bins), Span(Vocabulary, End, End + 1));
    }

    /// <summary>
    /// Grounds an instruction: <paramref name="imageTokens"/> <c>[N, Width]</c> are the image's language-model tokens,
    /// <paramref name="languageModel"/> runs the language model on <c>[n, Width]</c> rows.
    /// </summary>
    /// <returns>The boxes <c>[b, 4]</c> as normalized (x1, y1, x2, y2) and their scores <c>[b]</c>; empty when the
    /// model writes no box.</returns>
    public (Tensor<T> Boxes, Tensor<T> Scores) Ground(Tensor<T> imageTokens, IReadOnlyList<int> tokenIds,
        Func<Tensor<T>, Tensor<T>> languageModel, int maxBoxes)
    {
        if (imageTokens is null) throw new ArgumentNullException(nameof(imageTokens));
        if (languageModel is null) throw new ArgumentNullException(nameof(languageModel));
        var prefix = Engine.TensorConcatenate(new[] { AsRows(Engine, imageTokens), EmbedText(tokenIds) }, 0);
        var (tokens, probabilities) = DecodeGreedy(prefix, languageModel, _head,
            token => Rows(_coordinateEmbedding, OneHot(new[] { token }, Vocabulary)),
            written => Allowed(written, maxBoxes), End, 5 * Math.Max(0, maxBoxes) + 1);

        // Whole boxes only: four coordinates each, every one a coordinate value.
        int complete = 0;
        while (5 * complete + 4 <= tokens.Count && tokens.Skip(5 * complete).Take(4).All(t => t < Configuration.Bins))
            complete++;
        if (complete == 0) return (new Tensor<T>(new[] { 0, 4 }), new Tensor<T>(new[] { 0 }));

        var coordinates = new List<int>(4 * complete);
        var scores = new List<Tensor<T>>(complete);
        for (int b = 0; b < complete; b++)
        {
            coordinates.AddRange(tokens.Skip(5 * b).Take(4));
            var p = probabilities.Skip(5 * b).Take(4).ToList();
            scores.Add(Engine.TensorMultiply(Engine.TensorMultiply(p[0], p[1]), Engine.TensorMultiply(p[2], p[3])));
        }
        var values = new Tensor<int>(new[] { complete, 4 }, Vector<int>.FromList(coordinates)).Cast<T>();
        var boxes = Engine.TensorClamp(Engine.TensorDivideScalar(values, NumOps.FromDouble(Configuration.Divisor)), NumOps.Zero, NumOps.One);
        return (boxes, Engine.TensorConcatenate(scores.ToArray(), 0).Reshape(new[] { complete }));
    }
}
