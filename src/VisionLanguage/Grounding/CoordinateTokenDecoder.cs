using System.Globalization;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>How a coordinate-token decoder spells a coordinate value <c>k</c> in text.</summary>
internal enum CoordinateSpelling
{
    /// <summary><c>k / Divisor</c> as a decimal fraction with log10(Divisor) places (Shikra: "0.123").</summary>
    DecimalFraction,

    /// <summary><c>k</c> as a whole number without leading zeros (Ferret: "123").</summary>
    Integer,
}

/// <summary>The shape of a <see cref="CoordinateTokenDecoder{T}"/>.</summary>
/// <param name="Width">The language model's width.</param>
/// <param name="VocabSize">Text vocabulary size.</param>
/// <param name="MaxTextLength">Longest text the decoder embeds.</param>
/// <param name="Bins">How many coordinate values the decoder can write (Shikra: 1001 for 0.000-1.000; Ferret: 1000 for
/// 0-999).</param>
/// <param name="Divisor">The value a coordinate value is divided by to give a normalized coordinate (1000 for both).</param>
/// <param name="Spelling">How a coordinate value is written (Shikra: a decimal fraction; Ferret: a whole number).</param>
internal sealed record CoordinateTokenDecoderConfiguration(int Width, int VocabSize, int MaxTextLength, int Bins, double Divisor,
    CoordinateSpelling Spelling);

/// <summary>
/// Grounding by writing box coordinates as text, as Shikra (Chen et al. 2023, section 3: "the coordinates are expressed
/// in natural language as numbers", <c>[x1,y1,x2,y2]</c> normalized with three decimals) and Ferret / Ferret-v2 (You et
/// al. 2023, Zhang et al. 2024: discrete coordinates in [0, 999] written as text) do. The instruction's tokens follow
/// the image tokens; the language model then generates, greedily, each box character by character, digit by digit as
/// the LLaMA tokenizer splits numbers: <c>[</c>, four coordinates separated by <c>,</c>, <c>]</c>, with boxes separated
/// by <c>;</c> and the answer closed by the end token. A box's score is the probability the model gives its characters.
/// </summary>
/// <remarks>Decoding is constrained to well-formed boxes whose coordinates are values the model can write, as the
/// references' structured-output decoding does.</remarks>
internal sealed class CoordinateTokenDecoder<T> : GroundingHeadModule<T>
{
    // The output characters: the digits 0-9, then the punctuation of a box, then the end of the answer.
    private const int Point = 10;
    private const int Comma = 11;
    private const int Open = 12;
    private const int Close = 13;
    private const int Semicolon = 14;
    private const int EndToken = 15;
    private const int Vocabulary = 16;
    private const string Characters = "0123456789.,[];";

    private readonly DenseLayer<T> _textEmbedding;
    private readonly DenseLayer<T> _coordinateEmbedding;
    private readonly DenseLayer<T> _head;
    private readonly Dictionary<string, int> _values = new(StringComparer.Ordinal);
    private readonly HashSet<string> _prefixes = new(StringComparer.Ordinal);
    private readonly int _longestSpelling;

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
        int places = Math.Max(0, (int)Math.Round(Math.Log10(c.Divisor)));
        string format = "F" + places.ToString(CultureInfo.InvariantCulture);
        for (int k = 0; k < c.Bins; k++)
        {
            string spelling = c.Spelling == CoordinateSpelling.Integer
                ? k.ToString(CultureInfo.InvariantCulture)
                : (k / c.Divisor).ToString(format, CultureInfo.InvariantCulture);
            if (_values.ContainsKey(spelling)) continue;
            _values[spelling] = k;
            for (int length = 1; length <= spelling.Length; length++) _prefixes.Add(spelling.Substring(0, length));
            _longestSpelling = Math.Max(_longestSpelling, spelling.Length);
        }
    }

    public CoordinateTokenDecoderConfiguration Configuration { get; }

    /// <summary>The end-of-answer token.</summary>
    public int End => EndToken;

    public Tensor<T> EmbedText(IReadOnlyList<int> tokenIds)
    {
        if (tokenIds is null || tokenIds.Count == 0) throw new ArgumentException("The instruction has no tokens.", nameof(tokenIds));
        var ids = tokenIds.Take(Math.Min(tokenIds.Count, Configuration.MaxTextLength)).ToList();
        return Rows(_textEmbedding, OneHot(ids, Configuration.VocabSize));
    }

    /// <summary>The output tokens <c>[1, 16]</c> (as 0/1) that keep <paramref name="tokens"/> a well-formed answer.</summary>
    internal Tensor<T> Allowed(IReadOnlyList<int> tokens, int maxBoxes)
    {
        var allowed = new bool[Vocabulary];
        int boxes = 0, coordinate = -1;
        string current = string.Empty;
        foreach (int token in tokens)
        {
            if (token == Open) { coordinate = 0; current = string.Empty; }
            else if (token == Comma) { coordinate++; current = string.Empty; }
            else if (token == Close) { boxes++; coordinate = -1; }
            else if (token < Semicolon) current += Characters[token];
        }

        int last = tokens.Count > 0 ? tokens[tokens.Count - 1] : -1;
        if (coordinate < 0)
        {
            if (last == Semicolon) allowed[Open] = true;
            else
            {
                // At the start or after a box: another box (Open at the start, ';' between boxes) or the end.
                if (boxes < maxBoxes) allowed[last == Close ? Semicolon : Open] = true;
                allowed[EndToken] = true;
            }
        }
        else
        {
            for (int character = 0; character <= Point; character++)
                allowed[character] = _prefixes.Contains(current + Characters[character]);
            if (_values.ContainsKey(current))
            {
                if (coordinate < 3) allowed[Comma] = true;
                else allowed[Close] = true;
            }
        }

        var one = NumOps.One;
        var zero = NumOps.Zero;
        return new Tensor<T>(new[] { 1, Vocabulary }, Vector<T>.FromArray(allowed.Select(a => a ? one : zero).ToArray()));
    }

    /// <summary>
    /// Grounds a phrase: <paramref name="imageTokens"/> are the image's language-model tokens <c>[N, Width]</c>,
    /// <paramref name="languageModel"/> runs the language model on <c>[n, Width]</c> rows.
    /// </summary>
    /// <returns>The written boxes <c>[b, 4]</c> as normalized (x1, y1, x2, y2) and their scores <c>[b]</c>.</returns>
    public (Tensor<T> Boxes, Tensor<T> Scores) Ground(Tensor<T> imageTokens, IReadOnlyList<int> tokenIds,
        Func<Tensor<T>, Tensor<T>> languageModel, int maxBoxes)
    {
        if (imageTokens is null) throw new ArgumentNullException(nameof(imageTokens));
        if (languageModel is null) throw new ArgumentNullException(nameof(languageModel));
        var prefix = Engine.TensorConcatenate(new[] { AsRows(Engine, imageTokens), EmbedText(tokenIds) }, 0);
        int boxLength = 4 * _longestSpelling + 5;
        var (tokens, probabilities) = DecodeGreedy(prefix, languageModel, _head,
            token => Rows(_coordinateEmbedding, OneHot(new[] { token }, Vocabulary)),
            written => Allowed(written, maxBoxes), EndToken, boxLength * Math.Max(0, maxBoxes) + 1);

        // Whole boxes only: '[' four values ']', scored by the probability of every character from '[' to ']'.
        var coordinates = new List<int>();
        var scores = new List<Tensor<T>>();
        int open = -1;
        var values = new List<int>(4);
        string current = string.Empty;
        for (int i = 0; i < tokens.Count; i++)
        {
            int token = tokens[i];
            if (token == Open) { open = i; values.Clear(); current = string.Empty; }
            else if (open >= 0 && (token == Comma || token == Close))
            {
                values.Add(_values.TryGetValue(current, out int value) ? value : 0);
                current = string.Empty;
                if (token == Close)
                {
                    coordinates.AddRange(values);
                    scores.Add(probabilities.Skip(open).Take(i - open + 1).Aggregate((a, b) => Engine.TensorMultiply(a, b)));
                    open = -1;
                }
            }
            else if (open >= 0 && token <= Point) current += Characters[token];
        }
        int complete = scores.Count;
        if (complete == 0) return (new Tensor<T>(new[] { 0, 4 }), new Tensor<T>(new[] { 0 }));

        var bins = new Tensor<int>(new[] { complete, 4 }, Vector<int>.FromList(coordinates)).Cast<T>();
        var boxes = Engine.TensorClamp(Engine.TensorDivideScalar(bins, NumOps.FromDouble(Configuration.Divisor)), NumOps.Zero, NumOps.One);
        return (boxes, Engine.TensorConcatenate(scores.ToArray(), 0).Reshape(new[] { complete }));
    }
}
