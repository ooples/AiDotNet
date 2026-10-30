using AiDotNet.Agentic.Models.Local;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// The green-list language-model watermark of Kirchenbauer et al. (2023): generation-time logit bias toward a keyed
/// green list, detected with a one-proportion z-test.
/// </summary>
/// <remarks>
/// <para>
/// <b>Embedding.</b> Before each token is sampled, the preceding <c>contextWidth</c> token ids and the secret key are
/// hashed (HMAC-SHA256) into a seed, and the seed places every vocabulary entry on the green list with probability
/// gamma (<c>greenListFraction</c>). delta = 4 x <c>watermarkStrength</c> is added to the logits of the green tokens.
/// This is the paper's "soft" watermark with its LeftHash context (Algorithm 2); with the defaults gamma = 0.25 and
/// delta = 2.0 it is one of the settings the paper evaluates. Low-entropy steps, where one token dominates, are barely
/// affected, which is what keeps the text quality.
/// </para>
/// <para>
/// <b>Detection.</b> Each scored token is green with probability gamma in text the key did not shape, so the count
/// |s|_G of green tokens among T scored tokens gives z = (|s|_G - gamma T) / sqrt(T gamma (1 - gamma)) (the paper's
/// Eq. 2). Following the authors' follow-up (Kirchenbauer et al., ICLR 2024), each (context, token) pair is scored
/// once, so repeated phrases do not inflate z. The confidence is 0 up to z = 3 (false-positive rate 0.135%) and 1 from
/// z = 6; <see cref="TextWatermarkerBase{T}.EvaluateText"/> reports from z = 3.9 by default.
/// </para>
/// <para>
/// <b>Where the green list differs from the paper.</b> The paper draws an exact gamma-fraction of the vocabulary with
/// a seeded permutation; here each token is green independently with probability gamma, decided by a keyed hash of
/// (seed, token id). The detection statistic and its null distribution are the same, the cost is O(1) per token
/// instead of a permutation per step, and the decision for one token can be checked without the vocabulary size.
/// </para>
/// <para>
/// <b>Hooks.</b> Embedding needs the generator: wrap an <see cref="ICausalLanguageModel{T}"/> with
/// <see cref="WatermarkModel"/> (it keeps <see cref="IIncrementalCausalLanguageModel{T}"/> models incremental) or call
/// <see cref="BiasLogits"/> in your own decode loop. Detecting a string needs the generator's tokenizer
/// (<c>tokenizer</c>); without one, <see cref="Detect"/> has no token ids to score and reports no evidence, and
/// <see cref="DetectTokens"/> scores token ids directly.
/// </para>
/// <para>
/// <b>For Beginners:</b> while the AI writes, this watermarker nudges it toward a secret list of words that changes
/// at every position. A reader cannot tell, but with the key you can count how often the AI picked listed words:
/// about a quarter of the time by chance, far more often when the watermark is present.
/// </para>
/// <para>
/// <b>References:</b>
/// - Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein. A Watermark for Large Language Models (ICML 2023)
/// - Kirchenbauer et al. On the Reliability of Watermarks for Large Language Models (ICLR 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Watermark for Large Language Models",
    "https://arxiv.org/abs/2301.10226",
    Year = 2023,
    Authors = "John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein")]
public class SamplingWatermarker<T> : TextWatermarkerBase<T>, IGenerationWatermarker<T>
{
    // Separates this watermark's seeds from other watermarkers sharing the key ("KGW1").
    private const int SeedSalt = 0x4B475731;

    private readonly int? _vocabSize;
    private readonly double _gamma;
    private readonly double _delta;
    private readonly int _contextWidth;
    private readonly bool _scoreRepeatedNgrams;
    private readonly IGenerationTokenizer? _tokenizer;

    /// <inheritdoc />
    public override string ModuleName => "SamplingWatermarker";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Green-list token watermark";

    /// <summary>
    /// Initializes a new green-list watermarker.
    /// </summary>
    /// <param name="watermarkStrength">
    /// Strength of the logit bias: delta = 4 x strength is added to green-token logits. Default: 0.5 (delta = 2.0).
    /// </param>
    /// <param name="vocabSize">
    /// The generator's vocabulary size: token ids outside [0, vocabSize) are not scored, and <see cref="BiasLogits"/>
    /// rejects a longer logits vector. <c>null</c> (the default) accepts any non-negative id.
    /// </param>
    /// <param name="greenListFraction">The green-list fraction gamma, in (0, 1). Default: 0.25.</param>
    /// <param name="key">The secret key; <c>null</c> uses the published <see cref="WatermarkKey.Public"/> key.</param>
    /// <param name="tokenizer">
    /// The generator's tokenizer, used to turn text back into the token ids the watermark was embedded in.
    /// <c>null</c> means text cannot be scored, only token ids (<see cref="DetectTokens"/>).
    /// </param>
    /// <param name="contextWidth">The number of preceding tokens hashed into the green-list seed (h). Default: 1.</param>
    /// <param name="scoreRepeatedNgrams">
    /// Whether a repeated (context, token) pair is scored every time it occurs. Default: <c>false</c>, which keeps
    /// repetitive text from inflating the z-score.
    /// </param>
    public SamplingWatermarker(
        double watermarkStrength = 0.5,
        int? vocabSize = null,
        double greenListFraction = 0.25,
        string? key = null,
        IGenerationTokenizer? tokenizer = null,
        int contextWidth = 1,
        bool scoreRepeatedNgrams = false)
        : base(watermarkStrength, key)
    {
        if (watermarkStrength <= 0)
            throw new ArgumentOutOfRangeException(nameof(watermarkStrength), "Watermark strength must be positive.");
        if (vocabSize is { } v && v <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabSize), "The vocabulary size must be positive.");
        if (double.IsNaN(greenListFraction) || greenListFraction <= 0 || greenListFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(greenListFraction), "The green-list fraction must lie strictly between 0 and 1.");
        if (contextWidth < 1)
            throw new ArgumentOutOfRangeException(nameof(contextWidth), "The context width must be at least 1.");

        _vocabSize = vocabSize;
        _gamma = greenListFraction;
        _delta = 4.0 * watermarkStrength;
        _contextWidth = contextWidth;
        _scoreRepeatedNgrams = scoreRepeatedNgrams;
        _tokenizer = tokenizer;
    }

    /// <summary>The green-list fraction gamma.</summary>
    public double GreenListFraction => _gamma;

    /// <summary>The logit bias delta added to green tokens.</summary>
    public double GreenListBias => _delta;

    /// <summary>The number of preceding tokens hashed into the green-list seed.</summary>
    public int ContextWidth => _contextWidth;

    /// <summary>
    /// Whether <paramref name="token"/> is on the green list that follows <paramref name="context"/>, whose last
    /// <see cref="ContextWidth"/> entries are used.
    /// </summary>
    /// <exception cref="ArgumentException">The context is shorter than <see cref="ContextWidth"/>.</exception>
    public bool IsGreen(IReadOnlyList<int> context, int token)
    {
        if (context is null) throw new ArgumentNullException(nameof(context));
        if (context.Count < _contextWidth)
            throw new ArgumentException($"The green list needs {_contextWidth} preceding tokens; got {context.Count}.", nameof(context));
        return WatermarkKey.Uniform(ContextSeed(context, context.Count), token) < _gamma;
    }

    /// <inheritdoc />
    public void BiasLogits(Vector<T> logits, IReadOnlyList<int> context)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (context is null) throw new ArgumentNullException(nameof(context));
        if (_vocabSize is { } v && logits.Length > v)
            throw new ArgumentException($"The logits cover {logits.Length} tokens but the watermarker was configured for a vocabulary of {v}.", nameof(logits));
        if (context.Count < _contextWidth) return;

        ulong seed = ContextSeed(context, context.Count);
        T delta = NumOps.FromDouble(_delta);
        for (int token = 0; token < logits.Length; token++)
        {
            if (WatermarkKey.Uniform(seed, token) < _gamma)
                logits[token] = NumOps.Add(logits[token], delta);
        }
    }

    /// <inheritdoc />
    public ICausalLanguageModel<T> WatermarkModel(ICausalLanguageModel<T> model)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (_vocabSize is { } v && model.VocabularySize > v)
            throw new ArgumentException($"The model's vocabulary of {model.VocabularySize} exceeds the configured {v}.", nameof(model));
        return model is IIncrementalCausalLanguageModel<T> incremental
            ? new WatermarkedIncrementalCausalLanguageModel<T>(incremental, this)
            : new WatermarkedCausalLanguageModel<T>(model, this);
    }

    /// <inheritdoc />
    public WatermarkResult DetectTokens(IReadOnlyList<int> tokenIds)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));
        int green = 0, scored = 0;
        var seen = _scoreRepeatedNgrams ? null : new HashSet<string>(StringComparer.Ordinal);
        for (int i = _contextWidth; i < tokenIds.Count; i++)
        {
            int token = tokenIds[i];
            if (token < 0 || (_vocabSize is { } v && token >= v)) continue;
            if (seen is not null && !seen.Add(NgramKey(tokenIds, i))) continue;
            if (WatermarkKey.Uniform(ContextSeed(tokenIds, i), token) < _gamma) green++;
            scored++;
        }

        return TextResult(WatermarkStatistics.CountZ(green, scored, _gamma), scored);
    }

    /// <inheritdoc />
    /// <remarks>
    /// The text is re-tokenized with the generator's tokenizer. Without a tokenizer there are no token ids to score
    /// and the result reports no evidence.
    /// </remarks>
    public override WatermarkResult Detect(string text)
    {
        if (_tokenizer is null || string.IsNullOrWhiteSpace(text)) return TextResult(0, 0);
        return DetectTokens(_tokenizer.Encode(text));
    }

    // HMAC of the `width` tokens before `position`.
    private ulong ContextSeed(IReadOnlyList<int> tokens, int position)
    {
        var context = new int[_contextWidth];
        for (int j = 0; j < _contextWidth; j++) context[j] = tokens[position - _contextWidth + j];
        return Key.Seed(SeedSalt, context);
    }

    private string NgramKey(IReadOnlyList<int> tokens, int position)
    {
        var parts = new string[_contextWidth + 1];
        for (int j = 0; j <= _contextWidth; j++) parts[j] = tokens[position - _contextWidth + j].ToString(System.Globalization.CultureInfo.InvariantCulture);
        return string.Join(",", parts);
    }
}
