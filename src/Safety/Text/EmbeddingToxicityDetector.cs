using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects toxic text using embedding-based cosine similarity to known toxic concept vectors.
/// </summary>
/// <remarks>
/// <para>
/// Computes a lightweight embedding (character n-gram hash vectors) of the input text and
/// measures cosine similarity against pre-built concept vectors for known toxic categories.
/// This approach catches semantically similar content that regex-based approaches miss.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of looking for specific bad words, this module converts text
/// into a mathematical representation (embedding) and measures how "close" it is to known
/// examples of toxic content. This catches rephrasings, misspellings, and subtle toxicity
/// that keyword matching would miss.
/// </para>
/// <para>
/// <b>References:</b>
/// - MetaTox knowledge graph for enhanced LLM toxicity detection (2024, arxiv:2412.15268)
/// - LLM-extracted rationales for interpretable hate speech detection (2024, arxiv:2403.12403)
/// - GPT-4o/LLaMA-3 zero-shot hate speech detection (2025, arxiv:2506.12744)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph",
    "https://arxiv.org/abs/2412.15268",
    Year = 2024,
    Authors = "Yifan Li, Zhengnan Hou, et al.")]
public class EmbeddingToxicityDetector<T> : TextSafetyModuleBase<T>
{

    private readonly T _threshold;
    private readonly int _embeddingDim;

    // Pre-built toxic concept vectors (character n-gram hash space)
    [Buffer]
    private readonly Vector<T>[] _toxicConceptVectors;
    private readonly SafetyCategory[] _toxicCategories;

    /// <inheritdoc />
    public override string ModuleName => "EmbeddingToxicityDetector";

    /// <summary>
    /// Initializes a new embedding-based toxicity detector.
    /// </summary>
    /// <param name="threshold">Cosine similarity threshold (0-1). Default: 0.6.</param>
    /// <param name="embeddingDim">Dimension of the hash embedding. Default: 128.</param>
    public EmbeddingToxicityDetector(double threshold = 0.6, int embeddingDim = 128)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        if (embeddingDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(embeddingDim),
                "Embedding dimension must be positive.");
        }

        _threshold = NumOps.FromDouble(threshold);
        _embeddingDim = embeddingDim;

        // Build toxic concept vectors from known toxic phrase sets
        var concepts = BuildToxicConcepts();
        _toxicConceptVectors = new Vector<T>[concepts.Count];
        _toxicCategories = new SafetyCategory[concepts.Count];

        for (int i = 0; i < concepts.Count; i++)
        {
            _toxicConceptVectors[i] = ComputeConceptEmbedding(concepts[i].Phrases);
            _toxicCategories[i] = concepts[i].Category;
        }
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text))
        {
            return findings;
        }

        // Compute embedding for input text
        var inputEmbedding = ComputeTextEmbedding(text);
        if (_toxicConceptVectors.Length == 0) return findings;

        // Compare against every toxic concept at once
        var concepts = Engine.TensorStack(Array.ConvertAll(_toxicConceptVectors, v => Tensor<T>.FromVector(v)), 0);
        var similarities = SafetyTensorOps<T>.CosineSimilarities(Tensor<T>.FromVector(inputEmbedding), concepts);
        var similarityValues = similarities.ToArray();

        foreach (int i in SafetyTensorOps<T>.IndicesAtLeast(similarities, _threshold))
        {
            double simDouble = NumOps.ToDouble(similarityValues[i]);
            findings.Add(new SafetyFinding
            {
                Category = _toxicCategories[i],
                Severity = simDouble >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = simDouble,
                Description = $"Text embedding similarity to {_toxicCategories[i]} concept: {simDouble:F3}.",
                RecommendedAction = simDouble >= 0.8 ? SafetyAction.Block : SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>
    /// Computes a character n-gram hash embedding for the given text.
    /// Uses feature hashing (hashing trick) to map n-grams to a fixed-size vector.
    /// </summary>
    private Vector<T> ComputeTextEmbedding(string text)
    {
        string normalized = text.ToLowerInvariant().Trim();

        // Character trigram and 4-gram hashing; the sign of the hash is a random projection
        var hashes = SafetyTensorOps<T>.NgramHashes(normalized, 3);
        hashes.AddRange(SafetyTensorOps<T>.NgramHashes(normalized, 4));
        var embedding = SafetyTensorOps<T>.HashedCounts(_embeddingDim, hashes, signed: true);

        // L2 normalize
        return SafetyTensorOps<T>.NormalizeUnlessTiny(embedding).ToVector();
    }

    /// <summary>
    /// Computes a concept embedding by averaging embeddings of representative phrases.
    /// </summary>
    private Vector<T> ComputeConceptEmbedding(string[] phrases)
    {
        if (phrases.Length == 0) return new Vector<T>(_embeddingDim);

        var phraseEmbeddings = Engine.TensorStack(
            Array.ConvertAll(phrases, phrase => Tensor<T>.FromVector(ComputeTextEmbedding(phrase))), 0);
        var centroid = Engine.ReduceSum(phraseEmbeddings, new[] { 0 }, false);
        return SafetyTensorOps<T>.NormalizeUnlessTiny(centroid).ToVector();
    }

    private static List<ToxicConcept> BuildToxicConcepts()
    {
        return new List<ToxicConcept>
        {
            new(SafetyCategory.ViolenceThreat, new[]
            {
                "i will hurt you", "kill you", "make you suffer",
                "destroy you", "find where you live", "physical harm",
                "beat you up", "coming after you"
            }),
            new(SafetyCategory.HateSpeech, new[]
            {
                "all of them are", "those people are inferior",
                "should be removed from", "don't belong here",
                "subhuman", "vermin", "parasites", "infestation"
            }),
            new(SafetyCategory.Harassment, new[]
            {
                "you are worthless", "nobody likes you",
                "you should be ashamed", "ugly and stupid",
                "waste of space", "pathetic loser"
            }),
            new(SafetyCategory.ViolenceSelfHarm, new[]
            {
                "hurt yourself", "end it all", "not worth living",
                "better off without you", "no reason to go on"
            }),
            new(SafetyCategory.SocialEngineering, new[]
            {
                "send me your password", "click this link urgently",
                "verify your account immediately", "your account will be suspended",
                "wire transfer urgent", "act now or lose access"
            })
        };
    }

    private readonly struct ToxicConcept
    {
        public readonly SafetyCategory Category;
        public readonly string[] Phrases;

        public ToxicConcept(SafetyCategory category, string[] phrases)
        {
            Category = category;
            Phrases = phrases;
        }
    }
}
