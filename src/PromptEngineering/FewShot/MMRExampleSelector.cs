using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.PromptEngineering.FewShot;

/// <summary>
/// Selects examples using Maximum Marginal Relevance (MMR) to balance relevance and diversity.
/// </summary>
/// <typeparam name="T">The type of numeric data used for similarity scoring.</typeparam>
/// <remarks>
/// <para>
/// MMR combines relevance (similarity to query) and diversity (difference from already-selected examples).
/// It iteratively selects examples that maximize: lambda * relevance - (1 - lambda) * max_similarity_to_selected
/// </para>
/// <para><b>For Beginners:</b> Picks examples that are both relevant AND different from each other.
///
/// Think of it like building a playlist:
/// - Pure relevance: All songs sound almost identical (boring!)
/// - Pure diversity: Random songs that don't fit together (confusing!)
/// - MMR: Similar style, but each song brings something different (perfect!)
///
/// Example:
/// <code>
/// var selector = new MMRExampleSelector&lt;double&gt;(embeddingFunction, lambda: 0.7);
///
/// selector.AddExample(new FewShotExample { Input = "How to sort?", Output = "Use sort()" });
/// selector.AddExample(new FewShotExample { Input = "How to sort descending?", Output = "Use sort(reverse=True)" });
/// selector.AddExample(new FewShotExample { Input = "How to filter?", Output = "Use filter()" });
/// selector.AddExample(new FewShotExample { Input = "What is Python?", Output = "A language" });
///
/// // MMR picks relevant but diverse examples
/// var examples = selector.SelectExamples("How to sort numbers?", 2);
/// // Might return "How to sort?" and "How to filter?" (relevant to lists, but different operations)
/// </code>
///
/// The lambda parameter controls the balance:
/// - lambda = 1.0: Pure relevance (like SemanticSimilarity)
/// - lambda = 0.5: Equal balance
/// - lambda = 0.0: Pure diversity (like Diversity)
/// </para>
/// </remarks>
public class MMRExampleSelector<T> : FewShotExampleSelectorBase<T>
{
    private readonly Func<string, Vector<T>> _embeddingFunction;
    private readonly Dictionary<FewShotExample, Vector<T>> _exampleEmbeddings;
    private readonly T _lambda;

    /// <summary>
    /// Initializes a new instance of the MMRExampleSelector class.
    /// </summary>
    /// <param name="embeddingModel">Embedding model used to convert text to embedding vectors.</param>
    public MMRExampleSelector(IEmbeddingModel<T> embeddingModel)
        : this(embeddingModel is null ? throw new ArgumentNullException(nameof(embeddingModel)) : embeddingModel.Embed, NumOps.FromDouble(0.7))
    {
    }

    /// <summary>
    /// Initializes a new instance of the MMRExampleSelector class.
    /// </summary>
    /// <param name="embeddingModel">Embedding model used to convert text to embedding vectors.</param>
    /// <param name="lambda">Balance between relevance (1.0) and diversity (0.0).</param>
    public MMRExampleSelector(IEmbeddingModel<T> embeddingModel, T lambda)
        : this(embeddingModel is null ? throw new ArgumentNullException(nameof(embeddingModel)) : embeddingModel.Embed, lambda)
    {
    }

    /// <summary>
    /// Initializes a new instance of the MMRExampleSelector class.
    /// </summary>
    /// <param name="embeddingFunction">Function to convert text to embedding vectors.</param>
    public MMRExampleSelector(Func<string, Vector<T>> embeddingFunction)
        : this(embeddingFunction, NumOps.FromDouble(0.7))
    {
    }

    /// <summary>
    /// Initializes a new instance of the MMRExampleSelector class.
    /// </summary>
    /// <param name="embeddingFunction">Function to convert text to embedding vectors.</param>
    /// <param name="lambda">Balance between relevance (1.0) and diversity (0.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lambda controls the relevance/diversity tradeoff.
    ///
    /// Common values:
    /// - 0.7: Favor relevance with some diversity (recommended starting point)
    /// - 0.5: Equal weight to both
    /// - 0.3: Favor diversity with some relevance
    ///
    /// Experiment to find the best value for your use case.
    /// </para>
    /// </remarks>
    public MMRExampleSelector(Func<string, Vector<T>> embeddingFunction, T lambda)
    {
        Guard.NotNull(embeddingFunction);
        _embeddingFunction = embeddingFunction;
        _lambda = ClampToUnitInterval(lambda);
        _exampleEmbeddings = new Dictionary<FewShotExample, Vector<T>>();
    }

    /// <summary>
    /// Gets the lambda value used for MMR scoring.
    /// </summary>
    public T Lambda => _lambda;

    /// <summary>
    /// Called when an example is added. Pre-computes the embedding.
    /// </summary>
    protected override void OnExampleAdded(FewShotExample example)
    {
        _exampleEmbeddings[example] = _embeddingFunction(example.Input);
    }

    /// <summary>
    /// Called when an example is removed. Removes the cached embedding.
    /// </summary>
    protected override void OnExampleRemoved(FewShotExample example)
    {
        _exampleEmbeddings.Remove(example);
    }

    /// <summary>
    /// Selects examples using the MMR algorithm.
    /// </summary>
    protected override IReadOnlyList<FewShotExample> SelectExamplesCore(string query, int count)
    {
        var queryEmbedding = _embeddingFunction(query);
        var selected = new List<FewShotExample>();
        var selectedEmbeddings = new List<Vector<T>>();
        var remaining = new HashSet<FewShotExample>(Examples);

        // Pre-calculate relevance scores
        var relevanceScores = new Dictionary<FewShotExample, T>();
        foreach (var example in Examples)
        {
            relevanceScores[example] = CosineSimilarity(queryEmbedding, _exampleEmbeddings[example]);
        }

        // Iteratively select examples using MMR
        var oneMinusLambda = NumOps.Subtract(NumOps.One, _lambda);
        while (selected.Count < count && remaining.Count > 0)
        {
            FewShotExample? bestCandidate = null;
            T bestScore = NumOps.Zero;
            bool hasBestScore = false;

            foreach (var candidate in remaining)
            {
                var candidateEmbedding = _exampleEmbeddings[candidate];
                var relevance = relevanceScores[candidate];

                // Calculate max similarity to already-selected examples
                T maxSimilarityToSelected = NumOps.Zero;
                foreach (var selectedEmbedding in selectedEmbeddings)
                {
                    var similarity = CosineSimilarity(candidateEmbedding, selectedEmbedding);
                    if (NumOps.GreaterThan(similarity, maxSimilarityToSelected))
                    {
                        maxSimilarityToSelected = similarity;
                    }
                }

                // MMR score: lambda * relevance - (1 - lambda) * maxSimilarityToSelected
                var mmrScore = NumOps.Subtract(
                    NumOps.Multiply(_lambda, relevance),
                    NumOps.Multiply(oneMinusLambda, maxSimilarityToSelected));

                if (!hasBestScore || NumOps.GreaterThan(mmrScore, bestScore))
                {
                    bestScore = mmrScore;
                    bestCandidate = candidate;
                    hasBestScore = true;
                }
            }

            if (bestCandidate is not null)
            {
                selected.Add(bestCandidate);
                selectedEmbeddings.Add(_exampleEmbeddings[bestCandidate]);
                remaining.Remove(bestCandidate);
            }
            else
            {
                break;
            }
        }

        return selected.AsReadOnly();
    }
}
