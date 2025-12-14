using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.FewShot;

/// <summary>
/// Selects examples based on semantic similarity to the query.
/// </summary>
/// <typeparam name="T">The type of numeric data used for similarity scoring.</typeparam>
/// <remarks>
/// <para>
/// This selector uses embedding vectors to find examples that are semantically similar to the query.
/// It converts both the query and examples into vector representations and selects those with the
/// highest cosine similarity.
/// </para>
/// <para><b>For Beginners:</b> Picks examples most similar in meaning to your query.
///
/// How it works:
/// 1. Convert query and examples to mathematical vectors (embeddings)
/// 2. Calculate similarity between query and each example
/// 3. Return the most similar examples
///
/// Example:
/// <code>
/// var selector = new SemanticSimilarityExampleSelector&lt;double&gt;(embeddingFunction);
///
/// selector.AddExample(new FewShotExample { Input = "How do I sort a list?", Output = "Use list.sort()" });
/// selector.AddExample(new FewShotExample { Input = "What is the weather today?", Output = "Check forecast" });
/// selector.AddExample(new FewShotExample { Input = "How to reverse a list?", Output = "Use list.reverse()" });
///
/// // Query about lists will return list-related examples
/// var examples = selector.SelectExamples("How to filter a list?", 2);
/// // Returns the sorting and reversing examples (both about lists)
/// </code>
///
/// Use this when:
/// - Query types vary significantly
/// - Relevant examples improve performance
/// - You have an embedding model available
/// </para>
/// </remarks>
public class SemanticSimilarityExampleSelector<T> : FewShotExampleSelectorBase<T>
{
    private readonly Func<string, double[]> _embeddingFunction;
    private readonly Dictionary<FewShotExample, double[]> _exampleEmbeddings;

    /// <summary>
    /// Initializes a new instance of the SemanticSimilarityExampleSelector class.
    /// </summary>
    /// <param name="embeddingFunction">Function to convert text to embedding vectors.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The embedding function converts text to numbers.
    ///
    /// Example embedding functions:
    /// - OpenAI text-embedding-ada-002
    /// - Sentence Transformers
    /// - Local embedding models
    ///
    /// The function takes a string and returns an array of numbers (the embedding).
    /// </para>
    /// </remarks>
    public SemanticSimilarityExampleSelector(Func<string, double[]> embeddingFunction)
    {
        _embeddingFunction = embeddingFunction ?? throw new ArgumentNullException(nameof(embeddingFunction));
        _exampleEmbeddings = new Dictionary<FewShotExample, double[]>();
    }

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
    /// Selects the most semantically similar examples.
    /// </summary>
    protected override IReadOnlyList<FewShotExample> SelectExamplesCore(string query, int count)
    {
        var queryEmbedding = _embeddingFunction(query);

        var scoredExamples = Examples
            .Select(example => new
            {
                Example = example,
                Score = CosineSimilarity(queryEmbedding, _exampleEmbeddings[example])
            })
            .OrderByDescending(x => x.Score)
            .Take(count)
            .Select(x => x.Example)
            .ToList();

        return scoredExamples.AsReadOnly();
    }

    /// <summary>
    /// Calculates cosine similarity between two vectors.
    /// </summary>
    private static double CosineSimilarity(double[] a, double[] b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must have the same length.");
        }

        double dotProduct = 0;
        double magnitudeA = 0;
        double magnitudeB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }

        double magnitude = Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB);
        return magnitude > 0 ? dotProduct / magnitude : 0;
    }
}
