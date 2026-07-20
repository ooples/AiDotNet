using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// An agentic multi-hop retriever that wraps a base <see cref="IRetriever{T}"/> and an
/// <see cref="IGenerator{T}"/> to answer questions requiring several rounds of evidence gathering.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// For each hop (up to <c>maxHops</c>) the retriever fetches documents for the current query, then
/// asks the generator to propose a follow-up sub-query given the original question and the evidence
/// gathered so far. Newly retrieved documents are accumulated with de-duplication by document id.
/// The loop stops early as soon as a hop contributes no new documents, or when the generator returns
/// an empty / unchanged follow-up query. The final result is the merged set ordered by descending
/// relevance score.
/// </para>
/// <para>
/// Because the wrapped <see cref="IRetriever{T}"/> and <see cref="IGenerator{T}"/> contracts are
/// synchronous, the hop loop runs synchronously; <see cref="RetrieveAsync(string, int, CancellationToken)"/>
/// exposes the same behaviour with cooperative cancellation between hops.
/// </para>
/// <para>
/// <b>topK semantics:</b> <c>topK</c> controls how many documents each individual hop retrieves. The
/// returned collection is the merged, de-duplicated union across all hops and may therefore contain
/// more than <c>topK</c> documents — this is intentional for multi-hop evidence accumulation.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a researcher who does not stop at the first search.
/// It searches, reads what it found, decides what is still missing, searches again for that, and
/// keeps going for a few rounds — collecting all the useful documents along the way and never listing
/// the same one twice. It stops as soon as a new round finds nothing new.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Retriever)]
[PipelineStage(PipelineStage.Retrieval)]
public class MultiHopRetriever<T> : RetrieverBase<T>
{
    private readonly IRetriever<T> _baseRetriever;
    private readonly IGenerator<T> _generator;
    private readonly int _maxHops;

    /// <summary>
    /// Gets the maximum number of retrieval hops performed.
    /// </summary>
    public int MaxHops => _maxHops;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiHopRetriever{T}"/> class.
    /// </summary>
    /// <param name="baseRetriever">The underlying retriever used for each hop.</param>
    /// <param name="generator">The generator used to propose follow-up sub-queries.</param>
    /// <param name="maxHops">The maximum number of hops to perform (default: 3).</param>
    /// <param name="defaultTopK">The default number of documents to retrieve per hop (default: 5).</param>
    public MultiHopRetriever(
        IRetriever<T> baseRetriever,
        IGenerator<T> generator,
        int maxHops = 3,
        int defaultTopK = 5)
        : base(defaultTopK)
    {
        if (baseRetriever == null)
            throw new ArgumentNullException(nameof(baseRetriever));
        if (generator == null)
            throw new ArgumentNullException(nameof(generator));
        if (maxHops <= 0)
            throw new ArgumentException("maxHops must be greater than zero", nameof(maxHops));

        _baseRetriever = baseRetriever;
        _generator = generator;
        _maxHops = maxHops;
    }

    /// <summary>
    /// Executes the multi-hop retrieval loop for a query.
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="topK">The validated number of documents to retrieve per hop.</param>
    /// <param name="metadataFilters">The validated metadata filters applied to every hop.</param>
    /// <returns>The merged, de-duplicated, relevance-ordered set of documents across all hops.</returns>
    protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
    {
        return ExecuteHops(query, topK, metadataFilters, CancellationToken.None);
    }

    /// <summary>
    /// Asynchronously performs multi-hop retrieval using the default TopK, honoring cancellation.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="cancellationToken">A token observed between hops for cooperative cancellation.</param>
    /// <returns>The merged, de-duplicated, relevance-ordered set of documents across all hops.</returns>
    public Task<IEnumerable<Document<T>>> RetrieveAsync(string query, CancellationToken cancellationToken = default)
    {
        return RetrieveAsync(query, DefaultTopK, cancellationToken);
    }

    /// <summary>
    /// Asynchronously performs multi-hop retrieval, honoring cancellation between hops.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="topK">The number of documents to retrieve per hop.</param>
    /// <param name="cancellationToken">A token observed between hops for cooperative cancellation.</param>
    /// <returns>The merged, de-duplicated, relevance-ordered set of documents across all hops.</returns>
    public Task<IEnumerable<Document<T>>> RetrieveAsync(string query, int topK, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));
        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be greater than zero");

        cancellationToken.ThrowIfCancellationRequested();
        var results = ExecuteHops(query, topK, new Dictionary<string, object>(), cancellationToken);
        return Task.FromResult<IEnumerable<Document<T>>>(results);
    }

    private IEnumerable<Document<T>> ExecuteHops(
        string query,
        int topK,
        Dictionary<string, object> metadataFilters,
        CancellationToken cancellationToken)
    {
        var accumulated = new List<Document<T>>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var currentQuery = query;

        for (int hop = 0; hop < _maxHops; hop++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var hopResults = _baseRetriever.Retrieve(currentQuery, topK, metadataFilters);

            int addedThisHop = 0;
            foreach (var doc in hopResults)
            {
                if (seen.Add(DedupKey(doc)))
                {
                    accumulated.Add(doc);
                    addedThisHop++;
                }
            }

            // Stop early when a hop adds nothing new.
            if (addedThisHop == 0)
                break;

            // No need to plan a further hop after the last one.
            if (hop == _maxHops - 1)
                break;

            var followUp = ProposeFollowUpQuery(query, accumulated);

            // Stop if the generator cannot propose a non-empty next query. A repeated (unchanged)
            // query is allowed to proceed: it will retrieve the same documents and be caught by the
            // "adds nothing new" check on the following hop.
            if (string.IsNullOrWhiteSpace(followUp))
                break;

            currentQuery = followUp.Trim();
        }

        return accumulated
            .OrderByDescending(d => d.HasRelevanceScore ? Convert.ToDouble(d.RelevanceScore) : double.NegativeInfinity)
            .ToList();
    }

    /// <summary>
    /// Returns the merged multi-hop set without truncating to <paramref name="topK"/>.
    /// </summary>
    /// <param name="results">The already merged and ordered results.</param>
    /// <param name="topK">The per-hop retrieval size (not used to truncate the merged union).</param>
    /// <returns>The merged results, materialized.</returns>
    protected override IEnumerable<Document<T>> PostProcessResults(IEnumerable<Document<T>> results, int topK)
    {
        // Multi-hop deliberately returns the full de-duplicated union across hops, which can exceed
        // the per-hop topK. Do not truncate here.
        return results as IList<Document<T>> ?? results.ToList();
    }

    private static string DedupKey(Document<T> doc)
    {
        return !string.IsNullOrEmpty(doc.Id) ? "id:" + doc.Id : "content:" + (doc.Content ?? string.Empty);
    }

    private string ProposeFollowUpQuery(string originalQuestion, IReadOnlyList<Document<T>> evidence)
    {
        var prompt = new StringBuilder();
        prompt.AppendLine("You are performing multi-hop retrieval. Given the original question and the");
        prompt.AppendLine("evidence gathered so far, propose the single best follow-up search query for");
        prompt.AppendLine("the next step to fill remaining information gaps. Return only the query text.");
        prompt.AppendLine();
        prompt.AppendLine("Original Query: " + originalQuestion);
        prompt.AppendLine();
        prompt.AppendLine("Evidence so far:");

        int shown = 0;
        foreach (var doc in evidence)
        {
            if (shown >= 5)
                break;

            var content = doc.Content ?? string.Empty;
            if (content.Length > 200)
                content = content.Substring(0, 200) + "...";

            prompt.AppendLine("- " + content);
            shown++;
        }

        prompt.AppendLine();
        prompt.Append("What should the next step search for?");

        return _generator.Generate(prompt.ToString()) ?? string.Empty;
    }
}
