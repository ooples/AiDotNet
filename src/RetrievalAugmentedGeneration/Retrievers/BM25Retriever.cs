using System.Text.RegularExpressions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Implements the BM25 (Best Matching 25) ranking algorithm for sparse retrieval.
/// </summary>
/// <typeparam name="T">The numeric data type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// BM25 is a probabilistic keyword-based retrieval algorithm that ranks documents based on
/// term frequency and inverse document frequency. It's particularly effective at finding
/// documents that contain specific keywords and often outperforms simple TF-IDF for retrieval tasks.
/// </para>
/// <para><b>For Beginners:</b> BM25 finds documents containing your search words.
/// 
/// Think of it like a smart keyword search:
/// - Looks for documents that contain your search terms
/// - Gives higher scores to rare words (more discriminative)
/// - Adjusts for document length (longer docs aren't unfairly penalized)
/// - Considers how often a word appears (but with diminishing returns)
/// 
/// How it works:
/// 1. Break query into words: "machine learning" → ["machine", "learning"]
/// 2. For each document, calculate a score based on:
///    - How many times each query word appears
///    - How rare each word is across all documents
///    - The document's length
/// 3. Return top-scoring documents
/// 
/// Real-world example:
/// - Query: "neural network training"
/// - Doc A: Contains "neural" 5 times, "network" 3 times, "training" 2 times
/// - Doc B: Contains "neural" 1 time, "network" 1 time (no "training")
/// - BM25 Score: Doc A gets higher score because it has all terms and more occurrences
/// 
/// When to use BM25:
/// - When you need exact keyword matching
/// - When you know specific terms users will search for
/// - As part of a hybrid search (combine with vector search for best results)
/// - When you need explainable results (you can see which keywords matched)
/// 
/// Why BM25 is powerful:
/// - Fast: No neural networks, just math
/// - Effective: Often beats simple TF-IDF
/// - Interpretable: You can see why a document was retrieved
/// - Language-agnostic: Works with any language (with proper tokenization)
/// </para>
/// </remarks>
public class BM25Retriever<T> : RetrieverBase<T>
{
    private readonly double _k1;
    private readonly double _b;
    private readonly Dictionary<string, Document<T>> _documents;
    private readonly Dictionary<string, Dictionary<string, int>> _termFrequencies;
    private readonly Dictionary<string, int> _documentFrequencies;
    private readonly Dictionary<string, int> _documentLengths;
    private double _averageDocumentLength;

    /// <summary>
    /// Initializes a new instance of the BM25Retriever class.
    /// </summary>
    /// <param name="k1">Controls term frequency saturation (typical range: 1.2-2.0, default: 1.5).</param>
    /// <param name="b">Controls document length normalization (0-1, default: 0.75).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The parameters k1 and b control how BM25 behaves:
    /// 
    /// **k1 (Term Saturation):**
    /// - Higher k1 (e.g., 2.0): Cares more about term frequency
    /// - Lower k1 (e.g., 1.2): Less importance on multiple occurrences
    /// - Default 1.5 works well for most use cases
    /// 
    /// **b (Length Normalization):**
    /// - b = 1.0: Full length penalty (shorter docs favored)
    /// - b = 0.0: No length penalty
    /// - Default 0.75 provides balanced normalization
    /// 
    /// Start with defaults and tune if needed!
    /// </para>
    /// </remarks>
    public BM25Retriever(double k1 = 1.5, double b = 0.75)
    {
        if (k1 <= 0)
            throw new ArgumentException("k1 must be greater than zero", nameof(k1));
        
        if (b < 0 || b > 1)
            throw new ArgumentException("b must be between 0 and 1", nameof(b));

        _k1 = k1;
        _b = b;
        _documents = new Dictionary<string, Document<T>>();
        _termFrequencies = new Dictionary<string, Dictionary<string, int>>();
        _documentFrequencies = new Dictionary<string, int>();
        _documentLengths = new Dictionary<string, int>();
        _averageDocumentLength = 0;
    }

    /// <summary>
    /// Indexes documents for BM25 retrieval.
    /// </summary>
    /// <param name="documents">The documents to index.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This must be called before retrieving documents.
    /// 
    /// What it does:
    /// 1. Tokenizes each document (splits into words)
    /// 2. Counts how often each term appears in each document
    /// 3. Counts in how many documents each term appears
    /// 4. Calculates statistics needed for BM25 scoring
    /// 
    /// This is a one-time operation per document set. Call it whenever
    /// your documents change.
    /// </para>
    /// </remarks>
    public void IndexDocuments(IEnumerable<Document<T>> documents)
    {
        if (documents == null)
            throw new ArgumentNullException(nameof(documents));

        var docList = documents.ToList();
        if (docList.Count == 0)
            return;

        _termFrequencies.Clear();
        _documentFrequencies.Clear();
        _documentLengths.Clear();
        _documents.Clear();

        int totalLength = 0;

        foreach (var doc in docList)
        {
            _documents[doc.Id] = doc;
            var tokens = Tokenize(doc.Content);
            var termFreq = new Dictionary<string, int>();

            foreach (var token in tokens)
            {
                if (!termFreq.ContainsKey(token))
                {
                    termFreq[token] = 0;
                    
                    if (!_documentFrequencies.ContainsKey(token))
                        _documentFrequencies[token] = 0;
                    
                    _documentFrequencies[token]++;
                }
                termFreq[token]++;
            }

            _termFrequencies[doc.Id] = termFreq;
            _documentLengths[doc.Id] = tokens.Count;
            totalLength += tokens.Count;
        }

        _averageDocumentLength = totalLength / (double)docList.Count;
    }

    /// <summary>
    /// Core retrieval logic that finds top-K documents using BM25 scoring.
    /// </summary>
    protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
    {
        if (_termFrequencies.Count == 0)
            throw new InvalidOperationException("No documents indexed. Call IndexDocuments first.");

        var queryTokens = Tokenize(query);
        var scores = new Dictionary<string, T>();

        foreach (var docId in _termFrequencies.Keys)
        {
            var score = CalculateBM25Score(queryTokens, docId);
            scores[docId] = score;
        }

        var topDocIds = scores
            .OrderByDescending(kvp => Convert.ToDouble(kvp.Value))
            .Take(topK)
            .Select(kvp => kvp.Key)
            .ToList();

        var results = new List<Document<T>>();
        foreach (var docId in topDocIds)
        {
            if (_documents.ContainsKey(docId))
            {
                var doc = _documents[docId];
                
                // Create a new document with the score
                var scoredDoc = new Document<T>(doc.Id, doc.Content, doc.Metadata)
                {
                    RelevanceScore = scores[docId],
                    HasRelevanceScore = true
                };
                
                // Apply metadata filters if any
                if (ApplyMetadataFilters(scoredDoc, metadataFilters))
                {
                    results.Add(scoredDoc);
                }
            }
        }

        return results;
    }

    /// <summary>
    /// Calculates the BM25 score for a document given a query.
    /// </summary>
    private T CalculateBM25Score(List<string> queryTokens, string docId)
    {
        var score = NumOps.Zero;
        var termFreq = _termFrequencies[docId];
        var docLength = _documentLengths[docId];
        var totalDocs = _termFrequencies.Count;

        foreach (var term in queryTokens.Distinct())
        {
            if (!_documentFrequencies.ContainsKey(term))
                continue;

            var df = _documentFrequencies[term];
            var tf = termFreq.ContainsKey(term) ? termFreq[term] : 0;

            // IDF calculation: log((N - df + 0.5) / (df + 0.5))
            var idf = Math.Log((totalDocs - df + 0.5) / (df + 0.5));
            
            // TF calculation with BM25 saturation
            var tfNorm = (tf * (_k1 + 1)) / 
                        (tf + _k1 * (1 - _b + _b * (docLength / _averageDocumentLength)));

            var termScore = NumOps.FromDouble(idf * tfNorm);
            score = NumOps.Add(score, termScore);
        }

        return score;
    }

    /// <summary>
    /// Applies metadata filters to a document.
    /// </summary>
    private bool ApplyMetadataFilters(Document<T> doc, Dictionary<string, object> filters)
    {
        if (filters == null || filters.Count == 0)
            return true;

        foreach (var filter in filters)
        {
            if (!doc.Metadata.ContainsKey(filter.Key))
                return false;

            if (!doc.Metadata[filter.Key].Equals(filter.Value))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Tokenizes text into lowercase terms.
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <returns>A list of tokens.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Simple tokenization:
    /// 
    /// What it does:
    /// 1. Converts to lowercase: "Machine Learning" → "machine learning"
    /// 2. Removes non-alphanumeric characters: "it's" → "its"
    /// 3. Splits on whitespace: "machine learning" → ["machine", "learning"]
    /// 4. Removes empty tokens
    /// 
    /// This is a basic implementation. For production, consider:
    /// - Stemming (running, runs → run)
    /// - Stopword removal (removing "the", "a", "is")
    /// - Language-specific tokenizers
    /// </para>
    /// </remarks>
    private List<string> Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new List<string>();

        var cleaned = Regex.Replace(text.ToLowerInvariant(), @"[^a-z0-9\s]", " ");
        return cleaned.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).ToList();
    }
}
