
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// FLARE (Forward-Looking Active REtrieval) pattern that actively decides when and what to retrieve during generation.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FLARE (Forward-Looking Active REtrieval augmented generation) is an advanced RAG pattern that monitors
/// the language model's confidence during generation. When uncertainty is detected (low confidence on
/// next tokens), FLARE automatically retrieves additional relevant information to improve answer quality.
/// This creates a dynamic retrieval loop where retrieval happens only when needed, rather than all upfront.
/// </para>
/// <para><b>For Beginners:</b> Think of FLARE like asking follow-up questions when you're unsure.
/// 
/// Normal RAG:
/// - Question: "What is quantum computing?"
/// - Step 1: Retrieve all documents about quantum computing
/// - Step 2: Generate complete answer from those documents
/// - Problem: Might miss specific details or retrieve too much irrelevant info
/// 
/// FLARE:
/// - Question: "What is quantum computing?"
/// - Step 1: Start generating answer...
/// - Step 2: "Quantum computing uses quantum bits or..." (confident, keep going)
/// - Step 3: "...which leverage principles like..." (uncertain - what principles exactly?)
/// - Step 4: RETRIEVE more docs about "quantum principles superposition entanglement"
/// - Step 5: Continue with new information: "...superposition and entanglement..."
/// - Result: More focused retrieval, better coverage of uncertainty areas
/// 
/// It's like having a conversation where you ask for clarification only when you need it,
/// rather than reading an entire encyclopedia upfront.
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Setup
/// var generator = new StubGenerator&lt;double&gt;(); // Or real LLM
/// var retriever = new DenseRetriever&lt;double&gt;(embeddingModel, documentStore);
/// 
/// // Create FLARE retriever with uncertainty threshold
/// var flare = new FLARERetriever&lt;double&gt;(
///     generator,
///     retriever,
///     uncertaintyThreshold: 0.5  // Retrieve when confidence drops below 50%
/// );
/// 
/// // Generate answer with active retrieval
/// var answer = flare.GenerateWithActiveRetrieval(
///     "Explain how CRISPR gene editing works and its applications"
/// );
/// 
/// // FLARE will:
/// // 1. Start generating about CRISPR
/// // 2. Detect uncertainty about specific mechanisms
/// // 3. Retrieve more docs about "CRISPR Cas9 mechanism"
/// // 4. Continue generating with new info
/// // 5. Detect uncertainty about applications
/// // 6. Retrieve docs about "CRISPR medical applications"
/// // 7. Complete answer with all retrieved knowledge
/// </code>
/// </para>
/// <para><b>How It Works:</b>
/// The retrieval process is:
/// 1. Initial retrieval - Get top-3 relevant documents
/// 2. Start generating answer with initial context
/// 3. Monitor confidence - Check for uncertainty signals (keywords like "I'm not sure", "unclear")
/// 4. Active retrieval - When uncertain, extract missing topics and retrieve more docs
/// 5. Integrate new information - Continue generating with expanded context
/// 6. Repeat - Maximum 5 iterations to prevent infinite loops
/// 7. Return complete answer assembled from all iterations
/// 
/// Current implementation uses keyword detection for uncertainty. Production version would use:
/// - Token-level confidence scores (logprobs from LLM)
/// - Attention weights to identify knowledge gaps
/// - Explicit uncertainty statements from the model
/// </para>
/// <para><b>Benefits:</b>
/// - More efficient retrieval - Only fetches what's needed
/// - Better coverage - Addresses uncertainty areas specifically
/// - Reduced noise - Avoids retrieving irrelevant documents upfront
/// - Adaptive - Responds to complexity of the question dynamically
/// - Cost-effective - Fewer total documents retrieved vs exhaustive upfront retrieval
/// </para>
/// <para><b>Limitations:</b>
/// - Requires LLM with confidence scores (logprobs) for best results
/// - Multiple LLM calls increase latency
/// - May miss information if uncertainty detection fails
/// - Current implementation uses simple keyword matching (needs improvement with real LLM logprobs)
/// </para>
/// </remarks>
public class FLARERetriever<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IGenerator<T> _generator;
    private readonly RetrieverBase<T> _baseRetriever;
    private readonly double _uncertaintyThreshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="FLARERetriever{T}"/> class.
    /// </summary>
    /// <param name="generator">The LLM generator (use StubGenerator or real LLM).</param>
    /// <param name="baseRetriever">The underlying retriever to use.</param>
    /// <param name="uncertaintyThreshold">Threshold for triggering retrieval (0.0-1.0, default 0.5).</param>
    public FLARERetriever(
        IGenerator<T> generator,
        RetrieverBase<T> baseRetriever,
        double uncertaintyThreshold = 0.5)
    {
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _baseRetriever = baseRetriever ?? throw new ArgumentNullException(nameof(baseRetriever));

        if (uncertaintyThreshold < 0.0 || uncertaintyThreshold > 1.0)
            throw new ArgumentOutOfRangeException(nameof(uncertaintyThreshold), "Threshold must be between 0 and 1");

        _uncertaintyThreshold = uncertaintyThreshold;
    }

    /// <summary>
    /// Generates an answer with active retrieval triggered by detected uncertainty.
    /// </summary>
    /// <param name="query">The user's question to answer.</param>
    /// <returns>Complete answer generated iteratively with active retrieval when needed.</returns>
    /// <exception cref="ArgumentException">Thrown when query is null or whitespace.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the FLARE algorithm: it generates an answer incrementally,
    /// monitoring for signs of uncertainty at each step. When uncertainty exceeds the threshold,
    /// it retrieves additional relevant documents and continues generation with the new context.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main method that implements "generate with retrieval on demand".
    /// 
    /// Process:
    /// 1. Initial Setup: Retrieve top-3 documents for the query
    /// 2. Generate Partial Answer: Create answer segment with current context
    /// 3. Check Confidence: Look for uncertainty signals ("I'm not sure", "unclear", etc.)
    /// 4. If Uncertain:
    ///    - Extract what information is missing
    ///    - Retrieve documents about that specific topic
    ///    - Add to context
    /// 5. Repeat: Up to 5 times or until confident
    /// 6. Return: Complete answer assembled from all iterations
    /// 
    /// Example Flow:
    /// - Query: "How does photosynthesis work?"
    /// - Iteration 1: "Photosynthesis is..." (confident)
    /// - Iteration 2: "...but the exact chemical process of..." (UNCERTAIN!)
    /// - Retrieval: Get docs about "photosynthesis chemical reactions"
    /// - Iteration 3: "...the light-dependent reactions convert..." (confident with new info)
    /// - Done: Return complete answer
    /// </para>
    /// </remarks>
    public string GenerateWithActiveRetrieval(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        var answer = new StringBuilder();
        var currentQuery = query;
        var maxIterations = 5; // Prevent infinite loops
        var iteration = 0;

        // Initial retrieval
        var initialDocs = _baseRetriever.Retrieve(query, topK: 3).ToList();
        var allRetrievedDocs = new List<Document<T>>(initialDocs);

        while (iteration < maxIterations)
        {
            iteration++;

            // Generate partial answer
            var partialPrompt = $@"Query: {currentQuery}

Context: {string.Join("\n\n", allRetrievedDocs.Select(d => d.Content.Substring(0, Math.Min(200, d.Content.Length))))}

Please provide a partial answer. If you need more information, indicate what is missing.";

            var partialAnswer = _generator.Generate(partialPrompt);
            answer.AppendLine(partialAnswer);

            // Detect if more information is needed (simplified confidence check)
            var confidence = CalculateConfidence(partialAnswer, allRetrievedDocs);

            if (confidence >= _uncertaintyThreshold)
            {
                // Confident enough, stop
                break;
            }

            // Extract what information is needed
            var missingInfo = ExtractMissingInformation(partialAnswer);
            if (string.IsNullOrEmpty(missingInfo))
            {
                // No clear indication of missing info, stop
                break;
            }

            // Retrieve additional documents
            var additionalDocs = _baseRetriever.Retrieve(missingInfo, topK: 2).ToList();

            if (additionalDocs.Count == 0)
            {
                // No more documents found, stop
                break;
            }

            // Add to retrieved documents
            foreach (var doc in additionalDocs)
            {
                if (!allRetrievedDocs.Any(d => d.Id == doc.Id))
                {
                    allRetrievedDocs.Add(doc);
                }
            }

            // Update current query for next iteration
            currentQuery = missingInfo;
        }

        return answer.ToString().Trim();
    }

    private double CalculateConfidence(string generatedText, List<Document<T>> retrievedDocs)
    {
        // Multi-factor confidence calculation
        // In production, this would use token-level confidence from the LLM (logprobs)

        if (string.IsNullOrWhiteSpace(generatedText))
            return 0.0;

        var lowerText = generatedText.ToLowerInvariant();

        // Uncertainty indicators (negative signals)
        var uncertaintyPhrases = new[]
        {
            "not sure", "don't know", "i'm not certain", "need more information",
            "unclear", "missing", "uncertain", "cannot determine", "insufficient",
            "would need", "it's difficult to say", "without more context",
            "i don't have enough", "unable to", "not enough information"
        };

        // Confidence indicators (positive signals)
        var confidencePhrases = new[]
        {
            "based on the context", "according to", "the documents show",
            "clearly", "specifically", "in summary", "therefore", "thus",
            "this means", "the answer is", "it is", "we can see that",
            "the evidence suggests", "as stated"
        };

        // Count occurrences of uncertainty vs confidence phrases
        var uncertaintyCount = uncertaintyPhrases.Count(phrase => lowerText.Contains(phrase));
        var confidenceCount = confidencePhrases.Count(phrase => lowerText.Contains(phrase));

        // Calculate phrase-based score (weighted toward uncertainty detection)
        double phraseScore;
        if (uncertaintyCount > 0 && confidenceCount == 0)
        {
            // Only uncertainty phrases found
            phraseScore = Math.Max(0.1, 0.4 - (uncertaintyCount * 0.1));
        }
        else if (uncertaintyCount > confidenceCount)
        {
            // More uncertainty than confidence
            phraseScore = 0.3 + (confidenceCount * 0.05);
        }
        else if (confidenceCount > 0)
        {
            // More or equal confidence phrases
            phraseScore = Math.Min(0.9, 0.5 + (confidenceCount * 0.1));
        }
        else
        {
            // Neutral - no strong indicators either way
            phraseScore = 0.5;
        }

        // Length score (normalized, with minimum threshold for meaningful content)
        // Short answers (<100 chars) get penalized, long answers (>300 chars) get full credit
        double lengthScore;
        if (generatedText.Length < 50)
        {
            lengthScore = 0.2;
        }
        else if (generatedText.Length < 100)
        {
            lengthScore = 0.4;
        }
        else if (generatedText.Length < 200)
        {
            lengthScore = 0.6;
        }
        else
        {
            lengthScore = Math.Min(1.0, 0.7 + (generatedText.Length / 1000.0));
        }

        // Document relevance score
        var relevanceScores = retrievedDocs
            .Where(d => d.HasRelevanceScore)
            .Select(d => Convert.ToDouble(d.RelevanceScore))
            .ToList();

        var avgRelevance = relevanceScores.Any() ? relevanceScores.Average() : 0.5;

        // Context coverage score - check if generated text references document content
        double coverageScore = 0.5;
        if (retrievedDocs.Count > 0)
        {
            var docKeywords = retrievedDocs
                .SelectMany(d => d.Content.Split(new[] { ' ', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries))
                .Where(w => w.Length > 4) // Only meaningful words
                .Select(w => w.ToLowerInvariant())
                .Distinct()
                .ToList();

            var generatedWords = generatedText.Split(new[] { ' ', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(w => w.Length > 4)
                .Select(w => w.ToLowerInvariant())
                .Distinct()
                .ToList();

            if (docKeywords.Count > 0 && generatedWords.Count > 0)
            {
                var overlap = generatedWords.Count(w => docKeywords.Contains(w));
                coverageScore = Math.Min(1.0, (double)overlap / Math.Min(generatedWords.Count, 20));
            }
        }

        // Weighted combination of all factors
        // Phrase score is most important (40%), then relevance (25%), coverage (20%), length (15%)
        var finalScore = (phraseScore * 0.40) +
                        (avgRelevance * 0.25) +
                        (coverageScore * 0.20) +
                        (lengthScore * 0.15);

        return Math.Max(0.0, Math.Min(1.0, finalScore));
    }

    private string ExtractMissingInformation(string generatedText)
    {
        // Look for phrases indicating missing information
        var patterns = new[]
        {
            @"need more information about (.+?)[\.\n]",
            @"unclear about (.+?)[\.\n]",
            @"missing details on (.+?)[\.\n]",
            @"don't know about (.+?)[\.\n]",
            @"requires (.+?)[\.\n]"
        };

        foreach (var pattern in patterns)
        {
            var match = System.Text.RegularExpressions.Regex.Match(
                generatedText,
                pattern,
                System.Text.RegularExpressions.RegexOptions.IgnoreCase,
                RegexTimeout);

            if (match.Success)
            {
                return match.Groups[1].Value.Trim();
            }
        }

        return string.Empty;
    }
}
