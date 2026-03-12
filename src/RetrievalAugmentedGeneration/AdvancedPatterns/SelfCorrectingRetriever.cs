
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// Self-correcting retriever that iteratively refines answers through critique, error detection, and targeted re-retrieval.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This advanced RAG pattern implements a self-correction loop: retrieve documents, generate answer,
/// critique the answer for errors or gaps, retrieve additional targeted documents, and repeat until
/// the answer is satisfactory. This mirrors how humans refine their understanding through iteration.
/// </para>
/// <para><b>For Beginners:</b> Think of this like writing an essay with self-editing.
/// 
/// Normal approach:
/// - Research topic once → Write essay → Submit (might have errors!)
/// 
/// Self-correcting approach:
/// - Research → Write draft → Read and critique → "Wait, I'm missing data about X"
/// - Research X specifically → Add to essay → Critique again → "This part contradicts that part"
/// - Research to resolve → Fix contradiction → Final review → Submit when satisfied
/// 
/// Example:
/// Question: "What caused the fall of the Roman Empire?"
/// 
/// Iteration 1:
/// - Retrieved: General docs about Roman Empire
/// - Answer: "Economic problems and barbarian invasions caused the fall"
/// - Critique: "Too vague - which economic problems? When did invasions happen?"
/// - Satisfied: NO
/// 
/// Iteration 2:
/// - Re-retrieve: "Roman Empire economic problems inflation"
/// - Answer: "Currency debasement and inflation in the 3rd century, plus Germanic invasions in 410 AD"
/// - Critique: "Better, but missing Eastern vs Western Empire distinction"
/// - Satisfied: NO
/// 
/// Iteration 3:
/// - Re-retrieve: "Western Roman Empire Eastern Byzantine"
/// - Answer: "Western Empire fell in 476 AD due to economics + invasions; Eastern continued as Byzantine"
/// - Critique: "Complete and accurate!"
/// - Satisfied: YES → Return answer
/// </para>
/// <para><b>How It Works:</b>
/// The self-correction process:
/// 1. Initial Retrieval - Get top-K relevant documents for query
/// 2. Generate Answer - Create initial answer from retrieved documents
/// 3. Generate Critique - LLM critiques its own answer for errors/gaps
/// 4. Check Satisfaction - Parse critique for approval keywords
/// 5. If Not Satisfied:
///    a. Extract gaps - Identify what information is missing
///    b. Re-retrieve - Get documents about missing topics
///    c. Generate improved answer with all documents
///    d. Repeat critique (max 3 iterations)
/// 6. Return final answer
/// 
/// Current implementation uses keyword detection for satisfaction.
/// Production should use structured critique (JSON) with explicit quality scores.
/// </para>
/// <para><b>Benefits:</b>
/// - Higher accuracy through iterative refinement
/// - Catches and corrects initial mistakes
/// - Identifies and fills knowledge gaps automatically
/// - More comprehensive answers
/// - Transparent - shows reasoning through critique
/// </para>
/// <para><b>Limitations:</b>
/// - Multiple LLM calls (higher cost/latency)
/// - May not converge if critique is inconsistent
/// - Depends heavily on LLM's self-critique ability
/// - Limited to max iterations (prevents infinite loops)
/// </para>
/// </remarks>
public class SelfCorrectingRetriever<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IGenerator<T> _generator;
    private readonly RetrieverBase<T> _baseRetriever;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="SelfCorrectingRetriever{T}"/> class.
    /// </summary>
    /// <param name="generator">The LLM generator (use StubGenerator or real LLM).</param>
    /// <param name="baseRetriever">The underlying retriever to use.</param>
    /// <param name="maxIterations">Maximum number of correction iterations (default: 3).</param>
    public SelfCorrectingRetriever(
        IGenerator<T> generator,
        RetrieverBase<T> baseRetriever,
        int maxIterations = 3)
    {
        Guard.NotNull(generator);
        _generator = generator;
        Guard.NotNull(baseRetriever);
        _baseRetriever = baseRetriever;

        if (maxIterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxIterations), "Max iterations must be positive");

        _maxIterations = maxIterations;
    }

    /// <summary>
    /// Retrieves documents and generates a self-corrected, refined answer through iterative critique.
    /// </summary>
    /// <param name="query">The user's question requiring a high-quality answer.</param>
    /// <param name="topK">Number of documents to retrieve in each iteration.</param>
    /// <param name="metadataFilters">Optional metadata filters for document scoping (e.g., tenant ID, permissions).</param>
    /// <returns>Final refined answer after self-correction iterations.</returns>
    /// <exception cref="ArgumentException">Thrown when query is null or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when topK is not positive.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the self-correction loop: retrieve → generate → critique → refine.
    /// It continues iterating until the answer passes self-critique or max iterations is reached.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having a built-in editor that keeps improving your answer.
    /// 
    /// The process:
    /// 1. Retrieve: Get topK documents about the query
    /// 2. Generate: Create answer from those documents
    /// 3. Critique: Ask LLM "Is this answer complete and accurate?"
    /// 4. Evaluate Critique:
    ///    - If critique contains "satisfactory", "complete", "accurate" → Done!
    ///    - If critique mentions gaps/errors → Continue improving
    /// 5. Extract Gaps: What information is missing from critique?
    /// 6. Re-Retrieve: Get more documents about the gaps
    /// 7. Re-Generate: Create improved answer with ALL documents (old + new)
    /// 8. Repeat: Back to step 3 (max 3 times total)
    /// 9. Return: Best answer achieved
    /// 
    /// Safety: Maximum 3 iterations prevent infinite loops if model keeps finding issues.
    /// </para>
    /// </remarks>
    public string RetrieveAndAnswer(string query, int topK, Dictionary<string, object>? metadataFilters = null)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK < 1)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        metadataFilters ??= new Dictionary<string, object>();

        // Step 1: Initial retrieval
        var documents = _baseRetriever.Retrieve(query, topK, metadataFilters).ToList();

        if (documents.Count == 0)
        {
            return "I don't have enough information to answer this question.";
        }

        var currentAnswer = string.Empty;
        var iteration = 0;

        while (iteration < _maxIterations)
        {
            iteration++;

            // Step 2: Generate answer from current documents
            var groundedAnswer = _generator.GenerateGrounded(query, documents);
            currentAnswer = groundedAnswer.Answer;

            // Step 3: Critique the answer
            var critiquePrompt = $@"Query: {query}

Answer: {currentAnswer}

Please critique this answer:
1. Are there any factual errors?
2. Are there gaps in coverage?
3. What additional information would improve it?
4. Is it complete and accurate?

Provide a brief critique.";

            var critique = _generator.Generate(critiquePrompt);

            // Step 4: Check if answer is satisfactory
            if (IsAnswerSatisfactory(critique))
            {
                // Answer is good, stop iterating
                break;
            }

            // Step 5: Identify missing information
            var missingInfo = ExtractMissingInformation(critique);

            if (string.IsNullOrEmpty(missingInfo))
            {
                // No clear improvement path, stop
                break;
            }

            // Step 6: Retrieve additional documents
            var additionalDocs = _baseRetriever.Retrieve(missingInfo, topK: 2, metadataFilters).ToList();

            if (additionalDocs.Count == 0)
            {
                // No new documents found, stop
                break;
            }

            // Step 7: Add new documents to the set
            foreach (var doc in additionalDocs)
            {
                if (!documents.Any(d => d.Id == doc.Id))
                {
                    documents.Add(doc);
                }
            }

            // Continue to next iteration with expanded document set
        }

        return currentAnswer;
    }

    private bool IsAnswerSatisfactory(string critique)
    {
        // Check for positive indicators
        var positiveIndicators = new[]
        {
            "complete", "accurate", "comprehensive", "satisfactory",
            "no errors", "well-covered", "sufficient"
        };

        // Check for negative indicators
        var negativeIndicators = new[]
        {
            "missing", "incorrect", "incomplete", "error", "gap",
            "unclear", "needs", "lacks", "wrong"
        };

        var critiqueLower = critique.ToLower();
        var positiveCount = positiveIndicators.Count(indicator => critiqueLower.Contains(indicator));
        var negativeCount = negativeIndicators.Count(indicator => critiqueLower.Contains(indicator));

        // Satisfactory if more positive than negative indicators
        return positiveCount > negativeCount;
    }

    private string ExtractMissingInformation(string critique)
    {
        // Look for phrases indicating missing information
        var patterns = new[]
        {
            @"missing (.+?)[\.\n]",
            @"needs (.+?)[\.\n]",
            @"should include (.+?)[\.\n]",
            @"lacks (.+?)[\.\n]",
            @"gap in (.+?)[\.\n]",
            @"additional information about (.+?)[\.\n]"
        };

        foreach (var pattern in patterns)
        {
            var match = System.Text.RegularExpressions.Regex.Match(
                critique,
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
