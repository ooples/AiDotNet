using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// Verified reasoning retriever that validates each reasoning step with critic models.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This advanced retrieval pattern adds verification and self-refinement to chain-of-thought
/// reasoning. Each reasoning step is evaluated by a critic model to ensure it's well-supported
/// by retrieved evidence. If a step is found to be weak or unsupported, the system can
/// refine it or generate alternative reasoning paths.
/// </para>
/// <para><b>For Beginners:</b> Think of this like having a fact-checker review each step
/// of your reasoning.
///
/// Regular Chain-of-Thought:
/// - Generate reasoning steps
/// - Retrieve documents
/// - Return results
///
/// Verified Reasoning:
/// - Generate a reasoning step
/// - Retrieve supporting documents
/// - Ask a critic: "Is this step well-supported by the documents?"
/// - If not, refine the step or try a different approach
/// - Continue only with verified steps
///
/// This is useful when:
/// - Accuracy is critical (medical, legal, scientific domains)
/// - You want to avoid hallucinations or unsupported claims
/// - You need transparent, verifiable reasoning chains
/// </para>
/// </remarks>
public class VerifiedReasoningRetriever<T> : RetrieverBase<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IGenerator<T> _generator;
    private readonly RetrieverBase<T> _baseRetriever;
    private readonly double _verificationThreshold;
    private readonly int _maxRefinementAttempts;

    /// <summary>
    /// Initializes a new instance of the <see cref="VerifiedReasoningRetriever{T}"/> class.
    /// </summary>
    /// <param name="generator">The LLM generator for reasoning and critique.</param>
    /// <param name="baseRetriever">The underlying retriever to use for document retrieval.</param>
    /// <param name="verificationThreshold">Minimum verification score to accept a step (0-1, default: 0.7).</param>
    /// <param name="maxRefinementAttempts">Maximum attempts to refine a weak step (default: 2).</param>
    /// <param name="defaultTopK">Default number of documents to retrieve (default: 5).</param>
    public VerifiedReasoningRetriever(
        IGenerator<T> generator,
        RetrieverBase<T> baseRetriever,
        double verificationThreshold = 0.7,
        int maxRefinementAttempts = 2,
        int defaultTopK = 5) : base(defaultTopK)
    {
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _baseRetriever = baseRetriever ?? throw new ArgumentNullException(nameof(baseRetriever));

        if (verificationThreshold < 0 || verificationThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(verificationThreshold), "Threshold must be between 0 and 1");

        if (maxRefinementAttempts < 0)
            throw new ArgumentOutOfRangeException(nameof(maxRefinementAttempts), "Max refinement attempts must be non-negative");

        _verificationThreshold = verificationThreshold;
        _maxRefinementAttempts = maxRefinementAttempts;
    }

    /// <summary>
    /// Core retrieval logic using verified reasoning.
    /// </summary>
    protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
    {
        var result = RetrieveWithVerification(query, topK, metadataFilters);
        return result.Documents;
    }

    /// <summary>
    /// Retrieves documents using verified reasoning with critic feedback.
    /// Returns detailed verification results including reasoning steps and scores.
    /// </summary>
    /// <param name="query">The query to retrieve documents for.</param>
    /// <param name="topK">Maximum number of documents to return.</param>
    /// <param name="metadataFilters">Metadata filters to apply during retrieval.</param>
    /// <returns>Verified reasoning result with documents and verification details.</returns>
    public VerifiedReasoningResult<T> RetrieveWithVerification(
        string query,
        int topK,
        Dictionary<string, object>? metadataFilters = null)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK < 1)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        metadataFilters ??= new Dictionary<string, object>();

        // Step 1: Generate initial reasoning chain
        var reasoningSteps = GenerateReasoningChain(query);

        // Step 2: Verify and refine each step
        var verifiedSteps = new List<VerifiedReasoningStep<T>>();
        int refinedCount = 0;

        foreach (var stepText in reasoningSteps)
        {
            var verifiedStep = VerifyAndRefineStep(stepText, query, metadataFilters);
            verifiedSteps.Add(verifiedStep);

            if (verifiedStep.RefinementAttempts > 0)
                refinedCount++;
        }

        // Step 3: Collect all documents from verified steps
        var allDocuments = new Dictionary<string, Document<T>>();
        foreach (var step in verifiedSteps.Where(s => s.IsVerified))
        {
            foreach (var doc in step.SupportingDocuments.Where(d => !allDocuments.ContainsKey(d.Id)))
            {
                allDocuments[doc.Id] = doc;
            }
        }

        // Step 4: Return results
        var documents = allDocuments.Values
            .OrderByDescending(d => d.HasRelevanceScore ? d.RelevanceScore : default(T))
            .Take(topK);

        return new VerifiedReasoningResult<T>
        {
            Documents = documents,
            VerifiedSteps = verifiedSteps.AsReadOnly(),
            AverageVerificationScore = verifiedSteps.Count > 0
                ? verifiedSteps.Average(s => s.VerificationScore)
                : 0.0,
            RefinedStepsCount = refinedCount
        };
    }

    /// <summary>
    /// Generates initial reasoning chain for the query.
    /// </summary>
    private List<string> GenerateReasoningChain(string query)
    {
        var prompt = $@"Question: {query}

Please break this question into a step-by-step reasoning chain.
Each step should be a specific, verifiable claim or sub-question.

Format your response as:
Step 1: [reasoning statement]
Step 2: [reasoning statement]
Step 3: [reasoning statement]";

        var response = _generator.Generate(prompt);
        return ParseReasoningSteps(response);
    }

    /// <summary>
    /// Verifies a reasoning step and refines it if necessary.
    /// </summary>
    private VerifiedReasoningStep<T> VerifyAndRefineStep(
        string statement,
        string originalQuery,
        Dictionary<string, object> metadataFilters)
    {
        var step = new VerifiedReasoningStep<T>
        {
            Statement = statement,
            OriginalStatement = statement
        };

        for (int attempt = 0; attempt <= _maxRefinementAttempts; attempt++)
        {
            // Retrieve supporting documents
            step.SupportingDocuments = _baseRetriever
                .Retrieve(step.Statement, topK: 5, metadataFilters)
                .ToList();

            // Verify the step with critic model
            var verification = VerifyStep(step.Statement, step.SupportingDocuments, originalQuery);
            step.VerificationScore = verification.score;
            step.CritiqueFeedback = verification.feedback;
            step.IsVerified = verification.score >= _verificationThreshold;

            if (step.IsVerified || attempt >= _maxRefinementAttempts)
            {
                step.RefinementAttempts = attempt;
                break;
            }

            // Refine the step based on critic feedback
            step.Statement = RefineStep(step.Statement, verification.feedback, originalQuery);
        }

        return step;
    }

    /// <summary>
    /// Verifies a reasoning step using a critic model.
    /// </summary>
    private (double score, string feedback) VerifyStep(
        string statement,
        List<Document<T>> supportingDocs,
        string originalQuery)
    {
        var docsContext = supportingDocs.Count > 0
            ? string.Join("\n\n", supportingDocs.Select((d, i) => $"[{i + 1}] {d.Content.Substring(0, Math.Min(200, d.Content.Length))}..."))
            : "No supporting documents found.";

        var criticPrompt = $@"You are a critical evaluator of reasoning steps.

Original Question: {originalQuery}

Reasoning Step to Evaluate: {statement}

Supporting Documents:
{docsContext}

Evaluate this reasoning step on the following criteria:
1. Relevance: Is it relevant to answering the original question?
2. Support: Is it well-supported by the provided documents?
3. Clarity: Is it clear and specific?
4. Validity: Is it logically sound?

Provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect)
2. Brief feedback explaining the score

Format your response as:
Score: [0.0-1.0]
Feedback: [your critique]";

        var response = _generator.Generate(criticPrompt);
        return ParseCriticResponse(response);
    }

    /// <summary>
    /// Refines a reasoning step based on critic feedback.
    /// </summary>
    private string RefineStep(string statement, string feedback, string originalQuery)
    {
        var refinementPrompt = $@"Original Question: {originalQuery}

Current Reasoning Step: {statement}

Critic Feedback: {feedback}

Please refine the reasoning step to address the feedback.
Make it more specific, better supported, and more directly relevant to answering the question.

Provide only the refined reasoning step:";

        var response = _generator.Generate(refinementPrompt);
        return response.Trim();
    }

    /// <summary>
    /// Parses reasoning steps from LLM response.
    /// </summary>
    private List<string> ParseReasoningSteps(string response)
    {
        var steps = new List<string>();
        var lines = response.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var line in lines)
        {
            var trimmed = line.Trim();

            // Match "Step N:" format
            var match = System.Text.RegularExpressions.Regex.Match(
                trimmed,
                @"^Step\s+\d+\s*:\s*(.+)$",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase,
                RegexTimeout
            );

            if (match.Success && match.Groups[1].Value.Length > 10)
            {
                steps.Add(match.Groups[1].Value.Trim());
            }
            // Also match numbered lists
            else
            {
                match = System.Text.RegularExpressions.Regex.Match(
                    trimmed,
                    @"^\d+[\.\)]\s*(.+)$",
                    System.Text.RegularExpressions.RegexOptions.None,
                    RegexTimeout
                );

                if (match.Success && match.Groups[1].Value.Length > 10)
                {
                    steps.Add(match.Groups[1].Value.Trim());
                }
            }
        }

        // If no steps found, treat the whole response as one step
        if (steps.Count == 0 && !string.IsNullOrWhiteSpace(response))
        {
            steps.Add(response.Trim());
        }

        return steps;
    }

    /// <summary>
    /// Parses critic response to extract score and feedback.
    /// </summary>
    private (double score, string feedback) ParseCriticResponse(string response)
    {
        double score = 0.5; // Default middle score
        string feedback = response;

        // Try to extract score
        var scoreMatch = System.Text.RegularExpressions.Regex.Match(
            response,
            @"Score\s*:\s*([0-9]*\.?[0-9]+)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase,
            RegexTimeout
        );

        if (scoreMatch.Success && double.TryParse(scoreMatch.Groups[1].Value, out double parsedScore))
        {
            score = Math.Max(0.0, Math.Min(1.0, parsedScore)); // Clamp to [0, 1]
        }

        // Try to extract feedback
        var feedbackMatch = System.Text.RegularExpressions.Regex.Match(
            response,
            @"Feedback\s*:\s*(.+)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase | System.Text.RegularExpressions.RegexOptions.Singleline,
            RegexTimeout
        );

        if (feedbackMatch.Success)
        {
            feedback = feedbackMatch.Groups[1].Value.Trim();
        }

        return (score, feedback);
    }
}
