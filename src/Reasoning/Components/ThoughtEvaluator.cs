using System.Text.RegularExpressions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Reasoning.Components;

/// <summary>
/// Evaluates the quality and promise of thoughts or reasoning steps.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The thought evaluator judges how good or promising a thought is.
/// It's like a coach evaluating different strategies - which one is most likely to succeed?
///
/// The evaluator considers:
/// - Relevance to the original problem
/// - Logical soundness
/// - Progress toward a solution
/// - Feasibility and practicality
///
/// Returns a score (typically 0.0 to 1.0) where higher means more promising.
/// </para>
/// </remarks>
internal class ThoughtEvaluator<T> : IThoughtEvaluator<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IChatModel<T> _chatModel;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="ThoughtEvaluator{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used to evaluate thoughts.</param>
    public ThoughtEvaluator(IChatModel<T> chatModel)
    {
        _chatModel = chatModel ?? throw new ArgumentNullException(nameof(chatModel));
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public async Task<T> EvaluateThoughtAsync(
        AiDotNet.Reasoning.Models.ThoughtNode<T> node,
        string originalQuery,
        ReasoningConfig config,
        CancellationToken cancellationToken = default)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        if (string.IsNullOrWhiteSpace(originalQuery))
            throw new ArgumentException("Original query cannot be empty", nameof(originalQuery));

        // Build context
        var pathFromRoot = node.GetPathFromRoot();
        string reasoningPath = string.Join(" → ", pathFromRoot);

        // Create evaluation prompt
        string prompt = BuildEvaluationPrompt(node.Thought, originalQuery, reasoningPath);

        // Get evaluation from LLM
        string response = await _chatModel.GenerateResponseAsync(prompt);

        // Parse the score
        double score = ParseEvaluationScore(response);

        // Store evaluation details in metadata
        node.Metadata["evaluation_response"] = response;
        node.Metadata["evaluation_score"] = score;

        return _numOps.FromDouble(score);
    }

    /// <summary>
    /// Builds the prompt for evaluating a thought.
    /// </summary>
    private string BuildEvaluationPrompt(string thought, string originalQuery, string reasoningPath)
    {
        return $@"You are evaluating the quality and promise of a reasoning step.

Original Problem: {originalQuery}

Reasoning Path So Far:
{reasoningPath}

Current Thought to Evaluate: ""{thought}""

Evaluate this thought on a scale from 0.0 to 1.0 based on:
1. **Relevance**: Does it address the original problem?
2. **Logic**: Is it logically sound and well-reasoned?
3. **Progress**: Does it make meaningful progress toward a solution?
4. **Feasibility**: Is it practical and achievable?
5. **Correctness**: Is the reasoning accurate?

Respond in JSON format:
{{
  ""score"": 0.85,
  ""reasoning"": ""Brief explanation of the score""
}}

Evaluate the thought:";
    }

    /// <summary>
    /// Parses the evaluation score from the LLM response.
    /// </summary>
    private double ParseEvaluationScore(string response)
    {
        try
        {
            // Try JSON parsing
            string jsonContent = ExtractJsonFromResponse(response);
            var root = JObject.Parse(jsonContent);

            if (root["score"] != null)
            {
                double score = root["score"]!.Value<double>();
                return MathHelper.Clamp(score, 0.0, 1.0);
            }
        }
        catch (JsonException)
        {
            // Continue to fallback
        }

        // Fallback: look for decimal numbers in the response
        var numberMatch = Regex.Match(response, @"(?:score|rating|evaluation)[\s:]*([0-9]*\.?[0-9]+)", RegexOptions.IgnoreCase, RegexTimeout);
        if (numberMatch.Success && double.TryParse(numberMatch.Groups[1].Value, out double fallbackScore))
        {
            // Normalize if needed (might be out of 10, 5, etc.)
            if (fallbackScore > 1.0 && fallbackScore <= 10.0)
            {
                fallbackScore /= 10.0;
            }
            else if (fallbackScore > 10.0 && fallbackScore <= 100.0)
            {
                fallbackScore /= 100.0;
            }

            return MathHelper.Clamp(fallbackScore, 0.0, 1.0);
        }

        // Default to middle score if parsing fails
        return 0.5;
    }

    /// <summary>
    /// Extracts JSON content from markdown code blocks.
    /// </summary>
    private string ExtractJsonFromResponse(string response)
    {
        // Remove markdown code block markers
        var jsonMatch = Regex.Match(response, @"```(?:json)?\s*(\{[\s\S]*?\})\s*```", RegexOptions.Multiline, RegexTimeout);
        if (jsonMatch.Success)
        {
            return jsonMatch.Groups[1].Value;
        }

        // Try to find JSON object
        var jsonObjectMatch = Regex.Match(response, @"\{[\s\S]*?\}", RegexOptions.None, RegexTimeout);
        if (jsonObjectMatch.Success)
        {
            return jsonObjectMatch.Value;
        }

        return response;
    }
}
