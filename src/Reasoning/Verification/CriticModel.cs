using System.Text.RegularExpressions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Reasoning.Verification;

/// <summary>
/// Implements a critic model that evaluates and provides feedback on reasoning quality.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A critic model acts like a teacher or peer reviewer, examining
/// reasoning steps and providing constructive feedback. This is a key component of verified
/// reasoning systems like DeepSeek-R1.
///
/// **What it does:**
/// - Evaluates each reasoning step for correctness and quality
/// - Provides a score (0.0 to 1.0) indicating quality
/// - Offers specific feedback on strengths and weaknesses
/// - Suggests improvements
///
/// **Example:**
/// Step: "15% of 240 is 36 because 240 × 0.15 = 36"
///
/// Critique:
/// - Score: 0.95 (excellent)
/// - Strengths: ["Correct conversion of percentage", "Accurate calculation", "Clear explanation"]
/// - Weaknesses: ["Could show intermediate steps more clearly"]
/// - Suggestions: ["Show: 15/100 = 0.15, then 0.15 × 240 = 36"]
///
/// This enables self-refinement loops where weak steps are improved based on feedback.
/// </para>
/// </remarks>
internal class CriticModel<T> : ICriticModel<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IChatModel<T> _chatModel;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="CriticModel{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for critique generation.</param>
    public CriticModel(IChatModel<T> chatModel)
    {
        _chatModel = chatModel ?? throw new ArgumentNullException(nameof(chatModel));
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public async Task<CritiqueResult<T>> CritiqueStepAsync(
        ReasoningStep<T> step,
        ReasoningContext context,
        CancellationToken cancellationToken = default)
    {
        if (step == null)
            throw new ArgumentNullException(nameof(step));

        if (context == null)
            throw new ArgumentNullException(nameof(context));

        // Build critique prompt
        string prompt = BuildCritiquePrompt(step, context);

        // Get critique from LLM
        string response = await _chatModel.GenerateResponseAsync(prompt);

        // Parse the critique
        return ParseCritique(response);
    }

    /// <inheritdoc/>
    public async Task<CritiqueResult<T>> CritiqueChainAsync(
        ReasoningChain<T> chain,
        CancellationToken cancellationToken = default)
    {
        if (chain == null)
            throw new ArgumentNullException(nameof(chain));

        // Build chain critique prompt
        string prompt = BuildChainCritiquePrompt(chain);

        // Get critique from LLM
        string response = await _chatModel.GenerateResponseAsync(prompt);

        // Parse the critique
        return ParseCritique(response);
    }

    /// <summary>
    /// Builds a prompt for critiquing a single step.
    /// </summary>
    private string BuildCritiquePrompt(ReasoningStep<T> step, ReasoningContext context)
    {
        string previousStepsText = context.PreviousSteps.Count > 0
            ? $"\n\nPrevious steps:\n{string.Join("\n", context.PreviousSteps.Select((s, i) => $"{i + 1}. {s}"))}"
            : "";

        string evidenceText = context.SupportingEvidence.Count > 0
            ? $"\n\nSupporting evidence:\n{string.Join("\n", context.SupportingEvidence)}"
            : "";

        string domainText = !string.IsNullOrEmpty(context.Domain)
            ? $"\n\nDomain: {context.Domain}"
            : "";

        return $@"You are a careful critic evaluating the quality of a reasoning step.

Original problem: {context.Query}{domainText}{previousStepsText}

Current step to critique:
Step {step.StepNumber}: {step.Content}{evidenceText}

Evaluate this reasoning step and provide:
1. **Score** (0.0-1.0): Overall quality score
2. **Feedback**: Detailed evaluation explanation
3. **Strengths**: What this step does well (list 1-3 specific strengths)
4. **Weaknesses**: Problems or gaps in the reasoning (list 1-3 specific weaknesses, or empty if none)
5. **Suggestions**: How to improve this step (list 1-3 specific suggestions, or empty if step is already excellent)

Evaluation criteria:
- **Correctness**: Is the reasoning logically sound and factually accurate?
- **Clarity**: Is the step clearly explained and easy to follow?
- **Relevance**: Does it address the original problem?
- **Completeness**: Are there missing justifications or leaps in logic?
- **Consistency**: Does it align with previous steps?

Respond in JSON format:
{{
  ""score"": 0.85,
  ""feedback"": ""This step correctly identifies..., however..."",
  ""strengths"": [
    ""Logical deduction is sound"",
    ""Calculation is accurate""
  ],
  ""weaknesses"": [
    ""Could provide more justification for the assumption""
  ],
  ""suggestions"": [
    ""Add an explanation of why this approach was chosen""
  ]
}}

Provide your critique:";
    }

    /// <summary>
    /// Builds a prompt for critiquing an entire chain.
    /// </summary>
    private string BuildChainCritiquePrompt(ReasoningChain<T> chain)
    {
        string stepsText = string.Join("\n", chain.Steps.Select(s => $"Step {s.StepNumber}: {s.Content}"));

        return $@"You are a careful critic evaluating the overall quality of a reasoning chain.

Original problem: {chain.Query}

Complete reasoning chain:
{stepsText}

Final answer: {chain.FinalAnswer}

Evaluate this complete reasoning chain and provide:
1. **Score** (0.0-1.0): Overall chain quality
2. **Feedback**: Assessment of the reasoning as a whole
3. **Strengths**: What the chain does well (1-3 items)
4. **Weaknesses**: Overall problems with the reasoning (1-3 items, or empty if none)
5. **Suggestions**: How to improve the reasoning (1-3 items, or empty if already excellent)

Evaluation criteria:
- **Logical flow**: Do steps follow logically from each other?
- **Completeness**: Are all necessary steps present?
- **Correctness**: Is the final answer correct?
- **Coherence**: Is the reasoning coherent and well-structured?

Respond in JSON format:
{{
  ""score"": 0.85,
  ""feedback"": ""The reasoning chain...,""strengths"": [...],
  ""weaknesses"": [...],
  ""suggestions"": [...]
}}

Provide your critique:";
    }

    /// <summary>
    /// Parses a critique response into a structured result.
    /// </summary>
    private CritiqueResult<T> ParseCritique(string response)
    {
        var result = new CritiqueResult<T>();

        try
        {
            // Try JSON parsing
            string jsonContent = ExtractJsonFromResponse(response);
            var root = JObject.Parse(jsonContent);

            // Parse score
            if (root["score"] != null)
            {
                double score = root["score"]!.Value<double>();
                result.Score = _numOps.FromDouble(MathHelper.Clamp(score, 0.0, 1.0));
            }

            // Parse feedback
            result.Feedback = root["feedback"]?.Value<string>() ?? "";

            // Parse strengths
            if (root["strengths"] is JArray strengthsArray)
            {
                result.Strengths = strengthsArray.Select(s => s.Value<string>() ?? "").Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
            }

            // Parse weaknesses
            if (root["weaknesses"] is JArray weaknessesArray)
            {
                result.Weaknesses = weaknessesArray.Select(w => w.Value<string>() ?? "").Where(w => !string.IsNullOrWhiteSpace(w)).ToList();
            }

            // Parse suggestions
            if (root["suggestions"] is JArray suggestionsArray)
            {
                result.Suggestions = suggestionsArray.Select(s => s.Value<string>() ?? "").Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
            }
        }
        catch (JsonException)
        {
            // Fallback parsing
            result = ParseCritiqueFromText(response);
        }

        // Determine if passes threshold (typically 0.7)
        double scoreValue = Convert.ToDouble(result.Score);
        result.PassesThreshold = scoreValue >= 0.7;

        return result;
    }

    /// <summary>
    /// Fallback text-based parsing for non-JSON responses.
    /// </summary>
    private CritiqueResult<T> ParseCritiqueFromText(string response)
    {
        var result = new CritiqueResult<T>
        {
            Feedback = response
        };

        // Try to extract a score
        var scoreMatch = Regex.Match(response, @"(?:score|rating)[\s:]*([0-9]*\.?[0-9]+)", RegexOptions.IgnoreCase, RegexTimeout);
        if (scoreMatch.Success && double.TryParse(scoreMatch.Groups[1].Value, out double score))
        {
            if (score > 1.0 && score <= 10.0) score /= 10.0;
            result.Score = _numOps.FromDouble(MathHelper.Clamp(score, 0.0, 1.0));
        }
        else
        {
            result.Score = _numOps.FromDouble(0.7); // Default moderate score
        }

        result.PassesThreshold = Convert.ToDouble(result.Score) >= 0.7;

        return result;
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
