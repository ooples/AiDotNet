using System.Text.RegularExpressions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Reasoning.Components;

/// <summary>
/// Detects logical contradictions within reasoning chains.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A contradiction detector finds statements that conflict with each other.
/// If your reasoning says two things that can't both be true, that's a contradiction - and a serious error.
///
/// **Examples of contradictions:**
/// - Step 2: "x is greater than 10"
/// - Step 5: "x equals 5"
/// → **Contradiction!** x can't be both > 10 and equal to 5
///
/// - Step 1: "All cats are animals"
/// - Step 3: "Fluffy is a cat that is not an animal"
/// → **Contradiction!** Violates the rule from step 1
///
/// - Step 4: "The answer is 36"
/// - Step 7: "The answer is 42"
/// → **Contradiction!** Can't have two different final answers
///
/// **Why it matters:**
/// - Contradictions indicate flawed reasoning
/// - Essential for logical consistency
/// - Critical in mathematics, science, and formal logic
/// - Helps identify where reasoning went wrong
///
/// **Used in:**
/// - Verified reasoning systems
/// - Mathematical proof checkers
/// - Logical reasoning tasks
/// - Self-consistency validation
/// </para>
/// </remarks>
internal class ContradictionDetector<T> : IContradictionDetector<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IChatModel<T> _chatModel;

    /// <summary>
    /// Initializes a new instance of the <see cref="ContradictionDetector{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for contradiction detection.</param>
    public ContradictionDetector(IChatModel<T> chatModel)
    {
        _chatModel = chatModel ?? throw new ArgumentNullException(nameof(chatModel));
    }

    /// <inheritdoc/>
    public async Task<List<Contradiction>> DetectContradictionsAsync(
        ReasoningChain<T> chain,
        CancellationToken cancellationToken = default)
    {
        if (chain == null)
            throw new ArgumentNullException(nameof(chain));

        var contradictions = new List<Contradiction>();

        if (chain.Steps.Count < 2)
        {
            return contradictions; // Need at least 2 steps to have contradictions
        }

        // Check each pair of steps for contradictions
        // Use sliding window to prioritize checking adjacent and nearby steps
        for (int i = 0; i < chain.Steps.Count; i++)
        {
            for (int j = i + 1; j < chain.Steps.Count && j <= i + 5; j++) // Check up to 5 steps ahead
            {
                cancellationToken.ThrowIfCancellationRequested();

                bool isContradictory = await AreContradictoryAsync(
                    chain.Steps[i],
                    chain.Steps[j],
                    cancellationToken);

                if (isContradictory)
                {
                    var contradiction = await AnalyzeContradictionAsync(
                        chain.Steps[i],
                        chain.Steps[j],
                        cancellationToken);

                    contradictions.Add(contradiction);
                }
            }
        }

        // Also do some spot checks for non-adjacent steps if chain is long
        if (chain.Steps.Count > 10)
        {
            var random = RandomHelper.CreateSeededRandom(42); // Deterministic for reproducibility

            for (int attempt = 0; attempt < 5; attempt++)
            {
                int i = random.Next(chain.Steps.Count);
                int j = random.Next(chain.Steps.Count);

                if (i == j || Math.Abs(i - j) <= 5)
                    continue; // Skip same or already checked

                cancellationToken.ThrowIfCancellationRequested();

                bool isContradictory = await AreContradictoryAsync(
                    chain.Steps[i],
                    chain.Steps[j],
                    cancellationToken);

                if (isContradictory)
                {
                    var contradiction = await AnalyzeContradictionAsync(
                        chain.Steps[i],
                        chain.Steps[j],
                        cancellationToken);

                    contradictions.Add(contradiction);
                }
            }
        }

        return contradictions;
    }

    /// <inheritdoc/>
    public async Task<bool> AreContradictoryAsync(
        ReasoningStep<T> step1,
        ReasoningStep<T> step2,
        CancellationToken cancellationToken = default)
    {
        if (step1 == null || step2 == null)
            return false;

        // Quick heuristic checks first (avoid LLM call if obvious)
        if (HasObviousContradiction(step1.Content, step2.Content))
        {
            return true;
        }

        // Use LLM for deeper analysis
        string prompt = BuildContradictionCheckPrompt(step1, step2);
        string response = await _chatModel.GenerateResponseAsync(prompt, cancellationToken);

        return ParseContradictionResponse(response);
    }

    /// <summary>
    /// Performs detailed analysis of a contradiction.
    /// </summary>
    private async Task<Contradiction> AnalyzeContradictionAsync(
        ReasoningStep<T> step1,
        ReasoningStep<T> step2,
        CancellationToken cancellationToken)
    {
        string prompt = BuildContradictionAnalysisPrompt(step1, step2);
        string response = await _chatModel.GenerateResponseAsync(prompt, cancellationToken);

        return ParseContradictionAnalysis(response, step1.StepNumber, step2.StepNumber);
    }

    /// <summary>
    /// Quick heuristic check for obvious contradictions.
    /// </summary>
    private bool HasObviousContradiction(string text1, string text2)
    {
        if (string.IsNullOrWhiteSpace(text1) || string.IsNullOrWhiteSpace(text2))
            return false;

        string lower1 = text1.ToLowerInvariant();
        string lower2 = text2.ToLowerInvariant();

        // Check for contradictions: same subject with different values
        // Pattern 1: "X is Y" vs "X is Z" (where Y != Z)
        var isPattern = @"(\w+)\s+is\s+(\w+)";
        var match1 = Regex.Match(lower1, isPattern, RegexOptions.None, RegexTimeout);
        var match2 = Regex.Match(lower2, isPattern, RegexOptions.None, RegexTimeout);
        if (match1.Success && match2.Success &&
            match1.Groups[1].Value == match2.Groups[1].Value &&
            match1.Groups[2].Value != match2.Groups[2].Value)
        {
            return true;
        }

        // Pattern 2: "X is not Y" vs "X is Y" (explicit negation)
        var isNotPattern = @"(\w+)\s+is\s+not\s+(\w+)";
        match1 = Regex.Match(lower1, isNotPattern, RegexOptions.None, RegexTimeout);
        match2 = Regex.Match(lower2, isPattern, RegexOptions.None, RegexTimeout);
        if (match1.Success && match2.Success &&
            match1.Groups[1].Value == match2.Groups[1].Value &&
            match1.Groups[2].Value == match2.Groups[2].Value)
        {
            return true;
        }

        // Pattern 3: "X equals N" vs "X equals M" (where N != M)
        var equalsPattern = @"(\w+)\s+equals?\s+([0-9\.]+)";
        match1 = Regex.Match(lower1, equalsPattern, RegexOptions.None, RegexTimeout);
        match2 = Regex.Match(lower2, equalsPattern, RegexOptions.None, RegexTimeout);
        if (match1.Success && match2.Success &&
            match1.Groups[1].Value == match2.Groups[1].Value &&
            match1.Groups[2].Value != match2.Groups[2].Value)
        {
            return true;
        }

        // Pattern 4: "answer is N" vs "answer is M" (where N != M)
        var answerPattern = @"answer\s+is\s+([0-9\.]+)";
        match1 = Regex.Match(lower1, answerPattern, RegexOptions.None, RegexTimeout);
        match2 = Regex.Match(lower2, answerPattern, RegexOptions.None, RegexTimeout);
        if (match1.Success && match2.Success &&
            match1.Groups[1].Value != match2.Groups[1].Value)
        {
            return true;
        }

        return false;
    }

    /// <summary>
    /// Builds prompt for checking if two steps contradict.
    /// </summary>
    private string BuildContradictionCheckPrompt(ReasoningStep<T> step1, ReasoningStep<T> step2)
    {
        return $@"Determine if these two reasoning steps contradict each other.

Step {step1.StepNumber}: {step1.Content}

Step {step2.StepNumber}: {step2.Content}

Two statements contradict if they cannot both be true at the same time.

Examples of contradictions:
- ""x > 10"" and ""x = 5"" (mutually exclusive values)
- ""All cats are animals"" and ""Fluffy is a cat that is not an animal"" (logical inconsistency)
- ""The answer is 36"" and ""The answer is 42"" (conflicting conclusions)

Respond in JSON format:
{{
  ""contradictory"": true,
  ""brief_reason"": ""Explanation if contradictory""
}}

Analyze:";
    }

    /// <summary>
    /// Builds prompt for analyzing a contradiction in detail.
    /// </summary>
    private string BuildContradictionAnalysisPrompt(ReasoningStep<T> step1, ReasoningStep<T> step2)
    {
        return $@"Analyze this contradiction between two reasoning steps.

Step {step1.StepNumber}: {step1.Content}

Step {step2.StepNumber}: {step2.Content}

Provide:
1. Clear explanation of why they contradict
2. Severity (0.0 = minor inconsistency, 1.0 = direct logical contradiction)

Respond in JSON format:
{{
  ""explanation"": ""Detailed explanation of the contradiction"",
  ""severity"": 0.85
}}

Analyze:";
    }

    /// <summary>
    /// Parses whether steps are contradictory from LLM response.
    /// </summary>
    private bool ParseContradictionResponse(string response)
    {
        try
        {
            string jsonContent = ExtractJsonFromResponse(response);
            var root = JObject.Parse(jsonContent);

            if (root["contradictory"] != null)
            {
                return root["contradictory"]!.Value<bool>();
            }
        }
        catch (JsonException)
        {
            // Fallback to text analysis
        }

        // Fallback: look for positive indicators
        if (string.IsNullOrWhiteSpace(response))
            return false;

        string lower = response.ToLowerInvariant();
        return lower.Contains("yes") ||
               lower.Contains("contradictory") ||
               lower.Contains("contradict") ||
               lower.Contains("inconsistent");
    }

    /// <summary>
    /// Parses contradiction analysis from LLM response.
    /// </summary>
    private Contradiction ParseContradictionAnalysis(string response, int step1Num, int step2Num)
    {
        var contradiction = new Contradiction
        {
            Step1Number = step1Num,
            Step2Number = step2Num
        };

        try
        {
            string jsonContent = ExtractJsonFromResponse(response);
            var root = JObject.Parse(jsonContent);

            contradiction.Explanation = root["explanation"]?.Value<string>() ?? response;

            if (root["severity"] != null)
            {
                try
                {
                    contradiction.Severity = MathHelper.Clamp(root["severity"]!.Value<double>(), 0.0, 1.0);
                }
                catch (FormatException)
                {
                    contradiction.Severity = 0.8; // Default if non-numeric
                }
            }
            else
            {
                contradiction.Severity = 0.8; // Default high severity
            }
        }
        catch (JsonException)
        {
            contradiction.Explanation = response;
            contradiction.Severity = 0.8;
        }

        return contradiction;
    }

    /// <summary>
    /// Extracts JSON from response.
    /// </summary>
    private string ExtractJsonFromResponse(string response)
    {
        var jsonMatch = Regex.Match(response, @"```(?:json)?\s*(\{[\s\S]*?\})\s*```", RegexOptions.Multiline, RegexTimeout);
        if (jsonMatch.Success)
        {
            return jsonMatch.Groups[1].Value;
        }

        var jsonObjectMatch = Regex.Match(response, @"\{[\s\S]*?\}", RegexOptions.None, RegexTimeout);
        if (jsonObjectMatch.Success)
        {
            return jsonObjectMatch.Value;
        }

        return response;
    }
}
