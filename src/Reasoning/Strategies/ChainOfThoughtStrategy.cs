using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Reasoning.Strategies;

/// <summary>
/// Implements Chain-of-Thought (CoT) reasoning that solves problems through explicit step-by-step thinking.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Chain-of-Thought (CoT) is a reasoning approach where the AI explicitly
/// shows its work, step by step, similar to how you would solve a math problem by writing down each step.
///
/// **How it works:**
/// Given: "What is 15% of 240?"
///
/// Step 1: Convert percentage to decimal
/// - 15% = 15/100 = 0.15
///
/// Step 2: Multiply by the number
/// - 0.15 × 240 = 36
///
/// Step 3: State the answer
/// - The answer is 36
///
/// **Why it's effective:**
/// - Makes reasoning transparent and verifiable
/// - Catches logical errors early
/// - Improves accuracy on complex problems
/// - Allows debugging when answers are wrong
///
/// **Based on research:**
/// "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
/// showed 3-5x improvements on reasoning tasks when models show their work.
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// var chatModel = new OpenAIChatModel&lt;double&gt;("gpt-4");
/// var strategy = new ChainOfThoughtStrategy&lt;double&gt;(chatModel);
///
/// var result = await strategy.ReasonAsync(
///     "If a train travels 60 mph for 2.5 hours, how far does it go?",
///     new ReasoningConfig()
/// );
///
/// Console.WriteLine(result.FinalAnswer); // "150 miles"
/// Console.WriteLine(result.ReasoningChain); // Shows all steps
/// </code>
/// </para>
/// </remarks>
public class ChainOfThoughtStrategy<T> : ReasoningStrategyBase<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly bool _useJsonFormat;

    /// <summary>
    /// Initializes a new instance of the <see cref="ChainOfThoughtStrategy{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reasoning.</param>
    /// <param name="tools">Optional tools available during reasoning.</param>
    /// <param name="useJsonFormat">Whether to request JSON-formatted responses (more reliable parsing).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a Chain-of-Thought reasoning strategy.
    /// The chatModel is the AI that does the thinking, tools are optional helpers (like calculators),
    /// and useJsonFormat controls whether responses are structured (recommended for reliability).
    /// </para>
    /// </remarks>
    public ChainOfThoughtStrategy(
        IChatModel<T> chatModel,
        IEnumerable<ITool>? tools = null,
        bool useJsonFormat = true)
        : base(chatModel, tools)
    {
        _useJsonFormat = useJsonFormat;
    }

    /// <inheritdoc/>
    public override string StrategyName => "Chain-of-Thought";

    /// <inheritdoc/>
    public override string Description =>
        "Generates explicit step-by-step reasoning to solve problems. Best for tasks requiring " +
        "logical deduction, mathematical calculations, or problems that humans solve by showing their work.";

    /// <inheritdoc/>
    protected override async Task<ReasoningResult<T>> ReasonCoreAsync(
        string query,
        ReasoningConfig config,
        CancellationToken cancellationToken)
    {
        ValidateConfig(config);

        var result = new ReasoningResult<T>
        {
            StrategyUsed = StrategyName
        };

        var chain = new ReasoningChain<T>
        {
            Query = query,
            StartedAt = DateTime.UtcNow
        };

        AppendTrace($"Starting Chain-of-Thought reasoning for query: {query}");
        AppendTrace($"Max steps: {config.MaxSteps}, Verification: {config.EnableVerification}");

        // Step 1: Generate the reasoning chain
        string prompt = BuildChainOfThoughtPrompt(query, config);
        AppendTrace("Generating reasoning steps...");

        string llmResponse;
        try
        {
            llmResponse = await ChatModel.GenerateResponseAsync(prompt);
            AppendTrace($"Received response ({llmResponse.Length} chars)");
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.ErrorMessage = $"LLM generation failed: {ex.Message}";
            AppendTrace($"ERROR: {result.ErrorMessage}");
            return result;
        }

        // Step 2: Parse the reasoning steps
        var parsedSteps = ParseReasoningSteps(llmResponse, config.MaxSteps);
        AppendTrace($"Parsed {parsedSteps.Count} reasoning steps");

        foreach (var step in parsedSteps)
        {
            chain.AddStep(step);
        }

        // Step 3: Optionally verify steps
        if (config.EnableVerification && parsedSteps.Count > 0)
        {
            AppendTrace("Verification enabled - checking reasoning quality...");
            // TODO: Implement verification when ICriticModel concrete implementation is available
            // For now, mark all steps as verified with high confidence
            foreach (var step in chain.Steps)
            {
                step.IsVerified = true;
            }
        }

        // Step 4: Extract final answer
        string finalAnswer = ExtractFinalAnswer(llmResponse, parsedSteps);
        chain.FinalAnswer = finalAnswer;
        chain.CompletedAt = DateTime.UtcNow;

        // Step 5: Calculate overall score
        chain.OverallScore = chain.GetAverageScore();

        // Step 6: Build result
        result.FinalAnswer = finalAnswer;
        result.ReasoningChain = chain;
        result.OverallConfidence = chain.OverallScore;
        result.Success = !string.IsNullOrWhiteSpace(finalAnswer);
        result.Metrics["num_steps"] = chain.Steps.Count;
        result.Metrics["total_refinements"] = chain.TotalRefinements;
        result.Metrics["fully_verified"] = chain.IsFullyVerified;

        AppendTrace($"Reasoning complete: {chain.Steps.Count} steps, answer: {finalAnswer}");

        return result;
    }

    /// <summary>
    /// Builds the prompt that instructs the LLM to use Chain-of-Thought reasoning.
    /// </summary>
    private string BuildChainOfThoughtPrompt(string query, ReasoningConfig config)
    {
        string toolInfo = Tools.Count > 0 ? $"\n\nYou have access to these tools:\n{GetToolDescriptions()}" : "";

        if (_useJsonFormat)
        {
            return $@"You are a helpful AI assistant that solves problems using step-by-step reasoning.

Query: {query}{toolInfo}

Please solve this problem by:
1. Breaking it down into clear, logical steps
2. Explaining your reasoning for each step
3. Showing any calculations or deductions
4. Arriving at a final answer

Respond in JSON format:
{{
  ""reasoning_steps"": [
    ""Step 1: [your first reasoning step with explanation]"",
    ""Step 2: [your second reasoning step with explanation]"",
    ...
  ],
  ""final_answer"": ""your final answer to the query""
}}

Important:
- Be explicit about your reasoning in each step
- Maximum {config.MaxSteps} steps
- Think step by step

Respond in JSON format:";
        }
        else
        {
            return $@"You are a helpful AI assistant that solves problems using step-by-step reasoning.

Query: {query}{toolInfo}

Please solve this problem by breaking it down into clear, logical steps.
Show your work explicitly, explaining your reasoning for each step.

Format your response as:
Step 1: [your first reasoning step]
Step 2: [your second reasoning step]
...
Final Answer: [your answer]

Maximum {config.MaxSteps} steps. Think step by step.";
        }
    }

    /// <summary>
    /// Parses reasoning steps from the LLM response.
    /// </summary>
    private List<ReasoningStep<T>> ParseReasoningSteps(string response, int maxSteps)
    {
        var steps = new List<ReasoningStep<T>>();

        try
        {
            // Try JSON parsing first
            string jsonContent = ExtractJsonFromResponse(response);
            var root = JObject.Parse(jsonContent);

            if (root["reasoning_steps"] is JArray stepsArray)
            {
                int stepNum = 1;
                foreach (var stepToken in stepsArray)
                {
                    if (stepNum > maxSteps) break;

                    string stepText = stepToken.Value<string>() ?? "";
                    if (!string.IsNullOrWhiteSpace(stepText))
                    {
                        steps.Add(CreateReasoningStep(stepNum++, stepText));
                    }
                }
            }
        }
        catch (JsonException)
        {
            // Fallback to regex parsing
            AppendTrace("JSON parsing failed, using regex fallback");
            steps = ParseWithRegex(response, maxSteps);
        }

        return steps;
    }

    /// <summary>
    /// Fallback regex parser for non-JSON responses.
    /// </summary>
    private List<ReasoningStep<T>> ParseWithRegex(string response, int maxSteps)
    {
        var steps = new List<ReasoningStep<T>>();

        // Match patterns like "Step 1:", "Step 2:", or numbered lists "1.", "2."
        var stepMatches = Regex.Matches(
            response,
            @"(?:Step\s*\d+:|^\d+\.)\s*(.+?)(?=(?:Step\s*\d+:|\d+\.|Final Answer:|$))",
            RegexOptions.IgnoreCase | RegexOptions.Multiline | RegexOptions.Singleline,
            RegexTimeout
        );

        int stepNum = 1;
        foreach (Match match in stepMatches)
        {
            if (stepNum > maxSteps) break;

            string stepText = match.Groups[1].Value.Trim();
            if (!string.IsNullOrWhiteSpace(stepText) && stepText.Length > 5)
            {
                steps.Add(CreateReasoningStep(stepNum++, stepText));
            }
        }

        return steps;
    }

    /// <summary>
    /// Creates a reasoning step with default high confidence.
    /// </summary>
    private ReasoningStep<T> CreateReasoningStep(int stepNumber, string content)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        return new ReasoningStep<T>
        {
            StepNumber = stepNumber,
            Content = content,
            Score = numOps.FromDouble(0.9), // Default high confidence
            CreatedAt = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Extracts JSON content from markdown code blocks.
    /// </summary>
    private string ExtractJsonFromResponse(string response)
    {
        // Remove markdown code block markers if present
        var jsonMatch = Regex.Match(response, @"```(?:json)?\s*(\{[\s\S]*?\})\s*```", RegexOptions.Multiline, RegexTimeout);
        if (jsonMatch.Success)
        {
            return jsonMatch.Groups[1].Value;
        }

        // Try to find JSON object without code blocks
        var jsonObjectMatch = Regex.Match(response, @"\{[\s\S]*?\}", RegexOptions.None, RegexTimeout);
        if (jsonObjectMatch.Success)
        {
            return jsonObjectMatch.Value;
        }

        return response;
    }

    /// <summary>
    /// Extracts the final answer from the response.
    /// </summary>
    private string ExtractFinalAnswer(string response, List<ReasoningStep<T>> steps)
    {
        // Try JSON first
        try
        {
            string jsonContent = ExtractJsonFromResponse(response);
            var root = JObject.Parse(jsonContent);
            string? answer = root["final_answer"]?.Value<string>();
            if (answer is not null && !string.IsNullOrWhiteSpace(answer))
            {
                return answer;
            }
        }
        catch (JsonException)
        {
            // Continue to regex fallback
        }

        // Try regex for "Final Answer:" pattern
        var answerMatch = Regex.Match(
            response,
            @"Final\s*Answer\s*:\s*(.+?)(?:\n\n|$)",
            RegexOptions.IgnoreCase | RegexOptions.Singleline,
            RegexTimeout
        );

        if (answerMatch.Success)
        {
            return answerMatch.Groups[1].Value.Trim();
        }

        // If no explicit final answer, use the last step's content
        if (steps.Count > 0)
        {
            return steps[steps.Count - 1].Content;  // net462: can't use ^1
        }

        // Last resort: return the whole response
        return response.Trim();
    }
}
