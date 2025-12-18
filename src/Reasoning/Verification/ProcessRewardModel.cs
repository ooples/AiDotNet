using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Reasoning.Verification;

/// <summary>
/// Implements a Process Reward Model (PRM) that scores individual reasoning steps.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A Process Reward Model (PRM) is like giving points for showing your work correctly,
/// not just getting the final answer right. This is crucial for training better reasoning systems.
///
/// **Process vs Outcome Rewards:**
///
/// **Process Reward Model (PRM):**
/// - Scores each individual reasoning step
/// - Example: Step 1: +0.9, Step 2: +0.7, Step 3: +0.95
/// - Helps identify exactly where reasoning goes wrong
/// - More informative for training
///
/// **Outcome Reward Model (ORM):**
/// - Scores only the final answer
/// - Example: Final answer correct: +1.0, incorrect: 0.0
/// - Simpler but less informative
///
/// **Why PRMs are important:**
/// - Used in training GPT-o1, DeepSeek-R1, and other reasoning models
/// - Enables reinforcement learning for reasoning
/// - Helps models learn step-by-step thinking
/// - Catches errors early in the reasoning process
///
/// **Research basis:**
/// "Let's Verify Step by Step" (Lightman et al., 2023) from OpenAI showed that
/// PRMs significantly outperform ORMs for mathematical reasoning.
/// </para>
/// </remarks>
internal class ProcessRewardModel<T> : IRewardModel<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IChatModel<T> _chatModel;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="ProcessRewardModel{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reward scoring.</param>
    public ProcessRewardModel(IChatModel<T> chatModel)
    {
        _chatModel = chatModel ?? throw new ArgumentNullException(nameof(chatModel));
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public RewardModelType ModelType => RewardModelType.Process;

    /// <inheritdoc/>
    public string ModelName => "Process Reward Model (PRM)";

    /// <inheritdoc/>
    public async Task<T> CalculateStepRewardAsync(
        ReasoningStep<T> step,
        ReasoningContext context,
        CancellationToken cancellationToken = default)
    {
        if (step == null)
            throw new ArgumentNullException(nameof(step));

        // Check cancellation before making LLM call
        cancellationToken.ThrowIfCancellationRequested();

        // Build evaluation prompt
        string prompt = BuildStepRewardPrompt(step, context);

        // Get reward score from LLM (pass cancellationToken)
        string response = await _chatModel.GenerateResponseAsync(prompt, cancellationToken);

        // Check cancellation after LLM call
        cancellationToken.ThrowIfCancellationRequested();

        // Parse the reward score
        double reward = ParseRewardScore(response);

        return _numOps.FromDouble(reward);
    }

    /// <inheritdoc/>
    public async Task<T> CalculateChainRewardAsync(
        ReasoningChain<T> chain,
        string? correctAnswer = null,
        CancellationToken cancellationToken = default)
    {
        if (chain == null)
            throw new ArgumentNullException(nameof(chain));

        // For PRM, chain reward is the average of step rewards
        // (could also use minimum, or weighted average based on position)

        if (chain.Steps.Count == 0)
        {
            return _numOps.Zero;
        }

        // Calculate reward for each step
        var stepRewards = new List<T>();

        for (int i = 0; i < chain.Steps.Count; i++)
        {
            var step = chain.Steps[i];

            var context = new ReasoningContext
            {
                Query = chain.Query,
                PreviousSteps = chain.Steps.Take(i).Select(s => s.Content).ToList()
            };

            // Use existing score if verified (zero is a valid verified score)
            if (step.IsVerified)
            {
                stepRewards.Add(step.Score);
            }
            else
            {
                T reward = await CalculateStepRewardAsync(step, context, cancellationToken);
                stepRewards.Add(reward);
            }
        }

        // Calculate average reward
        var rewardVector = new Vector<T>(stepRewards);
        T avgReward = rewardVector.Mean();

        // If correct answer provided, adjust reward based on correctness
        if (!string.IsNullOrEmpty(correctAnswer))
        {
            bool answerCorrect = chain.FinalAnswer is not null &&
                                chain.FinalAnswer.Equals(correctAnswer, StringComparison.OrdinalIgnoreCase);
            if (!answerCorrect)
            {
                // Penalize if final answer is wrong (even if process was good)
                avgReward = _numOps.Multiply(avgReward, _numOps.FromDouble(0.5));
            }
        }

        return avgReward;
    }

    /// <summary>
    /// Builds a prompt for scoring a reasoning step.
    /// </summary>
    private string BuildStepRewardPrompt(ReasoningStep<T> step, ReasoningContext context)
    {
        string previousStepsText = context.PreviousSteps.Count > 0
            ? $"\n\nPrevious steps:\n{string.Join("\n", context.PreviousSteps.Select((s, i) => $"{i + 1}. {s}"))}"
            : "";

        return $@"You are a reward model evaluating the quality of a reasoning step.

Original problem: {context.Query}{previousStepsText}

Current step to evaluate:
Step {step.StepNumber}: {step.Content}

Assign a reward score from 0.0 to 1.0 based on:

**High Reward (0.8-1.0):**
- Step is logically sound and correct
- Clear explanation and reasoning
- Makes meaningful progress toward solution
- Well-justified and supported

**Medium Reward (0.5-0.8):**
- Step is mostly correct but has minor issues
- Reasoning is somewhat unclear
- Makes some progress but could be more direct
- Partially justified

**Low Reward (0.0-0.5):**
- Step contains logical errors or is incorrect
- Unclear or confusing reasoning
- Makes little/no progress toward solution
- Poorly justified or unsupported

Respond in JSON format:
{{
  ""reward"": 0.85,
  ""reasoning"": ""Brief explanation of the reward score""
}}

Evaluate the step:";
    }

    /// <summary>
    /// Parses the reward score from the LLM response.
    /// </summary>
    private double ParseRewardScore(string response)
    {
        try
        {
            // Try JSON parsing
            string jsonContent = ExtractJsonFromResponse(response);
            var root = JObject.Parse(jsonContent);

            if (root["reward"] != null)
            {
                double rewardValue = root["reward"]!.Value<double>();
                return MathHelper.Clamp(rewardValue, 0.0, 1.0);
            }
        }
        catch (JsonException)
        {
            // Continue to fallback
        }

        // Fallback: look for numbers
        var numberMatch = Regex.Match(response, @"(?:reward|score)[\s:]*([0-9]*\.?[0-9]+)", RegexOptions.IgnoreCase, RegexTimeout);
        if (numberMatch.Success && double.TryParse(numberMatch.Groups[1].Value, out double reward))
        {
            // Normalize if needed
            if (reward > 1.0 && reward <= 10.0)
            {
                reward /= 10.0;
            }

            return MathHelper.Clamp(reward, 0.0, 1.0);
        }

        // Default to moderate reward if can't parse
        Debug.WriteLine($"Warning: Failed to parse reward from LLM response. Defaulting to 0.5. Response: {response.Substring(0, Math.Min(100, response.Length))}...");
        return 0.5;
    }

    /// <summary>
    /// Extracts JSON content from markdown code blocks.
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
