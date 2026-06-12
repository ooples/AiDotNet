using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// An <see cref="ITrajectoryEvaluator"/> that scores a run with a model acting as judge ("LLM-as-judge"):
/// it shows a judging model the task and the agent's answer (and an optional rubric) and parses a 0–1 score
/// from the reply. This is the general, model-based reward signal the self-improving layer can optimize
/// against without a hand-written metric.
/// </summary>
/// <typeparam name="T">The numeric type of the judging <see cref="IChatClient{T}"/>.</typeparam>
/// <remarks>
/// <para>
/// The judge can be any connector — including the in-process <c>LocalEngineChatClient</c>, so evaluation runs
/// fully offline. (The reasoning reward models in <c>Reasoning/Verification</c> can likewise be adapted to
/// this interface; LLM-as-judge is the connector-native default.) Scores are clamped to [0, 1].
/// </para>
/// <para><b>For Beginners:</b> Instead of writing rules to grade an answer, you ask another model "how good
/// is this, from 0 to 1?". This wraps that into a grader the rest of the system uses like any other.
/// </para>
/// </remarks>
public sealed class ChatClientTrajectoryEvaluator<T> : ITrajectoryEvaluator
{
    private static readonly Regex ScorePattern = new(@"[-+]?\d+(\.\d+)?", RegexOptions.Compiled);

    private readonly IChatClient<T> _judge;
    private readonly string? _rubric;

    /// <summary>
    /// Initializes a new LLM-as-judge evaluator.
    /// </summary>
    /// <param name="judge">The model used to grade trajectories.</param>
    /// <param name="rubric">Optional grading rubric/criteria included in the prompt.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="judge"/> is <c>null</c>.</exception>
    public ChatClientTrajectoryEvaluator(IChatClient<T> judge, string? rubric = null)
    {
        Guard.NotNull(judge);
        _judge = judge;
        _rubric = rubric;
    }

    /// <inheritdoc/>
    public async Task<double> EvaluateAsync(AgentTrajectory trajectory, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(trajectory);

        var messages = new[]
        {
            ChatMessage.System(
                "You are a strict evaluator. Rate the assistant's answer to the task on a scale from 0 to 1, " +
                "where 1 is perfect and 0 is completely wrong. Reply with only the number."),
            ChatMessage.User(BuildJudgePrompt(trajectory)),
        };

        var options = new ChatOptions { Temperature = 0.0 };
        var response = await _judge.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        return ParseScore(response.Text);
    }

    private string BuildJudgePrompt(AgentTrajectory trajectory)
    {
        var builder = new StringBuilder();
        if (_rubric is { } rubric && rubric.Trim().Length > 0)
        {
            builder.Append("Rubric: ").AppendLine(rubric).AppendLine();
        }

        builder.AppendLine("Task and conversation:");
        foreach (var message in trajectory.Messages)
        {
            if (message.Role == ChatRole.User || message.Role == ChatRole.System)
            {
                builder.Append(message.Role).Append(": ").AppendLine(message.Text);
            }
        }

        builder.AppendLine();
        builder.Append("Assistant's final answer: ").AppendLine(trajectory.FinalText);
        builder.AppendLine();
        builder.Append("Score (0 to 1):");
        return builder.ToString();
    }

    private static double ParseScore(string text)
    {
        var match = ScorePattern.Match(text);
        if (!match.Success || !double.TryParse(match.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var score))
        {
            return 0.0;
        }

        if (score < 0)
        {
            return 0.0;
        }

        return score > 1 ? 1.0 : score;
    }
}
