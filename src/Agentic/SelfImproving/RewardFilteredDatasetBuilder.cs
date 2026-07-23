using System.Text;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Builds a <see cref="FineTuningDataset"/> from captured trajectories by keeping only those whose reward
/// meets a threshold and turning each into a (prompt, completion) pair — reward-filtered behavior cloning,
/// the data-preparation half of online LoRA self-improvement.
/// </summary>
/// <remarks>
/// <para>
/// The prompt is the trajectory's conversation up to (but excluding) its final turn; the completion is the
/// final answer. The resulting dataset is then handed to the repository's LoRA / fine-tuning trainer to
/// produce an adapter for the local model — the trainer is the model-layer step; this is the agentic step
/// that decides <em>what</em> to learn from (only high-reward runs).
/// </para>
/// <para><b>For Beginners:</b> Sift the logbook for the runs that scored well, and from each make a
/// question→good-answer pair. Those pairs become the lesson plan for fine-tuning the local model on its own
/// best work.
/// </para>
/// </remarks>
public sealed class RewardFilteredDatasetBuilder
{
    private readonly double _minReward;

    /// <summary>
    /// Initializes a new builder.
    /// </summary>
    /// <param name="minReward">The minimum reward a trajectory must have to be included. Default 0.5.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="minReward"/> is NaN or infinite.</exception>
    public RewardFilteredDatasetBuilder(double minReward = 0.5)
    {
        // double.IsFinite is unavailable on net471 — spell out both halves.
        if (double.IsNaN(minReward) || double.IsInfinity(minReward))
        {
            throw new ArgumentOutOfRangeException(
                nameof(minReward), minReward, "Minimum reward must be a finite number.");
        }

        _minReward = minReward;
    }

    /// <summary>
    /// Builds the dataset from the given trajectories, keeping only graded runs at or above the threshold
    /// that have both a prompt context and a non-empty completion.
    /// </summary>
    /// <param name="trajectories">The trajectories to distill.</param>
    /// <returns>The reward-filtered fine-tuning dataset.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="trajectories"/> is <c>null</c>.</exception>
    public FineTuningDataset Build(IEnumerable<AgentTrajectory> trajectories)
    {
        Guard.NotNull(trajectories);

        var examples = new List<FineTuningExample>();
        foreach (var trajectory in trajectories)
        {
            // NaN compares false against the threshold, so `reward < _minReward`
            // alone would let malformed (non-finite) rewards through and poison
            // the dataset — require a finite grade explicitly.
            if (trajectory.Reward is not { } reward
                || double.IsNaN(reward)
                || double.IsInfinity(reward)
                || reward < _minReward)
            {
                continue;
            }

            if (trajectory.Messages.Count < 2 || trajectory.FinalText.Trim().Length == 0)
            {
                continue;
            }

            var prompt = RenderPrompt(trajectory.Messages);
            if (string.IsNullOrWhiteSpace(prompt))
            {
                // No usable prompt context — a degenerate pair would be
                // rejected by FineTuningExample anyway.
                continue;
            }

            examples.Add(new FineTuningExample(prompt, trajectory.FinalText, reward));
        }

        return new FineTuningDataset(examples);
    }

    private static string RenderPrompt(IReadOnlyList<ChatMessage> messages)
    {
        var builder = new StringBuilder();
        for (var i = 0; i < messages.Count - 1; i++)
        {
            if (i > 0)
            {
                builder.Append('\n');
            }

            builder.Append(messages[i].Role).Append(": ").Append(messages[i].Text);
        }

        return builder.ToString();
    }
}
