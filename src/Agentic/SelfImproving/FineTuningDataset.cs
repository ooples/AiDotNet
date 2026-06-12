namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// A set of reward-filtered <see cref="FineTuningExample"/>s ready to hand to a LoRA / fine-tuning trainer —
/// the bridge from "the agent did well on these runs" to "make the local model better at them".
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A study packet of known-good question/answer pairs. Feed it to the existing
/// LoRA fine-tuning trainer and the local model improves at the kinds of task it already handled well —
/// learning from its own successes (reward-filtered behavior cloning).
/// </para>
/// </remarks>
public sealed class FineTuningDataset
{
    /// <summary>
    /// Initializes a new dataset.
    /// </summary>
    /// <param name="examples">The fine-tuning examples.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="examples"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="examples"/> contains a <c>null</c> entry.</exception>
    public FineTuningDataset(IReadOnlyList<FineTuningExample> examples)
    {
        Guard.NotNull(examples);
        // Snapshot the caller's list: MeanReward is computed once here, so a
        // later mutation of a live List<> would silently desynchronize
        // Count/Examples from the reported mean. Null entries are rejected
        // now rather than failing later with a less precise exception.
        var snapshot = new FineTuningExample[examples.Count];
        for (var i = 0; i < examples.Count; i++)
        {
            snapshot[i] = examples[i] ?? throw new ArgumentException(
                $"Example at index {i} cannot be null.", nameof(examples));
        }

        Examples = Array.AsReadOnly(snapshot);
        MeanReward = snapshot.Length == 0 ? 0.0 : snapshot.Average(e => e.Reward);
    }

    /// <summary>Gets the fine-tuning examples.</summary>
    public IReadOnlyList<FineTuningExample> Examples { get; }

    /// <summary>Gets the number of examples.</summary>
    public int Count => Examples.Count;

    /// <summary>Gets the mean reward across the examples (0 when empty).</summary>
    public double MeanReward { get; }
}
