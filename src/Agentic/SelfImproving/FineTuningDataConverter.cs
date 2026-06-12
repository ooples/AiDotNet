using AiDotNet.Models.Options;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Bridges the self-improving layer to the fine-tuning framework: converts a reward-filtered
/// <see cref="FineTuningDataset"/> (prompt → good-completion pairs) into the framework's supervised
/// <see cref="FineTuningData{T, TInput, TOutput}"/> shape so it can be handed to
/// <c>SupervisedFineTuning</c> (or any SFT-capable fine-tuner) to fine-tune / LoRA-adapt a model on the
/// agent's own best behavior.
/// </summary>
/// <remarks>
/// <para>
/// Prompts become <c>Inputs</c>, completions become <c>Outputs</c>, and each example's reward is carried as a
/// <c>SampleWeight</c> so the trainer can weight higher-reward examples more heavily. The resulting data uses
/// <c>string</c> inputs/outputs; a model that consumes text (or a tokenization step in front of a tensor
/// model) completes the loop — the training execution itself is a model-pipeline concern.
/// </para>
/// <para><b>For Beginners:</b> Turns the "good runs" you collected into the exact format the fine-tuning
/// trainer expects, so you can teach a model to imitate its own successes.
/// </para>
/// </remarks>
public static class FineTuningDataConverter
{
    /// <summary>
    /// Converts a reward-filtered dataset to supervised fine-tuning data (string in/out), carrying rewards as
    /// sample weights.
    /// </summary>
    /// <typeparam name="T">The numeric type of the fine-tuning data.</typeparam>
    /// <param name="dataset">The reward-filtered dataset.</param>
    /// <returns>The supervised <see cref="FineTuningData{T, TInput, TOutput}"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="dataset"/> is <c>null</c>.</exception>
    public static FineTuningData<T, string, string> ToSupervisedData<T>(FineTuningDataset dataset)
    {
        Guard.NotNull(dataset);

        var count = dataset.Examples.Count;
        var inputs = new string[count];
        var outputs = new string[count];
        var weights = new double[count];
        for (var i = 0; i < count; i++)
        {
            inputs[i] = dataset.Examples[i].Prompt;
            outputs[i] = dataset.Examples[i].Completion;
            weights[i] = dataset.Examples[i].Reward;
        }

        return new FineTuningData<T, string, string>
        {
            Inputs = inputs,
            Outputs = outputs,
            SampleWeights = weights,
        };
    }
}
