using AiDotNet.FineTuning;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Runs the online self-improvement loop's final step: fine-tune (e.g. LoRA-adapt) a model on a reward-filtered
/// <see cref="FineTuningDataset"/> by converting it to supervised data and invoking a fine-tuner. This closes
/// the loop capture → evaluate → reward-filter → <em>train</em>.
/// </summary>
/// <remarks>
/// <para>
/// The fine-tuner is any <see cref="FineTuningBase{T, TInput, TOutput}"/> over string in/out (notably
/// <c>SupervisedFineTuning</c>) and the model any <see cref="IFullModel{T, TInput, TOutput}"/> that consumes
/// text. The actual training (LoRA via the configured <see cref="FineTuningOptions{T}"/>) and its hardware cost
/// live in the fine-tuner; this is the thin, deterministic bridge from the agentic dataset to that call.
/// </para>
/// <para><b>For Beginners:</b> Hand it your "good runs" dataset, a fine-tuner, and a model, and it trains the
/// model to do more of what worked.
/// </para>
/// </remarks>
public static class LoRAFineTuner
{
    /// <summary>
    /// Fine-tunes a model on a reward-filtered dataset.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="fineTuner">The fine-tuner (e.g. SupervisedFineTuning) to run.</param>
    /// <param name="model">The model to fine-tune.</param>
    /// <param name="dataset">The reward-filtered dataset.</param>
    /// <param name="cancellationToken">Token used to cancel training.</param>
    /// <returns>The fine-tuned model.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="fineTuner"/>, <paramref name="model"/>, or <paramref name="dataset"/> is <c>null</c>.</exception>
    public static Task<IFullModel<T, string, string>> FineTuneFromDatasetAsync<T>(
        FineTuningBase<T, string, string> fineTuner,
        IFullModel<T, string, string> model,
        FineTuningDataset dataset,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(fineTuner);
        Guard.NotNull(model);
        Guard.NotNull(dataset);

        var trainingData = FineTuningDataConverter.ToSupervisedData<T>(dataset);
        return fineTuner.FineTuneAsync(model, trainingData, cancellationToken);
    }
}
