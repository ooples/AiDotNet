namespace AiDotNet.MetaLearning;

using AiDotNet.Interfaces;
using AiDotNet.Models.Inputs;

/// <summary>
/// Strategy for meta-learning (e.g., SEAL) that can orchestrate episodic training and optionally adapt inference,
/// composing with an existing model and optimizer.
/// </summary>
public interface IMetaLearningStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Prepare or transform the optimization input data (e.g., form episodes, augment, or reweight) before Optimize().
    /// Must not mutate the provided model or optimizer directly; return a new or adjusted OptimizationInputData.
    /// </summary>
    OptimizationInputData<T, TInput, TOutput> Prepare(
        OptimizationInputData<T, TInput, TOutput> input,
        IFullModel<T, TInput, TOutput> model,
        IOptimizer<T, TInput, TOutput> optimizer);

    /// <summary>
    /// Optional inference-time adaptation hook. Implementations may perform fast adaptation before prediction.
    /// Return true if adaptation was applied and the prediction was produced; otherwise false to fall back.
    /// </summary>
    bool TryPredict(
        TInput newData,
        PredictionModelResult<T, TInput, TOutput> modelResult,
        out TOutput prediction);
}

