using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for nested learning algorithms - a multi-level optimization paradigm for continual learning.
/// Based on Google's Nested Learning research.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">Input data type</typeparam>
/// <typeparam name="TOutput">Output data type</typeparam>
public interface INestedLearner<T, TInput, TOutput>
{
    /// <summary>
    /// Performs a nested learning step with multi-level optimization.
    /// </summary>
    MetaTrainingStepResult<T> NestedStep(TInput input, TOutput expectedOutput, int level = 0);

    /// <summary>
    /// Trains the model using nested learning across multiple levels.
    /// </summary>
    MetaTrainingResult<T> Train(IEnumerable<(TInput Input, TOutput Output)> trainingData, int maxIterations = 1000);

    /// <summary>
    /// Adapts the model to a new task without catastrophic forgetting.
    /// </summary>
    MetaAdaptationResult<T> AdaptToNewTask(IEnumerable<(TInput Input, TOutput Output)> newTaskData, T preservationStrength);

    /// <summary>
    /// Gets the number of nested optimization levels.
    /// </summary>
    int NumberOfLevels { get; }

    /// <summary>
    /// Gets the update frequency for each level.
    /// </summary>
    int[] UpdateFrequencies { get; }
}
