using System;
using System.Collections.Generic;
using System.Numerics;
using AiDotNet.Models;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for nested learning algorithms that treat models as interconnected,
    /// multi-level learning problems optimized simultaneously.
    /// Based on Google's Nested Learning paradigm for continual learning.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.)</typeparam>
    /// <typeparam name="TInput">Input data type</typeparam>
    /// <typeparam name="TOutput">Output data type</typeparam>
    public interface INestedLearner<T, TInput, TOutput>
        where T : struct, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>
    {
        /// <summary>
        /// Performs a nested learning step with multi-level optimization.
        /// </summary>
        /// <param name="input">Input data</param>
        /// <param name="expectedOutput">Expected output</param>
        /// <param name="level">Optimization level (0 = fastest updates, higher = slower updates)</param>
        /// <returns>Result containing loss and metrics for each level</returns>
        NestedLearningStepResult<T> NestedStep(TInput input, TOutput expectedOutput, int level = 0);

        /// <summary>
        /// Trains the model using nested learning across multiple levels.
        /// </summary>
        /// <param name="trainingData">Training dataset</param>
        /// <param name="numLevels">Number of nested optimization levels</param>
        /// <param name="maxIterations">Maximum iterations per level</param>
        /// <returns>Training result with performance metrics per level</returns>
        NestedLearningResult<T> Train(
            IEnumerable<(TInput Input, TOutput Output)> trainingData,
            int numLevels = 3,
            int maxIterations = 1000);

        /// <summary>
        /// Adapts the model to a new task using nested learning's continual learning capabilities.
        /// </summary>
        /// <param name="newTaskData">Data for the new task</param>
        /// <param name="preservationStrength">How strongly to preserve previous knowledge (0-1)</param>
        /// <returns>Adaptation result</returns>
        NestedAdaptationResult<T> AdaptToNewTask(
            IEnumerable<(TInput Input, TOutput Output)> newTaskData,
            T preservationStrength = default);

        /// <summary>
        /// Gets the number of nested optimization levels.
        /// </summary>
        int NumberOfLevels { get; }

        /// <summary>
        /// Gets the update frequency for each level.
        /// Level 0 updates every step, level 1 every N steps, etc.
        /// </summary>
        int[] UpdateFrequencies { get; }
    }

    /// <summary>
    /// Result of a single nested learning step.
    /// </summary>
    public class NestedLearningStepResult<T>
        where T : struct, IFloatingPoint<T>
    {
        /// <summary>Loss value for this step</summary>
        public T Loss { get; set; }

        /// <summary>Loss value per nested level</summary>
        public Dictionary<int, T> LossPerLevel { get; set; } = new();

        /// <summary>Gradients computed at each level</summary>
        public Dictionary<int, Vector<T>> GradientsPerLevel { get; set; } = new();

        /// <summary>Which levels were updated in this step</summary>
        public HashSet<int> UpdatedLevels { get; set; } = new();
    }

    /// <summary>
    /// Result of nested learning training.
    /// </summary>
    public class NestedLearningResult<T>
        where T : struct, IFloatingPoint<T>
    {
        /// <summary>Final training loss</summary>
        public T FinalLoss { get; set; }

        /// <summary>Loss history per level</summary>
        public Dictionary<int, List<T>> LossHistoryPerLevel { get; set; } = new();

        /// <summary>Number of iterations performed</summary>
        public int Iterations { get; set; }

        /// <summary>Training duration</summary>
        public TimeSpan Duration { get; set; }

        /// <summary>Whether training converged</summary>
        public bool Converged { get; set; }
    }

    /// <summary>
    /// Result of adapting to a new task using nested learning.
    /// </summary>
    public class NestedAdaptationResult<T>
        where T : struct, IFloatingPoint<T>
    {
        /// <summary>Loss on new task after adaptation</summary>
        public T NewTaskLoss { get; set; }

        /// <summary>Loss on previous tasks (to measure forgetting)</summary>
        public T PreviousTasksLoss { get; set; }

        /// <summary>Forgetting metric (how much performance degraded on old tasks)</summary>
        public T ForgettingMetric { get; set; }

        /// <summary>Number of adaptation steps performed</summary>
        public int AdaptationSteps { get; set; }
    }
}
