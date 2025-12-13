using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Interface for optimization passes that transform computation graphs to improve inference performance.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public interface IOptimizationPass<T> where T : struct
{
    /// <summary>
    /// The type of optimization pass.
    /// </summary>
    OptimizationPassType PassType { get; }

    /// <summary>
    /// The name of this optimization pass.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Applies the optimization pass to the computation graph.
    /// </summary>
    /// <param name="graph">The computation graph to optimize</param>
    /// <returns>True if the graph was modified, false otherwise</returns>
    bool Apply(IOptimizationGraph<T> graph);

    /// <summary>
    /// Checks if this pass can be applied to the graph.
    /// </summary>
    bool CanApply(IOptimizationGraph<T> graph);
}
