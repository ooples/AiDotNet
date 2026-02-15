using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for Context Flow mechanism - maintains distinct information pathways
/// and update rates for each nested optimization level.
/// Core component of nested learning paradigm.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ContextFlow")]
public interface IContextFlow<T>
{
    /// <summary>
    /// Propagates context through the flow network at a specific optimization level.
    /// Each level has its own distinct set of information from which it learns.
    /// </summary>
    Vector<T> PropagateContext(Vector<T> input, int currentLevel);

    /// <summary>
    /// Computes gradients with respect to context flow for backpropagation.
    /// </summary>
    Vector<T> ComputeContextGradients(Vector<T> upstreamGradient, int level);

    /// <summary>
    /// Updates the context flow based on multi-level optimization.
    /// </summary>
    void UpdateFlow(Vector<T>[] gradients, T[] learningRates);

    /// <summary>
    /// Gets the current context state for a specific optimization level.
    /// </summary>
    Vector<T> GetContextState(int level);

    /// <summary>
    /// Compresses internal context flows (deep learning compression mechanism).
    /// </summary>
    Vector<T> CompressContext(Vector<T> context, int targetLevel);

    /// <summary>
    /// Resets the context flow to initial state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets the number of context flow levels.
    /// </summary>
    int NumberOfLevels { get; }
}
