using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for Continuum Memory System (CMS) - a spectrum of memory modules
/// operating at different frequencies for nested learning.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public interface IContinuumMemorySystem<T>
{
    /// <summary>
    /// Stores a representation in the continuum memory at the specified frequency level.
    /// </summary>
    void Store(Vector<T> representation, int frequencyLevel);

    /// <summary>
    /// Retrieves a representation from the continuum memory.
    /// </summary>
    Vector<T> Retrieve(Vector<T> query, int frequencyLevel);

    /// <summary>
    /// Updates memory based on current context and frequency.
    /// </summary>
    void Update(Vector<T> context, bool[] updateMask);

    /// <summary>
    /// Consolidates memories across frequency levels.
    /// </summary>
    void Consolidate();

    /// <summary>
    /// Gets the number of frequency levels in the continuum.
    /// </summary>
    int NumberOfFrequencyLevels { get; }

    /// <summary>
    /// Gets or sets the decay rate for each frequency level.
    /// </summary>
    T[] DecayRates { get; set; }

    /// <summary>
    /// Gets the current memory state at each frequency level.
    /// </summary>
    Vector<T>[] MemoryStates { get; }
}
