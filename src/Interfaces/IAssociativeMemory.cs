using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for Associative Memory modules used in nested learning.
/// Models both backpropagation and attention mechanisms as associative memory.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
[AiDotNet.Configuration.YamlConfigurable("AssociativeMemory")]
public interface IAssociativeMemory<T>
{
    /// <summary>
    /// Associates an input with a target output (learns the mapping).
    /// In backpropagation context: maps data point to local error.
    /// In attention context: maps queries to key-value pairs.
    /// </summary>
    void Associate(Vector<T> input, Vector<T> target);

    /// <summary>
    /// Retrieves the associated output for a given input query.
    /// </summary>
    Vector<T> Retrieve(Vector<T> query);

    /// <summary>
    /// Updates the memory based on new associations.
    /// </summary>
    void Update(Vector<T> input, Vector<T> target, T learningRate);

    /// <summary>
    /// Gets the memory capacity.
    /// </summary>
    int Capacity { get; }

    /// <summary>
    /// Clears all stored associations.
    /// </summary>
    void Clear();
}
