using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// A buffer for storing and replaying experiences in reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ReplayBuffer<T>
{
    private readonly int _capacity;
    private readonly List<Experience<T>> _buffer;
    private readonly Random _random;

    /// <summary>
    /// Gets the current number of experiences in the buffer.
    /// </summary>
    public int Count => _buffer.Count;

    /// <summary>
    /// Initializes a new instance of the ReplayBuffer class.
    /// </summary>
    /// <param name="capacity">The maximum number of experiences to store.</param>
    /// <param name="seed">Optional random seed.</param>
    public ReplayBuffer(int capacity, int? seed = null)
    {
        _capacity = capacity;
        _buffer = new List<Experience<T>>(capacity);
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Adds a new experience to the buffer.
    /// </summary>
    /// <param name="experience">The experience to add.</param>
    public void Add(Experience<T> experience)
    {
        if (_buffer.Count >= _capacity)
        {
            _buffer.RemoveAt(0); // FIFO
        }
        _buffer.Add(experience);
    }

    /// <summary>
    /// Samples a batch of experiences from the buffer.
    /// </summary>
    /// <param name="batchSize">The number of experiences to sample.</param>
    /// <returns>A list of sampled experiences.</returns>
    public List<Experience<T>> Sample(int batchSize)
    {
        var batch = new List<Experience<T>>(batchSize);
        int count = _buffer.Count;
        
        if (count == 0) return batch;

        for (int i = 0; i < batchSize; i++)
        {
            int index = _random.Next(count);
            batch.Add(_buffer[index]);
        }

        return batch;
    }

    /// <summary>
    /// Clears the buffer.
    /// </summary>
    public void Clear()
    {
        _buffer.Clear();
    }
}
