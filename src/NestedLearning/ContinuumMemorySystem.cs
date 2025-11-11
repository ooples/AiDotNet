using System;
using System.Numerics;
using AiDotNet.Interfaces;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.NestedLearning
{
    /// <summary>
    /// Implementation of Continuum Memory System (CMS) for nested learning.
    /// Provides a spectrum of memory modules operating at different frequencies.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class ContinuumMemorySystem<T> : IContinuumMemorySystem<T>
        where T : struct, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>
    {
        private readonly int _numFrequencyLevels;
        private readonly int _memoryDimension;
        private Tensor<T>[] _memoryStates;
        private T[] _decayRates;

        /// <summary>
        /// Initializes a new Continuum Memory System.
        /// </summary>
        /// <param name="memoryDimension">Dimension of memory representations</param>
        /// <param name="numFrequencyLevels">Number of frequency levels</param>
        /// <param name="decayRates">Optional custom decay rates per level</param>
        public ContinuumMemorySystem(
            int memoryDimension,
            int numFrequencyLevels = 3,
            T[]? decayRates = null)
        {
            _memoryDimension = memoryDimension;
            _numFrequencyLevels = numFrequencyLevels;

            // Initialize memory states
            _memoryStates = new Tensor<T>[numFrequencyLevels];
            for (int i = 0; i < numFrequencyLevels; i++)
            {
                _memoryStates[i] = Tensor<T>.CreateFromArray(
                    new T[memoryDimension],
                    new[] { memoryDimension });
            }

            // Initialize decay rates - higher levels decay slower
            _decayRates = decayRates ?? CreateDefaultDecayRates();
        }

        private T[] CreateDefaultDecayRates()
        {
            var rates = new T[_numFrequencyLevels];
            for (int i = 0; i < _numFrequencyLevels; i++)
            {
                // Exponentially decreasing decay: 0.9, 0.95, 0.98, ...
                double rate = 0.9 + (i * 0.05);
                rates[i] = T.CreateChecked(Math.Min(rate, 0.99));
            }
            return rates;
        }

        /// <inheritdoc/>
        public void Store(Tensor<T> representation, int frequencyLevel)
        {
            if (frequencyLevel < 0 || frequencyLevel >= _numFrequencyLevels)
            {
                throw new ArgumentException($"Invalid frequency level: {frequencyLevel}");
            }

            // Store with exponential moving average based on decay rate
            T decay = _decayRates[frequencyLevel];
            T oneMinusDecay = T.One - decay;

            var currentMemory = _memoryStates[frequencyLevel].ToArray();
            var newRepresentation = representation.ToArray();

            for (int i = 0; i < Math.Min(currentMemory.Length, newRepresentation.Length); i++)
            {
                currentMemory[i] = currentMemory[i] * decay + newRepresentation[i] * oneMinusDecay;
            }

            _memoryStates[frequencyLevel] = Tensor<T>.CreateFromArray(
                currentMemory,
                _memoryStates[frequencyLevel].Shape);
        }

        /// <inheritdoc/>
        public Tensor<T> Retrieve(Tensor<T> query, int frequencyLevel)
        {
            if (frequencyLevel < 0 || frequencyLevel >= _numFrequencyLevels)
            {
                throw new ArgumentException($"Invalid frequency level: {frequencyLevel}");
            }

            // For now, simple retrieval returns the memory state
            // In a more advanced implementation, this could use attention mechanism
            return _memoryStates[frequencyLevel];
        }

        /// <inheritdoc/>
        public void Update(Tensor<T> context, bool[] updateMask)
        {
            if (updateMask.Length != _numFrequencyLevels)
            {
                throw new ArgumentException(
                    $"Update mask length ({updateMask.Length}) must match number of frequency levels ({_numFrequencyLevels})");
            }

            for (int i = 0; i < _numFrequencyLevels; i++)
            {
                if (updateMask[i])
                {
                    Store(context, i);
                }
            }
        }

        /// <inheritdoc/>
        public void Consolidate()
        {
            // Memory consolidation: transfer information from faster to slower levels
            for (int i = 0; i < _numFrequencyLevels - 1; i++)
            {
                var fastMemory = _memoryStates[i].ToArray();
                var slowMemory = _memoryStates[i + 1].ToArray();

                // Transfer rate decreases with level (faster levels transfer more)
                T transferRate = T.CreateChecked(0.05 / (i + 1)); // 0.05, 0.025, 0.0167, ...

                for (int j = 0; j < Math.Min(fastMemory.Length, slowMemory.Length); j++)
                {
                    // Blend fast memory into slow memory
                    slowMemory[j] = slowMemory[j] * (T.One - transferRate) + fastMemory[j] * transferRate;
                }

                _memoryStates[i + 1] = Tensor<T>.CreateFromArray(
                    slowMemory,
                    _memoryStates[i + 1].Shape);
            }
        }

        /// <inheritdoc/>
        public int NumberOfFrequencyLevels => _numFrequencyLevels;

        /// <inheritdoc/>
        public T[] DecayRates
        {
            get => _decayRates;
            set
            {
                if (value.Length != _numFrequencyLevels)
                {
                    throw new ArgumentException(
                        $"Decay rates length ({value.Length}) must match number of frequency levels ({_numFrequencyLevels})");
                }
                _decayRates = value;
            }
        }

        /// <inheritdoc/>
        public Tensor<T>[] MemoryStates => _memoryStates;

        /// <summary>
        /// Resets all memory states to zero.
        /// </summary>
        public void Reset()
        {
            for (int i = 0; i < _numFrequencyLevels; i++)
            {
                _memoryStates[i] = Tensor<T>.CreateFromArray(
                    new T[_memoryDimension],
                    new[] { _memoryDimension });
            }
        }

        /// <summary>
        /// Gets the memory dimension.
        /// </summary>
        public int MemoryDimension => _memoryDimension;
    }
}
