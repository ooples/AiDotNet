using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NestedLearning;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Continuum Memory System (CMS) layer for neural networks.
/// Implements multi-frequency memory for nested learning.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class ContinuumMemorySystemLayer<T> : LayerBase<T>
{
    private readonly int _memoryDim;
    private readonly int _numFrequencyLevels;
    private readonly ContinuumMemorySystem<T> _memorySystem;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public ContinuumMemorySystemLayer(
        int[] inputShape,
        int memoryDim,
        int numFrequencyLevels = 3,
        T[]? decayRates = null)
        : base(inputShape, new[] { memoryDim }, null, null)
    {
        _memoryDim = memoryDim;
        _numFrequencyLevels = numFrequencyLevels;
        _memorySystem = new ContinuumMemorySystem<T>(memoryDim, numFrequencyLevels, decayRates);

        // Initialize parameters as empty vector
        Parameters = new Vector<T>(0);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        LastInput = input;

        var inputVector = input.ToVector();

        // Aggregate outputs from all frequency levels
        var output = new Vector<T>(_memoryDim);

        for (int level = 0; level < _numFrequencyLevels; level++)
        {
            // Store input in memory
            _memorySystem.Store(inputVector, level);

            // Retrieve from memory
            var retrieved = _memorySystem.Retrieve(inputVector, level);

            // Accumulate
            for (int i = 0; i < _memoryDim; i++)
            {
                output[i] = _numOps.Add(output[i], retrieved[i]);
            }
        }

        LastOutput = new Tensor<T>(new[] { _memoryDim }, output);
        return LastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Simple pass-through gradient for now
        return outputGradient;
    }

    public void ConsolidateMemory()
    {
        _memorySystem.Consolidate();
    }

    public void ResetMemory()
    {
        _memorySystem.Reset();
    }

    public Vector<T>[] GetMemoryStates() => _memorySystem.MemoryStates;
}
