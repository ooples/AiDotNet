using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Continuum Memory System (CMS) layer for neural networks.
/// Implements a sequential chain of MLP blocks with different update frequencies.
/// Based on Equations 30-31 from "Nested Learning" paper.
/// yt = MLP^(fk)(MLP^(fk-1)(...MLP^(f1)(xt)))
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class ContinuumMemorySystemLayer<T> : LayerBase<T>
{
    private readonly DenseLayer<T>[] _mlpBlocks;
    private readonly int[] _updateFrequencies;
    private readonly int[] _chunkSizes;
    private readonly T[] _learningRates;
    private readonly Vector<T>[] _accumulatedGradients;
    private readonly int[] _stepCounters;
    private int _globalStep;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a CMS layer as a chain of MLP blocks.
    /// </summary>
    /// <param name="inputShape">Input shape</param>
    /// <param name="hiddenDim">Hidden dimension for each MLP block</param>
    /// <param name="numLevels">Number of frequency levels (k in the paper)</param>
    /// <param name="updateFrequencies">Update frequencies for each level (f1, f2, ..., fk)</param>
    /// <param name="learningRates">Learning rates per level</param>
    public ContinuumMemorySystemLayer(
        int[] inputShape,
        int hiddenDim,
        int numFrequencyLevels = 3,
        int[]? updateFrequencies = null,
        T[]? learningRates = null)
        : base(inputShape, new[] { hiddenDim }, null, null)
    {
        // Default update frequencies: 1, 10, 100, ...
        _updateFrequencies = updateFrequencies ?? CreateDefaultUpdateFrequencies(numFrequencyLevels);

        // Calculate chunk sizes: C(ℓ) = max_ℓ C(ℓ) / fℓ
        int maxChunkSize = _updateFrequencies[numFrequencyLevels - 1];
        _chunkSizes = new int[numFrequencyLevels];
        for (int i = 0; i < numFrequencyLevels; i++)
        {
            _chunkSizes[i] = maxChunkSize / _updateFrequencies[i];
        }

        // Default learning rates: decrease with level
        _learningRates = learningRates ?? CreateDefaultLearningRates(numFrequencyLevels);

        // Create chain of MLP blocks (DenseLayer with ReLU activation)
        _mlpBlocks = new DenseLayer<T>[numFrequencyLevels];
        int currentDim = inputShape[0];

        for (int i = 0; i < numFrequencyLevels; i++)
        {
            _mlpBlocks[i] = new DenseLayer<T>(
                inputShape: new[] { currentDim },
                outputUnits: hiddenDim,
                activation: ActivationFunction.ReLU);
            currentDim = hiddenDim;
        }

        // Initialize gradient accumulation buffers
        _accumulatedGradients = new Vector<T>[numFrequencyLevels];
        _stepCounters = new int[numFrequencyLevels];
        for (int i = 0; i < numFrequencyLevels; i++)
        {
            int paramCount = _mlpBlocks[i].Parameters.Length;
            _accumulatedGradients[i] = new Vector<T>(paramCount);
            _stepCounters[i] = 0;
        }

        _globalStep = 0;
        Parameters = new Vector<T>(0); // CMS manages its own MLP parameters
    }

    private int[] CreateDefaultUpdateFrequencies(int numLevels)
    {
        var frequencies = new int[numLevels];
        for (int i = 0; i < numLevels; i++)
        {
            frequencies[i] = (int)Math.Pow(10, i); // 1, 10, 100, 1000, ...
        }
        return frequencies;
    }

    private T[] CreateDefaultLearningRates(int numLevels)
    {
        var rates = new T[numLevels];
        double baseLR = 0.01;
        for (int i = 0; i < numLevels; i++)
        {
            double rate = baseLR / Math.Pow(10, i);
            rates[i] = _numOps.FromDouble(rate);
        }
        return rates;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        LastInput = input;
        var current = input;

        // Sequential chain: yt = MLP^(fk)(MLP^(fk-1)(...MLP^(f1)(xt)))
        for (int level = 0; level < _mlpBlocks.Length; level++)
        {
            current = _mlpBlocks[level].Forward(current);
        }

        LastOutput = current;
        _globalStep++;
        return current;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var gradient = outputGradient;

        // Backprop through chain in reverse order
        for (int level = _mlpBlocks.Length - 1; level >= 0; level--)
        {
            gradient = _mlpBlocks[level].Backward(gradient);

            // Accumulate gradients for this level
            // Equation 31: θ^(fℓ)_{i+1} = θ^(fℓ)_i - Σ η^(ℓ)_t f(θ^(fℓ)_t; xt) if i ≡ 0 (mod C(ℓ))
            var mlpGradient = _mlpBlocks[level].Gradients;
            if (mlpGradient != null)
            {
                for (int i = 0; i < mlpGradient.Length; i++)
                {
                    _accumulatedGradients[level][i] = _numOps.Add(
                        _accumulatedGradients[level][i],
                        mlpGradient[i]);
                }
            }

            _stepCounters[level]++;

            // Update parameters when step count reaches chunk size
            if (_stepCounters[level] >= _chunkSizes[level])
            {
                UpdateLevelParameters(level);
                _stepCounters[level] = 0;

                // Reset gradient accumulation
                _accumulatedGradients[level] = new Vector<T>(_accumulatedGradients[level].Length);
            }
        }

        return gradient;
    }

    private void UpdateLevelParameters(int level)
    {
        var currentParams = _mlpBlocks[level].Parameters;
        var updated = new Vector<T>(currentParams.Length);

        T learningRate = _learningRates[level];

        for (int i = 0; i < currentParams.Length; i++)
        {
            // θ^(fℓ)_{i+1} = θ^(fℓ)_i - η^(ℓ) * Σ gradients
            T update = _numOps.Multiply(_accumulatedGradients[level][i], learningRate);
            updated[i] = _numOps.Subtract(currentParams[i], update);
        }

        _mlpBlocks[level].SetParameters(updated);
    }

    /// <summary>
    /// Consolidates memory from faster to slower levels.
    /// </summary>
    public void ConsolidateMemory()
    {
        // Transfer knowledge from faster (lower level) to slower (higher level) MLPs
        for (int i = 0; i < _mlpBlocks.Length - 1; i++)
        {
            var fastParams = _mlpBlocks[i].Parameters;
            var slowParams = _mlpBlocks[i + 1].Parameters;

            int minLen = Math.Min(fastParams.Length, slowParams.Length);
            T transferRate = _numOps.FromDouble(0.01);
            T oneMinusTransfer = _numOps.Subtract(_numOps.One, transferRate);

            var consolidated = new Vector<T>(slowParams.Length);
            for (int j = 0; j < slowParams.Length; j++)
            {
                if (j < minLen)
                {
                    T slow = _numOps.Multiply(slowParams[j], oneMinusTransfer);
                    T fast = _numOps.Multiply(fastParams[j], transferRate);
                    consolidated[j] = _numOps.Add(slow, fast);
                }
                else
                {
                    consolidated[j] = slowParams[j];
                }
            }

            _mlpBlocks[i + 1].SetParameters(consolidated);
        }
    }

    /// <summary>
    /// Resets all MLP blocks in the chain.
    /// </summary>
    public void ResetMemory()
    {
        foreach (var mlp in _mlpBlocks)
        {
            mlp.Reset();
        }

        for (int i = 0; i < _accumulatedGradients.Length; i++)
        {
            _accumulatedGradients[i] = new Vector<T>(_accumulatedGradients[i].Length);
            _stepCounters[i] = 0;
        }

        _globalStep = 0;
    }

    /// <summary>
    /// Gets the MLP blocks in the chain.
    /// </summary>
    public DenseLayer<T>[] GetMLPBlocks() => _mlpBlocks;

    /// <summary>
    /// Gets the update frequencies for each level.
    /// </summary>
    public int[] UpdateFrequencies => _updateFrequencies;

    /// <summary>
    /// Gets the chunk sizes for gradient accumulation.
    /// </summary>
    public int[] ChunkSizes => _chunkSizes;
}
