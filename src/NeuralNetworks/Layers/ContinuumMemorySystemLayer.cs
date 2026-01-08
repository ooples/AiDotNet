
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

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
    private Vector<T>[] _accumulatedGradients;
    private readonly int[] _stepCounters;
    private Tensor<T>[] _storedInputs;  // Store input to each MLP block for Modified GD
    private int _globalStep;
    private Tensor<T>? LastInput;
    private Tensor<T>? LastOutput;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Indicates whether this layer supports training. CMS always supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Indicates whether this layer supports GPU execution.
    /// CMS supports GPU because it chains DenseLayer blocks which all support GPU.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Creates a CMS layer as a chain of MLP blocks.
    /// </summary>
    /// <param name="inputShape">Input shape</param>
    /// <param name="hiddenDim">Hidden dimension for each MLP block</param>
    /// <param name="numLevels">Number of frequency levels (k in the paper)</param>
    /// <param name="updateFrequencies">Update frequencies for each level (f1, f2, ..., fk)</param>
    /// <param name="learningRates">Learning rates per level</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    public ContinuumMemorySystemLayer(
        int[] inputShape,
        int hiddenDim,
        int numFrequencyLevels = 3,
        int[]? updateFrequencies = null,
        T[]? learningRates = null,
        IEngine? engine = null)
        : base(inputShape, new[] { hiddenDim })
    {

        // Validate inputs
        if (inputShape == null || inputShape.Length == 0)
            throw new ArgumentException("Input shape cannot be null or empty", nameof(inputShape));

        if (inputShape[0] <= 0)
            throw new ArgumentException("Input dimension must be positive", nameof(inputShape));

        if (hiddenDim <= 0)
            throw new ArgumentException("Hidden dimension must be positive", nameof(hiddenDim));

        if (numFrequencyLevels <= 0)
            throw new ArgumentException("Number of frequency levels must be positive", nameof(numFrequencyLevels));

        if (numFrequencyLevels > 10)
            throw new ArgumentException("Number of frequency levels should not exceed 10 for practical purposes", nameof(numFrequencyLevels));

        // Validate custom update frequencies if provided
        if (updateFrequencies != null)
        {
            if (updateFrequencies.Length != numFrequencyLevels)
                throw new ArgumentException($"Update frequencies array length ({updateFrequencies.Length}) must match numFrequencyLevels ({numFrequencyLevels})", nameof(updateFrequencies));

            for (int i = 0; i < updateFrequencies.Length; i++)
            {
                if (updateFrequencies[i] <= 0)
                    throw new ArgumentException($"Update frequency at index {i} must be positive", nameof(updateFrequencies));
            }
        }

        // Validate custom learning rates if provided
        if (learningRates != null)
        {
            if (learningRates.Length != numFrequencyLevels)
                throw new ArgumentException($"Learning rates array length ({learningRates.Length}) must match numFrequencyLevels ({numFrequencyLevels})", nameof(learningRates));
        }

        // Default update frequencies: 1, 10, 100, ...
        _updateFrequencies = updateFrequencies ?? CreateDefaultUpdateFrequencies(numFrequencyLevels);

        // Calculate chunk sizes: C(ℓ) = max_ℓ C(ℓ) / fℓ
        int maxChunkSize = _updateFrequencies[numFrequencyLevels - 1];
        _chunkSizes = new int[numFrequencyLevels];
        for (int i = 0; i < numFrequencyLevels; i++)
        {
            _chunkSizes[i] = maxChunkSize / _updateFrequencies[i];
            if (_chunkSizes[i] <= 0)
                _chunkSizes[i] = 1; // Ensure at least 1 step per chunk
        }

        // Default learning rates: decrease with level
        _learningRates = learningRates ?? CreateDefaultLearningRates(numFrequencyLevels);

        // Create chain of MLP blocks (DenseLayer with ReLU activation)
        _mlpBlocks = new DenseLayer<T>[numFrequencyLevels];
        int currentDim = inputShape[0];

        for (int i = 0; i < numFrequencyLevels; i++)
        {
            _mlpBlocks[i] = new DenseLayer<T>(currentDim, hiddenDim, (IActivationFunction<T>)new ReLUActivation<T>());
            currentDim = hiddenDim;
        }

        // Initialize gradient accumulation buffers
        _accumulatedGradients = new Vector<T>[numFrequencyLevels];
        _stepCounters = new int[numFrequencyLevels];
        for (int i = 0; i < numFrequencyLevels; i++)
        {
            int paramCount = _mlpBlocks[i].ParameterCount;
            _accumulatedGradients[i] = new Vector<T>(paramCount);
            _accumulatedGradients[i].Fill(NumOps.Zero);
            _stepCounters[i] = 0;
        }

        // Initialize stored inputs for Modified GD
        _storedInputs = new Tensor<T>[numFrequencyLevels];

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
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (input.Shape == null || input.Shape.Length == 0)
            throw new ArgumentException("Input tensor must have a valid shape", nameof(input));

        LastInput = input;
        var current = input;

        // Sequential chain: yt = MLP^(fk)(MLP^(fk-1)(...MLP^(f1)(xt)))
        for (int level = 0; level < _mlpBlocks.Length; level++)
        {
            if (_mlpBlocks[level] == null)
                throw new InvalidOperationException($"MLP block at level {level} is null");

            // Store input for Modified GD optimizer
            _storedInputs[level] = current;

            current = _mlpBlocks[level].Forward(current);

            if (current == null)
                throw new InvalidOperationException($"MLP block at level {level} returned null output");
        }

        LastOutput = current;
        _globalStep++;
        return current;
    }

    /// <summary>
    /// GPU-accelerated forward pass chaining through all MLP blocks.
    /// Each DenseLayer block handles its own GPU operations (GEMM, bias, activation).
    /// yt = MLP^(fk)(MLP^(fk-1)(...MLP^(f1)(xt)))
    /// </summary>
    /// <param name="inputs">GPU-resident input tensors (uses first input).</param>
    /// <returns>GPU-resident output tensor after chaining through all MLP blocks.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var currentGpu = inputs[0];

        // Store CPU tensor for potential backward pass
        if (IsTrainingMode)
        {
            LastInput = currentGpu.ToTensor();
        }

        // Sequential chain through all MLP blocks on GPU
        // yt = MLP^(fk)(MLP^(fk-1)(...MLP^(f1)(xt)))
        for (int level = 0; level < _mlpBlocks.Length; level++)
        {
            if (_mlpBlocks[level] == null)
                throw new InvalidOperationException($"MLP block at level {level} is null");

            // Store input for Modified GD optimizer during training
            if (IsTrainingMode)
            {
                _storedInputs[level] = currentGpu.ToTensor();
            }

            // Each DenseLayer handles its own GPU operations (GEMM + bias + activation)
            currentGpu = _mlpBlocks[level].ForwardGpu(currentGpu);
        }

        // Store output for potential backward pass
        if (IsTrainingMode)
        {
            LastOutput = currentGpu.ToTensor();
        }

        _globalStep++;
        return currentGpu;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (outputGradient == null)
            throw new ArgumentNullException(nameof(outputGradient));

        var gradient = outputGradient;

        // Backprop through chain in reverse order
        for (int level = _mlpBlocks.Length - 1; level >= 0; level--)
        {
            if (_mlpBlocks[level] == null)
                throw new InvalidOperationException($"MLP block at level {level} is null");

            gradient = _mlpBlocks[level].Backward(gradient);

            // === Vectorized Gradient Accumulation using IEngine (Phase B: US-GPU-015) ===
            // Equation 31: θ^(fℓ)_{i+1} = θ^(fℓ)_i - Σ η^(ℓ)_t f(θ^(fℓ)_t; xt) if i ≡ 0 (mod C(ℓ))
            var mlpGradient = _mlpBlocks[level].GetParameterGradients();
            if (mlpGradient != null && mlpGradient.Length > 0)
            {
                int expectedLength = _accumulatedGradients[level].Length;
                if (mlpGradient.Length != expectedLength)
                {
                    throw new InvalidOperationException(
                        $"Gradient length mismatch at level {level}: expected {expectedLength}, got {mlpGradient.Length}");
                }

                // Vectorized accumulation with engine vector ops (no tensor conversion)
                _accumulatedGradients[level] = Engine.Add(_accumulatedGradients[level], mlpGradient);
            }

            _stepCounters[level]++;

            // Update parameters when step count reaches chunk size
            if (_stepCounters[level] >= _chunkSizes[level])
            {
                UpdateLevelParameters(level);
                _stepCounters[level] = 0;

                // Reset gradient accumulation
                int paramCount = _accumulatedGradients[level].Length;
                _accumulatedGradients[level] = new Vector<T>(paramCount);
                _accumulatedGradients[level].Fill(NumOps.Zero);
            }
        }

        return gradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation by delegating to each DenseLayer's own
    /// autodiff implementation. Since ContinuumMemorySystemLayer is a composite layer that
    /// chains multiple DenseLayer instances, we enable autodiff on each block and let them
    /// compute their own gradients.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (outputGradient == null)
            throw new ArgumentNullException(nameof(outputGradient));

        var gradient = outputGradient;

        // Enable autodiff on all MLP blocks
        bool[] originalAutodiffSettings = new bool[_mlpBlocks.Length];
        for (int i = 0; i < _mlpBlocks.Length; i++)
        {
            originalAutodiffSettings[i] = _mlpBlocks[i].UseAutodiff;
            _mlpBlocks[i].UseAutodiff = true;
        }

        try
        {
            // Backprop through chain in reverse order using autodiff
            for (int level = _mlpBlocks.Length - 1; level >= 0; level--)
            {
                if (_mlpBlocks[level] == null)
                    throw new InvalidOperationException($"MLP block at level {level} is null");

                // Let the DenseLayer use its own autodiff implementation
                gradient = _mlpBlocks[level].Backward(gradient);

                // === Vectorized Gradient Accumulation using IEngine (Phase B: US-GPU-015) ===
                var mlpGradient = _mlpBlocks[level].GetParameterGradients();
                if (mlpGradient != null && mlpGradient.Length > 0)
                {
                    int expectedLength = _accumulatedGradients[level].Length;
                    if (mlpGradient.Length != expectedLength)
                    {
                        throw new InvalidOperationException(
                            $"Gradient length mismatch at level {level}: expected {expectedLength}, got {mlpGradient.Length}");
                    }

                    // Vectorized accumulation with engine vector ops (no tensor conversion)
                    _accumulatedGradients[level] = Engine.Add(_accumulatedGradients[level], mlpGradient);
                }

                _stepCounters[level]++;

                // Update parameters when step count reaches chunk size
                if (_stepCounters[level] >= _chunkSizes[level])
                {
                    UpdateLevelParameters(level);
                    _stepCounters[level] = 0;

                    // Reset gradient accumulation
                    int paramCount = _accumulatedGradients[level].Length;
                    _accumulatedGradients[level] = new Vector<T>(paramCount);
                    _accumulatedGradients[level].Fill(NumOps.Zero);
                }
            }

            return gradient;
        }
        finally
        {
            // Restore original autodiff settings
            for (int i = 0; i < _mlpBlocks.Length; i++)
            {
                _mlpBlocks[i].UseAutodiff = originalAutodiffSettings[i];
            }
        }
    }

    private void UpdateLevelParameters(int level)
    {
        if (level < 0 || level >= _mlpBlocks.Length)
            throw new ArgumentOutOfRangeException(nameof(level), $"Level {level} is out of range [0, {_mlpBlocks.Length})");

        if (_mlpBlocks[level] == null)
            throw new InvalidOperationException($"MLP block at level {level} is null");

        var currentParams = _mlpBlocks[level].GetParameters();
        if (currentParams == null || currentParams.Length == 0)
            throw new InvalidOperationException($"MLP block at level {level} has no parameters");

        int accumulatedLength = _accumulatedGradients[level].Length;
        if (currentParams.Length != accumulatedLength)
        {
            throw new InvalidOperationException(
                $"Parameter count mismatch at level {level}: params={currentParams.Length}, gradients={accumulatedLength}");
        }

        T learningRate = _learningRates[level];

        // === Vectorized Standard Gradient Descent using Engine Tensor Operations ===
        // θ^(fℓ)_{i+1} = θ^(fℓ)_i - η^(ℓ) * Σ gradients
        var scaledGrad = Engine.Multiply(_accumulatedGradients[level], learningRate);
        var updated = Engine.Subtract(currentParams, scaledGrad);

        _mlpBlocks[level].SetParameters(updated);
    }

    /// <summary>
    /// Consolidates memory from faster to slower levels.
    /// Transfers knowledge from lower-level (faster) MLPs to higher-level (slower) MLPs.
    /// </summary>
    public void ConsolidateMemory()
    {
        if (_mlpBlocks == null || _mlpBlocks.Length == 0)
            throw new InvalidOperationException("MLP blocks are not initialized");

        // Transfer knowledge from faster (lower level) to slower (higher level) MLPs
        for (int i = 0; i < _mlpBlocks.Length - 1; i++)
        {
            if (_mlpBlocks[i] == null)
                throw new InvalidOperationException($"MLP block at level {i} is null");

            if (_mlpBlocks[i + 1] == null)
                throw new InvalidOperationException($"MLP block at level {i + 1} is null");

            var fastParams = _mlpBlocks[i].GetParameters();
            var slowParams = _mlpBlocks[i + 1].GetParameters();

            if (fastParams == null || fastParams.Length == 0)
                throw new InvalidOperationException($"Fast MLP at level {i} has no parameters");

            if (slowParams == null || slowParams.Length == 0)
                throw new InvalidOperationException($"Slow MLP at level {i + 1} has no parameters");

            // === Vectorized Memory Consolidation using Engine Tensor Operations ===
            int minLen = Math.Min(fastParams.Length, slowParams.Length);
            T transferRate = _numOps.FromDouble(0.01);
            T oneMinusTransfer = _numOps.Subtract(_numOps.One, transferRate);

            // Convert to tensors for Engine operations
            var consolidated = new Vector<T>(slowParams.Length);

            if (minLen > 0 && minLen == slowParams.Length && minLen == fastParams.Length)
            {
                // Same size - use full tensor operations
                // consolidated = slow * (1 - rate) + fast * rate
                var slowScaled = Engine.Multiply(slowParams, oneMinusTransfer);
                var fastScaled = Engine.Multiply(fastParams, transferRate);
                consolidated = Engine.Add(slowScaled, fastScaled);
            }
            else if (minLen > 0)
            {
                // Different sizes - handle overlapping portion
                // For simplicity, copy slow params first, then blend overlapping portion
                for (int j = 0; j < slowParams.Length; j++)
                {
                    consolidated[j] = slowParams[j];
                }

                // Blend overlapping portion
                for (int j = 0; j < minLen; j++)
                {
                    T slowVal = NumOps.Multiply(slowParams[j], oneMinusTransfer);
                    T fastVal = NumOps.Multiply(fastParams[j], transferRate);
                    consolidated[j] = NumOps.Add(slowVal, fastVal);
                }
            }
            else
            {
                // No overlap - just copy slow params
                for (int j = 0; j < slowParams.Length; j++)
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
        if (_mlpBlocks == null)
            throw new InvalidOperationException("MLP blocks are not initialized");

        foreach (var mlp in _mlpBlocks)
        {
            if (mlp == null)
                throw new InvalidOperationException("MLP block is null");

            mlp.ResetState();
        }

        for (int i = 0; i < _accumulatedGradients.Length; i++)
        {
            int paramCount = _accumulatedGradients[i].Length;
            _accumulatedGradients[i] = new Vector<T>(paramCount);
            _accumulatedGradients[i].Fill(NumOps.Zero);
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

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// This is a no-op for CMS because parameters are updated exclusively via UpdateLevelParameters
    /// when chunk counters trigger (i ≡ 0 mod C(ℓ)). Updating here would double-apply gradients.
    /// </summary>
    /// <param name="learningRate">Learning rate (unused - each level has its own rate)</param>
    public override void UpdateParameters(T learningRate)
    {
        // No-op: Parameters are updated via UpdateLevelParameters during Backward pass
        // when chunk counters reach their thresholds. Updating here would cause
        // double application of gradients since MLP blocks are already updated
        // in UpdateLevelParameters using Modified Gradient Descent (Equations 27-29).
    }

    /// <summary>
    /// Gets all parameters from all MLP blocks in the chain.
    /// Returns a concatenated vector of all parameters from all levels.
    /// </summary>
    /// <returns>Concatenated parameter vector</returns>
    public override Vector<T> GetParameters()
    {
        if (_mlpBlocks == null || _mlpBlocks.Length == 0)
            throw new InvalidOperationException("MLP blocks are not initialized");

        // Use Vector<T>.Concatenate for efficient parameter collection
        Vector<T> result = Vector<T>.Empty();

        foreach (var mlp in _mlpBlocks)
        {
            if (mlp == null)
                throw new InvalidOperationException("MLP block is null");

            var mlpParams = mlp.GetParameters();
            result = Vector<T>.Concatenate(result, mlpParams);
        }

        return result;
    }

    /// <summary>
    /// Sets all parameters for all MLP blocks in the chain.
    /// Distributes the parameter vector across all levels.
    /// </summary>
    /// <param name="parameters">Concatenated parameter vector</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        if (_mlpBlocks == null || _mlpBlocks.Length == 0)
            throw new InvalidOperationException("MLP blocks are not initialized");

        // Calculate total expected parameter count
        int totalParams = 0;
        foreach (var mlp in _mlpBlocks)
        {
            if (mlp == null)
                throw new InvalidOperationException("MLP block is null");

            totalParams += mlp.ParameterCount;
        }

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) does not match total parameters ({totalParams})",
                nameof(parameters));
        }

        // Distribute parameters to each MLP block
        int offset = 0;
        foreach (var mlp in _mlpBlocks)
        {
            int mlpParamCount = mlp.ParameterCount;
            var mlpParams = new Vector<T>(mlpParamCount);

            for (int i = 0; i < mlpParamCount; i++)
            {
                mlpParams[i] = parameters[offset + i];
            }

            mlp.SetParameters(mlpParams);
            offset += mlpParamCount;
        }

        // Update the base class Parameters property
        Parameters = parameters;
    }

    /// <summary>
    /// Resets the state of the layer (required by LayerBase).
    /// Resets all MLP blocks and clears gradient accumulation.
    /// </summary>
    public override void ResetState()
    {
        ResetMemory(); // Use existing ResetMemory implementation
    }

    /// <summary>
    /// Gets the parameter gradients for all MLP blocks.
    /// Returns concatenated gradients from all levels.
    /// </summary>
    public override Vector<T> GetParameterGradients()
    {
        if (_mlpBlocks == null || _mlpBlocks.Length == 0)
            throw new InvalidOperationException("MLP blocks are not initialized");

        // Calculate total parameter count
        int totalParams = 0;
        foreach (var mlp in _mlpBlocks)
        {
            if (mlp == null)
                throw new InvalidOperationException("MLP block is null");

            totalParams += mlp.ParameterCount;
        }

        // Concatenate all accumulated gradients
        var allGradients = new Vector<T>(totalParams);
        int offset = 0;

        for (int level = 0; level < _mlpBlocks.Length; level++)
        {
            var accGrad = _accumulatedGradients[level];
            for (int i = 0; i < accGrad.Length; i++)
            {
                allGradients[offset + i] = accGrad[i];
            }
            offset += accGrad.Length;
        }

        return allGradients;
    }

    /// <summary>
    /// Clears all accumulated gradients across all levels.
    /// </summary>
    public override void ClearGradients()
    {
        if (_mlpBlocks == null || _mlpBlocks.Length == 0)
            throw new InvalidOperationException("MLP blocks are not initialized");

        // Clear gradients in all MLP blocks
        foreach (var mlp in _mlpBlocks)
        {
            if (mlp == null)
                throw new InvalidOperationException("MLP block is null");

            mlp.ClearGradients();
        }

        // Clear accumulated gradients
        for (int i = 0; i < _accumulatedGradients.Length; i++)
        {
            int paramCount = _accumulatedGradients[i].Length;
            _accumulatedGradients[i] = new Vector<T>(paramCount);
            _accumulatedGradients[i].Fill(NumOps.Zero);
        }
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_mlpBlocks == null || _mlpBlocks.Length == 0)
            throw new InvalidOperationException("MLP blocks are not initialized.");

        // ContinuumMemorySystemLayer is a chain of DenseLayer (MLP) blocks
        // Since DenseLayer supports JIT compilation, we can chain them together
        // The update frequencies are only relevant during training, not inference

        var current = inputNodes[0];

        // Chain through all MLP blocks: yt = MLP^(fk)(MLP^(fk-1)(...MLP^(f1)(xt)))
        for (int level = 0; level < _mlpBlocks.Length; level++)
        {
            if (_mlpBlocks[level] == null)
                throw new InvalidOperationException($"MLP block at level {level} is null.");

            current = _mlpBlocks[level].ExportComputationGraph([current]);
        }

        return current;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> because ContinuumMemorySystemLayer is a chain of DenseLayer blocks,
    /// each of which supports JIT compilation. The update frequency logic is only used
    /// during training and does not affect inference.
    /// </value>
    public override bool SupportsJitCompilation => true;

}
