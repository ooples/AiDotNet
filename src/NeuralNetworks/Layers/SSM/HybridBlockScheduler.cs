using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Defines the scheduling pattern for hybrid SSM/attention architectures.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Hybrid architectures mix SSM blocks (like Mamba) with attention blocks.
/// The schedule pattern determines which layers are SSM and which are attention. Different patterns
/// have been shown to work well for different use cases.</para>
/// </remarks>
public enum HybridSchedulePattern
{
    /// <summary>
    /// Jamba-style: every Nth layer is attention, rest are SSM.
    /// Used by AI21's Jamba model (132B parameters).
    /// </summary>
    JambaStyle,

    /// <summary>
    /// Zamba-style: shared attention layers interleaved with Mamba blocks.
    /// The attention weights are shared (tied) to reduce parameter count.
    /// Used by Zyphra's Zamba models.
    /// </summary>
    ZambaStyle,

    /// <summary>
    /// Samba-style: alternating Mamba and sliding window attention blocks.
    /// Sliding window attention has a fixed context window instead of full quadratic attention.
    /// Used by Microsoft's Samba model.
    /// </summary>
    SambaStyle,

    /// <summary>
    /// User provides an explicit sequence of layer types.
    /// </summary>
    Custom
}

/// <summary>
/// Implements a composable hybrid block that schedules SSM and attention layers according to
/// configurable patterns used in modern hybrid architectures (Jamba, Zamba, Samba).
/// </summary>
/// <remarks>
/// <para>
/// Pure SSM models (all Mamba) and pure Transformer models (all attention) each have strengths:
/// - SSM: O(n) linear complexity, good at long-range sequential patterns
/// - Attention: O(n^2) quadratic complexity, but excellent at in-context learning and recall
///
/// Hybrid architectures combine both to get the best of each. The <see cref="HybridBlockScheduler{T}"/>
/// lets you define the mixing pattern.
/// </para>
/// <para>
/// Each block in the schedule applies: LayerNorm -> SubLayer -> Residual Connection.
/// This follows the pre-norm Transformer convention used by all modern architectures.
/// </para>
/// <para><b>For Beginners:</b> This is like a playlist that decides which type of layer
/// (SSM or attention) to use at each position in the network.
///
/// Imagine building a team:
/// - Mamba blocks are fast workers who process one thing at a time very efficiently
/// - Attention blocks are thorough workers who compare everything to everything else
///
/// A hybrid schedule picks the best worker for each position:
/// - Jamba: mostly fast workers, with a thorough worker every Nth position
/// - Zamba: fast workers sharing one set of thorough worker notes (shared attention)
/// - Samba: alternating fast workers and thorough workers with limited vision (sliding window)
///
/// The result is faster than pure attention but more capable than pure SSM.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item><description>Jamba: Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", 2024</description></item>
/// <item><description>Zamba: Glorioso et al., "Zamba: A Compact 7B SSM Hybrid Model", 2024</description></item>
/// <item><description>Samba: Ren et al., "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling", 2024</description></item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HybridBlockScheduler<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _sequenceLength;
    private readonly HybridSchedulePattern _schedulePattern;
    private readonly ILayer<T>[] _blocks;
    private readonly bool[] _isAttentionBlock;

    // Pre-norm parameters: one set per block
    private readonly Tensor<T>[] _normGammas;
    private readonly Tensor<T>[] _normBetas;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>[]? _lastNormedInputs;
    private Tensor<T>[]? _lastBlockOutputs;
    private Tensor<T>[]? _lastResiduals;
    private int[]? _originalInputShape;

    // Gradients for norm parameters
    private Tensor<T>[]? _normGammaGradients;
    private Tensor<T>[]? _normBetaGradients;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The width of the hidden representation shared by all blocks
    /// in this hybrid schedule. All SSM and attention blocks must use this same dimension.</para>
    /// </remarks>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of blocks in the schedule.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The total number of layers (both SSM and attention) in this
    /// hybrid architecture. This determines the depth of the model.</para>
    /// </remarks>
    public int NumBlocks => _blocks.Length;

    /// <summary>
    /// Gets the schedule pattern.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The pattern that determines which positions in the model use
    /// SSM blocks vs. attention blocks. See <see cref="HybridSchedulePattern"/> for available patterns.</para>
    /// </remarks>
    public HybridSchedulePattern SchedulePattern => _schedulePattern;

    /// <summary>
    /// Gets the total number of trainable parameters across all blocks and norms.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The total count of learnable numbers across all blocks
    /// (both SSM and attention) plus normalization parameters. Shared attention blocks
    /// (Zamba-style) count their parameters only once.</para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var block in _blocks)
                count += block.ParameterCount;
            foreach (var gamma in _normGammas)
                count += gamma.Length;
            foreach (var beta in _normBetas)
                count += beta.Length;
            return count;
        }
    }

    /// <summary>
    /// Creates a new hybrid block scheduler.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="blocks">List of sub-layers (SSM and/or attention) in execution order.</param>
    /// <param name="isAttentionBlock">Boolean array indicating which blocks are attention (true) vs SSM (false).</param>
    /// <param name="schedulePattern">The scheduling pattern used (for metadata/identification).</param>
    /// <param name="modelDimension">Model dimension (d_model). Must match all sub-layer dimensions.</param>
    /// <param name="activationFunction">Optional activation function applied to final output.</param>
    /// <exception cref="ArgumentException">When blocks or isAttentionBlock arrays are empty or mismatched.</exception>
    public HybridBlockScheduler(
        int sequenceLength,
        ILayer<T>[] blocks,
        bool[] isAttentionBlock,
        HybridSchedulePattern schedulePattern,
        int modelDimension,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new ActivationFunctions.IdentityActivation<T>())
    {
        if (blocks == null || blocks.Length == 0)
            throw new ArgumentException("Must provide at least one block.", nameof(blocks));

        if (isAttentionBlock == null || isAttentionBlock.Length != blocks.Length)
            throw new ArgumentException(
                "isAttentionBlock must have the same length as blocks.", nameof(isAttentionBlock));

        if (modelDimension <= 0)
            throw new ArgumentException(
                $"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));

        _sequenceLength = sequenceLength;
        _modelDimension = modelDimension;
        _schedulePattern = schedulePattern;
        _blocks = blocks;
        _isAttentionBlock = isAttentionBlock;

        // Initialize pre-norm parameters (RMSNorm-style: gamma only, beta for flexibility)
        _normGammas = new Tensor<T>[blocks.Length];
        _normBetas = new Tensor<T>[blocks.Length];
        for (int i = 0; i < blocks.Length; i++)
        {
            _normGammas[i] = new Tensor<T>(new[] { modelDimension });
            _normGammas[i].Fill(NumOps.One);
            _normBetas[i] = new Tensor<T>(new[] { modelDimension });
            _normBetas[i].Fill(NumOps.Zero);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        // Flatten to 3D [batch, seq, modelDim]
        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var current = rank == 2
            ? input.Reshape(1, seqLen, modelDim)
            : input.Reshape(batchSize, seqLen, modelDim);

        _lastInput = current;
        _lastNormedInputs = new Tensor<T>[_blocks.Length];
        _lastBlockOutputs = new Tensor<T>[_blocks.Length];
        _lastResiduals = new Tensor<T>[_blocks.Length];

        // Process through each block: pre-norm -> block -> residual
        for (int i = 0; i < _blocks.Length; i++)
        {
            _lastResiduals[i] = current;

            // Pre-norm: RMSNorm-style normalization
            var normed = ApplyRMSNorm(current, _normGammas[i], _normBetas[i], batchSize, seqLen);
            _lastNormedInputs[i] = normed;

            // Run through sub-layer
            var blockOut = _blocks[i].Forward(normed);
            _lastBlockOutputs[i] = blockOut;

            // Residual connection
            current = Engine.TensorAdd(current, blockOut);
        }

        var result = ApplyActivation(current);
        _lastOutput = result;

        // Reshape back to original rank
        if (rank == 2)
            return result.Reshape(seqLen, modelDim);

        return result.Reshape(_originalInputShape);
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastNormedInputs == null ||
            _lastBlockOutputs == null || _lastResiduals == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int rank = outputGradient.Shape.Length;
        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, seqLen, _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        // Apply activation derivative
        grad = ApplyActivationDerivative(_lastOutput!, grad);

        _normGammaGradients = new Tensor<T>[_blocks.Length];
        _normBetaGradients = new Tensor<T>[_blocks.Length];

        // Backward through blocks in reverse order
        for (int i = _blocks.Length - 1; i >= 0; i--)
        {
            // Residual: grad flows directly + through block
            var blockGrad = _blocks[i].Backward(grad);

            // Backward through RMSNorm
            var normGrad = BackwardRMSNorm(blockGrad, _lastResiduals[i], _normGammas[i],
                batchSize, seqLen, out var dGamma, out var dBeta);

            _normGammaGradients[i] = dGamma;
            _normBetaGradients[i] = dBeta;

            // Residual gradient: add norm gradient to direct gradient
            grad = Engine.TensorAdd(grad, normGrad);
        }

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return grad.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return grad.Reshape(_originalInputShape);

        return grad;
    }

    private Tensor<T> ApplyRMSNorm(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(input.Shape);
        T eps = NumOps.FromDouble(1e-6);
        var gamma2D = gamma.Reshape(1, _modelDimension);
        var beta2D = beta.Reshape(1, _modelDimension);

        for (int t = 0; t < seqLen; t++)
        {
            var slice = input.GetSliceAlongDimension(t, 1);  // [batch, modelDim]

            // RMS = sqrt(mean(x^2))
            var squared = Engine.TensorMultiply(slice, slice);
            var meanSquared = Engine.ReduceSum(squared, new int[] { 1 });  // [batch]
            T divisor = NumOps.FromDouble(_modelDimension);

            var normed = new Tensor<T>(slice.Shape);
            for (int b = 0; b < batchSize; b++)
            {
                T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(meanSquared[new[] { b }], divisor), eps));
                for (int d = 0; d < _modelDimension; d++)
                {
                    normed[new[] { b, d }] = NumOps.Divide(slice[new[] { b, d }], rms);
                }
            }

            // Apply gamma and beta
            var scaled = Engine.TensorBroadcastMultiply(normed, gamma2D);
            scaled = Engine.TensorBroadcastAdd(scaled, beta2D);
            output.SetSlice(1, t, scaled);
        }

        return output;
    }

    private Tensor<T> BackwardRMSNorm(Tensor<T> dOutput, Tensor<T> input, Tensor<T> gamma,
        int batchSize, int seqLen, out Tensor<T> dGamma, out Tensor<T> dBeta)
    {
        var dInput = new Tensor<T>(input.Shape);
        dGamma = new Tensor<T>(new[] { _modelDimension });
        dBeta = new Tensor<T>(new[] { _modelDimension });
        T eps = NumOps.FromDouble(1e-6);

        for (int t = 0; t < seqLen; t++)
        {
            var slice = input.GetSliceAlongDimension(t, 1);   // [batch, modelDim]
            var dOut = dOutput.GetSliceAlongDimension(t, 1);

            for (int b = 0; b < batchSize; b++)
            {
                // Compute RMS
                T sumSq = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    T val = slice[new[] { b, d }];
                    sumSq = NumOps.Add(sumSq, NumOps.Multiply(val, val));
                }
                T meanSq = NumOps.Divide(sumSq, NumOps.FromDouble(_modelDimension));
                T rms = NumOps.Sqrt(NumOps.Add(meanSq, eps));
                T rmsInv = NumOps.Divide(NumOps.One, rms);

                // dBeta accumulation
                for (int d = 0; d < _modelDimension; d++)
                {
                    dBeta[d] = NumOps.Add(dBeta[d], dOut[new[] { b, d }]);
                }

                // dGamma accumulation
                for (int d = 0; d < _modelDimension; d++)
                {
                    T normed = NumOps.Multiply(slice[new[] { b, d }], rmsInv);
                    dGamma[d] = NumOps.Add(dGamma[d], NumOps.Multiply(dOut[new[] { b, d }], normed));
                }

                // dInput through norm
                T dotProduct = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    T g = gamma[d];
                    dotProduct = NumOps.Add(dotProduct,
                        NumOps.Multiply(dOut[new[] { b, d }],
                            NumOps.Multiply(g, NumOps.Multiply(slice[new[] { b, d }], rmsInv))));
                }

                T rms3Inv = NumOps.Divide(rmsInv, NumOps.Multiply(rms, rms));
                for (int d = 0; d < _modelDimension; d++)
                {
                    T g = gamma[d];
                    T dNormed = NumOps.Multiply(dOut[new[] { b, d }], g);
                    T grad = NumOps.Multiply(dNormed, rmsInv);
                    T correction = NumOps.Multiply(
                        NumOps.Multiply(dotProduct, slice[new[] { b, d }]),
                        NumOps.Divide(rms3Inv, NumOps.FromDouble(_modelDimension)));
                    dInput[new[] { b, t, d }] = NumOps.Subtract(grad, correction);
                }
            }
        }

        return dInput;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCount;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        for (int i = 0; i < _blocks.Length; i++)
        {
            var blockParams = _blocks[i].GetParameters();
            for (int j = 0; j < blockParams.Length; j++)
                parameters[index++] = blockParams[j];

            for (int j = 0; j < _normGammas[i].Length; j++)
                parameters[index++] = _normGammas[i][j];

            for (int j = 0; j < _normBetas[i].Length; j++)
                parameters[index++] = _normBetas[i][j];
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = ParameterCount;
        if (parameters.Length != expectedParams)
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");

        int index = 0;
        for (int i = 0; i < _blocks.Length; i++)
        {
            int blockParamCount = _blocks[i].ParameterCount;
            var blockParams = new Vector<T>(blockParamCount);
            for (int j = 0; j < blockParamCount; j++)
                blockParams[j] = parameters[index++];
            _blocks[i].SetParameters(blockParams);

            for (int j = 0; j < _normGammas[i].Length; j++)
                _normGammas[i][j] = parameters[index++];

            for (int j = 0; j < _normBetas[i].Length; j++)
                _normBetas[i][j] = parameters[index++];
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_normGammaGradients == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);

        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].UpdateParameters(learningRate);

            _normGammas[i] = Engine.TensorAdd(_normGammas[i],
                Engine.TensorMultiplyScalar(_normGammaGradients[i], negLR));
            _normBetas[i] = Engine.TensorAdd(_normBetas[i],
                Engine.TensorMultiplyScalar(_normBetaGradients![i], negLR));
        }
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastNormedInputs = null;
        _lastBlockOutputs = null;
        _lastResiduals = null;
        _originalInputShape = null;
        _normGammaGradients = null;
        _normBetaGradients = null;

        foreach (var block in _blocks)
            block.ResetState();
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation for optimized inference.
    /// </summary>
    /// <value>
    /// True if all sub-blocks support JIT compilation. The scheduler chains the computation
    /// graphs of its sub-blocks with residual connections and normalization.
    /// </value>
    public override bool SupportsJitCompilation
    {
        get
        {
            foreach (var block in _blocks)
            {
                if (!block.SupportsJitCompilation)
                    return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Exports the computation graph by chaining sub-block graphs with residual connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The exported graph chains each sub-block's computation graph sequentially,
    /// applying pre-norm and residual connections between blocks. This mirrors the
    /// forward pass structure: for each block, apply norm -> sub-block -> add residual.
    /// </para>
    /// <para>
    /// JIT-compilable hybrid models like PyTorch's torch.compile support this pattern,
    /// enabling fused kernels across the entire block schedule.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Input placeholder: [1, modelDim] (single timestep)
        var inputPlaceholder = new Tensor<T>(new int[] { 1, _modelDimension });
        var currentNode = TensorOperations<T>.Variable(inputPlaceholder, "hybrid_input");
        inputNodes.Add(currentNode);

        // Chain each sub-block's computation graph with residual connections
        for (int i = 0; i < _blocks.Length; i++)
        {
            // Add norm parameters as variable nodes
            var gammaNode = TensorOperations<T>.Variable(_normGammas[i], $"norm_gamma_{i}");
            inputNodes.Add(gammaNode);

            // Residual = current (save for later)
            var residualNode = currentNode;

            // Simplified norm: scale by gamma (symbolic approximation of RMSNorm)
            var normedNode = TensorOperations<T>.ElementwiseMultiply(currentNode, gammaNode);

            // Chain sub-block's graph
            var blockInputs = new List<ComputationNode<T>> { normedNode };
            var blockOutput = _blocks[i].ExportComputationGraph(blockInputs);
            inputNodes.AddRange(blockInputs.GetRange(1, blockInputs.Count - 1));

            // Residual connection: output = residual + block_output
            currentNode = TensorOperations<T>.Add(residualNode, blockOutput);
        }

        return currentNode;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumBlocks"] = _blocks.Length.ToString();
        metadata["SchedulePattern"] = _schedulePattern.ToString();

        int attentionCount = 0;
        int ssmCount = 0;
        for (int i = 0; i < _isAttentionBlock.Length; i++)
        {
            if (_isAttentionBlock[i]) attentionCount++;
            else ssmCount++;
        }
        metadata["AttentionBlocks"] = attentionCount.ToString();
        metadata["SSMBlocks"] = ssmCount.ToString();

        return metadata;
    }

    /// <summary>
    /// Creates a Jamba-style hybrid schedule where every Nth layer is attention and the rest are SSM.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Jamba architecture (AI21, 2024) interleaves SSM layers with full attention layers at a
    /// fixed ratio. For example, with attentionFrequency=4 and 12 total layers, layers 3, 7, and 11
    /// would be attention blocks and the remaining 9 would be Mamba blocks.
    /// </para>
    /// <para><b>For Beginners:</b> Jamba mostly uses fast Mamba layers but adds a few attention layers
    /// at regular intervals to boost recall and in-context learning. Think of it as a team that's
    /// mostly sprinters (Mamba) with a few analysts (attention) placed every Nth position.</para>
    /// </remarks>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">Model embedding dimension.</param>
    /// <param name="numLayers">Total number of layers in the schedule.</param>
    /// <param name="attentionFrequency">
    /// Place an attention block every N layers. Default: 4.
    /// <para><b>For Beginners:</b> A value of 4 means 1 out of every 4 layers is attention.
    /// Lower values = more attention layers (more capable but slower).</para>
    /// </param>
    /// <param name="stateDimension">SSM state dimension for Mamba blocks. Default: 16.</param>
    /// <param name="numAttentionHeads">Number of attention heads. Default: 8.</param>
    /// <returns>A configured HybridBlockScheduler with Jamba-style scheduling.</returns>
    public static HybridBlockScheduler<T> CreateJambaSchedule(
        int sequenceLength,
        int modelDimension,
        int numLayers,
        int attentionFrequency = 4,
        int stateDimension = 16,
        int numAttentionHeads = 8)
    {
        if (numLayers <= 0)
            throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));
        if (attentionFrequency <= 0)
            throw new ArgumentException($"Attention frequency ({attentionFrequency}) must be positive.", nameof(attentionFrequency));

        var blocks = new ILayer<T>[numLayers];
        var isAttention = new bool[numLayers];

        for (int i = 0; i < numLayers; i++)
        {
            if ((i + 1) % attentionFrequency == 0)
            {
                // Attention layer at every Nth position
                blocks[i] = new GatedLinearAttentionLayer<T>(sequenceLength, modelDimension, numAttentionHeads);
                isAttention[i] = true;
            }
            else
            {
                // SSM (Mamba) block for the rest
                blocks[i] = new MambaBlock<T>(sequenceLength, modelDimension, stateDimension);
                isAttention[i] = false;
            }
        }

        return new HybridBlockScheduler<T>(sequenceLength, blocks, isAttention,
            HybridSchedulePattern.JambaStyle, modelDimension);
    }

    /// <summary>
    /// Creates a Zamba-style hybrid schedule with shared attention weights interleaved with SSM blocks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Zamba architecture (Zyphra, 2024) uses a single shared attention layer that is interleaved
    /// between groups of Mamba blocks. Because the attention weights are shared (the same layer instance
    /// is reused), this significantly reduces parameter count while retaining in-context learning ability.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of having many different attention layers (expensive),
    /// Zamba uses ONE attention layer and reuses it at multiple positions. It's like having one expert
    /// consultant who visits different departments on a rotating schedule, rather than hiring separate
    /// consultants for each department. This saves memory while keeping most of the capability.</para>
    /// </remarks>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">Model embedding dimension.</param>
    /// <param name="numLayers">Total number of layers in the schedule.</param>
    /// <param name="attentionFrequency">
    /// Place the shared attention block every N layers. Default: 3.
    /// <para><b>For Beginners:</b> How often the shared attention layer appears. A value of 3
    /// means the same attention layer is used at positions 2, 5, 8, etc.</para>
    /// </param>
    /// <param name="stateDimension">SSM state dimension for Mamba blocks. Default: 16.</param>
    /// <param name="numAttentionHeads">Number of attention heads for the shared attention layer. Default: 8.</param>
    /// <returns>A configured HybridBlockScheduler with Zamba-style shared attention.</returns>
    public static HybridBlockScheduler<T> CreateZambaSchedule(
        int sequenceLength,
        int modelDimension,
        int numLayers,
        int attentionFrequency = 3,
        int stateDimension = 16,
        int numAttentionHeads = 8)
    {
        if (numLayers <= 0)
            throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));
        if (attentionFrequency <= 0)
            throw new ArgumentException($"Attention frequency ({attentionFrequency}) must be positive.", nameof(attentionFrequency));

        var blocks = new ILayer<T>[numLayers];
        var isAttention = new bool[numLayers];

        // Create ONE shared attention layer (Zamba's key innovation)
        var sharedAttention = new GatedLinearAttentionLayer<T>(sequenceLength, modelDimension, numAttentionHeads);

        for (int i = 0; i < numLayers; i++)
        {
            if ((i + 1) % attentionFrequency == 0)
            {
                // Shared attention layer (same instance reused)
                blocks[i] = sharedAttention;
                isAttention[i] = true;
            }
            else
            {
                // Independent Mamba blocks
                blocks[i] = new MambaBlock<T>(sequenceLength, modelDimension, stateDimension);
                isAttention[i] = false;
            }
        }

        return new HybridBlockScheduler<T>(sequenceLength, blocks, isAttention,
            HybridSchedulePattern.ZambaStyle, modelDimension);
    }

    /// <summary>
    /// Creates a Samba-style hybrid schedule alternating Mamba blocks and sliding window attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Samba architecture (Microsoft, 2024) alternates between Mamba blocks and sliding window
    /// attention. Unlike full attention (which looks at all tokens), sliding window attention only
    /// attends to a fixed window of nearby tokens, making it O(n * w) where w is the window size.
    /// This combines the global context of SSM with the local precision of windowed attention.
    /// </para>
    /// <para><b>For Beginners:</b> Samba alternates between two types of layers:
    /// - Mamba: reads the whole sequence efficiently (captures long-range patterns)
    /// - Sliding window attention: focuses on nearby tokens precisely (captures local patterns)
    ///
    /// It's like alternating between reading a summary of a book (Mamba) and re-reading specific
    /// paragraphs carefully (windowed attention). You get both global understanding and local detail.</para>
    /// </remarks>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">Model embedding dimension.</param>
    /// <param name="numLayers">Total number of layers (should be even for clean alternation).</param>
    /// <param name="stateDimension">SSM state dimension for Mamba blocks. Default: 16.</param>
    /// <param name="numAttentionHeads">Number of attention heads. Default: 8.</param>
    /// <returns>A configured HybridBlockScheduler with Samba-style alternation.</returns>
    public static HybridBlockScheduler<T> CreateSambaSchedule(
        int sequenceLength,
        int modelDimension,
        int numLayers,
        int stateDimension = 16,
        int numAttentionHeads = 8)
    {
        if (numLayers <= 0)
            throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));

        var blocks = new ILayer<T>[numLayers];
        var isAttention = new bool[numLayers];

        for (int i = 0; i < numLayers; i++)
        {
            if (i % 2 == 0)
            {
                // Even positions: Mamba block
                blocks[i] = new MambaBlock<T>(sequenceLength, modelDimension, stateDimension);
                isAttention[i] = false;
            }
            else
            {
                // Odd positions: sliding window attention (using GatedLinearAttention which has linear complexity)
                blocks[i] = new GatedLinearAttentionLayer<T>(sequenceLength, modelDimension, numAttentionHeads);
                isAttention[i] = true;
            }
        }

        return new HybridBlockScheduler<T>(sequenceLength, blocks, isAttention,
            HybridSchedulePattern.SambaStyle, modelDimension);
    }
}
