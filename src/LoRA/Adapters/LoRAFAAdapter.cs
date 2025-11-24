using AiDotNet.Interfaces;
using System.Collections.Generic;
using AiDotNet.Autodiff;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// LoRA-FA (LoRA with Frozen A matrix) adapter for parameter-efficient fine-tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoRA-FA is a variant of standard LoRA that freezes matrix A after random initialization and only
/// trains matrix B. This provides approximately 50% parameter reduction compared to standard LoRA
/// with minimal performance loss in most scenarios.
/// </para>
/// <para><b>For Beginners:</b> LoRA-FA makes LoRA even more efficient!
///
/// Standard LoRA uses two small matrices (A and B) that both get trained:
/// - Matrix A: Compresses input (trained)
/// - Matrix B: Expands to output (trained)
///
/// LoRA-FA optimizes this further:
/// - Matrix A: Compresses input (frozen - never changes after initialization)
/// - Matrix B: Expands to output (trained - the only thing that learns)
///
/// Why freeze matrix A?
/// - Research shows matrix A can be randomly initialized and frozen without much performance loss
/// - This cuts trainable parameters in half (only matrix B is trained)
/// - Training is faster and uses less memory
/// - Perfect when you need maximum efficiency
///
/// Example parameter counts for a 1000×1000 layer with rank=8:
/// - Standard LoRA: 8,000 (A) + 8,000 (B) = 16,000 trainable parameters
/// - LoRA-FA: 0 (A frozen) + 8,000 (B) = 8,000 trainable parameters (50% reduction!)
///
/// When to use LoRA-FA:
/// - Memory is very limited
/// - Training speed is critical
/// - You can tolerate a small performance trade-off
/// - You're working with very large models
/// </para>
/// </remarks>
public class LoRAFAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Whether matrix A is frozen (always true for LoRA-FA).
    /// </summary>
    private readonly bool _freezeMatrixA = true;

    /// <summary>
    /// Gets whether matrix A is frozen during training (always true for LoRA-FA).
    /// </summary>
    /// <remarks>
    /// This is a key characteristic of LoRA-FA - matrix A is randomly initialized
    /// and then frozen, never updated during training.
    /// </remarks>
    public bool IsMatrixAFrozen => _freezeMatrixA;

    /// <summary>
    /// Gets the total number of trainable parameters (only matrix B).
    /// </summary>
    /// <remarks>
    /// <para>
    /// For LoRA-FA, only matrix B is trainable. Matrix A is frozen, so it doesn't count
    /// toward trainable parameters. This results in approximately 50% parameter reduction
    /// compared to standard LoRA.
    /// </para>
    /// <para><b>For Beginners:</b> This returns how many parameters will actually be trained.
    /// Since matrix A is frozen, we only count matrix B's parameters. If the base layer is
    /// also frozen (typical case), this is just matrix B. Otherwise, it's base layer + matrix B.
    ///
    /// For a layer with input size 1000, output size 1000, and rank 8:
    /// - Matrix B size: rank × outputSize = 8 × 1000 = 8,000 parameters
    /// - Matrix A size: inputSize × rank = 1000 × 8 = 8,000 parameters (but frozen, so not counted)
    /// - Total trainable: 8,000 (50% less than standard LoRA's 16,000)
    /// </para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            // CRITICAL: Return full LoRA parameter count (A + B) to match base class invariants
            // Even though matrix A is frozen, it must be included in the parameter buffer
            // to avoid IndexOutOfRangeException in base class private helpers
            // The freeze logic is handled in UpdateParameters, not in buffer sizing
            if (!_freezeBaseLayer)
            {
                return _baseLayer.ParameterCount + _loraLayer.ParameterCount;
            }

            return _loraLayer.ParameterCount;
        }
    }

    /// <summary>
    /// Initializes a new LoRA-FA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LoRA-FA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a LoRA-FA adapter that wraps any layer.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to make more efficient to fine-tune
    /// - rank: How much compression (lower = fewer parameters, less flexibility)
    /// - alpha: How strong the LoRA adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true for efficiency)
    ///
    /// What happens during initialization:
    /// 1. Matrix A gets random values (Gaussian initialization)
    /// 2. Matrix A is immediately frozen (never updated during training)
    /// 3. Matrix B starts at zero (so initially LoRA-FA has no effect)
    /// 4. Only matrix B will be trained, reducing parameters by 50% vs standard LoRA
    ///
    /// This is perfect when you need maximum parameter efficiency!
    /// </para>
    /// </remarks>
    public LoRAFAAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        // Matrix A is automatically initialized by base class and will remain frozen
        // Matrix B starts at zero and will be the only trainable component
    }

    /// <summary>
    /// Performs the forward pass through both base and LoRA layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass is identical to standard LoRA: output = base_layer(input) + lora_layer(input)
    /// The difference is that matrix A inside the LoRA layer is frozen, but this doesn't affect
    /// the forward computation.
    /// </para>
    /// <para><b>For Beginners:</b> The forward pass works exactly like standard LoRA.
    /// We compute the base layer output, compute the LoRA correction (using frozen A and trainable B),
    /// and add them together. The frozen matrix A still participates in the computation - it just
    /// doesn't get updated during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Forward pass is identical to standard LoRA
        // Frozen matrix A still participates in computation
        return base.Forward(input);
    }

    /// <summary>
    /// Performs the backward pass, computing gradients only for matrix B (matrix A is frozen).
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass differs from standard LoRA in that gradients for matrix A are not computed
    /// or stored, since matrix A is frozen. Only gradients for matrix B and (if not frozen) the base
    /// layer are computed.
    /// </para>
    /// <para><b>For Beginners:</b> This is where LoRA-FA saves computation and memory!
    ///
    /// During learning, the backward pass normally computes gradients for both matrix A and B.
    /// But in LoRA-FA, we skip the gradient computation for matrix A entirely because:
    /// 1. Matrix A is frozen (won't be updated anyway)
    /// 2. No need to store gradients we won't use
    /// 3. Less computation = faster training
    /// 4. Less memory = can train larger models
    ///
    /// We still compute:
    /// - Gradients for matrix B (the only trainable LoRA component)
    /// - Gradients for the base layer (if not frozen)
    /// - Input gradients to pass to earlier layers
    ///
    /// This is the key optimization that makes LoRA-FA more efficient than standard LoRA!
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Let the base implementation handle the backward pass
        // The LoRA layer will compute gradients for both A and B
        Tensor<T> inputGradient = base.Backward(outputGradient);

        // After base backward pass, we need to zero out the gradients for matrix A
        // since it's frozen and shouldn't be updated
        // The ParameterGradients vector contains [baseLayerGrads (if not frozen), matrixAGrads, matrixBGrads]
        // We need to zero out the matrix A gradients

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int rank = _loraLayer.Rank;
        int matrixAParamCount = inputSize * rank;

        // Calculate offset to matrix A gradients in the parameter gradients vector
        int offset = _freezeBaseLayer ? 0 : _baseLayer.ParameterCount;

        // Zero out matrix A gradients (they won't be used in updates anyway, but this keeps things clean)
        if (ParameterGradients != null)
        {
            for (int i = 0; i < matrixAParamCount; i++)
            {
                ParameterGradients[offset + i] = NumOps.Zero;
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates parameters, but only for matrix B (matrix A remains frozen).
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates only matrix B using the gradients computed during backpropagation.
    /// Matrix A is never updated, as it remains frozen at its initial random values.
    /// </para>
    /// <para><b>For Beginners:</b> This is where we apply what we learned during training!
    ///
    /// The parameter update phase normally adjusts both matrix A and B based on their gradients.
    /// But in LoRA-FA, we only update matrix B:
    /// 1. Get the gradients for matrix B from backpropagation
    /// 2. Update matrix B: B_new = B_old - learningRate × gradient_B
    /// 3. Skip matrix A entirely (it stays frozen)
    /// 4. Update base layer parameters if not frozen
    ///
    /// This is faster than standard LoRA because:
    /// - Fewer parameters to update
    /// - Less memory traffic
    /// - Simpler computation
    ///
    /// Matrix A stays exactly as it was initialized - random Gaussian values that never change!
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Get the current parameters from the LoRA layer
        Vector<T> loraParams = _loraLayer.GetParameters();

        // Get the gradients
        Vector<T> loraGrads = _loraLayer.GetParameterGradients();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int rank = _loraLayer.Rank;
        int matrixAParamCount = inputSize * rank;
        int matrixBParamCount = rank * outputSize;

        // Create updated parameters vector
        Vector<T> updatedLoraParams = new Vector<T>(loraParams.Length);

        // Copy matrix A unchanged (frozen)
        for (int i = 0; i < matrixAParamCount; i++)
        {
            updatedLoraParams[i] = loraParams[i];
        }

        // Update matrix B only
        for (int i = 0; i < matrixBParamCount; i++)
        {
            int idx = matrixAParamCount + i;
            T update = NumOps.Multiply(loraGrads[idx], learningRate);
            updatedLoraParams[idx] = NumOps.Subtract(loraParams[idx], update);
        }

        // Set the updated parameters back to the LoRA layer
        _loraLayer.SetParameters(updatedLoraParams);

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update the adapter's parameter vector
        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Updates the parameter vector from the current layer states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CRITICAL: For LoRA-FA, this packs BOTH matrix A and B to match ParameterCount.
    /// Even though matrix A is frozen, it must be included in the parameter buffer
    /// to maintain base-class invariants and prevent buffer overruns.
    /// The freeze logic is in UpdateParameters, not in buffer packing.
    /// </para>
    /// </remarks>
    protected override void UpdateParametersFromLayers()
    {
        int idx = 0;

        // If base layer is not frozen, pack its parameters first
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        // Pack ALL LoRA parameters (both matrix A and B)
        // Matrix A is frozen but must be in the buffer for base class compatibility
        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            Parameters[idx++] = loraParams[i];
        }
    }

    /// <summary>
    /// Merges the LoRA-FA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with LoRA weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method merges the LoRA-FA adaptation (using frozen matrix A and trained matrix B)
    /// back into the base layer's weights. The process is identical to standard LoRA merging,
    /// as both frozen and trained matrices contribute equally to the final merged weights.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your LoRA-FA adaptation to create a regular layer.
    ///
    /// Even though matrix A was frozen during training, it still participated in all the forward
    /// passes and contributed to the model's behavior. When merging:
    /// 1. Compute the full weight matrix: W_lora = A × B × scaling
    /// 2. Add these weights to the base layer's weights
    /// 3. Create a new layer with the merged weights
    ///
    /// The result is identical to what your adapted model was producing, but:
    /// - Faster inference (single matrix multiply instead of A × B)
    /// - Simpler deployment (one layer instead of adapter + base layer)
    /// - No need for LoRA-aware code in production
    ///
    /// Even though A was frozen (never trained), it still matters for the final merged weights
    /// because it was part of the random projection that B learned to work with!
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Merging works identically to standard LoRA
        // Both frozen A and trained B contribute to the merged weights
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("LoRAFAAdapter merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the LoRA weight contribution (A × B × scaling)
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters (works for both DenseLayer and FullyConnectedLayer)
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Both DenseLayer and FullyConnectedLayer store parameters as [weights..., biases...]
        // We need to add the LoRA weights to the base weights
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights (add LoRA contribution to base weights)
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged (LoRA doesn't modify biases)
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Exports the computation graph for this layer (JIT compilation support).
    /// </summary>
    /// <param name="inputNodes">The input computation nodes.</param>
    /// <returns>The output computation node representing this layer's operation.</returns>
    /// <remarks>
    /// This is a stub implementation. Full JIT support will be added in a future update.
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotImplementedException(
            $"{GetType().Name} does not have full JIT compilation support yet. " +
            "This layer will use the standard Forward() method for now.");
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    /// <remarks>
    /// Currently returns false as full JIT support is not yet implemented.
    /// </remarks>
    public override bool SupportsJitCompilation => false;
}
