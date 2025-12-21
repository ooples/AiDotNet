using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// DyLoRA (Dynamic LoRA) adapter that trains with multiple ranks simultaneously.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// DyLoRA extends the standard LoRA approach by training multiple rank configurations simultaneously
/// using a nested dropout technique. This allows a single trained adapter to be deployed at different
/// rank levels without retraining, providing flexibility for different hardware constraints or
/// performance requirements.
/// </para>
/// <para>
/// The key innovation is nested dropout: during training, for each forward pass, a random rank r
/// is selected from the active ranks, and only the first r components of matrices A and B are used.
/// This ensures that smaller ranks can function independently and don't rely on higher-rank components.
/// </para>
/// <para><b>For Beginners:</b> DyLoRA is like LoRA with a superpower - flexibility!
///
/// Standard LoRA problem:
/// - You choose rank=8 and train
/// - Later realize rank=4 would work fine (save memory/speed)
/// - Or need rank=16 for better quality
/// - Must retrain from scratch with the new rank
///
/// DyLoRA solution:
/// - Train once with multiple ranks (e.g., [2, 4, 8, 16])
/// - Deploy with ANY of those ranks without retraining
/// - Switch between ranks at runtime based on device capabilities
///
/// How it works:
/// 1. Train with MaxRank (e.g., 16) but randomly use smaller ranks during training
/// 2. Nested dropout ensures each rank works independently
/// 3. After training, pick deployment rank based on needs (2=fastest, 16=best quality)
///
/// Use cases:
/// - Deploy same model to mobile (rank=2) and server (rank=16)
/// - Dynamic quality scaling based on battery level
/// - A/B testing different rank/quality trade-offs
/// - Training once, deploying everywhere
///
/// Example: Train with ActiveRanks=[2,4,8], deploy with:
/// - Rank=2 for mobile devices (98% parameter reduction, good quality)
/// - Rank=4 for tablets (95% parameter reduction, better quality)
/// - Rank=8 for desktops (90% parameter reduction, best quality)
/// </para>
/// </remarks>
public class DyLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Maximum rank for the LoRA decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the highest rank that can be used during inference. The actual matrices A and B
    /// are sized for this maximum rank, but smaller ranks can be used by only accessing the
    /// first r columns/rows.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "full size" of your LoRA adapter. You can always
    /// use a smaller rank, but you can't exceed this maximum without retraining.
    /// </para>
    /// </remarks>
    private readonly int _maxRank;

    /// <summary>
    /// Array of ranks to train simultaneously during nested dropout.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During training, each forward pass randomly selects one of these ranks and only uses
    /// that many components. This ensures all these ranks are viable for deployment.
    /// </para>
    /// <para><b>For Beginners:</b> These are the rank options you can choose from after training.
    /// For example, [2, 4, 8, 16] means you can deploy with any of these four ranks.
    /// </para>
    /// </remarks>
    private readonly int[] _activeRanks;

    /// <summary>
    /// Current rank to use during inference (forward pass in eval mode).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This determines how many components of the LoRA matrices are used during inference.
    /// Can be changed at runtime to trade off between speed and quality.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "deployment rank" - the actual rank you're using
    /// right now for predictions. You can change this at any time without retraining!
    /// </para>
    /// </remarks>
    private int _currentDeploymentRank;

    /// <summary>
    /// Random number generator for nested dropout during training.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Whether the adapter is in training mode (uses nested dropout).
    /// </summary>
    private bool _isTraining;

    /// <summary>
    /// Cached input from the last forward pass for gradient computation.
    /// </summary>
    private Tensor<T>? _cachedInput;

    /// <summary>
    /// Cached active rank from the last forward pass for gradient computation.
    /// </summary>
    private int _cachedActiveRank;

    /// <summary>
    /// Cached LoRA parameter gradients computed in backward pass.
    /// </summary>
    private Vector<T>? _cachedLoRAGradients;

    /// <summary>
    /// Gets the maximum rank of the DyLoRA adapter.
    /// </summary>
    public int MaxRank => _maxRank;

    /// <summary>
    /// Gets the array of active ranks used during training.
    /// </summary>
    public int[] ActiveRanks => _activeRanks.ToArray();

    /// <summary>
    /// Gets or sets the current deployment rank used during inference.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when attempting to set a rank not in ActiveRanks.</exception>
    public int CurrentDeploymentRank
    {
        get => _currentDeploymentRank;
        set => SetDeploymentRank(value);
    }

    /// <summary>
    /// Gets or sets whether the adapter is in training mode.
    /// </summary>
    /// <remarks>
    /// When in training mode, nested dropout is applied. In eval mode, the deployment rank is used.
    /// </remarks>
    public bool IsTraining
    {
        get => _isTraining;
        set => _isTraining = value;
    }

    /// <summary>
    /// Initializes a new DyLoRA adapter with the specified parameters.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with DyLoRA.</param>
    /// <param name="maxRank">The maximum rank of the LoRA decomposition.</param>
    /// <param name="activeRanks">Array of ranks to train simultaneously (must be sorted ascending and all &lt;= maxRank).</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to maxRank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer or activeRanks is null.</exception>
    /// <exception cref="ArgumentException">Thrown when activeRanks is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a DyLoRA adapter that can train and deploy with multiple ranks.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to make flexible and efficient
    /// - maxRank: The maximum rank you might need (e.g., 16)
    /// - activeRanks: Which ranks to make available (e.g., [2, 4, 8, 16])
    /// - alpha: How strong the LoRA adaptation is (usually equals maxRank)
    /// - freezeBaseLayer: Whether to lock the original layer (usually true)
    ///
    /// Example:
    /// new DyLoRAAdapter(denseLayer, maxRank: 16, activeRanks: new[] { 2, 4, 8, 16 })
    /// This trains a single adapter that can deploy with ranks 2, 4, 8, or 16.
    /// </para>
    /// </remarks>
    public DyLoRAAdapter(
        ILayer<T> baseLayer,
        int maxRank,
        int[] activeRanks,
        double alpha = -1,
        bool freezeBaseLayer = true)
        : base(baseLayer, maxRank, alpha, freezeBaseLayer)
    {
        if (activeRanks == null)
        {
            throw new ArgumentNullException(nameof(activeRanks));
        }

        if (activeRanks.Length == 0)
        {
            throw new ArgumentException("ActiveRanks must contain at least one rank", nameof(activeRanks));
        }

        // Validate activeRanks are sorted and within bounds
        for (int i = 0; i < activeRanks.Length; i++)
        {
            if (activeRanks[i] <= 0)
            {
                throw new ArgumentException($"All ranks must be positive, but activeRanks[{i}] = {activeRanks[i]}", nameof(activeRanks));
            }

            if (activeRanks[i] > maxRank)
            {
                throw new ArgumentException($"All ranks must be <= maxRank ({maxRank}), but activeRanks[{i}] = {activeRanks[i]}", nameof(activeRanks));
            }

            if (i > 0 && activeRanks[i] <= activeRanks[i - 1])
            {
                throw new ArgumentException("ActiveRanks must be sorted in ascending order with no duplicates", nameof(activeRanks));
            }
        }

        _maxRank = maxRank;
        _activeRanks = activeRanks.ToArray();
        _currentDeploymentRank = activeRanks[activeRanks.Length - 1]; // Default to highest rank
        _random = RandomHelper.CreateSecureRandom();
        _isTraining = true; // Start in training mode
    }

    /// <summary>
    /// Sets the deployment rank for inference.
    /// </summary>
    /// <param name="rank">The rank to use (must be in ActiveRanks).</param>
    /// <exception cref="ArgumentException">Thrown when rank is not in ActiveRanks.</exception>
    /// <remarks>
    /// <para>
    /// This allows switching between different ranks at runtime without retraining.
    /// The rank must be one of the ActiveRanks that were trained.
    /// </para>
    /// <para><b>For Beginners:</b> This changes the quality/speed trade-off of your model.
    /// Higher rank = better quality but slower. Lower rank = faster but slightly lower quality.
    ///
    /// Example usage:
    /// - Battery low? adapter.SetDeploymentRank(2) for speed
    /// - Plugged in? adapter.SetDeploymentRank(16) for quality
    /// - On mobile? adapter.SetDeploymentRank(4) for balance
    /// </para>
    /// </remarks>
    public void SetDeploymentRank(int rank)
    {
        if (!_activeRanks.Contains(rank))
        {
            throw new ArgumentException(
                $"Deployment rank {rank} is not in ActiveRanks [{string.Join(", ", _activeRanks)}]. " +
                "Only trained ranks can be used for deployment.",
                nameof(rank));
        }

        _currentDeploymentRank = rank;
    }

    /// <summary>
    /// Performs the forward pass with dynamic rank selection.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and DyLoRA output.</returns>
    /// <remarks>
    /// <para>
    /// During training, a random rank is selected from ActiveRanks for nested dropout.
    /// During inference, the CurrentDeploymentRank is used consistently.
    /// </para>
    /// <para><b>For Beginners:</b> This processes input through both the base layer and DyLoRA:
    ///
    /// Training mode:
    /// - Randomly picks a rank from ActiveRanks each forward pass
    /// - Uses only that many components of A and B matrices
    /// - This trains all ranks to work independently
    ///
    /// Inference mode:
    /// - Always uses CurrentDeploymentRank
    /// - Consistent behavior for production
    /// - Can change rank without retraining
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Select rank for this forward pass
        int activeRank = _isTraining
            ? _activeRanks[_random.Next(_activeRanks.Length)]  // Random rank during training
            : _currentDeploymentRank;                          // Fixed rank during inference

        // Cache input and active rank for backward pass
        _cachedInput = input.Clone();
        _cachedActiveRank = activeRank;

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // CRITICAL: Prime _loraLayer caches by calling Forward, then mask to activeRank
        // This ensures _loraLayer.Backward will have fresh cached inputs for gradient computation
        Tensor<T> fullLoraOutput = _loraLayer.Forward(input);
        Tensor<T> loraOutput = MaskOutputToRank(fullLoraOutput, activeRank);

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Masks the full LoRA output to only include contributions from the first 'rank' components.
    /// </summary>
    /// <param name="fullOutput">The full LoRA output from all maxRank components.</param>
    /// <param name="rank">Number of components to use (nested dropout rank).</param>
    /// <returns>Masked LoRA output tensor using only first 'rank' components.</returns>
    /// <remarks>
    /// <para>
    /// This recomputes the LoRA output using only the first 'rank' columns of A and rows of B.
    /// By calling _loraLayer.Forward first (in Forward method), we ensure _loraLayer has fresh
    /// cached inputs for gradient computation, then we mask the output to match the nested dropout rank.
    /// </para>
    /// </remarks>
    private Tensor<T> MaskOutputToRank(Tensor<T> fullOutput, int rank)
    {
        // If using full rank, return as-is
        if (rank == _maxRank)
        {
            return fullOutput;
        }

        // Recompute with rank restriction using ForwardWithRank logic
        // (We must recompute rather than mask, since LoRA output is input*A*B, not linearly separable by rank)
        if (_cachedInput == null)
        {
            throw new InvalidOperationException("MaskOutputToRank called without cached input");
        }
        return ForwardWithRank(_cachedInput, rank);
    }

    /// <summary>
    /// Performs forward pass through LoRA layer using only the first 'rank' components.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="rank">Number of components to use.</param>
    /// <returns>LoRA output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This restricts the LoRA computation to use only the first 'rank' columns of A and rows of B,
    /// implementing the nested dropout mechanism.
    /// </para>
    /// <para><b>For Beginners:</b> This is the core of DyLoRA's flexibility. Instead of using all
    /// components of A and B, we only use the first 'rank' of them. This simulates what would happen
    /// if we had trained with that specific rank from the start.
    /// </para>
    /// </remarks>
    private Tensor<T> ForwardWithRank(Tensor<T> input, int rank)
    {
        // Get matrices A and B from the LoRA layer
        Matrix<T> fullA = _loraLayer.GetMatrixA();
        Matrix<T> fullB = _loraLayer.GetMatrixB();

        // Extract submatrices using only the first 'rank' components
        // A: [inputSize, maxRank] -> [inputSize, rank]
        // B: [maxRank, outputSize] -> [rank, outputSize]
        int inputSize = fullA.Rows;
        int outputSize = fullB.Columns;

        Matrix<T> subA = new Matrix<T>(inputSize, rank);
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                subA[i, j] = fullA[i, j];
            }
        }

        Matrix<T> subB = new Matrix<T>(rank, outputSize);
        for (int i = 0; i < rank; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                subB[i, j] = fullB[i, j];
            }
        }

        // Compute forward pass with submatrices
        int batchSize = input.Shape[0];
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = input[i * inputSize + j];
            }
        }

        // input * A_sub * B_sub * scaling
        T scaling = _loraLayer.Scaling;
        Matrix<T> intermediate = inputMatrix.Multiply(subA);
        Matrix<T> output = intermediate.Multiply(subB).Multiply(scaling);

        // Convert back to tensor
        Vector<T> outputData = new Vector<T>(batchSize * outputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                outputData[idx++] = output[i, j];
            }
        }

        return new Tensor<T>(new[] { batchSize, outputSize }, outputData);
    }

    /// <summary>
    /// Performs the backward pass with nested dropout training.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// During training, gradients are computed for all components, but the nested dropout ensures
    /// that only the active rank's components receive meaningful gradients. This trains all ranks
    /// simultaneously while ensuring each smaller rank can function independently.
    /// </para>
    /// <para><b>For Beginners:</b> This is where DyLoRA learning happens! During backpropagation:
    ///
    /// 1. Gradients flow back through whichever rank was used in the forward pass
    /// 2. Only those components get updated
    /// 3. Over many iterations, all ranks get trained
    /// 4. Smaller ranks learn to work without relying on larger rank components
    ///
    /// This is why you can deploy with any trained rank - each one was trained independently!
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_cachedInput == null)
        {
            throw new InvalidOperationException("Backward called without a preceding Forward call");
        }

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Backward through LoRA with restricted rank
        Tensor<T> loraInputGrad = BackwardWithRank(outputGradient, _cachedInput, _cachedActiveRank);

        // Sum input gradients
        Tensor<T> inputGrad = new Tensor<T>(baseInputGrad.Shape);
        for (int i = 0; i < baseInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(baseInputGrad[i], loraInputGrad[i]);
        }

        // Update parameter gradients from base and LoRA layers
        UpdateParameterGradientsFromLayers();

        return inputGrad;
    }

    /// <summary>
    /// Performs backward pass through LoRA layer using only the first 'rank' components.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from next layer.</param>
    /// <param name="input">Input from forward pass.</param>
    /// <param name="rank">Number of components that were used in forward pass.</param>
    /// <returns>Input gradient tensor.</returns>
    private Tensor<T> BackwardWithRank(Tensor<T> outputGradient, Tensor<T> input, int rank)
    {
        // Get matrices A and B from the LoRA layer
        Matrix<T> fullA = _loraLayer.GetMatrixA();
        Matrix<T> fullB = _loraLayer.GetMatrixB();

        int inputSize = fullA.Rows;
        int outputSize = fullB.Columns;

        // Extract submatrices using only the first 'rank' components
        Matrix<T> subA = new Matrix<T>(inputSize, rank);
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                subA[i, j] = fullA[i, j];
            }
        }

        Matrix<T> subB = new Matrix<T>(rank, outputSize);
        for (int i = 0; i < rank; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                subB[i, j] = fullB[i, j];
            }
        }

        // Convert tensors to matrices
        int batchSize = outputGradient.Shape[0];
        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradMatrix[b, j] = outputGradient[b, j];
            }
        }

        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[b, j] = input[b, j];
            }
        }

        // Compute gradients for submatrices
        // dL/dB = (gradMatrix^T @ inputMatrix @ subA^T) with scaling
        // dL/dA = (inputMatrix^T @ gradMatrix @ subB^T) with scaling
        Matrix<T> gradB = gradMatrix.Transpose().Multiply(inputMatrix).Multiply(subA.Transpose());
        Matrix<T> gradA = inputMatrix.Transpose().Multiply(gradMatrix).Multiply(subB.Transpose());

        // Scale by _loraLayer.Scaling (alpha/maxRank) to match forward pass
        // CRITICAL: Use same scaling as ForwardWithRank (line 394) for correct gradients
        T scaleT = _loraLayer.Scaling;
        for (int i = 0; i < gradB.Rows; i++)
        {
            for (int j = 0; j < gradB.Columns; j++)
            {
                gradB[i, j] = NumOps.Multiply(gradB[i, j], scaleT);
            }
        }
        for (int i = 0; i < gradA.Rows; i++)
        {
            for (int j = 0; j < gradA.Columns; j++)
            {
                gradA[i, j] = NumOps.Multiply(gradA[i, j], scaleT);
            }
        }

        // Update full gradient matrices (only the active rank components)
        Matrix<T> fullGradA = new Matrix<T>(inputSize, _maxRank);
        Matrix<T> fullGradB = new Matrix<T>(_maxRank, outputSize);

        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                fullGradA[i, j] = gradA[i, j];
            }
        }
        for (int i = 0; i < rank; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                fullGradB[i, j] = gradB[i, j];
            }
        }

        // Pack gradients into LoRA layer's parameter gradients
        // Store them in cache to be used by UpdateParameterGradientsFromLayers
        Vector<T> loraParamGrads = new Vector<T>(_loraLayer.ParameterCount);
        int idx = 0;
        for (int i = 0; i < fullGradA.Rows; i++)
        {
            for (int j = 0; j < fullGradA.Columns; j++)
            {
                loraParamGrads[idx++] = fullGradA[i, j];
            }
        }
        for (int i = 0; i < fullGradB.Rows; i++)
        {
            for (int j = 0; j < fullGradB.Columns; j++)
            {
                loraParamGrads[idx++] = fullGradB[i, j];
            }
        }
        // Cache the LoRA gradients for later use in UpdateParameterGradientsFromLayers
        _cachedLoRAGradients = loraParamGrads;

        // Compute input gradient: dL/dInput = gradMatrix @ (subB @ subA)^T
        Matrix<T> combinedWeight = subB.Multiply(subA.Transpose());
        // Use same scaling as ForwardWithRank (line 394) for correct gradients
        T alphaScaleT = _loraLayer.Scaling;
        for (int i = 0; i < combinedWeight.Rows; i++)
        {
            for (int j = 0; j < combinedWeight.Columns; j++)
            {
                combinedWeight[i, j] = NumOps.Multiply(combinedWeight[i, j], alphaScaleT);
            }
        }

        Matrix<T> inputGradMatrix = gradMatrix.Multiply(combinedWeight);

        // Convert back to tensor
        Vector<T> inputGradVector = new Vector<T>(batchSize * inputSize);
        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputGradVector[b * inputSize + j] = inputGradMatrix[b, j];
            }
        }

        return new Tensor<T>(new[] { batchSize, inputSize }, inputGradVector);
    }

    /// <summary>
    /// Updates the parameter gradients vector from the layer gradients.
    /// </summary>
    private void UpdateParameterGradientsFromLayers()
    {
        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // Base layer gradients (if not frozen)
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // LoRA layer gradients - use cached gradients computed in BackwardWithRank
        if (_cachedLoRAGradients != null)
        {
            for (int i = 0; i < _cachedLoRAGradients.Length; i++)
            {
                ParameterGradients[idx++] = _cachedLoRAGradients[i];
            }
        }
    }

    /// <summary>
    /// Updates parameters for the base layer and the LoRA layer using cached gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Manually update LoRA layer's parameters using cached gradients,
        // as the base UpdateParameters would use the LoRA layer's empty internal gradients.
        if (_cachedLoRAGradients != null)
        {
            if (_cachedLoRAGradients.Length == _loraLayer.ParameterCount)
            {
                Vector<T> loraParams = _loraLayer.GetParameters();
                for (int i = 0; i < loraParams.Length; i++)
                {
                    T update = NumOps.Multiply(_cachedLoRAGradients[i], learningRate);
                    loraParams[i] = NumOps.Subtract(loraParams[i], update);
                }
                _loraLayer.SetParameters(loraParams);
            }

            // Clear the cache after use.
            _cachedLoRAGradients = null;
        }
    }

    /// <summary>
    /// Trains the adapter with nested dropout across all active ranks.
    /// </summary>
    /// <param name="inputs">Training input tensors.</param>
    /// <param name="targets">Training target tensors.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <param name="lossFunction">Loss function to minimize.</param>
    /// <remarks>
    /// <para>
    /// This training method ensures that all active ranks are trained by randomly selecting
    /// a rank for each forward pass. This implements the nested dropout technique that makes
    /// DyLoRA flexible for different deployment ranks.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper method for training your DyLoRA adapter.
    ///
    /// During training:
    /// - Each forward pass randomly uses a different rank
    /// - This trains all ranks simultaneously
    /// - After training, you can deploy with any of the active ranks
    ///
    /// Think of it like training a team where each member can work alone or together.
    /// The random selection ensures everyone learns to be independent.
    /// </para>
    /// </remarks>
    public void TrainWithNestedDropout(
        Tensor<T>[] inputs,
        Tensor<T>[] targets,
        int epochs,
        T learningRate,
        Func<Tensor<T>, Tensor<T>, T> lossFunction)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (targets == null)
        {
            throw new ArgumentNullException(nameof(targets));
        }

        if (inputs.Length != targets.Length)
        {
            throw new ArgumentException("Inputs and targets must have the same length");
        }

        // Ensure we're in training mode
        bool wasTraining = _isTraining;
        _isTraining = true;

        try
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                T totalLoss = NumOps.Zero;

                for (int i = 0; i < inputs.Length; i++)
                {
                    // Forward pass (uses random rank due to training mode)
                    Tensor<T> output = Forward(inputs[i]);

                    // Compute loss
                    T loss = lossFunction(output, targets[i]);
                    totalLoss = NumOps.Add(totalLoss, loss);

                    // Compute output gradient (simplified - assumes MSE loss)
                    Tensor<T> outputGrad = new Tensor<T>(output.Shape);
                    for (int j = 0; j < output.Length; j++)
                    {
                        outputGrad[j] = NumOps.Subtract(output[j], targets[i][j]);
                    }

                    // Backward pass
                    Backward(outputGrad);

                    // Update parameters
                    UpdateParameters(learningRate);
                }

                // Optional: Print epoch loss
                // Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {totalLoss}");
            }
        }
        finally
        {
            // Restore original training state
            _isTraining = wasTraining;
        }
    }

    /// <summary>
    /// Merges the DyLoRA adaptation into the base layer using the current deployment rank.
    /// </summary>
    /// <returns>A new layer with DyLoRA weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method merges only the components up to CurrentDeploymentRank, creating a layer
    /// that's equivalent to the DyLoRA adapter at that specific rank.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your DyLoRA adaptation at the current rank.
    ///
    /// After training:
    /// 1. Set the deployment rank you want: adapter.SetDeploymentRank(8)
    /// 2. Merge to create a standard layer: mergedLayer = adapter.MergeToOriginalLayer()
    /// 3. Use the merged layer for faster inference
    ///
    /// Benefits of merging:
    /// - Faster inference (no separate LoRA computation)
    /// - Simpler deployment (single layer instead of adapter + base)
    /// - Compatible with systems that don't support LoRA
    ///
    /// Note: You can merge at different ranks to create multiple versions:
    /// - Mobile version: SetDeploymentRank(2), then merge
    /// - Desktop version: SetDeploymentRank(16), then merge
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("DyLoRAAdapter merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the LoRA matrices and extract submatrices for current deployment rank
        Matrix<T> fullA = _loraLayer.GetMatrixA();
        Matrix<T> fullB = _loraLayer.GetMatrixB();

        int inputSize = fullA.Rows;
        int outputSize = fullB.Columns;
        int rank = _currentDeploymentRank;

        // Extract submatrices
        Matrix<T> subA = new Matrix<T>(inputSize, rank);
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                subA[i, j] = fullA[i, j];
            }
        }

        Matrix<T> subB = new Matrix<T>(rank, outputSize);
        for (int i = 0; i < rank; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                subB[i, j] = fullB[i, j];
            }
        }

        // Compute merged weights: W_lora = A_sub * B_sub * scaling
        T scaling = _loraLayer.Scaling;
        Matrix<T> loraWeights = subA.Multiply(subB).Multiply(scaling).Transpose();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Sets the adapter to training mode (enables nested dropout).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this before training to enable random rank selection.
    /// This is what makes DyLoRA train all ranks simultaneously.
    /// </para>
    /// </remarks>
    public void Train()
    {
        _isTraining = true;
    }

    /// <summary>
    /// Sets the adapter to evaluation mode (uses fixed deployment rank).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this before inference/prediction to use a consistent rank.
    /// This ensures predictable behavior in production.
    /// </para>
    /// </remarks>
    public void Eval()
    {
        _isTraining = false;
    }
}
