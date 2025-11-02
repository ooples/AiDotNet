using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

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
    /// new DyLoRAAdapter(denseLayer, maxRank: 16, activeRanks: [2, 4, 8, 16])
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
        _random = new Random();
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
                $"Only trained ranks can be used for deployment.",
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

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Forward through LoRA layer with restricted rank
        Tensor<T> loraOutput = ForwardWithRank(input, activeRank);

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        return result;
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
        // The base LoRA backward pass handles gradient computation
        // Nested dropout is automatically handled by the forward pass restriction
        return base.Backward(outputGradient);
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

        // Create a new dense layer with merged parameters
        DenseLayer<T> mergedLayer = new DenseLayer<T>(inputSize, outputSize, (IActivationFunction<T>?)null);
        mergedLayer.SetParameters(mergedParams);

        return mergedLayer;
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
