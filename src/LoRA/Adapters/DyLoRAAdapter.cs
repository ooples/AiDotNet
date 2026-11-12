using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Training;

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
[AutoParameters]
public partial class DyLoRAAdapter<T> : LoRAAdapterBase<T>
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
    /// Cached active rank from the last forward pass for gradient computation.
    /// </summary>
    private int _cachedActiveRank;

    /// <summary>
    /// Cached LoRA parameter gradients computed in backward pass.
    /// </summary>
    [Scratch]
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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Select rank for this forward pass
        int activeRank = _isTraining
            ? _activeRanks[_random.Next(_activeRanks.Length)]  // Random rank during training
            : _currentDeploymentRank;                          // Fixed rank during inference

        // Record the active rank
        _cachedActiveRank = activeRank;

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // LoRA output using only the first activeRank components. The masked factors are built
        // from the live LoRA tensors, so a gradient tape sees zero derivatives above activeRank.
        Tensor<T> loraOutput = ForwardWithRank(input, activeRank);

        // Sum the outputs
        Tensor<T> result = Engine.TensorAdd(baseOutput, loraOutput);

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
    /// implementing the nested dropout mechanism. Instead of copying sub-matrices, the live factors
    /// are multiplied by 0/1 masks, which gives the same output (the masked terms add exact zeros)
    /// and keeps the computation connected to the trainable tensors: under a gradient tape the
    /// columns of A and rows of B above 'rank' receive exactly zero gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This is the core of DyLoRA's flexibility. Instead of using all
    /// components of A and B, we only use the first 'rank' of them. This simulates what would happen
    /// if we had trained with that specific rank from the start.
    /// </para>
    /// </remarks>
    private Tensor<T> ForwardWithRank(Tensor<T> input, int rank)
    {
        // Using full rank: the plain LoRA forward
        if (rank == _maxRank)
        {
            return _loraLayer.Forward(input);
        }

        // Live factors: A [inputSize, maxRank], B [maxRank, outputSize]
        var factors = _loraLayer.GetTrainableParameters();
        Tensor<T> fullA = factors[0];
        Tensor<T> fullB = factors[1];
        int inputSize = fullA.Shape[0];
        int outputSize = fullB.Shape[1];

        // Keep the first 'rank' columns of A and rows of B
        Tensor<T> maskA = Engine.TensorConcatenate(
            [Tensor<T>.CreateDefault([inputSize, rank], NumOps.One), new Tensor<T>([inputSize, _maxRank - rank])], 1);
        Tensor<T> maskB = Engine.TensorConcatenate(
            [Tensor<T>.CreateDefault([rank, outputSize], NumOps.One), new Tensor<T>([_maxRank - rank, outputSize])], 0);

        // input * A_sub * B_sub * scaling
        return _loraLayer.ForwardWithFactors(
            input, Engine.TensorMultiply(fullA, maskA), Engine.TensorMultiply(fullB, maskB));
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
                Vector<T> loraParams = Engine.Subtract(
                    _loraLayer.GetParameters(), Engine.Multiply(_cachedLoRAGradients, learningRate));
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
                for (int i = 0; i < inputs.Length; i++)
                {
                    // One tape training step on this sample. The forward samples rank b, the
                    // masked factors give zero gradient above b, and the base layer is trained
                    // through the same tape when it is not frozen. The loss differentiated is
                    // 0.5 * ||output - target||², whose output gradient is (output - target)
                    // (simplified - assumes MSE loss); the supplied loss function is still
                    // evaluated per sample, as before (nothing reads an epoch total).
                    TapeTrainingStep<T>.Step(
                        new ITrainableLayer<T>[] { this },
                        inputs[i],
                        targets[i],
                        learningRate,
                        sample => Forward(sample),
                        (output, target) =>
                        {
                            _ = lossFunction(output, target);
                            var diff = Engine.TensorSubtract(output, Engine.Reshape(target, output.Shape.ToArray()));
                            var squared = Engine.Reshape(Engine.TensorMultiply(diff, diff), [diff.Length]);
                            return Engine.TensorMultiplyScalar(
                                Engine.ReduceSum(squared, [0], keepDims: true), NumOps.FromDouble(0.5));
                        });
                }

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

        // Compute merged weights: W_lora = A_sub * B_sub * scaling ([inputSize, outputSize])
        T scaling = _loraLayer.Scaling;
        Matrix<T> loraWeights = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(
                Engine.TensorNarrow(Tensor<T>.FromMatrix(fullA), 1, 0, rank),
                Engine.TensorNarrow(Tensor<T>.FromMatrix(fullB), 0, 0, rank)),
            scaling).ToMatrix();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;

        // Merge weights. DenseLayer stores its weights input-major and FullyConnectedLayer
        // output-major ([outputSize, inputSize]), so the latter uses the transpose.
        // Biases are kept unchanged.
        bool isOutputMajor = _baseLayer is FullyConnectedLayer<T>;
        Vector<T> mergedParams = AddFlatWeightsToParameters(
            baseParams, isOutputMajor ? loraWeights.Transpose() : loraWeights, weightCount);

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
