using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// LoRA+ adapter that uses optimized learning rates for faster convergence and better performance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoRA+ (February 2024) improves upon standard LoRA by using different learning rates for the A and B matrices.
/// The key insight is that matrix B (which starts at zero) needs faster updates than matrix A (which starts random).
/// This simple modification leads to significantly faster convergence and improved final performance.
/// </para>
/// <para><b>For Beginners:</b> LoRA+ is an enhanced version of LoRA that trains faster and better.
///
/// In standard LoRA:
/// - Both matrix A and B are updated with the same learning rate
/// - Matrix B starts at zero, so it needs time to "catch up"
/// - Matrix A starts random, so it's already contributing from the start
///
/// LoRA+ recognizes this asymmetry:
/// - Matrix A is updated with a base learning rate (e.g., 0.0001)
/// - Matrix B is updated with a higher learning rate (e.g., 0.0016 = 16x higher)
/// - This accelerates learning without instability
///
/// Key parameters:
/// - BaseLearningRate: Learning rate for matrix A (the "slow" matrix)
/// - LearningRateRatio: Multiplier for matrix B (typically 16.0)
/// - ScaledLearningRate: Computed as BaseLearningRate * LearningRateRatio
///
/// Research shows LoRA+ typically achieves:
/// - 2x faster convergence
/// - Better final performance
/// - No additional parameters compared to standard LoRA
///
/// Example: If base learning rate is 0.0001 and ratio is 16.0:
/// - Matrix A updates with learning rate 0.0001
/// - Matrix B updates with learning rate 0.0016
///
/// Reference: LoRA+: Efficient Low Rank Adaptation of Large Models (February 2024)
/// </para>
/// </remarks>
public class LoRAPlusAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// The ratio of learning rates between matrix B and matrix A.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This ratio determines how much faster matrix B is updated compared to matrix A.
    /// Typical values range from 8.0 to 32.0, with 16.0 being the recommended default.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much faster the B matrix learns.
    /// A ratio of 16.0 means B learns 16x faster than A. Higher values mean even faster
    /// B updates, but too high can cause instability.
    /// </para>
    /// </remarks>
    private double _learningRateRatio;

    /// <summary>
    /// The base learning rate applied to matrix A.
    /// </summary>
    /// <remarks>
    /// This is the slower learning rate applied to matrix A, which already has random
    /// initialization and contributes from the start of training.
    /// </remarks>
    private T _baseLearningRate;

    /// <summary>
    /// The scaled learning rate applied to matrix B (BaseLearningRate * LearningRateRatio).
    /// </summary>
    /// <remarks>
    /// This is the faster learning rate applied to matrix B, which starts at zero
    /// and needs accelerated updates to catch up with matrix A.
    /// </remarks>
    private T _scaledLearningRate;

    /// <summary>
    /// Gets or sets the learning rate ratio between matrix B and matrix A.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default value is 16.0 as recommended by the LoRA+ paper. Valid range is typically 1.0 to 32.0.
    /// </para>
    /// <para><b>For Beginners:</b> This is the multiplier that makes matrix B learn faster.
    /// - 1.0 = same speed as standard LoRA (no benefit)
    /// - 8.0 = moderate speedup
    /// - 16.0 = recommended default
    /// - 32.0 = aggressive speedup (may be unstable)
    /// </para>
    /// </remarks>
    public double LearningRateRatio
    {
        get => _learningRateRatio;
        set
        {
            if (value < 1.0)
            {
                throw new ArgumentException("Learning rate ratio must be at least 1.0", nameof(value));
            }
            _learningRateRatio = value;
            UpdateScaledLearningRate();
        }
    }

    /// <summary>
    /// Gets the base learning rate for matrix A.
    /// </summary>
    public T BaseLearningRate => _baseLearningRate;

    /// <summary>
    /// Gets the scaled learning rate for matrix B.
    /// </summary>
    public T ScaledLearningRate => _scaledLearningRate;

    /// <summary>
    /// Initializes a new LoRA+ adapter with optimized dual learning rates.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LoRA+.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="learningRateRatio">The ratio of B's learning rate to A's learning rate (default: 16.0).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when learningRateRatio is less than 1.0.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a LoRA+ adapter that will train faster than standard LoRA.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to efficiently fine-tune
    /// - rank: How much compression (lower = fewer parameters)
    /// - alpha: How strong the LoRA effect is
    /// - learningRateRatio: How much faster B learns than A (16.0 is recommended)
    /// - freezeBaseLayer: Whether to lock the original weights (usually true)
    ///
    /// The learning rate ratio is the key differentiator from standard LoRA. Higher ratios
    /// mean faster convergence but require careful tuning to avoid instability.
    /// </para>
    /// </remarks>
    public LoRAPlusAdapter(
        ILayer<T> baseLayer,
        int rank,
        double alpha = -1,
        double learningRateRatio = 16.0,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (learningRateRatio < 1.0)
        {
            throw new ArgumentException("Learning rate ratio must be at least 1.0", nameof(learningRateRatio));
        }

        _learningRateRatio = learningRateRatio;
        _baseLearningRate = NumOps.Zero;
        _scaledLearningRate = NumOps.Zero;
    }

    /// <summary>
    /// Sets the learning rates for this adapter.
    /// </summary>
    /// <param name="baseLearningRate">The base learning rate for matrix A.</param>
    /// <remarks>
    /// <para>
    /// This method sets the base learning rate and automatically computes the scaled
    /// learning rate for matrix B using the current learning rate ratio.
    /// </para>
    /// <para><b>For Beginners:</b> Call this to configure how fast the adapter learns.
    /// You only need to provide the base learning rate - the higher learning rate for
    /// matrix B is calculated automatically using the ratio you specified.
    ///
    /// Example: If you call SetLearningRates(0.0001) with ratio 16.0:
    /// - Matrix A will use learning rate 0.0001
    /// - Matrix B will use learning rate 0.0016 (16x faster)
    /// </para>
    /// </remarks>
    public void SetLearningRates(T baseLearningRate)
    {
        _baseLearningRate = baseLearningRate;
        UpdateScaledLearningRate();
    }

    /// <summary>
    /// Updates the scaled learning rate based on the current base learning rate and ratio.
    /// </summary>
    private void UpdateScaledLearningRate()
    {
        _scaledLearningRate = NumOps.Multiply(_baseLearningRate, NumOps.FromDouble(_learningRateRatio));
    }

    /// <summary>
    /// Performs the forward pass through both base and LoRA layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass is identical to standard LoRA: output = base_layer(input) + lora_layer(input).
    /// The dual learning rate optimization only affects the backward pass and parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This works exactly like standard LoRA during the forward pass.
    /// The magic of LoRA+ happens during training (backward pass), not inference.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Forward pass is identical to base LoRA implementation
        return base.Forward(input);
    }

    /// <summary>
    /// Performs the backward pass through both layers with dual learning rate scaling.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients for both matrices but applies different scaling
    /// factors to prepare for the dual learning rate update. Matrix B gradients are implicitly
    /// prepared for faster updates during the UpdateParameters call.
    /// </para>
    /// <para><b>For Beginners:</b> This is where LoRA+ differs from standard LoRA!
    /// During backpropagation, we compute gradients for both A and B matrices, but we'll
    /// apply different learning rates when actually updating the parameters. This prepares
    /// the gradients for the dual learning rate optimization.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // The base backward implementation computes gradients correctly
        // The dual learning rate is applied in UpdateParameters
        return base.Backward(outputGradient);
    }

    /// <summary>
    /// Updates parameters using dual learning rates (base rate for A, scaled rate for B).
    /// </summary>
    /// <param name="learningRate">This parameter is used as the base learning rate for matrix A.</param>
    /// <remarks>
    /// <para>
    /// This method overrides the standard LoRA parameter update to apply different learning rates:
    /// - Matrix A is updated with the base learning rate
    /// - Matrix B is updated with the scaled learning rate (base * ratio)
    /// - Base layer is updated with the base learning rate if not frozen
    /// </para>
    /// <para><b>For Beginners:</b> This is where the dual learning rate magic happens!
    /// Instead of updating both matrices at the same speed, we:
    /// 1. Update matrix A slowly (with the base learning rate)
    /// 2. Update matrix B quickly (with the scaled learning rate)
    ///
    /// This asymmetry accelerates training because:
    /// - Matrix A already has random values and is contributing
    /// - Matrix B starts at zero and needs to catch up
    /// - Giving B a higher learning rate helps it catch up faster
    ///
    /// The result is faster convergence and better final performance!
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Store the base learning rate for matrix A
        SetLearningRates(learningRate);

        // Get the LoRA layer's parameter gradients
        Vector<T> loraGrads = _loraLayer.GetParameterGradients();

        // Calculate dimensions
        int matrixASize = _loraLayer.GetMatrixA().Rows * _loraLayer.GetMatrixA().Columns;
        int matrixBSize = _loraLayer.GetMatrixB().Rows * _loraLayer.GetMatrixB().Columns;

        // Get current LoRA parameters
        Vector<T> loraParams = _loraLayer.GetParameters();

        // Update matrix A with base learning rate
        for (int i = 0; i < matrixASize; i++)
        {
            T update = NumOps.Multiply(loraGrads[i], _baseLearningRate);
            loraParams[i] = NumOps.Subtract(loraParams[i], update);
        }

        // Update matrix B with scaled learning rate (higher rate)
        for (int i = matrixASize; i < matrixASize + matrixBSize; i++)
        {
            T update = NumOps.Multiply(loraGrads[i], _scaledLearningRate);
            loraParams[i] = NumOps.Subtract(loraParams[i], update);
        }

        // Apply updated parameters to LoRA layer
        _loraLayer.SetParameters(loraParams);

        // Update base layer if not frozen (using base learning rate)
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(_baseLearningRate);
        }

        // Update the adapter's parameter vector
        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Merges the LoRA+ adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with LoRA weights merged into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// For LoRA+, merging works exactly like standard LoRA - the dual learning rates only
    /// affect training, not the final merged weights.
    /// </para>
    /// <para><b>For Beginners:</b> After training with LoRA+, you can merge the weights just like
    /// standard LoRA. The faster training doesn't change the final result, it just gets you there quicker!
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // LoRA+ merging is identical to standard LoRA
        // For Dense layers, delegate to DenseLoRAAdapter logic
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("LoRAPlusAdapter currently only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Calculate dimensions
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
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
    /// Updates the parameter vector from the current layer states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This private helper method synchronizes the adapter's parameter vector with the current state
    /// of the base and LoRA layers after updates.
    /// </para>
    /// </remarks>
    private void UpdateParametersFromLayers()
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

        // Pack LoRA parameters
        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            Parameters[idx++] = loraParams[i];
        }
    }
}
