using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Tied-LoRA adapter - LoRA with weight tying for extreme parameter efficiency across deep networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Tied-LoRA achieves even greater parameter efficiency than standard LoRA by:
/// - Sharing the same LoRA matrices (A and B) across multiple layers
/// - Training only layer-specific scaling factors
/// - Particularly effective for very deep networks with many similar layers
/// </para>
/// <para>
/// The forward computation is: output = base_layer(input) + layerScaling * (B_shared * A_shared * input)
/// where layerScaling is a trainable scalar unique to each layer, and A and B are shared trainable matrices.
/// </para>
/// <para><b>For Beginners:</b> Tied-LoRA is an ultra-efficient variant of LoRA for deep networks.
///
/// Think of the difference this way:
/// - Standard LoRA: Each layer has its own pair of small matrices (A and B) that are trained
/// - VeRA: ALL layers share the same random matrices (A and B) which are frozen. Only tiny
///   scaling vectors are trained per layer.
/// - Tied-LoRA: ALL layers share the same matrices (A and B) which ARE trained. Only a single
///   scaling factor is trained per layer.
///
/// Example parameter comparison for 10 layers of 1000x1000 with rank=8:
/// - Full fine-tuning: 10,000,000 parameters
/// - Standard LoRA (rank=8): 160,000 parameters (10 layers × 16,000 params each)
/// - Tied-LoRA (rank=8): ~16,010 parameters (shared 16,000 + 10 scaling factors)
///
/// Benefits of Tied-LoRA:
/// - ✅ Extreme parameter efficiency for deep networks (scales with depth)
/// - ✅ Shared matrices enforce consistency across layers
/// - ✅ Still trainable (unlike VeRA's frozen matrices)
/// - ✅ Very low memory footprint
/// - ✅ Faster training (fewer parameters to update)
///
/// Trade-offs:
/// - ⚠️ Less flexible than standard LoRA (shared adaptation across layers)
/// - ⚠️ Assumes layers benefit from similar adaptations
/// - ⚠️ May underperform standard LoRA on heterogeneous architectures
///
/// When to use Tied-LoRA:
/// - Very deep networks (transformers with many similar layers)
/// - Extreme memory constraints
/// - When layers have similar structure and function
/// - Rapid prototyping with minimal parameter overhead
/// - Fine-tuning massive models (GPT, BERT-style architectures)
///
/// Research insight: Tied-LoRA works well because in deep networks, many layers learn similar
/// transformations. By sharing the LoRA matrices and only varying the strength per layer,
/// we capture most of the adaptation capability with minimal parameters.
/// </para>
/// </remarks>
public class TiedLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Shared trainable matrix A (inputSize × rank) used by all Tied-LoRA adapters.
    /// </summary>
    /// <remarks>
    /// This matrix is shared across all Tied-LoRA layers and IS trained during fine-tuning.
    /// Unlike VeRA, this matrix is not frozen - it learns the common adaptation pattern.
    /// </remarks>
    private static Matrix<T>? _sharedMatrixA;

    /// <summary>
    /// Shared trainable matrix B (rank × outputSize) used by all Tied-LoRA adapters.
    /// </summary>
    /// <remarks>
    /// This matrix is shared across all Tied-LoRA layers and IS trained during fine-tuning.
    /// Unlike VeRA, this matrix is not frozen - it learns the common adaptation pattern.
    /// </remarks>
    private static Matrix<T>? _sharedMatrixB;

    /// <summary>
    /// Gradients for shared matrix A accumulated from all layers.
    /// </summary>
    private static Matrix<T>? _sharedMatrixAGradient;

    /// <summary>
    /// Gradients for shared matrix B accumulated from all layers.
    /// </summary>
    private static Matrix<T>? _sharedMatrixBGradient;

    /// <summary>
    /// Lock object for thread-safe shared matrix access and updates.
    /// </summary>
    private static readonly object _sharedLock = new object();

    /// <summary>
    /// Layer-specific scaling factor - the only trainable parameter unique to this layer.
    /// </summary>
    /// <remarks>
    /// This single scalar value controls how strongly this layer's output is affected by
    /// the shared LoRA adaptation. Different layers can have different scaling factors,
    /// allowing the network to modulate the shared adaptation per layer.
    /// </remarks>
    private T _layerScaling;

    /// <summary>
    /// Gradient for the layer-specific scaling factor.
    /// </summary>
    private T _layerScalingGradient;

    /// <summary>
    /// Layer index identifying this adapter's position in the network.
    /// </summary>
    /// <remarks>
    /// This helps track which layer this adapter belongs to, useful for debugging and
    /// analysis of how different layers utilize the shared adaptation.
    /// </remarks>
    private readonly int _layerIndex;

    /// <summary>
    /// Stored input from the forward pass, needed for gradient computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored intermediate value (B_shared * A_shared * input) from forward pass.
    /// </summary>
    private Matrix<T>? _lastIntermediate;

    /// <summary>
    /// Flag indicating whether this adapter instance has completed initialization.
    /// </summary>
    private bool _isInitialized;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// Tied-LoRA only trains a single scaling factor per layer (plus the base layer if not frozen).
    /// The shared matrices contribute to the parameter count only once across all layers.
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            // Guard against being called during base class construction before initialization
            if (!_isInitialized)
            {
                // During construction, delegate to base which computes full parameter count
                return base.ParameterCount;
            }

            // Only the layer scaling factor is unique to this layer
            int tiedLoraParams = 1; // Single scaling factor
            int baseParams = (_baseLayer != null && !_freezeBaseLayer) ? _baseLayer.ParameterCount : 0;
            return baseParams + tiedLoraParams;
        }
    }

    /// <summary>
    /// Gets the layer-specific scaling factor.
    /// </summary>
    public double LayerScaling => Convert.ToDouble(_layerScaling);

    /// <summary>
    /// Gets the layer index.
    /// </summary>
    public int LayerIndex => _layerIndex;

    /// <summary>
    /// Initializes a new Tied-LoRA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with Tied-LoRA.</param>
    /// <param name="rank">The rank of the low-rank decomposition (shared across all Tied-LoRA layers).</param>
    /// <param name="layerIndex">The index of this layer in the network (for tracking and debugging).</param>
    /// <param name="alpha">The scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when rank is invalid or shared matrices are not initialized.</exception>
    /// <remarks>
    /// <para>
    /// Before creating any Tied-LoRA adapters, you must call InitializeSharedMatrices() once to set up
    /// the shared trainable matrices that all Tied-LoRA layers will use.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a Tied-LoRA adapter for a layer. You must initialize
    /// the shared matrices first by calling:
    ///
    /// TiedLoRAAdapter&lt;T&gt;.InitializeSharedMatrices(inputSize, outputSize, rank);
    ///
    /// This needs to be done once before creating any Tied-LoRA adapters.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt
    /// - rank: How much compression (lower = fewer parameters)
    /// - layerIndex: Which layer this is (0, 1, 2, etc.) for tracking
    /// - alpha: How strong the Tied-LoRA adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true)
    ///
    /// The layerIndex helps identify which layer this adapter belongs to, which is useful
    /// for debugging and understanding how different layers use the shared adaptation.
    /// </para>
    /// </remarks>
    public TiedLoRAAdapter(ILayer<T> baseLayer, int rank, int layerIndex = 0, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (baseLayer == null)
        {
            throw new ArgumentNullException(nameof(baseLayer));
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        _layerIndex = layerIndex;

        // Ensure shared matrices are initialized
        lock (_sharedLock)
        {
            if (_sharedMatrixA == null || _sharedMatrixB == null)
            {
                throw new InvalidOperationException(
                    "Shared matrices must be initialized before creating Tied-LoRA adapters. " +
                    "Call TiedLoRAAdapter<T>.InitializeSharedMatrices(inputSize, outputSize, rank) first.");
            }

            // Validate shared matrix dimensions match this layer
            if (_sharedMatrixA.Rows != inputSize || _sharedMatrixA.Columns != rank)
            {
                throw new ArgumentException(
                    $"Shared matrix A dimensions ({_sharedMatrixA.Rows}×{_sharedMatrixA.Columns}) " +
                    $"do not match required dimensions ({inputSize}×{rank})", nameof(baseLayer));
            }

            if (_sharedMatrixB.Rows != rank || _sharedMatrixB.Columns != outputSize)
            {
                throw new ArgumentException(
                    $"Shared matrix B dimensions ({_sharedMatrixB.Rows}×{_sharedMatrixB.Columns}) " +
                    $"do not match required dimensions ({rank}×{outputSize})", nameof(baseLayer));
            }

            // Initialize shared gradient matrices if not already done
            if (_sharedMatrixAGradient == null)
            {
                _sharedMatrixAGradient = new Matrix<T>(inputSize, rank);
            }
            if (_sharedMatrixBGradient == null)
            {
                _sharedMatrixBGradient = new Matrix<T>(rank, outputSize);
            }
        }

        // Initialize layer-specific scaling factor to 1.0 (no initial effect)
        _layerScaling = NumOps.One;
        _layerScalingGradient = NumOps.Zero;

        // Mark as initialized so ParameterCount returns the reduced count
        _isInitialized = true;

        // Reallocate Parameters to the reduced size (just scaling factor + base if not frozen)
        Parameters = new Vector<T>(ParameterCount);

        // Update parameter vector with the scaling factor
        UpdateParametersFromScaling();
    }

    /// <summary>
    /// Initializes the shared trainable matrices used by all Tied-LoRA adapters.
    /// </summary>
    /// <param name="inputSize">The input dimension for the layers.</param>
    /// <param name="outputSize">The output dimension for the layers.</param>
    /// <param name="rank">The rank of the low-rank decomposition.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// This method must be called once before creating any Tied-LoRA adapters. It initializes the
    /// shared matrices A and B with random values that will be trained during fine-tuning.
    /// </para>
    /// <para>
    /// The shared matrices are initialized with Gaussian random values similar to Kaiming initialization
    /// for matrix A, and zeros for matrix B (so Tied-LoRA starts with no effect).
    /// </para>
    /// <para><b>For Beginners:</b> Call this once at the start before creating any Tied-LoRA layers:
    ///
    /// // Initialize shared trainable matrices (do this once)
    /// TiedLoRAAdapter&lt;double&gt;.InitializeSharedMatrices(inputSize: 784, outputSize: 128, rank: 8);
    ///
    /// // Now create Tied-LoRA adapters (they will use the shared matrices)
    /// var adapter1 = new TiedLoRAAdapter&lt;double&gt;(layer1, rank: 8, layerIndex: 0);
    /// var adapter2 = new TiedLoRAAdapter&lt;double&gt;(layer2, rank: 8, layerIndex: 1);
    ///
    /// All adapters share the same A and B matrices, but each has its own scaling factor!
    /// During training, the shared matrices learn the common adaptation pattern, while
    /// each layer's scaling factor controls how much to use that pattern.
    /// </para>
    /// </remarks>
    public static void InitializeSharedMatrices(int inputSize, int outputSize, int rank, int? seed = null)
    {
        lock (_sharedLock)
        {
            Random rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
            var ops = MathHelper.GetNumericOperations<T>();

            // Initialize matrix A (inputSize × rank) with Gaussian random values
            _sharedMatrixA = new Matrix<T>(inputSize, rank);
            T stddevA = ops.Sqrt(ops.Divide(ops.One, ops.FromDouble(rank)));
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < rank; j++)
                {
                    _sharedMatrixA[i, j] = ops.Multiply(ops.FromDouble(rng.NextGaussian()), stddevA);
                }
            }

            // Initialize matrix B (rank × outputSize) with zeros (no initial effect)
            _sharedMatrixB = new Matrix<T>(rank, outputSize);
            for (int i = 0; i < rank; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    _sharedMatrixB[i, j] = ops.Zero;
                }
            }

            // Initialize gradient matrices
            _sharedMatrixAGradient = new Matrix<T>(inputSize, rank);
            _sharedMatrixBGradient = new Matrix<T>(rank, outputSize);
        }
    }

    /// <summary>
    /// Resets the shared matrices and gradients (useful for testing or reinitializing).
    /// </summary>
    public static void ResetSharedMatrices()
    {
        lock (_sharedLock)
        {
            _sharedMatrixA = null;
            _sharedMatrixB = null;
            _sharedMatrixAGradient = null;
            _sharedMatrixBGradient = null;
        }
    }

    /// <summary>
    /// Resets the accumulated gradients for the shared matrices.
    /// Should be called after each optimization step.
    /// </summary>
    public static void ResetSharedGradients()
    {
        lock (_sharedLock)
        {
            var ops = MathHelper.GetNumericOperations<T>();

            if (_sharedMatrixAGradient != null)
            {
                int rows = _sharedMatrixAGradient.Rows;
                int cols = _sharedMatrixAGradient.Columns;
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        _sharedMatrixAGradient[i, j] = ops.Zero;
                    }
                }
            }

            if (_sharedMatrixBGradient != null)
            {
                int rows = _sharedMatrixBGradient.Rows;
                int cols = _sharedMatrixBGradient.Columns;
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        _sharedMatrixBGradient[i, j] = ops.Zero;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Updates the shared matrices using accumulated gradients.
    /// Should be called once after all layers have performed backward pass.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public static void UpdateSharedMatrices(T learningRate)
    {
        lock (_sharedLock)
        {
            if (_sharedMatrixA == null || _sharedMatrixB == null ||
                _sharedMatrixAGradient == null || _sharedMatrixBGradient == null)
            {
                return;
            }

            var ops = MathHelper.GetNumericOperations<T>();

            // Update matrix A
            for (int i = 0; i < _sharedMatrixA.Rows; i++)
            {
                for (int j = 0; j < _sharedMatrixA.Columns; j++)
                {
                    T update = ops.Multiply(_sharedMatrixAGradient[i, j], learningRate);
                    _sharedMatrixA[i, j] = ops.Subtract(_sharedMatrixA[i, j], update);
                }
            }

            // Update matrix B
            for (int i = 0; i < _sharedMatrixB.Rows; i++)
            {
                for (int j = 0; j < _sharedMatrixB.Columns; j++)
                {
                    T update = ops.Multiply(_sharedMatrixBGradient[i, j], learningRate);
                    _sharedMatrixB[i, j] = ops.Subtract(_sharedMatrixB[i, j], update);
                }
            }
        }
    }

    /// <summary>
    /// Gets whether the shared matrices have been initialized.
    /// </summary>
    public static bool AreSharedMatricesInitialized => _sharedMatrixA != null && _sharedMatrixB != null;

    /// <summary>
    /// Creates a Tied-LoRA-specific layer (not used since Tied-LoRA doesn't use standard LoRALayer).
    /// </summary>
    protected override LoRALayer<T> CreateLoRALayer(int rank, double alpha)
    {
        // Tied-LoRA doesn't use a standard LoRA layer, but we need to satisfy the base class
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        return new LoRALayer<T>(inputSize, outputSize, rank, alpha);
    }

    /// <summary>
    /// Performs the forward pass through the Tied-LoRA adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and Tied-LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// The Tied-LoRA forward pass computes:
    /// output = base_layer(input) + layerScaling * (B_shared * A_shared * input) * (alpha/rank)
    /// </para>
    /// <para><b>For Beginners:</b> This processes input through both the original layer and the
    /// Tied-LoRA adaptation:
    /// 1. Base layer processes the input (original behavior)
    /// 2. Tied-LoRA computes: input → A_shared (trainable) → B_shared (trainable) → layerScaling
    /// 3. The outputs are added together
    ///
    /// The key difference from standard LoRA: A and B are shared across all layers and ARE trained,
    /// but each layer only has one trainable parameter (layerScaling) to control the strength!
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input.Clone();

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        lock (_sharedLock)
        {
            if (_sharedMatrixA == null || _sharedMatrixB == null)
            {
                throw new InvalidOperationException("Shared matrices are not initialized");
            }

            // Tied-LoRA forward: layerScaling * (B_shared * A_shared * input) * (alpha/rank)
            int batchSize = input.Shape[0];
            int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
            int outputSize = GetOutputShape()[0];

            // Convert input to matrix [batchSize, inputSize]
            Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    inputMatrix[i, j] = input[i * inputSize + j];
                }
            }

            // Compute: input * A_shared → [batchSize, rank]
            Matrix<T> afterA = inputMatrix.Multiply(_sharedMatrixA);

            // Compute: afterA * B_shared → [batchSize, outputSize]
            Matrix<T> afterB = afterA.Multiply(_sharedMatrixB);
            _lastIntermediate = afterB.Clone(); // Store for backward pass

            // Apply layer-specific scaling and alpha/rank scaling
            T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));
            T totalScaling = NumOps.Multiply(_layerScaling, scaling);

            Matrix<T> scaled = new Matrix<T>(batchSize, outputSize);
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    scaled[i, j] = NumOps.Multiply(afterB[i, j], totalScaling);
                }
            }

            // Convert back to tensor
            Vector<T> tiedLoraOutputData = new Vector<T>(batchSize * outputSize);
            int idx = 0;
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    tiedLoraOutputData[idx++] = scaled[i, j];
                }
            }

            Tensor<T> tiedLoraOutput = new Tensor<T>(new[] { batchSize, outputSize }, tiedLoraOutputData);

            // Sum base output and Tied-LoRA output
            Tensor<T> result = new Tensor<T>(baseOutput.Shape);
            for (int i = 0; i < baseOutput.Length; i++)
            {
                result[i] = NumOps.Add(baseOutput[i], tiedLoraOutput[i]);
            }

            return result;
        }
    }

    /// <summary>
    /// Performs the backward pass through the Tied-LoRA adapter.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients for:
    /// 1. Layer-specific scaling factor (local to this layer)
    /// 2. Shared matrices A and B (accumulated across all layers)
    /// </para>
    /// <para><b>For Beginners:</b> This is where Tied-LoRA learns! During backpropagation:
    /// 1. Compute gradient for this layer's scaling factor
    /// 2. Accumulate gradients for shared matrices A and B (these are summed across all layers)
    /// 3. Update base layer if not frozen
    /// 4. Pass gradients back to earlier layers
    ///
    /// The shared matrices are updated once after all layers have computed their gradients,
    /// using the accumulated gradients from all layers.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastIntermediate == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        int batchSize = _lastInput.Shape[0];
        int inputSize = _lastInput.Shape.Length > 1 ? _lastInput.Shape[1] : _lastInput.Length;
        int outputSize = GetOutputShape()[0];
        int rank = Rank;

        // Convert gradient to matrix
        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradMatrix[i, j] = outputGradient[i * outputSize + j];
            }
        }

        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));

        lock (_sharedLock)
        {
            if (_sharedMatrixA == null || _sharedMatrixB == null ||
                _sharedMatrixAGradient == null || _sharedMatrixBGradient == null)
            {
                throw new InvalidOperationException("Shared matrices are not initialized");
            }

            // Compute gradient for layer scaling: sum over batch of (gradMatrix * _lastIntermediate * scaling)
            _layerScalingGradient = NumOps.Zero;
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    T grad = NumOps.Multiply(gradMatrix[i, j], _lastIntermediate[i, j]);
                    grad = NumOps.Multiply(grad, scaling);
                    _layerScalingGradient = NumOps.Add(_layerScalingGradient, grad);
                }
            }

            // Propagate gradient back through layer scaling: grad_afterB = gradMatrix * layerScaling * scaling
            T totalScaling = NumOps.Multiply(_layerScaling, scaling);
            Matrix<T> gradAfterB = new Matrix<T>(batchSize, outputSize);
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    gradAfterB[i, j] = NumOps.Multiply(gradMatrix[i, j], totalScaling);
                }
            }

            // Propagate through shared B: grad_afterA = gradAfterB * B^T
            Matrix<T> gradAfterA = gradAfterB.Multiply(_sharedMatrixB.Transpose());

            // Convert input to matrix for gradient computation
            Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    inputMatrix[i, j] = _lastInput[i * inputSize + j];
                }
            }

            // Compute intermediate: input * A_shared
            Matrix<T> afterA = inputMatrix.Multiply(_sharedMatrixA);

            // Accumulate gradient for shared B: B_grad += afterA^T * gradAfterB
            Matrix<T> afterATranspose = afterA.Transpose();
            Matrix<T> bGrad = afterATranspose.Multiply(gradAfterB);
            for (int i = 0; i < rank; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    _sharedMatrixBGradient[i, j] = NumOps.Add(_sharedMatrixBGradient[i, j], bGrad[i, j]);
                }
            }

            // Accumulate gradient for shared A: A_grad += input^T * gradAfterA
            Matrix<T> inputTranspose = inputMatrix.Transpose();
            Matrix<T> aGrad = inputTranspose.Multiply(gradAfterA);
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < rank; j++)
                {
                    _sharedMatrixAGradient[i, j] = NumOps.Add(_sharedMatrixAGradient[i, j], aGrad[i, j]);
                }
            }

            // Propagate through shared A: grad_input_tied = gradAfterA * A^T
            Matrix<T> tiedInputGrad = gradAfterA.Multiply(_sharedMatrixA.Transpose());

            // Backward through base layer
            Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

            // Sum input gradients from Tied-LoRA and base layer
            Vector<T> inputGradData = new Vector<T>(batchSize * inputSize);
            int idx = 0;
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    T tiedGrad = tiedInputGrad[i, j];
                    T baseGrad = baseInputGrad[i * inputSize + j];
                    inputGradData[idx++] = NumOps.Add(tiedGrad, baseGrad);
                }
            }

            // Update parameter gradients
            UpdateParameterGradientsFromScaling();

            return new Tensor<T>(new[] { batchSize, inputSize }, inputGradData);
        }
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// Tied-LoRA updates the layer-specific scaling factor locally, but shared matrices
    /// must be updated separately using UpdateSharedMatrices() after all layers have
    /// performed their backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This updates only the layer-specific scaling factor.
    /// The shared matrices A and B need to be updated separately after all layers finish
    /// their backward pass, because they accumulate gradients from all layers.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update layer-specific scaling factor
        T update = NumOps.Multiply(_layerScalingGradient, learningRate);
        _layerScaling = NumOps.Subtract(_layerScaling, update);

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromScaling();
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing parameters (layer scaling factor only, or base + scaling if base not frozen).</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateScalingFromParameters();
    }

    /// <summary>
    /// Updates the parameter vector from the current scaling factor value.
    /// </summary>
    private void UpdateParametersFromScaling()
    {
        int idx = 0;

        // Pack base layer parameters if not frozen
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        // Pack layer scaling factor
        Parameters[idx] = _layerScaling;
    }

    /// <summary>
    /// Updates the scaling factor from the parameter vector.
    /// </summary>
    private void UpdateScalingFromParameters()
    {
        int idx = 0;

        // Unpack base layer parameters if not frozen
        if (!_freezeBaseLayer)
        {
            int baseParamCount = _baseLayer.ParameterCount;
            Vector<T> baseParams = new Vector<T>(baseParamCount);
            for (int i = 0; i < baseParamCount; i++)
            {
                baseParams[i] = Parameters[idx++];
            }
            _baseLayer.SetParameters(baseParams);
        }

        // Unpack layer scaling factor
        _layerScaling = Parameters[idx];
    }

    /// <summary>
    /// Updates the parameter gradients vector from the scaling factor gradient.
    /// </summary>
    private void UpdateParameterGradientsFromScaling()
    {
        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // Pack base layer gradients if not frozen
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // Pack layer scaling gradient
        ParameterGradients[idx] = _layerScalingGradient;
    }

    /// <summary>
    /// Merges the Tied-LoRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with Tied-LoRA weights merged into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// This computes the full weight contribution from Tied-LoRA:
    /// W_tied = layerScaling * (B_shared * A_shared) * (alpha/rank)
    /// and adds it to the base layer's weights.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" the Tied-LoRA adaptation for deployment.
    /// After training, you can merge the adaptation into the original weights for faster inference.
    /// The merged layer will behave identically but without the Tied-LoRA overhead.
    ///
    /// Each layer gets a different merged result because the layer-specific scaling factor
    /// modulates how much of the shared adaptation is applied to that layer.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        lock (_sharedLock)
        {
            if (_sharedMatrixA == null || _sharedMatrixB == null)
            {
                throw new InvalidOperationException("Shared matrices are not initialized");
            }

            // Support DenseLayer and FullyConnectedLayer
            DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
            FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

            if (denseBase == null && fcBase == null)
            {
                throw new InvalidOperationException("TiedLoRAAdapter currently only supports DenseLayer or FullyConnectedLayer base layers");
            }

            int inputSize = GetInputShape()[0];
            int outputSize = GetOutputShape()[0];

            // Compute Tied-LoRA weight contribution: layerScaling * (B_shared * A_shared) * (alpha/rank)
            T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));
            T totalScaling = NumOps.Multiply(_layerScaling, scaling);

            // Multiply: intermediate = A_shared * B_shared
            Matrix<T> intermediate = _sharedMatrixA.Multiply(_sharedMatrixB);

            // Apply total scaling: W_tied = intermediate * totalScaling
            Matrix<T> tiedWeights = new Matrix<T>(inputSize, outputSize);
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    tiedWeights[i, j] = NumOps.Multiply(intermediate[i, j], totalScaling);
                }
            }

            // Transpose to match DenseLayer format [outputSize, inputSize]
            Matrix<T> tiedWeightsTransposed = tiedWeights.Transpose();

            // Get base layer parameters
            Vector<T> baseParams = _baseLayer.GetParameters();
            int weightCount = inputSize * outputSize;

            // Create merged parameters
            Vector<T> mergedParams = new Vector<T>(baseParams.Length);

            // Merge weights
            for (int i = 0; i < weightCount; i++)
            {
                int row = i / inputSize;
                int col = i % inputSize;
                mergedParams[i] = NumOps.Add(baseParams[i], tiedWeightsTransposed[row, col]);
            }

            // Copy biases unchanged
            for (int i = weightCount; i < baseParams.Length; i++)
            {
                mergedParams[i] = baseParams[i];
            }

            // Use helper method to clone base layer and preserve activation function
            return CreateMergedLayerWithClone(mergedParams);
        }
    }

    /// <summary>
    /// Resets the internal state of the Tied-LoRA adapter.
    /// </summary>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _lastInput = null;
        _lastIntermediate = null;
        _layerScalingGradient = NumOps.Zero;
    }
}
