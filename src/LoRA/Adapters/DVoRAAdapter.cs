using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// DVoRA (DoRA + VeRA) adapter - combines DoRA's magnitude-direction decomposition with VeRA's extreme parameter efficiency.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// DVoRA achieves the best of both worlds by:
/// - Applying DoRA's magnitude-direction decomposition for training stability
/// - Using VeRA's shared frozen matrices and scaling vectors for extreme parameter efficiency
/// - Applying the VeRA adaptation only to the direction component (not the magnitude)
/// </para>
/// <para>
/// <b>Mathematical Formulation:</b>
/// Given pre-trained weights W, DVoRA:
/// 1. Decomposes: W = m * d (magnitude and direction)
/// 2. Applies VeRA to direction: d' = d + d_scale * (B * A * input) * b_scale
/// 3. Normalizes direction: d_norm = d' / ||d'||
/// 4. Recomposes: W' = m * d_norm
///
/// Where:
/// - m: magnitude vector (trainable)
/// - d: direction matrix (normalized weight vectors)
/// - A, B: shared frozen random matrices (VeRA style)
/// - d_scale, b_scale: per-layer trainable scaling vectors (VeRA style)
/// </para>
/// <para>
/// <b>Research Context:</b>
/// DVoRA scores 5.0 vs VeRA's 4.3 (improvement of 16%) while maintaining ultra-low parameter counts.
/// It combines DoRA's superior training stability with VeRA's extreme parameter efficiency.
/// </para>
/// <para>
/// <b>For Beginners:</b> DVoRA is the ultimate parameter-efficient adapter.
///
/// Think of it as a hybrid technique:
/// - From DoRA: Separate magnitude (strength) from direction for stability
/// - From VeRA: Use shared random matrices and tiny scaling vectors for efficiency
/// - The magic: Apply VeRA's adaptation only to the direction, not the magnitude
///
/// Parameter comparison for 1000x1000 layer with rank=8:
/// - Full fine-tuning: 1,000,000 parameters
/// - Standard LoRA: 16,000 parameters (98.4% reduction)
/// - DoRA: 17,000 parameters (LoRA + magnitude vector)
/// - VeRA: 1,600 parameters (99.84% reduction)
/// - DVoRA: ~1,600 parameters (same as VeRA!) but with better performance (5.0 vs 4.3)
///
/// Benefits:
/// - ✅ Extremely parameter-efficient (10x fewer than standard LoRA, same as VeRA)
/// - ✅ Better performance than VeRA alone (5.0 vs 4.3 score)
/// - ✅ Training stability from DoRA's magnitude-direction decomposition
/// - ✅ Shared matrices reduce storage when adapting many layers
/// - ✅ Best choice for extreme memory constraints with quality requirements
///
/// Trade-offs:
/// - ⚠️ Requires shared matrix initialization before use
/// - ⚠️ Slightly more computation than VeRA (due to normalization)
/// - ⚠️ More complex than standard adapters (combines two techniques)
///
/// When to use DVoRA:
/// - Extreme memory constraints but need better quality than VeRA
/// - Mobile/edge deployment with limited resources
/// - Fine-tuning many layers efficiently
/// - When you want the absolute best parameter efficiency + quality balance
/// </para>
/// <para>
/// <b>References:</b>
/// - DoRA: "Weight-Decomposed Low-Rank Adaptation" (ICML 2024 Oral)
/// - VeRA: "Vector-based Random Matrix Adaptation"
/// - DVoRA: Combines both techniques for optimal efficiency and performance
/// </para>
/// </remarks>
public class DVoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Shared frozen random matrix A (inputSize × rank) used by all DVoRA adapters.
    /// </summary>
    /// <remarks>
    /// This matrix is initialized once globally and shared across all DVoRA layers.
    /// It is NEVER trained - it remains frozen at its random initialization values.
    /// This is the VeRA component of DVoRA.
    /// </remarks>
    private static Matrix<T>? _sharedMatrixA;

    /// <summary>
    /// Shared frozen random matrix B (rank × outputSize) used by all DVoRA adapters.
    /// </summary>
    /// <remarks>
    /// This matrix is initialized once globally and shared across all DVoRA layers.
    /// It is NEVER trained - it remains frozen at its random initialization values.
    /// This is the VeRA component of DVoRA.
    /// </remarks>
    private static Matrix<T>? _sharedMatrixB;

    /// <summary>
    /// Lock object for thread-safe shared matrix initialization.
    /// </summary>
    private static readonly object _initLock = new object();

    /// <summary>
    /// Magnitude component of the decomposed weights (scalar per output neuron).
    /// Trainable per-layer parameter.
    /// </summary>
    /// <remarks>
    /// The magnitude vector stores the L2 norm of each weight vector (one per output neuron).
    /// This is the DoRA component of DVoRA.
    /// </remarks>
    private Vector<T> _magnitude;

    /// <summary>
    /// Scaling vector d (outputSize) - trainable per-layer parameter.
    /// </summary>
    /// <remarks>
    /// This vector scales the VeRA output on a per-dimension basis.
    /// This is the VeRA component of DVoRA.
    /// </remarks>
    private Vector<T> _scalingVectorD;

    /// <summary>
    /// Scaling vector b (rank) - trainable per-layer parameter.
    /// </summary>
    /// <remarks>
    /// This vector scales the intermediate rank-dimensional representation.
    /// This is the VeRA component of DVoRA.
    /// </remarks>
    private Vector<T> _scalingVectorB;

    /// <summary>
    /// Gradient for magnitude vector computed during backpropagation.
    /// </summary>
    private Vector<T>? _magnitudeGradient;

    /// <summary>
    /// Gradient for scaling vector d computed during backpropagation.
    /// </summary>
    private Vector<T>? _scalingVectorDGradient;

    /// <summary>
    /// Gradient for scaling vector b computed during backpropagation.
    /// </summary>
    private Vector<T>? _scalingVectorBGradient;

    /// <summary>
    /// Cached normalized direction from the last forward pass, used in backpropagation.
    /// </summary>
    private Matrix<T>? _lastNormalizedDirection;

    /// <summary>
    /// Stored input from the forward pass, needed for gradient computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored intermediate value from forward pass, needed for backward pass.
    /// </summary>
    private Matrix<T>? _lastIntermediate;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// DVoRA parameters = base (if unfrozen) + LoRA layer + magnitude (outputSize) + d_scale (outputSize) + b_scale (rank).
    /// This is only slightly more than VeRA (adds magnitude vector) but much fewer than DoRA (no full LoRA matrices).
    /// Handles pre-initialization state by using fallback values when fields are null.
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            // Guard against pre-initialization state when base class constructor calls this property.
            int baseCount = _freezeBaseLayer ? 0 : _baseLayer.ParameterCount;
            int inputSize = GetInputShape()[0];
            int outputSize = GetOutputShape()[0];

            // We must include the LoRA slice size for base class compatibility, even though DVoRA doesn't train it.
            // The base constructor allocates parameter vectors based on this count, and packing/unpacking
            // methods expect the LoRA slice to be present, causing an IndexOutOfRange exception if omitted.
            int loraCount = _loraLayer?.ParameterCount ?? (inputSize * Rank + outputSize * Rank);

            int magnitudeCount = _magnitude?.Length ?? outputSize;
            int scalingDCount = _scalingVectorD?.Length ?? outputSize;
            int scalingBCount = _scalingVectorB?.Length ?? Rank;

            return baseCount + loraCount + magnitudeCount + scalingDCount + scalingBCount;
        }
    }

    /// <summary>
    /// Initializes a new DVoRA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with DVoRA.</param>
    /// <param name="rank">The rank of the low-rank decomposition (shared across all DVoRA layers).</param>
    /// <param name="alpha">The scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when shared matrices are not initialized.</exception>
    /// <remarks>
    /// <para>
    /// Before creating any DVoRA adapters, you must call InitializeSharedMatrices() once to set up
    /// the shared random matrices that all DVoRA layers will use.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a DVoRA adapter for a layer. Unlike standard LoRA,
    /// you must initialize the shared random matrices first by calling:
    ///
    /// DVoRAAdapter<T>.InitializeSharedMatrices(inputSize, outputSize, rank);
    ///
    /// This needs to be done once before creating any DVoRA adapters.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt
    /// - rank: How much compression (lower = fewer parameters)
    /// - alpha: How strong the adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true)
    /// </para>
    /// </remarks>
    public DVoRAAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (baseLayer == null)
        {
            throw new ArgumentNullException(nameof(baseLayer));
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Ensure shared matrices are initialized
        if (_sharedMatrixA == null || _sharedMatrixB == null)
        {
            throw new InvalidOperationException(
                "Shared matrices must be initialized before creating DVoRA adapters. " +
                "Call DVoRAAdapter<T>.InitializeSharedMatrices(inputSize, outputSize, rank) first.");
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

        // Initialize magnitude from base layer weights (DoRA component)
        _magnitude = new Vector<T>(outputSize);
        DecomposeWeights();

        // Initialize scaling vectors to ones (VeRA component - no initial effect)
        _scalingVectorD = new Vector<T>(outputSize);
        _scalingVectorB = new Vector<T>(rank);

        for (int i = 0; i < outputSize; i++)
        {
            _scalingVectorD[i] = NumOps.One;
        }

        for (int i = 0; i < rank; i++)
        {
            _scalingVectorB[i] = NumOps.One;
        }

        // Update parameter vector
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromComponents();
    }

    /// <summary>
    /// Initializes the shared random matrices used by all DVoRA adapters.
    /// </summary>
    /// <param name="inputSize">The input dimension for the layers.</param>
    /// <param name="outputSize">The output dimension for the layers.</param>
    /// <param name="rank">The rank of the low-rank decomposition.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// This method must be called once before creating any DVoRA adapters. It initializes the
    /// shared matrices A and B with random values that are frozen (never trained).
    /// </para>
    /// <para><b>For Beginners:</b> Call this once at the start before creating any DVoRA layers:
    ///
    /// // Initialize shared random matrices (do this once)
    /// DVoRAAdapter<double>.InitializeSharedMatrices(inputSize: 784, outputSize: 128, rank: 8);
    ///
    /// // Now create DVoRA adapters (they will use the shared matrices)
    /// var adapter1 = new DVoRAAdapter<double>(layer1, rank: 8);
    /// var adapter2 = new DVoRAAdapter<double>(layer2, rank: 8);
    ///
    /// All adapters share the same random A and B matrices, saving memory!
    /// </para>
    /// </remarks>
    public static void InitializeSharedMatrices(int inputSize, int outputSize, int rank, int? seed = null)
    {
        lock (_initLock)
        {
            if (inputSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(inputSize), "Input size must be greater than zero.");
            }
            if (outputSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputSize), "Output size must be greater than zero.");
            }
            if (rank <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(rank), "Rank must be greater than zero.");
            }

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

            // Initialize matrix B (rank × outputSize) with Gaussian random values
            _sharedMatrixB = new Matrix<T>(rank, outputSize);
            T stddevB = ops.Sqrt(ops.Divide(ops.One, ops.FromDouble(rank)));
            for (int i = 0; i < rank; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    _sharedMatrixB[i, j] = ops.Multiply(ops.FromDouble(rng.NextGaussian()), stddevB);
                }
            }
        }
    }

    /// <summary>
    /// Resets the shared matrices (useful for testing or reinitializing).
    /// </summary>
    public static void ResetSharedMatrices()
    {
        lock (_initLock)
        {
            _sharedMatrixA = null;
            _sharedMatrixB = null;
        }
    }

    /// <summary>
    /// Gets whether the shared matrices have been initialized.
    /// </summary>
    public static bool AreSharedMatricesInitialized => _sharedMatrixA != null && _sharedMatrixB != null;

    /// <summary>
    /// Decomposes the base layer's weights into magnitude and direction components.
    /// </summary>
    /// <remarks>
    /// This is the DoRA component of DVoRA. For each output neuron:
    /// 1. Extract the weight vector
    /// 2. Compute the L2 norm (magnitude)
    /// 3. Store the magnitude
    ///
    /// The direction is implicitly W/||W|| and doesn't need to be stored separately.
    /// </remarks>
    private void DecomposeWeights()
    {
        Vector<T> baseParams = _baseLayer.GetParameters();
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // For each output neuron, compute the magnitude of its weight vector
        for (int i = 0; i < outputSize; i++)
        {
            T sumSquares = NumOps.Zero;

            // Sum squares of all weights for this output neuron
            for (int j = 0; j < inputSize; j++)
            {
                int idx = i * inputSize + j;
                if (idx < weightCount && idx < baseParams.Length)
                {
                    T weight = baseParams[idx];
                    sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(weight, weight));
                }
            }

            // Magnitude is the L2 norm
            _magnitude[i] = NumOps.Sqrt(sumSquares);

            // Ensure magnitude is never zero (for numerical stability)
            if (NumOps.Equals(_magnitude[i], NumOps.Zero))
            {
                _magnitude[i] = NumOps.FromDouble(1e-8);
            }
        }
    }

    /// <summary>
    /// Normalizes a matrix row-wise (each row becomes a unit vector).
    /// </summary>
    /// <param name="matrix">The matrix to normalize.</param>
    /// <returns>Row-normalized matrix where each row has unit L2 norm.</returns>
    private Matrix<T> NormalizeRows(Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        Matrix<T> normalized = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            // Compute L2 norm of row
            T sumSquares = NumOps.Zero;
            for (int j = 0; j < cols; j++)
            {
                T val = matrix[i, j];
                sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(val, val));
            }

            T norm = NumOps.Sqrt(sumSquares);

            // Avoid division by zero
            if (NumOps.Equals(norm, NumOps.Zero))
            {
                norm = NumOps.FromDouble(1e-8);
            }

            // Normalize row
            for (int j = 0; j < cols; j++)
            {
                normalized[i, j] = NumOps.Divide(matrix[i, j], norm);
            }
        }

        return normalized;
    }

    /// <summary>
    /// Recomposes weights from magnitude and direction components.
    /// </summary>
    /// <param name="direction">The normalized direction matrix.</param>
    /// <returns>The full weight matrix (magnitude * direction).</returns>
    private Matrix<T> RecomposeWeights(Matrix<T> direction)
    {
        int outputSize = direction.Rows;
        int inputSize = direction.Columns;
        Matrix<T> weights = new Matrix<T>(outputSize, inputSize);

        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                weights[i, j] = NumOps.Multiply(_magnitude[i], direction[i, j]);
            }
        }

        return weights;
    }

    /// <summary>
    /// Creates a dummy LoRA layer (not used since DVoRA uses custom logic).
    /// </summary>
    protected override LoRALayer<T> CreateLoRALayer(int rank, double alpha)
    {
        // DVoRA doesn't use a standard LoRA layer, but we need to satisfy the base class
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        return new LoRALayer<T>(inputSize, outputSize, rank, alpha);
    }

    /// <summary>
    /// Performs the forward pass through the DVoRA adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output combining base layer with DVoRA-adapted weights.</returns>
    /// <remarks>
    /// <para>
    /// The DVoRA forward pass combines DoRA and VeRA:
    /// 1. Gets base layer weights W
    /// 2. Computes direction: d = W / ||W|| (DoRA)
    /// 3. Applies VeRA to direction: d' = d + d_scale * (B * A * input) * b_scale (VeRA)
    /// 4. Normalizes adapted direction: d_norm = d' / ||d'|| (DoRA)
    /// 5. Recomposes weights: W' = m * d_norm (DoRA)
    /// 6. Computes output: y = input @ W'^T
    /// </para>
    /// <para><b>For Beginners:</b> This is where DVoRA combines both techniques:
    ///
    /// DoRA part:
    /// - Split weights into magnitude (strength) and direction
    /// - Keep magnitude separate, work only with direction
    ///
    /// VeRA part:
    /// - Apply shared random matrices + tiny scaling vectors to the direction
    ///
    /// Final step:
    /// - Normalize the adjusted direction
    /// - Multiply magnitude back in
    /// - Use these hybrid-adapted weights for prediction
    ///
    /// Result: Stability of DoRA + efficiency of VeRA = best of both worlds!
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input.Clone();

        if (!_freezeBaseLayer)
        {
            _baseLayer.Forward(input);
        }

        // Get base layer parameters and extract weights
        Vector<T> baseParams = _baseLayer.GetParameters();
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Extract weight matrix from base layer
        Matrix<T> baseWeights = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                int weightIdx = i * inputSize + j;
                if (weightIdx < weightCount && weightIdx < baseParams.Length)
                {
                    baseWeights[i, j] = baseParams[weightIdx];
                }
                else
                {
                    baseWeights[i, j] = NumOps.Zero;
                }
            }
        }

        // Compute base direction (W / ||W||) - DoRA component
        Matrix<T> baseDirection = NormalizeRows(baseWeights);

        // Apply VeRA to get direction delta
        int batchSize = input.Shape[0];
        int rank = _scalingVectorB.Length;

        // Convert input to matrix [batchSize, inputSize]
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = input[i * inputSize + j];
            }
        }

        // VeRA forward: (B * A * input) with scaling vectors
        // Compute: input * A (shared, frozen) → [batchSize, rank]
        Matrix<T> afterA = inputMatrix.Multiply(_sharedMatrixA!);

        // Apply scaling vector b element-wise: afterA * diag(b) → [batchSize, rank]
        Matrix<T> afterB = new Matrix<T>(batchSize, rank);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                afterB[i, j] = NumOps.Multiply(afterA[i, j], _scalingVectorB[j]);
            }
        }

        // Compute: afterB * B (shared, frozen) → [batchSize, outputSize]
        Matrix<T> afterSharedB = afterB.Multiply(_sharedMatrixB!);
        _lastIntermediate = afterSharedB.Clone();

        // Apply scaling vector d element-wise: afterSharedB * diag(d) → [batchSize, outputSize]
        Matrix<T> veraContribution = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                veraContribution[i, j] = NumOps.Multiply(afterSharedB[i, j], _scalingVectorD[j]);
            }
        }

        // Apply alpha/rank scaling
        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));

        // Compute VeRA weight delta directly from matrices: delta = d .* (B * A_scaled)^T
        // This is deterministic and independent of the input batch

        // First compute A_scaled = A * diag(b)
        Matrix<T> aScaled = new Matrix<T>(inputSize, rank);
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                aScaled[i, j] = NumOps.Multiply(_sharedMatrixA![i, j], _scalingVectorB[j]);
            }
        }

        // Compute intermediate = A_scaled * B → [inputSize, outputSize]
        Matrix<T> intermediate = aScaled.Multiply(_sharedMatrixB!);

        // Apply d scaling and transpose to get weight delta [outputSize, inputSize]
        Matrix<T> veraWeightDelta = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                veraWeightDelta[i, j] = NumOps.Multiply(
                    NumOps.Multiply(intermediate[j, i], _scalingVectorD[i]),
                    scaling);
            }
        }

        // Add VeRA delta to base direction: d' = d + delta
        Matrix<T> adaptedDirection = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                adaptedDirection[i, j] = NumOps.Add(baseDirection[i, j], veraWeightDelta[i, j]);
            }
        }

        // Normalize the adapted direction: d_norm = d' / ||d'|| - DoRA component
        _lastNormalizedDirection = NormalizeRows(adaptedDirection);

        // Recompose weights: W' = m * d_norm - DoRA component
        Matrix<T> finalWeights = RecomposeWeights(_lastNormalizedDirection);

        // Compute output: y = input @ W'^T + bias
        Matrix<T> outputMatrix = inputMatrix.Multiply(finalWeights.Transpose());

        // Extract biases from base layer parameters
        Vector<T> biases = new Vector<T>(outputSize);
        for (int i = 0; i < outputSize; i++)
        {
            int biasIdx = weightCount + i;
            biases[i] = biasIdx < baseParams.Length ? baseParams[biasIdx] : NumOps.Zero;
        }

        // Add bias to each row of output
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                outputMatrix[i, j] = NumOps.Add(outputMatrix[i, j], biases[j]);
            }
        }

        // Convert back to tensor
        Vector<T> outputData = new Vector<T>(batchSize * outputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                outputData[idx++] = outputMatrix[i, j];
            }
        }

        return new Tensor<T>(new[] { batchSize, outputSize }, outputData);
    }

    /// <summary>
    /// Performs the backward pass through the DVoRA adapter.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients for:
    /// 1. Magnitude parameters (DoRA component, one per output neuron)
    /// 2. Scaling vectors d and b (VeRA component, per-layer)
    /// 3. Base layer weights (if not frozen)
    ///
    /// The shared matrices A and B remain frozen and are never updated.
    /// </para>
    /// <para><b>For Beginners:</b> This is where DVoRA learns! During backpropagation:
    /// 1. Compute gradients for magnitude (DoRA learning)
    /// 2. Compute gradients for scaling vectors d and b (VeRA learning)
    /// 3. Shared matrices A and B stay frozen (VeRA efficiency)
    /// 4. Pass gradients back to earlier layers
    ///
    /// We only train: magnitude + d + b = very few parameters!
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastNormalizedDirection == null || _lastIntermediate == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        int batchSize = outputGradient.Shape[0];
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int rank = _scalingVectorB.Length;

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

        // Compute magnitude gradients (DoRA component)
        _magnitudeGradient = new Vector<T>(outputSize);
        for (int i = 0; i < outputSize; i++)
        {
            T gradSum = NumOps.Zero;
            // Get the normalized direction vector for the current output unit
            Vector<T> normalizedDirectionRow = _lastNormalizedDirection.GetRow(i);

            for (int b = 0; b < batchSize; b++)
            {
                // Extract the input activation row for the current batch
                Vector<T> inputActivationRow = new Vector<T>(inputSize);
                for (int k = 0; k < inputSize; k++)
                {
                    inputActivationRow[k] = _lastInput[b * inputSize + k];
                }

                // Compute scalar projection: proj = Dot(_lastNormalizedDirection[i], inputActivationRow[b])
                T proj = NumOps.Zero;
                for (int k = 0; k < inputSize; k++)
                {
                    proj = NumOps.Add(proj, NumOps.Multiply(normalizedDirectionRow[k], inputActivationRow[k]));
                }

                // Compute gradient contribution: gradContribution = NumOps.Mul(gradMatrix[b,i], proj)
                T gradContribution = NumOps.Multiply(gradMatrix[b, i], proj);

                // Accumulate gradContribution into _magnitudeGradient[i]
                gradSum = NumOps.Add(gradSum, gradContribution);
            }
            _magnitudeGradient[i] = gradSum;
        }

        // Compute gradient for scaling vector d (VeRA component)
        _scalingVectorDGradient = new Vector<T>(outputSize);
        for (int j = 0; j < outputSize; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < batchSize; i++)
            {
                T grad = NumOps.Multiply(gradMatrix[i, j], _lastIntermediate[i, j]);
                grad = NumOps.Multiply(grad, scaling);
                sum = NumOps.Add(sum, grad);
            }
            _scalingVectorDGradient[j] = sum;
        }

        // Propagate gradient back through d scaling
        Matrix<T> gradAfterSharedB = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradAfterSharedB[i, j] = NumOps.Multiply(
                    NumOps.Multiply(gradMatrix[i, j], _scalingVectorD[j]),
                    scaling);
            }
        }

        // Propagate through shared B
        Matrix<T> gradAfterB = gradAfterSharedB.Multiply(_sharedMatrixB!.Transpose());

        // Convert input to matrix for gradient computation
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = _lastInput[i * inputSize + j];
            }
        }

        // Compute intermediate: input * A
        Matrix<T> afterA = inputMatrix.Multiply(_sharedMatrixA!);

        // Compute gradient for scaling vector b (VeRA component)
        _scalingVectorBGradient = new Vector<T>(rank);
        for (int j = 0; j < rank; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < batchSize; i++)
            {
                T grad = NumOps.Multiply(gradAfterB[i, j], afterA[i, j]);
                sum = NumOps.Add(sum, grad);
            }
            _scalingVectorBGradient[j] = sum;
        }

        // Propagate gradient back through b scaling
        Matrix<T> gradAfterA = new Matrix<T>(batchSize, rank);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                gradAfterA[i, j] = NumOps.Multiply(gradAfterB[i, j], _scalingVectorB[j]);
            }
        }

        // Propagate through shared A
        Matrix<T> veraInputGrad = gradAfterA.Multiply(_sharedMatrixA!.Transpose());

        // Backward through base layer (if not frozen)
        Tensor<T> baseInputGrad;
        if (!_freezeBaseLayer)
        {
            baseInputGrad = _baseLayer.Backward(outputGradient);
        }
        else
        {
            // Create zero gradient for base layer
            baseInputGrad = new Tensor<T>(_lastInput.Shape);
        }

        // Sum input gradients from DVoRA and base layer
        Vector<T> inputGradData = new Vector<T>(batchSize * inputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                T dvoraGrad = veraInputGrad[i, j];
                T baseGrad = baseInputGrad[i * inputSize + j];
                inputGradData[idx++] = NumOps.Add(dvoraGrad, baseGrad);
            }
        }

        // Update parameter gradients
        UpdateParameterGradientsFromComponents();

        return new Tensor<T>(new[] { batchSize, inputSize }, inputGradData);
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_magnitudeGradient == null || _scalingVectorDGradient == null || _scalingVectorBGradient == null)
        {
            return;
        }

        // Update magnitude parameters (DoRA component)
        for (int i = 0; i < _magnitude.Length; i++)
        {
            T update = NumOps.Multiply(_magnitudeGradient[i], learningRate);
            _magnitude[i] = NumOps.Subtract(_magnitude[i], update);

            // Ensure magnitude stays positive
            if (NumOps.LessThan(_magnitude[i], NumOps.FromDouble(1e-8)))
            {
                _magnitude[i] = NumOps.FromDouble(1e-8);
            }
        }

        // Update scaling vector d (VeRA component)
        for (int i = 0; i < _scalingVectorD.Length; i++)
        {
            T update = NumOps.Multiply(_scalingVectorDGradient[i], learningRate);
            _scalingVectorD[i] = NumOps.Subtract(_scalingVectorD[i], update);
        }

        // Update scaling vector b (VeRA component)
        for (int i = 0; i < _scalingVectorB.Length; i++)
        {
            T update = NumOps.Multiply(_scalingVectorBGradient[i], learningRate);
            _scalingVectorB[i] = NumOps.Subtract(_scalingVectorB[i], update);
        }

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromComponents();
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing all DVoRA parameters (magnitude, d, b).</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateComponentsFromParameters();
    }

    /// <summary>
    /// Updates the parameter vector from the current component states.
    /// </summary>
    private void UpdateParametersFromComponents()
    {
        int idx = 0;

        // Pack base layer parameters (if not frozen)
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        // Pack LoRA parameters (required for base class compatibility)
        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            Parameters[idx++] = loraParams[i];
        }

        // Pack magnitude parameters
        for (int i = 0; i < _magnitude.Length; i++)
        {
            Parameters[idx++] = _magnitude[i];
        }

        // Pack scaling vector d
        for (int i = 0; i < _scalingVectorD.Length; i++)
        {
            Parameters[idx++] = _scalingVectorD[i];
        }

        // Pack scaling vector b
        for (int i = 0; i < _scalingVectorB.Length; i++)
        {
            Parameters[idx++] = _scalingVectorB[i];
        }
    }

    /// <summary>
    /// Updates the components from the parameter vector.
    /// </summary>
    private void UpdateComponentsFromParameters()
    {
        int idx = 0;

        // Unpack base layer parameters (if not frozen)
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

        // Unpack LoRA parameters (required for base class compatibility)
        int loraParamCount = _loraLayer.ParameterCount;
        Vector<T> loraParams = new Vector<T>(loraParamCount);
        for (int i = 0; i < loraParamCount; i++)
        {
            loraParams[i] = Parameters[idx++];
        }
        _loraLayer.SetParameters(loraParams);

        // Unpack magnitude parameters
        for (int i = 0; i < _magnitude.Length; i++)
        {
            _magnitude[i] = Parameters[idx++];
        }

        // Unpack scaling vector d
        for (int i = 0; i < _scalingVectorD.Length; i++)
        {
            _scalingVectorD[i] = Parameters[idx++];
        }

        // Unpack scaling vector b
        for (int i = 0; i < _scalingVectorB.Length; i++)
        {
            _scalingVectorB[i] = Parameters[idx++];
        }
    }

    /// <summary>
    /// Updates the parameter gradients vector from the component gradients.
    /// </summary>
    private void UpdateParameterGradientsFromComponents()
    {
        if (_magnitudeGradient == null || _scalingVectorDGradient == null || _scalingVectorBGradient == null)
        {
            return;
        }

        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // Pack base layer gradients (if not frozen)
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // Pack magnitude gradients
        for (int i = 0; i < _magnitudeGradient.Length; i++)
        {
            ParameterGradients[idx++] = _magnitudeGradient[i];
        }

        // Pack scaling vector d gradients
        for (int i = 0; i < _scalingVectorDGradient.Length; i++)
        {
            ParameterGradients[idx++] = _scalingVectorDGradient[i];
        }

        // Pack scaling vector b gradients
        for (int i = 0; i < _scalingVectorBGradient.Length; i++)
        {
            ParameterGradients[idx++] = _scalingVectorBGradient[i];
        }
    }

    /// <summary>
    /// Merges the DVoRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with DVoRA weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not supported for merging.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a final layer with the DVoRA adaptations baked in.
    /// The merged weights combine DoRA's magnitude-direction decomposition with VeRA's adaptation:
    /// W' = m * normalize(d + VeRA_contribution)
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your DVoRA adaptation for deployment.
    ///
    /// After training with DVoRA, you probably want to deploy a simpler model without
    /// all the DVoRA machinery. This method creates that simpler model by:
    /// 1. Computing the VeRA contribution to direction
    /// 2. Adding it to the base direction
    /// 3. Normalizing the result (DoRA)
    /// 4. Multiplying by magnitude (DoRA)
    /// 5. Creating a new layer with these merged weights
    ///
    /// The result is a standard layer that behaves like your DVoRA-adapted model
    /// but is faster to run because it doesn't need the DVoRA computation at runtime.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        if (_sharedMatrixA == null || _sharedMatrixB == null)
        {
            throw new InvalidOperationException("Shared matrices are not initialized");
        }

        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("DVoRAAdapter currently only supports DenseLayer or FullyConnectedLayer base layers for merging");
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int rank = _scalingVectorB.Length;

        // Get base layer weights
        Vector<T> baseParams = _baseLayer.GetParameters();
        Matrix<T> baseWeights = new Matrix<T>(outputSize, inputSize);
        int weightCount = inputSize * outputSize;

        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                int weightIdx = i * inputSize + j;
                if (weightIdx < weightCount && weightIdx < baseParams.Length)
                {
                    baseWeights[i, j] = baseParams[weightIdx];
                }
            }
        }

        // Compute base direction
        Matrix<T> baseDirection = NormalizeRows(baseWeights);

        // Compute VeRA weight contribution: d * B * A * b * scaling
        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));

        // Apply b scaling to A: A_scaled = A * diag(b)
        Matrix<T> aScaled = new Matrix<T>(inputSize, rank);
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                aScaled[i, j] = NumOps.Multiply(_sharedMatrixA[i, j], _scalingVectorB[j]);
            }
        }

        // Multiply by B: intermediate = A_scaled * B
        Matrix<T> intermediate = aScaled.Multiply(_sharedMatrixB);

        // Apply d scaling: W_vera = intermediate * diag(d) * scaling
        Matrix<T> veraWeights = new Matrix<T>(inputSize, outputSize);
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                veraWeights[i, j] = NumOps.Multiply(
                    NumOps.Multiply(intermediate[i, j], _scalingVectorD[j]),
                    scaling);
            }
        }

        // Transpose to match direction matrix format [outputSize, inputSize]
        Matrix<T> veraWeightsTransposed = veraWeights.Transpose();

        // Add VeRA contribution to base direction
        Matrix<T> adaptedDirection = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                adaptedDirection[i, j] = NumOps.Add(baseDirection[i, j], veraWeightsTransposed[i, j]);
            }
        }

        // Normalize the adapted direction
        Matrix<T> normalizedDirection = NormalizeRows(adaptedDirection);

        // Recompose with magnitude: W' = m * d_norm
        Matrix<T> finalWeights = RecomposeWeights(normalizedDirection);

        // Create merged parameters (weights + biases)
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Copy merged weights
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                int weightIdx = i * inputSize + j;
                mergedParams[weightIdx] = finalWeights[i, j];
            }
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
    /// Resets the internal state of the DVoRA adapter.
    /// </summary>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _lastInput = null;
        _lastNormalizedDirection = null;
        _lastIntermediate = null;
        _magnitudeGradient = null;
        _scalingVectorDGradient = null;
        _scalingVectorBGradient = null;
    }
}
