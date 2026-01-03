using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// VeRA (Vector-based Random Matrix Adaptation) adapter - an extreme parameter-efficient variant of LoRA.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// VeRA achieves 10x fewer trainable parameters than standard LoRA by:
/// - Using a single pair of random low-rank matrices (A and B) shared across ALL layers
/// - Freezing these shared matrices (they are never trained)
/// - Training only small scaling vectors (d and b) that are specific to each layer
/// </para>
/// <para>
/// The forward computation is: output = base_layer(input) + d * (B * A * input) * b
/// where d and b are trainable vectors, and A and B are frozen shared matrices.
/// </para>
/// <para><b>For Beginners:</b> VeRA is an ultra-efficient version of LoRA for extreme memory constraints.
///
/// Think of the difference this way:
/// - Standard LoRA: Each layer has its own pair of small matrices (A and B) that are trained
/// - VeRA: ALL layers share the same random matrices (A and B) which are frozen. Only tiny
///   scaling vectors are trained per layer.
///
/// Example parameter comparison for a 1000x1000 layer with rank=8:
/// - Full fine-tuning: 1,000,000 parameters
/// - Standard LoRA (rank=8): 16,000 parameters (98.4% reduction)
/// - VeRA (rank=8): ~1,600 parameters (99.84% reduction) - 10x fewer than LoRA!
///
/// Trade-offs:
/// - ✅ Extreme parameter efficiency (10x fewer than LoRA)
/// - ✅ Very low memory footprint
/// - ✅ Shared matrices reduce storage when adapting many layers
/// - ⚠️ Slightly less flexible than standard LoRA (shared random projection)
/// - ⚠️ Performance may be marginally lower than LoRA in some cases
///
/// When to use VeRA:
/// - Extreme memory constraints (mobile, edge devices)
/// - Fine-tuning many layers with limited resources
/// - Rapid prototyping with minimal parameter overhead
/// - When LoRA is still too expensive
/// </para>
/// </remarks>
public class VeRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Shared frozen random matrix A (inputSize × rank) used by all VeRA adapters.
    /// </summary>
    /// <remarks>
    /// This matrix is initialized once globally and shared across all VeRA layers.
    /// It is NEVER trained - it remains frozen at its random initialization values.
    /// </remarks>
    private static Matrix<T>? _sharedMatrixA;

    /// <summary>
    /// Shared frozen random matrix B (rank × outputSize) used by all VeRA adapters.
    /// </summary>
    /// <remarks>
    /// This matrix is initialized once globally and shared across all VeRA layers.
    /// It is NEVER trained - it remains frozen at its random initialization values.
    /// </remarks>
    private static Matrix<T>? _sharedMatrixB;

    /// <summary>
    /// Lock object for thread-safe shared matrix initialization.
    /// </summary>
    private static readonly object _initLock = new object();

    /// <summary>
    /// Scaling vector d (outputSize) - trainable per-layer parameter.
    /// </summary>
    /// <remarks>
    /// This vector scales the output of the shared matrices on a per-dimension basis.
    /// It is initialized to ones so VeRA has no effect initially.
    /// </remarks>
    private Vector<T> _scalingVectorD;

    /// <summary>
    /// Scaling vector b (rank) - trainable per-layer parameter.
    /// </summary>
    /// <remarks>
    /// This vector scales the intermediate rank-dimensional representation.
    /// It is initialized to ones so VeRA has no effect initially.
    /// </remarks>
    private Vector<T> _scalingVectorB;

    /// <summary>
    /// Gradient for scaling vector d computed during backpropagation.
    /// </summary>
    private Vector<T>? _scalingVectorDGradient;

    /// <summary>
    /// Gradient for scaling vector b computed during backpropagation.
    /// </summary>
    private Vector<T>? _scalingVectorBGradient;

    /// <summary>
    /// Stored input from the forward pass, needed for gradient computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored intermediate value (B * A * input) from forward pass, needed for backward pass.
    /// </summary>
    private Matrix<T>? _lastIntermediate;

    /// <summary>
    /// Gets the total number of trainable parameters (only the scaling vectors d and b).
    /// </summary>
    /// <remarks>
    /// VeRA only trains the scaling vectors, not the shared matrices.
    /// For a layer with outputSize and rank r, this is: outputSize + rank.
    /// This is typically 10x fewer parameters than standard LoRA.
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int veraParams = _scalingVectorD.Length + _scalingVectorB.Length;
            return _freezeBaseLayer ? veraParams : (_baseLayer.ParameterCount + veraParams);
        }
    }

    /// <summary>
    /// Initializes a new VeRA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with VeRA.</param>
    /// <param name="rank">The rank of the low-rank decomposition (shared across all VeRA layers).</param>
    /// <param name="alpha">The scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when rank is invalid or shared matrices are not initialized.</exception>
    /// <remarks>
    /// <para>
    /// Before creating any VeRA adapters, you must call InitializeSharedMatrices() once to set up
    /// the shared random matrices that all VeRA layers will use.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a VeRA adapter for a layer. Unlike standard LoRA,
    /// you must initialize the shared random matrices first by calling:
    ///
    /// VeRAAdapter&lt;T&gt;.InitializeSharedMatrices(inputSize, outputSize, rank);
    ///
    /// This needs to be done once before creating any VeRA adapters.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt
    /// - rank: How much compression (lower = fewer parameters)
    /// - alpha: How strong the VeRA adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true)
    /// </para>
    /// </remarks>
    public VeRAAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
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
                "Shared matrices must be initialized before creating VeRA adapters. " +
                "Call VeRAAdapter<T>.InitializeSharedMatrices(inputSize, outputSize, rank) first.");
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

        // Initialize scaling vectors to ones (so VeRA has no initial effect)
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

        // Update parameter vector with scaling vectors
        UpdateParametersFromVectors();
    }

    /// <summary>
    /// Initializes the shared random matrices used by all VeRA adapters.
    /// </summary>
    /// <param name="inputSize">The input dimension for the layers.</param>
    /// <param name="outputSize">The output dimension for the layers.</param>
    /// <param name="rank">The rank of the low-rank decomposition.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// This method must be called once before creating any VeRA adapters. It initializes the
    /// shared matrices A and B with random values that are frozen (never trained).
    /// </para>
    /// <para>
    /// The shared matrices are initialized with Gaussian random values similar to Kaiming initialization.
    /// Once initialized, they remain frozen and are shared across all VeRA adapters with matching dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> Call this once at the start before creating any VeRA layers:
    ///
    /// // Initialize shared random matrices (do this once)
    /// VeRAAdapter&lt;double&gt;.InitializeSharedMatrices(inputSize: 784, outputSize: 128, rank: 8);
    ///
    /// // Now create VeRA adapters (they will use the shared matrices)
    /// var adapter1 = new VeRAAdapter&lt;double&gt;(layer1, rank: 8);
    /// var adapter2 = new VeRAAdapter&lt;double&gt;(layer2, rank: 8);
    ///
    /// All adapters share the same random A and B matrices, saving memory!
    /// </para>
    /// </remarks>
    public static void InitializeSharedMatrices(int inputSize, int outputSize, int rank, int? seed = null)
    {
        lock (_initLock)
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
    /// Creates a VeRA-specific layer (not used since VeRA doesn't use LoRALayer).
    /// </summary>
    /// <remarks>
    /// VeRA doesn't use the standard LoRALayer, so this creates a dummy layer.
    /// The actual VeRA computation is handled in Forward() and Backward() methods.
    /// </remarks>
    protected override LoRALayer<T> CreateLoRALayer(int rank, double alpha)
    {
        // VeRA doesn't use a standard LoRA layer, but we need to satisfy the base class
        // Create a minimal LoRA layer that won't be used
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        return new LoRALayer<T>(inputSize, outputSize, rank, alpha);
    }

    /// <summary>
    /// Performs the forward pass through the VeRA adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and VeRA output.</returns>
    /// <remarks>
    /// <para>
    /// The VeRA forward pass computes: output = base_layer(input) + d * (B * A * input) * b * scaling
    /// where d and b are trainable scaling vectors, A and B are frozen shared matrices,
    /// and scaling = alpha/rank.
    /// </para>
    /// <para><b>For Beginners:</b> This processes input through both the original layer and the VeRA adaptation:
    /// 1. Base layer processes the input (original behavior)
    /// 2. VeRA computes: input → A (shared) → b (scale) → B (shared) → d (scale)
    /// 3. The outputs are added together
    ///
    /// The key difference from standard LoRA: A and B are shared and frozen, only d and b are trained!
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input.Clone();

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // VeRA forward: d * (B * A * input) * b * scaling
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

        // Compute: input * A (shared, frozen) → [batchSize, rank]
        Matrix<T> afterA = inputMatrix.Multiply(_sharedMatrixA!);

        // Apply scaling vector b element-wise: afterA * diag(b) → [batchSize, rank]
        Matrix<T> afterB = new Matrix<T>(batchSize, _scalingVectorB.Length);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < _scalingVectorB.Length; j++)
            {
                afterB[i, j] = NumOps.Multiply(afterA[i, j], _scalingVectorB[j]);
            }
        }

        // Compute: afterB * B (shared, frozen) → [batchSize, outputSize]
        Matrix<T> afterSharedB = afterB.Multiply(_sharedMatrixB!);
        _lastIntermediate = afterSharedB.Clone(); // Store for backward pass

        // Apply scaling vector d element-wise: afterSharedB * diag(d) → [batchSize, outputSize]
        Matrix<T> afterD = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                afterD[i, j] = NumOps.Multiply(afterSharedB[i, j], _scalingVectorD[j]);
            }
        }

        // Apply alpha/rank scaling
        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));
        afterD = afterD.Multiply(scaling);

        // Convert back to tensor
        Vector<T> veraOutputData = new Vector<T>(batchSize * outputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                veraOutputData[idx++] = afterD[i, j];
            }
        }

        Tensor<T> veraOutput = new Tensor<T>(new[] { batchSize, outputSize }, veraOutputData);

        // Sum base output and VeRA output
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], veraOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through the VeRA adapter.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients ONLY for the scaling vectors d and b.
    /// The shared matrices A and B remain frozen and are never updated.
    /// </para>
    /// <para><b>For Beginners:</b> This is where VeRA learns! During backpropagation:
    /// 1. Compute gradients for scaling vectors d and b (these are trained)
    /// 2. Shared matrices A and B are NOT updated (they stay frozen)
    /// 3. Pass gradients back to earlier layers
    ///
    /// This is why VeRA is so efficient - we only train tiny scaling vectors!
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

        // Compute gradient for d: sum over batch of (gradMatrix * _lastIntermediate * scaling)
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

        // Propagate gradient back through d scaling: grad_afterSharedB = gradMatrix * diag(d) * scaling
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

        // Propagate through shared B: grad_afterB = gradAfterSharedB * B^T
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

        // Compute gradient for b: sum over batch of (gradAfterB * afterA)
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

        // Propagate gradient back through b scaling: grad_afterA = gradAfterB * diag(b)
        Matrix<T> gradAfterA = new Matrix<T>(batchSize, rank);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                gradAfterA[i, j] = NumOps.Multiply(gradAfterB[i, j], _scalingVectorB[j]);
            }
        }

        // Propagate through shared A: grad_input_vera = gradAfterA * A^T
        Matrix<T> veraInputGrad = gradAfterA.Multiply(_sharedMatrixA!.Transpose());

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Sum input gradients from VeRA and base layer
        Vector<T> inputGradData = new Vector<T>(batchSize * inputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                T veraGrad = veraInputGrad[i, j];
                T baseGrad = baseInputGrad[i * inputSize + j];
                inputGradData[idx++] = NumOps.Add(veraGrad, baseGrad);
            }
        }

        // Update parameter gradients
        UpdateParameterGradientsFromVectors();

        return new Tensor<T>(new[] { batchSize, inputSize }, inputGradData);
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// VeRA only updates the scaling vectors d and b. The shared matrices A and B remain frozen.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_scalingVectorDGradient == null || _scalingVectorBGradient == null)
        {
            return;
        }

        // Update scaling vector d
        for (int i = 0; i < _scalingVectorD.Length; i++)
        {
            T update = NumOps.Multiply(_scalingVectorDGradient[i], learningRate);
            _scalingVectorD[i] = NumOps.Subtract(_scalingVectorD[i], update);
        }

        // Update scaling vector b
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
        UpdateParametersFromVectors();
    }

    /// <summary>
    /// Gets the current parameters as a vector (scaling vectors only).
    /// </summary>
    /// <returns>Vector containing VeRA parameters (d and b vectors).</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing VeRA parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateVectorsFromParameters();
    }

    /// <summary>
    /// Updates the parameter vector from the current scaling vector values.
    /// </summary>
    private void UpdateParametersFromVectors()
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
    /// Updates the scaling vectors from the parameter vector.
    /// </summary>
    private void UpdateVectorsFromParameters()
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
    /// Updates the parameter gradients vector from the scaling vector gradients.
    /// </summary>
    private void UpdateParameterGradientsFromVectors()
    {
        if (_scalingVectorDGradient == null || _scalingVectorBGradient == null)
        {
            return;
        }

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
    /// Merges the VeRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with VeRA weights merged into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// This computes the full weight contribution from VeRA: W_vera = d * B * A * b * scaling,
    /// and adds it to the base layer's weights.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" the VeRA adaptation for deployment.
    /// After training, you can merge the adaptation into the original weights for faster inference.
    /// The merged layer will behave identically but without the VeRA overhead.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
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
            throw new InvalidOperationException("VeRAAdapter currently only supports DenseLayer or FullyConnectedLayer base layers");
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int rank = _scalingVectorB.Length;

        // Compute VeRA weight contribution: d * B * A * b * scaling
        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));

        // First apply b scaling to A: A_scaled = A * diag(b)
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

        // Transpose to match DenseLayer format [outputSize, inputSize]
        Matrix<T> veraWeightsTransposed = veraWeights.Transpose();

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
            mergedParams[i] = NumOps.Add(baseParams[i], veraWeightsTransposed[row, col]);
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
    /// Resets the internal state of the VeRA adapter.
    /// </summary>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _lastInput = null;
        _lastIntermediate = null;
        _scalingVectorDGradient = null;
        _scalingVectorBGradient = null;
    }
}
