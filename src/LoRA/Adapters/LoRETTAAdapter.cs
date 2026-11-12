using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// LoRETTA (Low-Rank Economic Tensor-Train Adaptation) adapter for parameter-efficient fine-tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoRETTA extends LoRA by using tensor-train decomposition instead of simple matrix factorization.
/// Instead of representing weight updates as W = A × B, LoRETTA uses a tensor-train decomposition
/// that captures higher-order correlations with even fewer parameters.
/// </para>
/// <para>
/// Tensor-train decomposition represents a high-dimensional tensor as a sequence of lower-dimensional
/// "cores" that are contracted together. For a weight matrix W of size (m × n), the tensor-train
/// representation is:
///
/// W[i,j] = G1[i] × G2 × G3 × ... × Gd[j]
///
/// where each core Gk has dimensions (r_{k-1} × n_k × r_k), and r_k are the TT-ranks.
/// The boundary ranks are r_0 = r_d = 1.
/// </para>
/// <para><b>For Beginners:</b> LoRETTA is an advanced version of LoRA that uses "tensor-train decomposition"!
///
/// Standard LoRA uses two matrices (A and B) to approximate weight changes:
/// - Matrix A: Compresses input to rank dimensions
/// - Matrix B: Expands back to output dimensions
/// - Parameters: inputSize × rank + rank × outputSize
///
/// LoRETTA uses multiple small "cores" chained together:
/// - Instead of 2 large matrices, use many small tensors
/// - Each core captures local correlations
/// - The cores are "contracted" (multiplied in sequence)
/// - Can express more complex patterns with fewer parameters
///
/// Why tensor-train decomposition?
/// 1. More expressive: Can capture higher-order correlations
/// 2. More efficient: Fewer parameters than matrix factorization
/// 3. Better compression: Exploits structure in weight updates
/// 4. Scalable: Grows logarithmically with dimensions
///
/// Example parameter counts for 1000×1000 layer:
/// - Full update: 1,000,000 parameters
/// - Standard LoRA (rank=8): 16,000 parameters (98.4% reduction)
/// - LoRETTA (rank=4, 3 cores): ~6,000 parameters (99.4% reduction, even better!)
///
/// Key parameters:
/// - ttRank: Controls compression (like LoRA's rank but more powerful)
/// - numCores: How many tensor cores in the chain (typically 3-5)
/// - alpha: Scaling factor for the adaptation strength
///
/// When to use LoRETTA:
/// - Maximum parameter efficiency needed
/// - Weight updates have higher-order structure
/// - You have very large layers to adapt
/// - Standard LoRA isn't expressive enough at low ranks
///
/// Reference:
/// Tensor-train decomposition: I. V. Oseledets, "Tensor-train decomposition,"
/// SIAM J. Scientific Computing, 2011.
/// </para>
/// </remarks>
[AutoParameters]
public partial class LoRETTAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Tensor-train cores representing the weight decomposition.
    /// Core k has shape (ttRanks[k-1], coreShape[k], ttRanks[k]).
    /// </summary>
    [TrainableParameter]
    private readonly List<Tensor<T>> _ttCores;

    /// <summary>
    /// The ranks of the tensor-train decomposition.
    /// Length is numCores + 1, with ttRanks[0] = ttRanks[numCores] = 1.
    /// </summary>
    private readonly int[] _ttRanks;

    /// <summary>
    /// The shape of each core in the tensor-train.
    /// </summary>
    private readonly int[] _coreShapes;

    /// <summary>
    /// Number of cores in the tensor-train.
    /// </summary>
    private readonly int _numCores;

    /// <summary>
    /// Gradients for each TT core computed during backpropagation.
    /// </summary>
    [Scratch]
    private List<Tensor<T>>? _ttCoreGradients;

    /// <summary>
    /// Cached intermediate tensors from forward pass, needed for gradient computation.
    /// </summary>
    [Scratch]
    private List<Tensor<T>>? _forwardIntermediates;

    /// <summary>
    /// Gets the tensor-train rank.
    /// </summary>
    /// <remarks>
    /// This is the maximum rank in the tensor-train decomposition. Lower rank means
    /// more compression but less expressiveness.
    /// </remarks>
    public int TTRank => _ttRanks.Max();

    /// <summary>
    /// Gets the number of cores in the tensor-train.
    /// </summary>
    public int NumCores => _numCores;

    /// <summary>Construction state: the 'ttRank' the layer was built with.</summary>
    private readonly int _ttRank;

    /// <summary>
    /// Initializes a new LoRETTA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LoRETTA.</param>
    /// <param name="ttRank">The rank of the tensor-train decomposition.</param>
    /// <param name="numCores">Number of cores in the tensor-train (default: 3).</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to ttRank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when ttRank or numCores are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a LoRETTA adapter that wraps any layer.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt efficiently
    /// - ttRank: Controls compression (lower = fewer parameters, less flexibility)
    /// - numCores: How many tensor cores to use (more cores = more expressive but more params)
    /// - alpha: How strong the adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true)
    ///
    /// The cores are initialized carefully:
    /// - First and last cores connect to input/output dimensions
    /// - Middle cores have uniform shapes
    /// - All cores start with small random values (Gaussian initialization)
    /// - Designed so initial LoRETTA has minimal effect
    ///
    /// Recommended settings:
    /// - ttRank=4 to 8: Good balance of efficiency and expressiveness
    /// - numCores=3: Standard choice (input core, middle core, output core)
    /// - numCores=4-5: For very large layers or complex adaptations
    /// </para>
    /// </remarks>
    public LoRETTAAdapter(
        ILayer<T> baseLayer,
        int ttRank,
        int numCores = 3,
        double alpha = -1,
        bool freezeBaseLayer = true)
        : base(baseLayer, ttRank, alpha, freezeBaseLayer)
    {
        _ttRank = ttRank;
        if (ttRank <= 0)
        {
            throw new ArgumentException("TT-rank must be positive", nameof(ttRank));
        }

        if (numCores < 2)
        {
            throw new ArgumentException("Number of cores must be at least 2", nameof(numCores));
        }

        _numCores = numCores;

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        // Initialize TT-ranks: [1, ttRank, ttRank, ..., ttRank, 1]
        _ttRanks = new int[numCores + 1];
        _ttRanks[0] = 1;
        _ttRanks[numCores] = 1;
        for (int k = 1; k < numCores; k++)
        {
            _ttRanks[k] = ttRank;
        }

        // Compute core shapes by factorizing input and output dimensions
        _coreShapes = ComputeCoreShapes(inputSize, outputSize, numCores);

        // Initialize TT cores
        _ttCores = new List<Tensor<T>>(numCores);
        InitializeTTCores();

        // Tensor-train cores replace the standard LoRA factors and are emitted as one ordered
        // trainable collection by the generator.
        FreezeSubLayerParameters(_loraLayer);
    }

    /// <summary>
    /// Computes the shape of each core by factorizing the total dimension.
    /// </summary>
    /// <param name="inputSize">Input dimension.</param>
    /// <param name="outputSize">Output dimension.</param>
    /// <param name="numCores">Number of cores.</param>
    /// <returns>Array of core shapes.</returns>
    /// <remarks>
    /// <para>
    /// We need to factorize the total dimensionality (inputSize × outputSize) across the cores.
    /// The product of all core shapes should approximately equal inputSize × outputSize.
    ///
    /// Strategy: Use geometric decomposition
    /// - First core: ~inputSize^(1/2) × outputSize^(1/(numCores-1))
    /// - Last core: ~inputSize^(1/2) × outputSize^(1/(numCores-1))
    /// - Middle cores: uniform sizes based on geometric mean
    /// </para>
    /// </remarks>
    private int[] ComputeCoreShapes(int inputSize, int outputSize, int numCores)
    {
        int[] shapes = new int[numCores];

        // Total "logical" dimension to decompose
        double totalDim = Math.Sqrt((double)inputSize * outputSize);

        // Use geometric factorization
        double dimPerCore = Math.Pow(totalDim, 2.0 / numCores);

        // Ensure each core has at least dimension 2
        int baseDim = Math.Max(2, (int)Math.Ceiling(dimPerCore));

        // Distribute dimensions
        for (int k = 0; k < numCores; k++)
        {
            shapes[k] = baseDim;
        }

        // Adjust first and last cores to better match input/output sizes
        shapes[0] = Math.Max(2, (int)Math.Ceiling(Math.Sqrt(inputSize)));
        shapes[numCores - 1] = Math.Max(2, (int)Math.Ceiling(Math.Sqrt(outputSize)));

        return shapes;
    }

    /// <summary>
    /// Initializes all TT cores with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each core is initialized with Gaussian noise scaled by 1/sqrt(product of dimensions).
    /// This ensures the overall adaptation starts small.
    /// </para>
    /// </remarks>
    private void InitializeTTCores()
    {
        Random random = RandomHelper.CreateSeededRandom(42);

        for (int k = 0; k < _numCores; k++)
        {
            int leftRank = _ttRanks[k];
            int coreShape = _coreShapes[k];
            int rightRank = _ttRanks[k + 1];

            // Core has shape [leftRank, coreShape, rightRank]
            int[] shape = new int[] { leftRank, coreShape, rightRank };
            Tensor<T> core = new Tensor<T>(shape);

            // Initialize with small Gaussian noise
            double scale = 1.0 / Math.Sqrt(leftRank * coreShape * rightRank);

            for (int i = 0; i < core.Length; i++)
            {
                core[i] = NumOps.Multiply(NumOps.FromDouble(random.NextGaussian()), NumOps.FromDouble(scale));
            }

            _ttCores.Add(core);
        }
    }

    /// <summary>
    /// Performs the forward pass through the LoRETTA adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and LoRETTA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes the tensor-train contraction to produce the adaptation,
    /// then adds it to the base layer output.
    /// </para>
    /// <para><b>For Beginners:</b> This processes input through both the original layer and
    /// the LoRETTA adaptation, then combines them.
    ///
    /// The LoRETTA forward pass:
    /// 1. Forward through base layer (original behavior)
    /// 2. Contract tensor-train cores with input (compute adaptation)
    /// 3. Add base output + adaptation output
    ///
    /// The tensor contraction is done sequentially through the cores, which is efficient
    /// even though it looks complex mathematically.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Collect intermediates only when a manual backward will read them (the list stays null otherwise,
        // so ComputeTensorTrainForward skips recording)
        SaveForBackward(ref _forwardIntermediates, new List<Tensor<T>>());

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Compute LoRETTA adaptation via tensor-train contraction
        Tensor<T> ttOutput = ComputeTensorTrainForward(input);

        // Sum the outputs
        Tensor<T> result = Engine.TensorAdd(baseOutput, ttOutput);

        return result;
    }

    /// <summary>
    /// Computes the forward pass through the tensor-train decomposition.
    /// </summary>
    /// <param name="input">Input tensor of shape [batchSize, inputSize].</param>
    /// <returns>Output tensor of shape [batchSize, outputSize].</returns>
    /// <remarks>
    /// <para>
    /// This performs the tensor-train contraction:
    /// 1. Reshape input to match first core dimensions
    /// 2. Contract through each core sequentially
    /// 3. Reshape output to match expected output dimensions
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeTensorTrainForward(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;

        // Start with input reshaped to work with first core
        // For simplicity, we'll use a matrix-based contraction approach

        // Flatten input to [batchSize × inputSize]
        Matrix<T> currentMatrix = Engine.Reshape(input, [batchSize, inputSize]).ToMatrix();


        // Contract through each core
        for (int k = 0; k < _numCores; k++)
        {
            currentMatrix = ContractWithCore(currentMatrix, _ttCores[k], k);

            // Store intermediate for backward pass
            if (_forwardIntermediates != null)
            {
                _forwardIntermediates.Add(TensorFromMatrix(currentMatrix));
            }
        }

        // Extract output
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        // Direct extraction without modulo wrapping: the first outputSize columns, zero beyond
        Tensor<T> outputData = TakeColumnsZeroPadded(Tensor<T>.FromMatrix(currentMatrix), outputSize);

        // Apply scaling (alpha / rank)
        T scaling = NumOps.Divide(
            NumOps.FromDouble(Alpha),
            NumOps.FromDouble(TTRank)
        );

        return Engine.TensorMultiplyScalar(outputData, scaling);
        return new Tensor<T>(new[] { batchSize, outputSize }, outputData);
    }

    /// <summary>
    /// Contracts a matrix with a tensor-train core.
    /// </summary>
    /// <param name="input">Input matrix [batchSize, currentDim].</param>
    /// <param name="core">TT core tensor [leftRank, coreShape, rightRank].</param>
    /// <param name="coreIndex">Index of the core being processed.</param>
    /// <returns>Output matrix [batchSize, nextDim].</returns>
    private Matrix<T> ContractWithCore(Matrix<T> input, Tensor<T> core, int coreIndex)
    {
        int batchSize = input.Rows;
        int leftRank = _ttRanks[coreIndex];
        int coreShape = _coreShapes[coreIndex];
        int rightRank = _ttRanks[coreIndex + 1];

        // Simplified contraction: treat core as a sequence of matrices
        // Core shape: [leftRank, coreShape, rightRank]
        // We'll contract by reshaping and matrix multiplication

        int inputDim = input.Columns;
        int outputDim = coreShape * rightRank;

        // Simplified: use first 'leftRank' dimensions of input:
        // output[b, c * rightRank + r] = sum_{l < min(leftRank, inputDim)} input[b, l] * core[l, c, r]
        int usedRank = Math.Min(leftRank, inputDim);
        if (usedRank == 0)
        {
            return new Matrix<T>(batchSize, outputDim);
        }

        Matrix<T> output = Engine.TensorMatMul(
            Engine.TensorNarrow(Tensor<T>.FromMatrix(input), 1, 0, usedRank),
            Engine.TensorNarrow(Engine.Reshape(core, [leftRank, outputDim]), 0, 0, usedRank)).ToMatrix();

        return output;
    }

    /// <summary>
    /// Converts a matrix to a tensor.
    /// </summary>
    private Tensor<T> TensorFromMatrix(Matrix<T> matrix)
    {
        return Tensor<T>.FromMatrix(matrix);
    }

    /// <summary>
    /// Returns the first <paramref name="columns"/> columns of a [rows, cols] tensor, padding with
    /// zero columns when it has fewer.
    /// </summary>
    private Tensor<T> TakeColumnsZeroPadded(Tensor<T> matrix, int columns)
    {
        int rows = matrix.Shape[0];
        int kept = Math.Min(columns, matrix.Shape[1]);
        if (kept == 0)
        {
            return new Tensor<T>([rows, columns]);
        }

        Tensor<T> taken = Engine.TensorNarrow(matrix, 1, 0, kept);
        return kept < columns
            ? Engine.TensorConcatenate([taken, new Tensor<T>([rows, columns - kept])], 1)
            : taken;
    }
        return new Tensor<T>(new[] { matrix.Rows, matrix.Columns }, data);
    }

    /// <summary>
    /// Converts a tensor to a matrix.
    /// </summary>
    private Matrix<T> MatrixFromTensor(Tensor<T> tensor)
    {
        if (tensor.Shape.Length != 2)
        {
            throw new ArgumentException($"Expected 2D tensor, got {tensor.Shape.Length}D");
        }

        Matrix<T> matrix = tensor.ToMatrix();

        return matrix;
    }

    /// <summary>
    /// Computes the backward pass through the tensor-train decomposition.
    /// </summary>
    /// <param name="outputGradient">Gradient from the output.</param>
    /// <returns>Gradient with respect to input.</returns>
    private Tensor<T> ComputeTensorTrainBackward(Tensor<T> outputGradient)
    {
        // Initialize core gradients
        _ttCoreGradients = new List<Tensor<T>>();
        for (int k = 0; k < _numCores; k++)
        {
            _ttCoreGradients.Add(new Tensor<T>(_ttCores[k]._shape));
        }

        int batchSize = outputGradient.Shape[0];
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        // Apply inverse scaling to output gradient
        T scaling = NumOps.Divide(
            NumOps.FromDouble(Alpha),
            NumOps.FromDouble(TTRank)
        );

        // Scaled output gradient in matrix form [batchSize, outputSize]
        Tensor<T> gradMatrix = Engine.Reshape(
            Engine.TensorMultiplyScalar(outputGradient, scaling), [batchSize, outputSize]);

        // Backpropagate through cores in reverse order
        for (int k = _numCores - 1; k >= 0; k--)
        {
            int leftRank = _ttRanks[k];
            int coreShape = _coreShapes[k];
            int rightRank = _ttRanks[k + 1];

            Tensor<T> core = _ttCores[k];

            // Get the input to this core from forward intermediates (the first core gets zero input)
            bool hasCoreInput = k > 0 && _forwardIntermediates != null && k <= _forwardIntermediates.Count;

            // Only core slices c < min(coreShape, gradient columns) receive gradient; with
            // g = gradMatrix[:, 0 : usableCore * rightRank] and x = coreInput[:, 0 : min(leftRank, cols)]:
            // ∂L/∂core_k[l, c, r] = sum_b x[b, l] * g[b, c * rightRank + r]
            int usableCore = Math.Min(coreShape, gradMatrix.Shape[1]);
            int width = usableCore * rightRank;
            Tensor<T> g = width > 0 ? Engine.TensorNarrow(gradMatrix, 1, 0, width) : new Tensor<T>([batchSize, 0]);

            if (hasCoreInput && width > 0 && _forwardIntermediates != null)
            {
                Tensor<T> coreInput = _forwardIntermediates[k - 1]; // [batchSize, cols] from TensorFromMatrix

                int usedRank = Math.Min(leftRank, coreInput.Shape[1]);
                if (usedRank > 0)
                {
                    Tensor<T> partial = Engine.TensorMatMul(
                        Engine.TensorTranspose(Engine.TensorNarrow(coreInput, 1, 0, usedRank)), g); // [usedRank, width]
                    if (width < coreShape * rightRank)
                    {
                        partial = Engine.TensorConcatenate(
                            [partial, new Tensor<T>([usedRank, coreShape * rightRank - width])], 1);
                    }

                    if (usedRank < leftRank)
                    {
                        partial = Engine.TensorConcatenate(
                            [partial, new Tensor<T>([leftRank - usedRank, coreShape * rightRank])], 0);
                    }

                    _ttCoreGradients[k] = Engine.Reshape(partial, core.Shape.ToArray());
                }
            }

            // Compute gradient to pass to previous core:
            // prevGrad[b, l] = sum_{c < usableCore, r} core[l, c, r] * g[b, c * rightRank + r]
            if (k > 0)
            {
                gradMatrix = width > 0
                    ? Engine.TensorMatMul(
                        g,
                        Engine.TensorTranspose(Engine.TensorNarrow(
                            Engine.Reshape(core, [leftRank, coreShape * rightRank]), 1, 0, width)))
                    : new Tensor<T>([batchSize, leftRank]);
            }
        }

        // Convert final gradient matrix to input gradient tensor (first inputSize columns, zero beyond)
        Tensor<T> inputGradient = TakeColumnsZeroPadded(gradMatrix, inputSize);

        return inputGradient;
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This applies the gradients to update the TT cores.
    ///
    /// For each core:
    /// 1. Get the gradient computed during backpropagation
    /// 2. Update: core_new = core_old - learningRate × gradient
    /// 3. Update base layer if not frozen
    ///
    /// This is conceptually the same as standard gradient descent, but applied to
    /// the tensor-train cores instead of weight matrices.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_ttCoreGradients == null)
        {
            return;
        }

        // Update each TT core: core -= learningRate * dcore
        for (int k = 0; k < _numCores; k++)
        {
            Engine.TensorSubtractInPlace(_ttCores[k], Engine.TensorMultiplyScalar(_ttCoreGradients[k], learningRate));
        }

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

    }

    /// <summary>
    /// Merges the LoRETTA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with LoRETTA weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not supported.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This "bakes in" your LoRETTA adaptation to create a regular layer.
    ///
    /// After training:
    /// 1. Contract all TT cores to form a full weight matrix
    /// 2. Add this matrix to the base layer's weights
    /// 3. Create a new layer with the merged weights
    ///
    /// The result is a standard layer that behaves like your adapted model but:
    /// - Faster inference (no tensor-train contraction needed)
    /// - Simpler deployment (single layer instead of adapter)
    /// - Compatible with any framework
    ///
    /// The tensor-train cores are contracted to form a full weight update matrix,
    /// which is then added to the original weights.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Check base layer type
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException(
                "LoRETTAAdapter merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Contract TT cores to form full weight matrix
        Matrix<T> ttWeights = ContractTensorTrainToMatrix();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        int weightCount = inputSize * outputSize;

        // Merge weights (add LoRETTA contribution, [outputSize, inputSize] row-major, to the base
        // weights). Biases are kept unchanged.
        Vector<T> mergedParams = AddFlatWeightsToParameters(baseParams, ttWeights, weightCount);

        // Create a new dense layer with merged parameters
        DenseLayer<T> mergedLayer = new DenseLayer<T>(
            outputSize,
            (IActivationFunction<T>?)null);
        mergedLayer.SetParameters(mergedParams);

        return mergedLayer;
    }

    /// <summary>
    /// Contracts the tensor-train cores into a full weight matrix.
    /// </summary>
    /// <returns>Full weight matrix representing the TT decomposition.</returns>
    /// <remarks>
    /// This performs the full contraction of all TT cores to recover the
    /// complete weight update matrix. This is expensive but only needed for merging.
    /// </remarks>
    private Matrix<T> ContractTensorTrainToMatrix()
    {
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        // Perform sequential contraction of all TT cores
        // Start with first core: [r0=1, n1, r1] → effectively [n1, r1]
        int r0 = _ttRanks[0];  // Should be 1
        int n1 = _coreShapes[0];
        int r1 = _ttRanks[1];

        // Extract first core as matrix [n1, r1]: core [r0=1, n1, r1], index 0 * n1 * r1 + i * r1 + j
        Tensor<T> firstCore = _ttCores[0];
        Tensor<T> contracted = Engine.Reshape(
            Engine.TensorNarrow(Engine.Reshape(firstCore, [firstCore.Length]), 0, 0, n1 * r1), [n1, r1]);

        // Contract with remaining cores
        for (int k = 1; k < _numCores; k++)
        {
            int leftRank = _ttRanks[k];
            int coreShape = _coreShapes[k];
            int rightRank = _ttRanks[k + 1];

            Tensor<T> core = _ttCores[k];

            // Current contracted has shape [prevDim, leftRank]
            // Core has shape [leftRank, coreShape, rightRank]
            // Result will have shape [prevDim, coreShape, rightRank] → [prevDim * coreShape, rightRank]

            int prevDim = contracted.Shape[0];
            int newDim = prevDim * coreShape;

            // newContracted[i*coreShape + c, r] = sum_{l < min(leftRank, cols)} contracted[i, l] * core[l, c, r]
            int usedRank = Math.Min(leftRank, contracted.Shape[1]);
            Tensor<T> newContracted = usedRank == 0
                ? new Tensor<T>([newDim, rightRank])
                : Engine.Reshape(
                    Engine.TensorMatMul(
                        Engine.TensorNarrow(contracted, 1, 0, usedRank),
                        Engine.TensorNarrow(Engine.Reshape(core, [leftRank, coreShape * rightRank]), 0, 0, usedRank)),
                    [newDim, rightRank]);

            contracted = newContracted;
        }

        // Final contracted tensor has shape [totalDim, rd=1]
        // Reshape to [outputSize, inputSize]
        // Copy column 0 of the contraction (the last rank should be 1) into the first
        // min(totalDim, outputSize * inputSize) row-major positions; the rest stay zero.
        int totalDim = contracted.Shape[0];
        int matrixSize = outputSize * inputSize;
        int copied = Math.Min(totalDim, matrixSize);
        Tensor<T> flat = contracted.Shape[1] > 0 && copied > 0
            ? Engine.TensorNarrow(Engine.Reshape(Engine.TensorNarrow(contracted, 1, 0, 1), [totalDim]), 0, 0, copied)
            : new Tensor<T>([0]);
        if (copied < matrixSize)
        {
            flat = Engine.TensorConcatenate([flat, new Tensor<T>([matrixSize - copied])], 0);
        }

        Matrix<T> result = Engine.Reshape(flat, [outputSize, inputSize]).ToMatrix();

        // Apply scaling
        T scaling = NumOps.Divide(
            NumOps.FromDouble(Alpha),
            NumOps.FromDouble(TTRank)
        );

        return Engine.MatrixMultiplyScalar(result, scaling);
    }

    /// <summary>
    /// Resets the internal state of the adapter.
    /// </summary>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _forwardIntermediates = null;
        _ttCoreGradients = null;
    }

    /// <summary>
    /// Gets parameter efficiency metrics for this LoRETTA adapter.
    /// </summary>
    /// <returns>A formatted string with parameter efficiency statistics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows how efficient LoRETTA is compared to alternatives.
    ///
    /// The metrics include:
    /// - Total parameters in base layer (what full fine-tuning would require)
    /// - LoRETTA parameters (what you actually train)
    /// - Equivalent LoRA parameters (for comparison)
    /// - Parameter reduction percentage
    /// - Compression ratio
    ///
    /// These numbers help you understand the efficiency gains from using LoRETTA!
    /// </para>
    /// </remarks>
    public string GetParameterEfficiencyMetrics()
    {
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        int fullParams = inputSize * outputSize;
        int ttParams = (int)(ParameterCount - (_freezeBaseLayer ? 0 : _baseLayer.ParameterCount));
        int equivalentLoRAParams = (inputSize + outputSize) * TTRank;

        double reductionVsFull = 100.0 * (1.0 - (double)ttParams / fullParams);
        double reductionVsLoRA = 100.0 * (1.0 - (double)ttParams / equivalentLoRAParams);
        double compressionRatio = (double)fullParams / ttParams;

        return $"LoRETTA Parameter Efficiency:\n" +
               $"  Full parameters: {fullParams:N0}\n" +
               $"  LoRETTA parameters: {ttParams:N0}\n" +
               $"  Equivalent LoRA (rank={TTRank}): {equivalentLoRAParams:N0}\n" +
               $"  Reduction vs full: {reductionVsFull:F2}%\n" +
               $"  Reduction vs LoRA: {reductionVsLoRA:F2}%\n" +
               $"  Compression ratio: {compressionRatio:F1}x\n" +
               $"  TT-rank: {TTRank}\n" +
               $"  Number of cores: {NumCores}";
    }
}
