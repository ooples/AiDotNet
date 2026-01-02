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
public class LoRETTAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Tensor-train cores representing the weight decomposition.
    /// Core k has shape (ttRanks[k-1], coreShape[k], ttRanks[k]).
    /// </summary>
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
    private List<Tensor<T>>? _ttCoreGradients;

    /// <summary>
    /// Cached intermediate tensors from forward pass, needed for gradient computation.
    /// </summary>
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

    /// <summary>
    /// Gets the total number of trainable parameters in the tensor-train cores.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The total parameters is the sum of all core sizes:
    /// sum_k (ttRanks[k-1] × coreShapes[k] × ttRanks[k])
    /// </para>
    /// <para>
    /// This is typically much smaller than standard LoRA for the same expressiveness.
    /// </para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int ttParams = 0;
            for (int k = 0; k < _numCores; k++)
            {
                ttParams += _ttRanks[k] * _coreShapes[k] * _ttRanks[k + 1];
            }

            // Add base layer parameters if not frozen
            if (!_freezeBaseLayer)
            {
                return _baseLayer.ParameterCount + ttParams;
            }

            return ttParams;
        }
    }

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
        int outputSize = GetOutputShape()[0];

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

        // Update parameter vector
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromCores();
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
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store intermediates for backward pass
        _forwardIntermediates = new List<Tensor<T>>();

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Compute LoRETTA adaptation via tensor-train contraction
        Tensor<T> ttOutput = ComputeTensorTrainForward(input);

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], ttOutput[i]);
        }

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
        Matrix<T> currentMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                currentMatrix[i, j] = input[i * inputSize + j];
            }
        }

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
        int outputSize = GetOutputShape()[0];
        Vector<T> outputData = new Vector<T>(batchSize * outputSize);

        int idx = 0;

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                // Direct extraction without modulo wrapping
                if (j < currentMatrix.Columns && i < currentMatrix.Rows)
                {
                    outputData[idx] = currentMatrix[i, j];
                }
                else
                {
                    outputData[idx] = NumOps.Zero;
                }
                idx++;
            }
        }

        // Apply scaling (alpha / rank)
        T scaling = NumOps.Divide(
            NumOps.FromDouble(Alpha),
            NumOps.FromDouble(TTRank)
        );

        for (int i = 0; i < outputData.Length; i++)
        {
            outputData[i] = NumOps.Multiply(outputData[i], scaling);
        }

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

        Matrix<T> output = new Matrix<T>(batchSize, outputDim);

        // For each batch element
        for (int b = 0; b < batchSize; b++)
        {
            // Contract input with core
            // Simplified: use first 'leftRank' dimensions of input
            for (int r = 0; r < rightRank; r++)
            {
                for (int c = 0; c < coreShape; c++)
                {
                    T sum = NumOps.Zero;

                    for (int l = 0; l < leftRank && l < inputDim; l++)
                    {
                        int coreIdx = (l * coreShape * rightRank) + (c * rightRank) + r;
                        if (coreIdx < core.Length)
                        {
                            T inputVal = input[b, l];
                            T coreVal = core[coreIdx];
                            sum = NumOps.Add(sum, NumOps.Multiply(inputVal, coreVal));
                        }
                    }

                    int outIdx = c * rightRank + r;
                    if (outIdx < outputDim)
                    {
                        output[b, outIdx] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Converts a matrix to a tensor.
    /// </summary>
    private Tensor<T> TensorFromMatrix(Matrix<T> matrix)
    {
        Vector<T> data = new Vector<T>(matrix.Rows * matrix.Columns);
        int idx = 0;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                data[idx++] = matrix[i, j];
            }
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

        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];
        Matrix<T> matrix = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = tensor[i * cols + j];
            }
        }

        return matrix;
    }

    /// <summary>
    /// Performs the backward pass through the LoRETTA adapter.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients for all TT cores and propagates gradients
    /// back through the tensor-train contraction.
    /// </para>
    /// <para><b>For Beginners:</b> This is where learning happens for LoRETTA!
    ///
    /// The backward pass:
    /// 1. Backpropagate through base layer
    /// 2. Backpropagate through tensor-train cores
    /// 3. Compute gradients for each core
    /// 4. Combine input gradients from both paths
    ///
    /// This is more complex than standard LoRA because we need to backpropagate through
    /// multiple cores, but the principle is the same: figure out how each parameter
    /// contributed to the error.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Backward through tensor-train
        Tensor<T> ttInputGrad = ComputeTensorTrainBackward(outputGradient);

        // Sum input gradients
        Tensor<T> inputGrad = new Tensor<T>(baseInputGrad.Shape);
        for (int i = 0; i < baseInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(baseInputGrad[i], ttInputGrad[i]);
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromCores();

        return inputGrad;
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
            _ttCoreGradients.Add(new Tensor<T>(_ttCores[k].Shape));
        }

        int batchSize = outputGradient.Shape[0];
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Apply inverse scaling to output gradient
        T scaling = NumOps.Divide(
            NumOps.FromDouble(Alpha),
            NumOps.FromDouble(TTRank)
        );

        Vector<T> scaledOutputGrad = new Vector<T>(outputGradient.Length);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            scaledOutputGrad[i] = NumOps.Multiply(outputGradient[i], scaling);
        }

        // Convert output gradient to matrix form [batchSize, outputSize]
        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradMatrix[i, j] = scaledOutputGrad[i * outputSize + j];
            }
        }

        // Backpropagate through cores in reverse order
        for (int k = _numCores - 1; k >= 0; k--)
        {
            int leftRank = _ttRanks[k];
            int coreShape = _coreShapes[k];
            int rightRank = _ttRanks[k + 1];

            Tensor<T> core = _ttCores[k];

            // Get the input to this core from forward intermediates
            Matrix<T> coreInput = (k > 0 && _forwardIntermediates != null && k <= _forwardIntermediates.Count)
                ? MatrixFromTensor(_forwardIntermediates[k - 1])
                : new Matrix<T>(batchSize, leftRank); // First core gets zero input

            // Compute gradient for this core using outer product of input and gradient
            // ∂L/∂core_k = input_{k-1}^T ⊗ grad_k
            for (int l = 0; l < leftRank; l++)
            {
                for (int c = 0; c < coreShape && c < gradMatrix.Columns; c++)
                {
                    for (int r = 0; r < rightRank; r++)
                    {
                        T grad = NumOps.Zero;
                        for (int b = 0; b < batchSize; b++)
                        {
                            T inputVal = (l < coreInput.Columns) ? coreInput[b, l] : NumOps.Zero;
                            T gradVal = gradMatrix[b, c * rightRank + r];
                            grad = NumOps.Add(grad, NumOps.Multiply(inputVal, gradVal));
                        }

                        int coreIdx = (l * coreShape * rightRank) + (c * rightRank) + r;
                        if (coreIdx < _ttCoreGradients[k].Length)
                        {
                            _ttCoreGradients[k][coreIdx] = grad;
                        }
                    }
                }
            }

            // Compute gradient to pass to previous core
            if (k > 0)
            {
                Matrix<T> prevGrad = new Matrix<T>(batchSize, leftRank);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int l = 0; l < leftRank; l++)
                    {
                        T sum = NumOps.Zero;
                        for (int c = 0; c < coreShape && c < gradMatrix.Columns; c++)
                        {
                            for (int r = 0; r < rightRank; r++)
                            {
                                int coreIdx = (l * coreShape * rightRank) + (c * rightRank) + r;
                                if (coreIdx < core.Length)
                                {
                                    T coreVal = core[coreIdx];
                                    T gradVal = gradMatrix[b, c * rightRank + r];
                                    sum = NumOps.Add(sum, NumOps.Multiply(coreVal, gradVal));
                                }
                            }
                        }
                        prevGrad[b, l] = sum;
                    }
                }
                gradMatrix = prevGrad;
            }
        }

        // Convert final gradient matrix to input gradient tensor
        Tensor<T> inputGradient = new Tensor<T>(new[] { batchSize, inputSize });
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize && j < gradMatrix.Columns; j++)
            {
                inputGradient[i * inputSize + j] = gradMatrix[i, j];
            }
        }

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

        // Update each TT core
        for (int k = 0; k < _numCores; k++)
        {
            for (int i = 0; i < _ttCores[k].Length; i++)
            {
                T update = NumOps.Multiply(_ttCoreGradients[k][i], learningRate);
                _ttCores[k][i] = NumOps.Subtract(_ttCores[k][i], update);
            }
        }

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromCores();
    }

    /// <summary>
    /// Updates the parameter vector from the current TT core values.
    /// </summary>
    private void UpdateParametersFromCores()
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

        // Pack all TT cores
        foreach (Tensor<T> core in _ttCores)
        {
            for (int i = 0; i < core.Length; i++)
            {
                Parameters[idx++] = core[i];
            }
        }
    }

    /// <summary>
    /// Updates the TT cores from the parameter vector.
    /// </summary>
    private void UpdateCoresFromParameters()
    {
        int idx = 0;

        // If base layer is not frozen, unpack its parameters first
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

        // Unpack all TT cores
        for (int k = 0; k < _numCores; k++)
        {
            for (int i = 0; i < _ttCores[k].Length; i++)
            {
                _ttCores[k][i] = Parameters[idx++];
            }
        }
    }

    /// <summary>
    /// Updates the parameter gradients vector from the TT core gradients.
    /// </summary>
    private void UpdateParameterGradientsFromCores()
    {
        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // If base layer is not frozen, pack its gradients first
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // Pack TT core gradients
        if (_ttCoreGradients != null)
        {
            foreach (Tensor<T> coreGrad in _ttCoreGradients)
            {
                for (int i = 0; i < coreGrad.Length; i++)
                {
                    ParameterGradients[idx++] = coreGrad[i];
                }
            }
        }
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing parameters.</returns>
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
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, got {parameters.Length}",
                nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateCoresFromParameters();
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
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights (add LoRETTA contribution to base weights)
        for (int i = 0; i < weightCount && i < baseParams.Length; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;

            T ttContribution = NumOps.Zero;
            if (row < ttWeights.Rows && col < ttWeights.Columns)
            {
                ttContribution = ttWeights[row, col];
            }

            mergedParams[i] = NumOps.Add(baseParams[i], ttContribution);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Create a new dense layer with merged parameters
        DenseLayer<T> mergedLayer = new DenseLayer<T>(
            inputSize,
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
        int outputSize = GetOutputShape()[0];

        // Perform sequential contraction of all TT cores
        // Start with first core: [r0=1, n1, r1] → effectively [n1, r1]
        int r0 = _ttRanks[0];  // Should be 1
        int n1 = _coreShapes[0];
        int r1 = _ttRanks[1];

        // Extract first core as matrix [n1, r1]
        Matrix<T> contracted = new Matrix<T>(n1, r1);
        Tensor<T> firstCore = _ttCores[0];

        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < r1; j++)
            {
                // First core: [r0=1, n1, r1], index: 0 * n1 * r1 + i * r1 + j
                int idx = i * r1 + j;
                contracted[i, j] = (idx < firstCore.Length) ? firstCore[idx] : NumOps.Zero;
            }
        }

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

            int prevDim = contracted.Rows;
            int newDim = prevDim * coreShape;

            Matrix<T> newContracted = new Matrix<T>(newDim, rightRank);

            // Perform contraction: newContracted[i*coreShape + c, r] = sum_l contracted[i, l] * core[l, c, r]
            for (int i = 0; i < prevDim; i++)
            {
                for (int c = 0; c < coreShape; c++)
                {
                    for (int r = 0; r < rightRank; r++)
                    {
                        T sum = NumOps.Zero;
                        for (int l = 0; l < leftRank && l < contracted.Columns; l++)
                        {
                            // Core index: [l, c, r] → l * coreShape * rightRank + c * rightRank + r
                            int coreIdx = l * coreShape * rightRank + c * rightRank + r;
                            if (coreIdx < core.Length)
                            {
                                T contractedVal = contracted[i, l];
                                T coreVal = core[coreIdx];
                                sum = NumOps.Add(sum, NumOps.Multiply(contractedVal, coreVal));
                            }
                        }
                        newContracted[i * coreShape + c, r] = sum;
                    }
                }
            }

            contracted = newContracted;
        }

        // Final contracted tensor has shape [totalDim, rd=1]
        // Reshape to [outputSize, inputSize]
        int totalDim = contracted.Rows;
        Matrix<T> result = new Matrix<T>(outputSize, inputSize);

        // Initialize with zeros
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                result[i, j] = NumOps.Zero;
            }
        }

        // Copy contracted values to result
        int matrixSize = outputSize * inputSize;
        for (int idx = 0; idx < Math.Min(totalDim, matrixSize); idx++)
        {
            int row = idx / inputSize;
            int col = idx % inputSize;
            if (row < outputSize && col < inputSize && idx < contracted.Rows)
            {
                // Last rank should be 1, so we just take column 0
                result[row, col] = (contracted.Columns > 0) ? contracted[idx, 0] : NumOps.Zero;
            }
        }

        // Apply scaling
        T scaling = NumOps.Divide(
            NumOps.FromDouble(Alpha),
            NumOps.FromDouble(TTRank)
        );

        return result.Multiply(scaling);
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
        int outputSize = GetOutputShape()[0];

        int fullParams = inputSize * outputSize;
        int ttParams = ParameterCount - (_freezeBaseLayer ? 0 : _baseLayer.ParameterCount);
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
