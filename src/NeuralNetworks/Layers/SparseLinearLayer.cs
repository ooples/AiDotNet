using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a fully connected layer with sparse weight matrix for efficient computation.
/// </summary>
/// <remarks>
/// <para>
/// A sparse linear layer is similar to a dense layer but uses sparse weight storage.
/// This is efficient when most weights are zero (or can be pruned), reducing both
/// memory usage and computation time.
/// </para>
/// <para><b>For Beginners:</b> This layer works like a regular dense layer, but uses
/// sparse matrices to store weights more efficiently.
///
/// Benefits of sparse layers:
/// - Much less memory for large layers with few active connections
/// - Faster computation (only non-zero weights are used)
/// - Natural for network pruning and compression
///
/// Use cases:
/// - Graph neural networks (sparse adjacency)
/// - Recommender systems (sparse user-item matrices)
/// - Pruned neural networks
/// - Very large embedding layers
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (float or double).</typeparam>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 8, 0.5")]
public partial class SparseLinearLayer<T> : LayerBase<T>
{
    private readonly ISparseEngine _engine;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The sparse weight matrix.
    /// Shape: [OutputFeatures, InputFeatures]
    /// </summary>
    // No [TrainableParameter] — SparseTensor is incompatible with dense ParameterBuffer.
    // Weight updates are handled by SparseLinearLayer.UpdateParameters() directly.
    private SparseTensor<T> _weights;

    /// <summary>
    /// The bias values (dense, typically small).
    /// </summary>
    private Vector<T> _biases;

    /// <summary>
    /// The sparsity level (fraction of weights that are zero).
    /// </summary>
    private readonly double _sparsity;

    /// <summary>
    /// Stored input from forward pass for backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored pre-activation output for gradient computation.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gradient for weights, stored during backward pass.
    /// Stored as dense matrix for gradient accumulation, then sparsified.
    /// </summary>
    private Matrix<T>? _weightsGradient;

    /// <summary>
    /// Gradient for biases, stored during backward pass.
    /// </summary>
    private Vector<T>? _biasesGradient;

    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int OutputFeatures { get; }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// For sparse layers, this returns the number of non-zero weights plus biases.
    /// </remarks>
    public override int ParameterCount =>
        _weights.NonZeroCount + OutputFeatures;

    /// <summary>
    /// Gets whether this layer supports training. Returns <c>true</c>: the layer
    /// owns a working <see cref="UpdateParameters"/> that updates its sparse
    /// weight tensor and dense bias vector from gradients computed in
    /// <see cref="Backward"/>, and the legacy training path
    /// (<c>if (layer.SupportsTraining) layer.UpdateParameters(lr)</c>) trains
    /// the layer correctly. Biases are also registered as a tape trainable
    /// parameter, so tape-mode optimizers update them too.
    /// </summary>
    /// <remarks>
    /// <para><b>Tape-mode caveat:</b> the sparse <see cref="SparseTensor{T}"/>
    /// weight tensor is still not visible to the tape's
    /// <c>ParameterBuffer&lt;T&gt;</c>-based discovery — that contract requires
    /// dense storage. In tape mode, weight updates flow through the
    /// layer's own <see cref="Backward"/> + <see cref="UpdateParameters"/>
    /// when the optimizer falls back to the legacy update path; weight
    /// gradients do not appear in the tape's flat-buffer view. Closing that
    /// gap fully (sparse-aware <c>ParameterBuffer&lt;T&gt;</c>) is tracked as
    /// a follow-up.</para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the SparseLinearLayer.
    /// </summary>
    /// <param name="inputFeatures">Number of input features.</param>
    /// <param name="outputFeatures">Number of output features.</param>
    /// <param name="sparsity">Fraction of weights to be zero (0.0 to 1.0).</param>
    /// <param name="activationFunction">Optional activation function.</param>
    public SparseLinearLayer(
        int inputFeatures,
        int outputFeatures,
        double sparsity = 0.9,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [inputFeatures],
            [outputFeatures],
            activationFunction ?? new ReLUActivation<T>())
    {
        if (sparsity < 0 || sparsity >= 1.0)
        {
            throw new ArgumentException("Sparsity must be in [0, 1).", nameof(sparsity));
        }

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.Eager;
        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;
        _sparsity = sparsity;

        _engine = CpuSparseEngine.Instance;
        _numOps = MathHelper.GetNumericOperations<T>();

        _biases = new Vector<T>(outputFeatures);
        // Biases initialized to zero by default (standard practice for ReLU layers)
        _weights = InitializeSparseWeights();

        // Note: SparseTensor weights are NOT registered as trainable parameters because
        // ParameterBuffer requires dense tensors for contiguous buffer views.
        // SparseLinearLayer handles its own weight updates via UpdateParameters().
    }

    /// <summary>
    /// Initializes sparse weights using the layer's initialization strategy.
    /// </summary>
    private SparseTensor<T> InitializeSparseWeights()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Calculate number of non-zero weights
        int totalWeights = OutputFeatures * InputFeatures;
        int nonZeroCount = Math.Max(1, (int)(totalWeights * (1.0 - _sparsity)));

        // Generate random non-zero positions
        var indices = new HashSet<(int row, int col)>();
        while (indices.Count < nonZeroCount)
        {
            int row = random.Next(OutputFeatures);
            int col = random.Next(InputFeatures);
            indices.Add((row, col));
        }

        // Create a dense weight tensor initialized via the strategy, then extract sparse values
        var denseWeights = new Tensor<T>([OutputFeatures, InputFeatures]);
        InitializeLayerWeights(denseWeights, InputFeatures, OutputFeatures);

        // Convert to COO format, keeping only the selected non-zero positions
        var rowIndices = new int[nonZeroCount];
        var colIndices = new int[nonZeroCount];
        var values = new T[nonZeroCount];

        int idx = 0;
        foreach (var (row, col) in indices.OrderBy(x => x.row).ThenBy(x => x.col))
        {
            rowIndices[idx] = row;
            colIndices[idx] = col;
            values[idx] = denseWeights[row, col];
            idx++;
        }

        return new SparseTensor<T>(OutputFeatures, InputFeatures, rowIndices, colIndices, values);
    }

    /// <summary>
    /// Performs the forward pass through the layer.
    /// </summary>
    /// <param name="input">Input tensor with shape [inputFeatures] or [batch, inputFeatures].</param>
    /// <returns>Output tensor with shape [outputFeatures] or [batch, outputFeatures].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        bool wasSingleSample = input.Rank == 1;

        int batchSize;
        Matrix<T> inputMatrix;

        if (wasSingleSample)
        {
            // Single sample - validate input size
            if (input.Shape[0] != InputFeatures)
            {
                throw new ArgumentException(
                    $"Input size {input.Shape[0]} does not match expected {InputFeatures}.");
            }

            batchSize = 1;
            inputMatrix = new Matrix<T>(1, InputFeatures);
            for (int i = 0; i < InputFeatures; i++)
            {
                inputMatrix[0, i] = input[i];
            }
        }
        else
        {
            batchSize = input.Shape[0];
            int inputLen = input.Shape[1];

            if (inputLen != InputFeatures)
            {
                throw new ArgumentException(
                    $"Input size {inputLen} does not match expected {InputFeatures}.");
            }

            inputMatrix = new Matrix<T>(batchSize, InputFeatures);
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < InputFeatures; i++)
                {
                    inputMatrix[b, i] = input[b, i];
                }
            }
        }

        // Transpose input for SpMM: output = W * X^T, then transpose back
        var inputTransposed = TransposeMatrix(inputMatrix);

        // Sparse matrix multiplication
        var outputMatrix = _engine.SpMM(_weights, inputTransposed);

        // Transpose back and add biases (output fully overwritten, safe to rent)
        var output = TensorAllocator.Rent<T>([batchSize, OutputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < OutputFeatures; o++)
            {
                output[b, o] = _numOps.Add(outputMatrix[o, b], _biases[o]);
            }
        }

        // Apply activation function
        var activated = ApplyActivation(output);

        // Return 1D tensor for single sample input to match input rank
        if (wasSingleSample)
        {
            var result = new Tensor<T>([OutputFeatures]);
            for (int o = 0; o < OutputFeatures; o++)
            {
                result[o] = activated[0, o];
            }
            // Store _lastOutput with same rank as returned output for consistent backward pass
            _lastOutput = result;
            return result;
        }

        _lastOutput = activated;
        return activated;
    }

    /// <summary>
    /// Updates the layer's parameters using the calculated gradients.
    /// Maintains sparsity pattern during updates.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        // Update biases
        for (int o = 0; o < OutputFeatures; o++)
        {
            var scaledGrad = _numOps.Multiply(learningRate, _biasesGradient[o]);
            _biases[o] = _numOps.Subtract(_biases[o], scaledGrad);
        }

        // Update only the non-zero weight positions (maintain sparsity)
        var newValues = new T[_weights.NonZeroCount];
        for (int nz = 0; nz < _weights.NonZeroCount; nz++)
        {
            int row = _weights.RowIndices[nz];
            int col = _weights.ColumnIndices[nz];
            T grad = _weightsGradient[row, col];
            T scaledGrad = _numOps.Multiply(learningRate, grad);
            newValues[nz] = _numOps.Subtract(_weights.Values[nz], scaledGrad);
        }

        // Create new sparse tensor with updated values
        _weights = new SparseTensor<T>(
            _weights.Rows,
            _weights.Columns,
            _weights.RowIndices,
            _weights.ColumnIndices,
            newValues);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all non-zero weights and biases.</returns>
    public override Vector<T> GetParameters()
    {
        var paramArray = new T[ParameterCount];
        int idx = 0;

        // Add non-zero weights
        for (int nz = 0; nz < _weights.NonZeroCount; nz++)
        {
            paramArray[idx++] = _weights.Values[nz];
        }

        // Add biases
        for (int o = 0; o < OutputFeatures; o++)
        {
            paramArray[idx++] = _biases[o];
        }

        return new Vector<T>(paramArray);
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, but got {parameters.Length}");
        }

        int idx = 0;

        // Restore weights (non-zero values only)
        var newValues = new T[_weights.NonZeroCount];
        for (int nz = 0; nz < _weights.NonZeroCount; nz++)
        {
            newValues[nz] = parameters[idx++];
        }

        _weights = new SparseTensor<T>(
            _weights.Rows,
            _weights.Columns,
            _weights.RowIndices,
            _weights.ColumnIndices,
            newValues);

        // Restore biases
        for (int o = 0; o < OutputFeatures; o++)
        {
            _biases[o] = parameters[idx++];
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        // Match GetParameters layout: non-zero weights (CSR order) + biases
        var gradients = new Vector<T>(ParameterCount);
        int idx = 0;

        if (_weightsGradient != null)
        {
            // Iterate non-zero positions in same CSR order as GetParameters
            for (int nz = 0; nz < _weights.NonZeroCount; nz++)
            {
                int row = _weights.RowIndices[nz];
                int col = _weights.ColumnIndices[nz];
                gradients[idx++] = _weightsGradient[row, col];
            }
        }
        else
        {
            idx += _weights.NonZeroCount;
        }

        if (_biasesGradient != null)
        {
            for (int o = 0; o < OutputFeatures; o++)
                gradients[idx++] = _biasesGradient[o];
        }

        return gradients;
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _weightsGradient = null;
        _biasesGradient = null;
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

    /// <summary>
    /// Applies the activation function to a computation node.
    /// </summary>
    private ComputationNode<T> ApplyActivationToComputationNode(ComputationNode<T> node)
    {
        // ScalarActivation is guaranteed non-null here since this method is only called when ScalarActivation is not null
        if (ScalarActivation is null)
            throw new InvalidOperationException("ScalarActivation cannot be null when applying activation to computation node.");

        // Use ApplyToGraph - the layer's SupportsJitCompilation property ensures this is only
        // called when the activation supports JIT compilation
        if (ScalarActivation.SupportsJitCompilation)
        {
            return ScalarActivation.ApplyToGraph(node);
        }

        // This should never be reached if SupportsJitCompilation is checked before ExportComputationGraph
        throw new InvalidOperationException(
            $"Internal error: Activation function '{ScalarActivation.GetType().Name}' does not support JIT compilation. " +
            "This indicates the layer's SupportsJitCompilation property was not checked before calling ExportComputationGraph.");
    }

    /// <summary>
    /// Transposes a matrix using O(1) stride-based view.
    /// </summary>
    private Matrix<T> TransposeMatrix(Matrix<T> matrix)
    {
        return matrix.Transpose();
    }
}
