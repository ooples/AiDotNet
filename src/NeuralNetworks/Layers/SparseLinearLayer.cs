using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
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
public class SparseLinearLayer<T> : LayerBase<T>
{
    private readonly ISparseEngine _engine;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The sparse weight matrix.
    /// Shape: [OutputFeatures, InputFeatures]
    /// </summary>
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
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// Sparse operations are not yet supported for JIT.
    /// </summary>
    public override bool SupportsJitCompilation => false;

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
        IActivationFunction<T>? activationFunction = null)
        : base(
            [inputFeatures],
            [outputFeatures],
            activationFunction ?? new ReLUActivation<T>())
    {
        if (sparsity < 0 || sparsity >= 1.0)
        {
            throw new ArgumentException("Sparsity must be in [0, 1).", nameof(sparsity));
        }

        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;
        _sparsity = sparsity;

        _engine = CpuSparseEngine.Instance;
        _numOps = MathHelper.GetNumericOperations<T>();

        _biases = new Vector<T>(outputFeatures);
        _weights = InitializeSparseWeights();
    }

    /// <summary>
    /// Initializes sparse weights using Xavier/Glorot initialization.
    /// </summary>
    private SparseTensor<T> InitializeSparseWeights()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var scale = Math.Sqrt(2.0 / (InputFeatures + OutputFeatures));

        // Calculate number of non-zero weights
        int totalWeights = OutputFeatures * InputFeatures;
        int nonZeroCount = Math.Max(1, (int)(totalWeights * (1.0 - _sparsity)));

        // Generate random indices
        var indices = new HashSet<(int row, int col)>();
        while (indices.Count < nonZeroCount)
        {
            int row = random.Next(OutputFeatures);
            int col = random.Next(InputFeatures);
            indices.Add((row, col));
        }

        // Convert to COO format
        var rowIndices = new int[nonZeroCount];
        var colIndices = new int[nonZeroCount];
        var values = new T[nonZeroCount];

        int idx = 0;
        foreach (var (row, col) in indices.OrderBy(x => x.row).ThenBy(x => x.col))
        {
            rowIndices[idx] = row;
            colIndices[idx] = col;
            values[idx] = _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale);
            idx++;
        }

        // Initialize biases to zero
        for (int i = 0; i < OutputFeatures; i++)
        {
            _biases[i] = _numOps.Zero;
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

        // Transpose back and add biases
        var output = new Tensor<T>([batchSize, OutputFeatures]);
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
    /// Performs the backward pass through the layer.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }

        // Apply activation derivative to get the true gradient (chain rule)
        var delta = ApplyActivationDerivative(_lastOutput, outputGradient);

        bool wasSingleSample = delta.Rank == 1;
        int batchSize;
        Tensor<T> gradTensor;

        if (wasSingleSample)
        {
            batchSize = 1;
            // Convert 1D to 2D for processing
            gradTensor = new Tensor<T>([1, OutputFeatures]);
            for (int o = 0; o < OutputFeatures; o++)
            {
                gradTensor[0, o] = delta[o];
            }
        }
        else
        {
            batchSize = delta.Shape[0];
            gradTensor = delta;
        }

        // Initialize gradients
        _weightsGradient = new Matrix<T>(OutputFeatures, InputFeatures);
        _biasesGradient = new Vector<T>(OutputFeatures);

        // Compute gradients
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < OutputFeatures; o++)
            {
                T gradOut = gradTensor[b, o];

                // Bias gradient: sum over batch
                _biasesGradient[o] = _numOps.Add(_biasesGradient[o], gradOut);

                // Weight gradient: only update non-zero positions
                for (int nz = 0; nz < _weights.NonZeroCount; nz++)
                {
                    int row = _weights.RowIndices[nz];
                    int col = _weights.ColumnIndices[nz];
                    if (row == o)
                    {
                        T inputVal = _lastInput.Rank == 1 ? _lastInput[col] : _lastInput[b, col];
                        var contrib = _numOps.Multiply(gradOut, inputVal);
                        _weightsGradient[row, col] = _numOps.Add(_weightsGradient[row, col], contrib);
                    }
                }
            }
        }

        // Compute input gradient using transpose of weights
        var transposedWeights = _engine.SparseTranspose(_weights);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            var gradVector = new Vector<T>(OutputFeatures);
            for (int o = 0; o < OutputFeatures; o++)
            {
                gradVector[o] = gradTensor[b, o];
            }

            var inputGradVec = _engine.SpMV(transposedWeights, gradVector);

            for (int i = 0; i < InputFeatures; i++)
            {
                if (_lastInput.Rank == 1)
                {
                    inputGradient[i] = _numOps.Add(inputGradient[i], inputGradVec[i]);
                }
                else
                {
                    inputGradient[b, i] = inputGradVec[i];
                }
            }
        }

        return inputGradient;
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

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

    /// <summary>
    /// Exports the layer's forward pass as a JIT-compilable computation graph.
    /// Currently not supported for sparse layers.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node.</returns>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "SparseLinearLayer does not yet support JIT compilation. " +
            "Sparse operations require specialized IR operations.");
    }

    /// <summary>
    /// Transposes a matrix.
    /// </summary>
    private Matrix<T> TransposeMatrix(Matrix<T> matrix)
    {
        var result = new Matrix<T>(matrix.Columns, matrix.Rows);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[j, i] = matrix[i, j];
            }
        }
        return result;
    }
}
