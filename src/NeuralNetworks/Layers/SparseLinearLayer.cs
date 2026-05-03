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
    /// <remarks>
    /// NOT registered as a tape-trainable parameter — see the body of the
    /// constructor for the rationale. Updates flow through the manual
    /// <see cref="UpdateParameters(T)"/> path that <see cref="SparseNeuralNetwork{T}.Train"/>
    /// drives directly. Once <see cref="_biases"/> is promoted to a
    /// <see cref="Tensor{T}"/> AND a <c>SetTrainableParameters</c> override
    /// keeps this field in sync with any <see cref="ParameterBuffer{T}"/>
    /// view swaps, the registration can be re-enabled.
    /// </remarks>
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
    /// Gets whether this layer supports training. Returns <c>true</c>:
    /// training flows through the manual
    /// <see cref="LayerBase{T}.UpdateParameters(T)"/> path that
    /// <see cref="SparseNeuralNetwork{T}.Train"/> drives directly with
    /// pre-computed gradients. Tape-mode optimizer training is NOT yet
    /// supported on this layer because (a) <see cref="_biases"/> is a
    /// <see cref="Vector{T}"/> rather than a <see cref="Tensor{T}"/> so
    /// it can't be tape-registered, (b) <see cref="UpdateParameters(T)"/>
    /// replaces <see cref="_weights"/> with a new
    /// <see cref="SparseTensor{T}"/> instance, which would leave any
    /// <see cref="ParameterBuffer{T}"/>-aliased view stale, and (c) no
    /// <c>SetTrainableParameters</c> override is in place to re-sync the
    /// field on view swap. SparseNeuralNetwork therefore routes around
    /// the tape and calls the manual gradient/update pair directly.
    /// </summary>
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

        // Tape-mode trainable-parameter registration is intentionally OMITTED
        // on this layer until three preconditions are met:
        //   1. _biases promoted from Vector<T> to Tensor<T> and registered
        //      with PersistentTensorRole.Bias (the existing dense bias-add in
        //      Forward must move to an engine op so its gradient flows on
        //      the tape).
        //   2. UpdateParameters(T) stops creating a new SparseTensor instance
        //      on every step (it currently re-allocates the .Values buffer,
        //      which leaves any ParameterBuffer view aliasing the OLD instance
        //      stale). Either mutate Values in place, or hand ownership over
        //      to SetTrainableParameters and let the buffer drive updates.
        //   3. A SetTrainableParameters override is added that re-syncs the
        //      _weights field when ParameterBuffer hands back a view-aliased
        //      replacement (otherwise Forward keeps using the old reference
        //      while the optimizer updates the view, decoupling them).
        // Until all three are in place, registering _weights here would
        // produce a tape path that updates a stale view while bias updates
        // silently no-op — strictly worse than the manual path that
        // SparseNeuralNetwork.Train already drives correctly. Tracked as
        // follow-up work; for now SparseNeuralNetwork bypasses the tape and
        // calls ComputeGradients + UpdateParameters(learningRate) directly.
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
    /// Computes gradients with respect to weights, biases, and input.
    /// Stores _weightsGradient (dense matrix) and _biasesGradient internally;
    /// returns dL/dx for upstream propagation. SparseNeuralNetwork drives this
    /// directly because the tape can't see SparseTensor weights — see
    /// <see cref="SparseNeuralNetwork{T}.Train"/> for the integration.
    /// </summary>
    internal Tensor<T> ComputeGradients(Tensor<T> outputGradient)
    {
        if (_lastInput is null || _lastOutput is null)
            throw new InvalidOperationException("Forward pass must run before ComputeGradients.");

        // Validate outputGradient shape so a mismatched gradient doesn't
        // silently propagate wrong values via the rank-vs-batch helper
        // closures below. Rank can legitimately differ when _lastInput was
        // single-sample (rank-1) — accept either matching rank or the
        // [features] / [batch, features] equivalence.
        if (!ShapeMatchesLastOutput(outputGradient))
        {
            throw new ArgumentException(
                $"Output gradient shape [{string.Join(", ", outputGradient.Shape)}] does not match " +
                $"last output shape [{string.Join(", ", _lastOutput.Shape)}].",
                nameof(outputGradient));
        }

        var preActGradient = ComputeActivationDerivative(_lastOutput, outputGradient);

        bool wasSingleSample = _lastInput.Rank == 1;
        int batchSize = wasSingleSample ? 1 : _lastInput.Shape[0];

        _weightsGradient = new Matrix<T>(OutputFeatures, InputFeatures);
        _biasesGradient = new Vector<T>(OutputFeatures);

        // Read both gradient and input via index helpers so we don't have to
        // materialise reshaped views — single-sample tensors are rank-1
        // [features], batched are rank-2 [batch, features].
        T GetGrad(int b, int o) => wasSingleSample ? preActGradient[o] : preActGradient[b, o];
        T GetInput(int b, int i) => wasSingleSample ? _lastInput[i] : _lastInput[b, i];

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < OutputFeatures; o++)
            {
                T g = GetGrad(b, o);
                _biasesGradient[o] = _numOps.Add(_biasesGradient[o], g);
                for (int i = 0; i < InputFeatures; i++)
                {
                    _weightsGradient[o, i] = _numOps.Add(
                        _weightsGradient[o, i],
                        _numOps.Multiply(g, GetInput(b, i)));
                }
            }
        }

        var inputGradient = new Tensor<T>(_lastInput._shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int nz = 0; nz < _weights.NonZeroCount; nz++)
            {
                int o = _weights.RowIndices[nz];
                int i = _weights.ColumnIndices[nz];
                T contribution = _numOps.Multiply(GetGrad(b, o), _weights.Values[nz]);
                if (wasSingleSample)
                    inputGradient[i] = _numOps.Add(inputGradient[i], contribution);
                else
                    inputGradient[b, i] = _numOps.Add(inputGradient[b, i], contribution);
            }
        }

        return inputGradient;
    }

    private Tensor<T> ComputeActivationDerivative(Tensor<T> output, Tensor<T> upstream)
    {
        // Identity activation: dy/d(pre) = 1, gradient passes through unchanged.
        if (ScalarActivation is null && VectorActivation is null)
            return upstream;

        var derivative = new Tensor<T>(output._shape);
        if (ScalarActivation is not null)
        {
            for (int i = 0; i < output.Length; i++)
                derivative[i] = ScalarActivation.Derivative(output[i]);
        }
        else
        {
            for (int i = 0; i < output.Length; i++)
                derivative[i] = _numOps.One;
        }

        var result = new Tensor<T>(output._shape);
        for (int i = 0; i < output.Length; i++)
            result[i] = _numOps.Multiply(upstream[i], derivative[i]);
        return result;
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
    /// Transposes a matrix using O(1) stride-based view.
    /// </summary>
    private Matrix<T> TransposeMatrix(Matrix<T> matrix)
    {
        return matrix.Transpose();
    }

    /// <summary>
    /// Returns true when <paramref name="grad"/> shape is consistent with
    /// the last forward's output. <c>_lastOutput</c> can be rank-1
    /// <c>[features]</c> for single-sample input or rank-2
    /// <c>[batch, features]</c> for batched input. The check accepts BOTH
    /// the strict same-rank match AND the rank-1 ↔ rank-2 single-sample
    /// equivalence: rank-1 <c>[features]</c> matches rank-2
    /// <c>[1, features]</c> with the same feature count, since both
    /// represent the same single-sample output and the per-element
    /// gradient indexing in the manual backprop path treats them the
    /// same way.
    /// </summary>
    private bool ShapeMatchesLastOutput(Tensor<T> grad)
    {
        if (_lastOutput is null) return false;
        if (grad.Length != _lastOutput.Length) return false;

        if (grad.Rank == _lastOutput.Rank)
        {
            for (int i = 0; i < grad.Rank; i++)
            {
                if (grad.Shape[i] != _lastOutput.Shape[i]) return false;
            }
            return true;
        }

        // Cross-rank single-sample equivalence: rank-1 [features] vs
        // rank-2 [1, features]. Either side may carry the singleton.
        // Reject anything else (e.g. rank-3 vs rank-1) — that's a real
        // mismatch the manual backprop path can't handle.
        if (grad.Rank == 1 && _lastOutput.Rank == 2 && _lastOutput.Shape[0] == 1)
        {
            return grad.Shape[0] == _lastOutput.Shape[1];
        }
        if (grad.Rank == 2 && _lastOutput.Rank == 1 && grad.Shape[0] == 1)
        {
            return grad.Shape[1] == _lastOutput.Shape[0];
        }
        return false;
    }
}
