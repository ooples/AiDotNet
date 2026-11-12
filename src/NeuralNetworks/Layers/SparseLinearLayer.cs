using AiDotNet.Helpers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines.Autodiff;

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
// A dense projection that happens to store its weights sparsely. SPARSITY IS NOT A SHAPE PROPERTY:
// _sparsity decides how many of the [OutputFeatures, InputFeatures] entries are non-zero, not how wide
// the result is - the sparse matmul still produces a fully dense [batch, OutputFeatures]. So the
// contract is the ordinary linear one and says nothing about _sparsity.
//
// Ranks 1 and 2 only, per ForwardTraced's own <param> ("shape [inputFeatures] or [batch,
// inputFeatures]") and enforced there: it promotes rank 1 to [1, InputFeatures] and then reads
// input2d.Shape[1], so a rank-3 input would compare the wrong axis against InputFeatures and throw.
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class SparseLinearLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// From <c>ForwardTraced</c>: the chain ends at
    /// <c>TensorAdd(outBO, Reshape(_biases, [1, OutputFeatures]))</c>, giving
    /// <c>[batch, OutputFeatures]</c>, and a rank-1 input is reshaped back to <c>[OutputFeatures]</c>
    /// on the way out. <c>OutputFeatures</c> is the constructor argument that sizes the weight matrix's
    /// row count, so <c>Fixed</c> is reading the layer's own parameter, not an observed number.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (OutputFeatures <= 0) return null;

        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(OutputFeatures));

        return inputRank switch
        {
            1 => new[] { features },
            2 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                features,
            },
            _ => null,
        };
    }

    private readonly ISparseEngine _engine;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The sparse weight matrix.
    /// Shape: [OutputFeatures, InputFeatures]
    /// </summary>
    /// <remarks>
    /// Registered as a tape-trainable parameter via
    /// <see cref="LayerBase{T}.RegisterTrainableParameter"/>. Updates always
    /// mutate the existing instance's <c>Values</c> array in place (see
    /// <see cref="UpdateParameters(T)"/> and <see cref="SetParameters"/>) so
    /// any <see cref="ParameterBuffer{T}"/> view aliasing this field stays
    /// synchronized. <see cref="SetTrainableParameters"/> handles the case
    /// where the buffer hands back a replacement instance and re-syncs the
    /// field reference.
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private SparseTensor<T> _weights;

    /// <summary>
    /// The bias values, registered as a tape-trainable parameter alongside
    /// <see cref="_weights"/>. 1-D tensor of shape <c>[OutputFeatures]</c>;
    /// promoted from <c>Vector{T}</c> so the tape can register it directly.
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _biases;

    /// <summary>
    /// The sparsity level (fraction of weights that are zero).
    /// </summary>
    private readonly double _sparsity;

    /// <summary>
    /// Stored input from forward pass for backpropagation.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored pre-activation output for gradient computation.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gradient for weights, stored during backward pass.
    /// Stored as a dense [OutputFeatures, InputFeatures] tensor, then sparsified on update.
    /// </summary>
    [Scratch]
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Gradient for biases, stored during backward pass. 1-D tensor matching
    /// <see cref="_biases"/>'s shape so gradient layout stays consistent
    /// across the manual backprop path and tape-mode.
    /// </summary>
    [Scratch]
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int OutputFeatures { get; }

    /// <summary>
    /// Gets whether this layer supports training. Returns <c>true</c>: both
    /// the sparse weight tensor and the dense bias tensor are registered as
    /// trainable parameters and round-trip through the tape's
    /// <see cref="ParameterBuffer{T}"/>. Updates mutate the registered
    /// instances in place (no per-step re-allocation) so view aliasing
    /// stays valid across training steps. The legacy
    /// <see cref="UpdateParameters(T)"/> path is also kept as a fallback
    /// for callers that aren't tape-driven (e.g.,
    /// <see cref="SparseNeuralNetwork{T}.Train"/>'s manual loop).
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
        [LayerState] int inputFeatures,
        [LayerState] int outputFeatures,
        [LayerState] double sparsity = 0.9,
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

        _biases = new Tensor<T>([outputFeatures]);
        // Biases initialized to zero by default (standard practice for ReLU layers)
        _weights = InitializeSparseWeights();

        // Tape-mode trainable-parameter registration. Both tensors are
        // registered so ParameterBuffer-driven optimizers see the full
        // parameter set.
        //   - _weights: SparseTensor<T> inherits from Tensor<T>; the
        //     ParameterBuffer is sparse-aware in AiDotNet.Tensors 0.70.0
        //     and only allocates NonZeroCount worth of buffer space.
        //     UpdateParameters and SetParameters mutate _weights.Values in
        //     place (no per-step new SparseTensor) so view aliasing stays
        //     valid; SetTrainableParameters re-syncs the field reference
        //     if the buffer hands back a replacement instance.
        //   - _biases: dense 1-D tensor [OutputFeatures]; registered with
        //     the standard Bias role so the tape can update both
        //     parameters in lockstep.
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes sparse weights using the layer's initialization strategy.
    /// </summary>
    private SparseTensor<T> InitializeSparseWeights()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Calculate number of non-zero weights
        // WIDENED TO long. OutputFeatures * InputFeatures is int * int, so it silently wraps for a
        // large layer: 32768 x 16384 lands on -536870912 (-2^29), which then travelled onward as a
        // NEGATIVE shape dimension and failed far from here as "Shape dimension 0 must be
        // non-negative". A sparse layer is exactly the kind that is wide enough to overflow, and the
        // count below is a proportion of the dense size, so it must be computed at full width and
        // only narrowed once it is known to fit.
        long totalWeights = (long)OutputFeatures * InputFeatures;
        long requested = (long)(totalWeights * (1.0 - _sparsity));
        if (requested > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"{GetType().Name} would need {requested:N0} non-zero weights for a "
                    + $"{OutputFeatures}x{InputFeatures} layer at sparsity {_sparsity}, which exceeds "
                    + "what a single sparse tensor can index. Raise the sparsity or split the layer.");
        }

        int nonZeroCount = (int)Math.Max(1L, requested);

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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        SaveForBackward(ref _lastInput, input); // #1668: skip in inference (arena safety)
        bool wasSingleSample = input.Rank == 1;

        // Promote a single rank-1 [InputFeatures] sample to [1, InputFeatures] so the whole
        // path is one uniform [batch, InputFeatures] dense op chain.
        var input2d = wasSingleSample
            ? Engine.Reshape(input, new[] { 1, InputFeatures })
            : input;

        int inputLen = input2d.Shape[1];
        if (inputLen != InputFeatures)
        {
            throw new ArgumentException(
                $"Input size {inputLen} does not match expected {InputFeatures}.");
        }

        // output = input @ Wᵀ = (W @ inputᵀ)ᵀ + bias, with W the sparse weight [Out, In].
        // Built entirely from TAPE-TRACKED Engine ops: the sparse matmul auto-records via
        // AiDotNet.Tensors #758 ISparseEngine.SparseMatMul (dense-gradient variant), so the
        // gradient reaches the registered _weights / _biases. The prior implementation
        // copied into Matrix<T> element by element and called the non-differentiable
        // ISparseEngine.SpMM, detaching the tape — TapeGradient_ShouldReachAtLeastOne-
        // TrainableParameter observed every parameter get a zero gradient.
        var inputT = Engine.TensorTranspose(input2d);                        // [In, batch]
        var outT = _engine.SparseMatMul(_weights, inputT);                   // [Out, batch]
        var outBO = Engine.TensorTranspose(outT);                            // [batch, Out]
        var biased = Engine.TensorAdd(
            outBO, Engine.Reshape(_biases, new[] { 1, OutputFeatures }));    // [batch, Out]
        var activated = ApplyActivation(biased);

        // Return a rank-1 [OutputFeatures] tensor for a rank-1 input to match its rank.
        if (wasSingleSample)
        {
            var result = Engine.Reshape(activated, new[] { OutputFeatures });
            SaveForBackward(ref _lastOutput, result);
            return result;
        }

        SaveForBackward(ref _lastOutput, activated);
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

        // Manual gradients into fresh buffers, exactly as master: nothing here joins the tape.
        using var noGrad = new NoGradScope<T>();
        var gradient2d = Engine.Reshape(preActGradient, new[] { batchSize, OutputFeatures });   // [batch, Out]
        var input2d = Engine.Reshape(_lastInput, new[] { batchSize, InputFeatures });         // [batch, In]

        // dB[o] = sum_b g[b, o]; dW[o, i] = sum_b g[b, o] x[b, i] (dense, sparsified on update).
        _biasesGradient = Engine.ReduceSum(gradient2d, new[] { 0 });
        _weightsGradient = Engine.TensorMatMul(Engine.TensorTranspose(gradient2d), input2d);   // [Out, In]

        // dX = g @ W; the zero entries of the sparse weight contribute nothing.
        var inputGradient = Engine.TensorMatMul(gradient2d, _weights.ToDense());              // [batch, In]
        return Engine.Reshape(inputGradient, _lastInput._shape);
    }

    private Tensor<T> ComputeActivationDerivative(Tensor<T> output, Tensor<T> upstream)
    {
        // Identity activation: dy/d(pre) = 1, gradient passes through unchanged.
        if (ScalarActivation is null && VectorActivation is null)
            return upstream;

        // The derivative belongs at the PRE-activation value. Evaluating Derivative(output) gave
        // f'(f(x)) - wrong for sigmoid, tanh, GELU, ... - and a vector activation was treated as
        // the identity. The base helper uses the activation's output-form derivative when it has one,
        // the cached pre-activation otherwise, and the full Jacobian for vector activations.
        return ApplyActivationDerivativeFromOutput(output, upstream);
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

        using var noGrad = new NoGradScope<T>();

        // Update biases in place — preserves the registered tensor reference
        // so any ParameterBuffer view aliasing _biases stays valid.
        Engine.TensorCopy(
            Engine.TensorSubtract(_biases, Engine.TensorMultiplyScalar(_biasesGradient, learningRate)), _biases);

        // Update only the non-zero weight positions (maintain sparsity): gather the dense gradient at
        // each stored (row, col) and subtract it from the stored values.
        // Write into the sparse tensor's own value storage rather than constructing a new
        // SparseTensor — same reasoning as biases. A new instance would leave any ParameterBuffer
        // view aliasing the OLD _weights stale. SparseTensor.Values returns a COPY of the values, so
        // the previous `_weights.Values[nz] = ...` loop updated a temporary and never changed a weight.
        int nonZeroCount = _weights.NonZeroCount;
        var rows = new Tensor<int>(_weights.RowIndices, new[] { nonZeroCount });
        var columns = new Tensor<int>(_weights.ColumnIndices, new[] { nonZeroCount });
        var flatPositions = Engine.TensorAdd(Engine.TensorMultiplyScalar(rows, InputFeatures), columns);
        var storedGradients = Engine.TensorIndexSelect(
            Engine.Reshape(_weightsGradient, new[] { OutputFeatures * InputFeatures }), flatPositions, 0);
        var storedValues = new Tensor<T>(_weights.Values, new[] { nonZeroCount });
        var updatedValues = Engine.TensorSubtract(storedValues, Engine.TensorMultiplyScalar(storedGradients, learningRate));
        updatedValues.AsSpan().CopyTo(_weights.DataVector.AsWritableSpan());
        _weights.IncrementVersion();
    }

    /// <summary>
    /// Adds sparsity to the layer metadata so deserialization rebuilds the
    /// layer with the same sparsity ratio and activation function. The base
    /// override already emits <c>ScalarActivationType</c>; without
    /// <c>Sparsity</c> the deserializer constructs the layer at the default
    /// 0.9 and any non-default sparsity (e.g., the output-head 0.99 some
    /// pruned-network papers use) would silently change post-Clone.
    /// </summary>
    internal override System.Collections.Generic.Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Sparsity"] = _sparsity.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
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

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

    // SetTrainableParameters override is auto-generated by
    // TrainableParameterGenerator from the [TrainableParameter] attributes
    // on _weights and _biases. The generated code re-syncs both field
    // references when ParameterBuffer hands back view-aliased replacements.
    // See *.TrainableParameters.g.cs.

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
