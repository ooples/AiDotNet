using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the recurrent memory cell from the original HiPPO paper.
/// </summary>
/// <remarks>
/// <para>
/// The cell maintains a trainable hidden state and a fixed, non-trainable polynomial memory.
/// The memory is updated with the selected HiPPO operator, while learned input-to-memory,
/// memory-to-hidden, and sigmoid-gate projections reproduce the paper's HiPPO-RNN wiring.
/// </para>
/// <para>
/// Reference: Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (2020).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Recurrent)]
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, ChangesShape = true, Cost = ComputeCost.High,
    TestInputShape = "1, 4, 2", TestConstructorArgs = "4, 2, 4")]
// A recurrent cell: it consumes a SEQUENCE and emits one hidden state per step, so the time axis is
// carried through and only the trailing width changes. ForwardTraced spells both halves out - it reads
// sequenceLength = input.Shape[Rank-2] and requires input.Shape[Rank-1] == _inputSize, then returns
// outputShape = input.Shape with the last entry overwritten by _hiddenSize.
// BatchOptional covers rank 2 as well as rank 3, and rank 2 is genuinely supported rather than merely
// tolerated: ForwardTraced has an explicit `if (input.Rank == 2) return Reshape(result,
// [sequenceLength, _hiddenSize])` branch, i.e. an unbatched [Time, Features] sequence.
// Higher ranks run too - every axis before the last two is folded into batchSize - but each extra
// leading axis would need a distinct role for a relation to name it, and there is no second batch-like
// role to give it, so OutputAxesFor declines beyond rank 3 rather than guessing.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class HippoMemoryCellLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _hiddenSize;
    private readonly int _inputSize;
    private readonly int _memorySize;
    private readonly int _memoryOrder;
    private readonly string _measure;
    private readonly string _discretization;
    private readonly int _initialTime;
    private readonly double _timeStep;
    private readonly double _timescaleMin;
    private readonly double _timescaleMax;
    private readonly bool _useGate;

    // Fixed HiPPO operator, rebuilt from _measure and _order at construction. These are buffers,
    // not trainable parameters, and all three must stay declared the same way: _aTranspose alone
    // briefly carried [TrainableParameter], which the hand-written parameter pair below hid
    // because that pair made the generator skip this type entirely. Deleting the pair would have
    // promoted a fixed operator into the trainable vector and moved every downstream offset.
    private readonly Tensor<T> _a;
    private readonly Tensor<T> _aTranspose;
    private readonly Tensor<T> _bRow;

    // Original HiPPO-RNN learned projections.
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _memoryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _memoryBias;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _hiddenWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _hiddenBias;
    // The gate exists only when the layer is configured with one, so both of its tensors are
    // conditional. Without the condition the generated surface would always publish six tensors
    // while an ungated cell holds four, and the count would stop matching the vector.
    [TrainableParameter(Role = PersistentTensorRole.Weights, Condition = nameof(_useGate))]
    private Tensor<T> _gateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases, Condition = nameof(_useGate))]
    private Tensor<T> _gateBias;

    private readonly Tensor<T>? _ltiTransition;
    private readonly Tensor<T>? _ltiInput;
    [Scratch]
    private readonly Dictionary<int, (Tensor<T> A, Tensor<T> B)> _legsZohCache = new();

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Gets the hidden-state width.</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Gets the number of independent polynomial memories.</summary>
    public int MemorySize => _memorySize;

    /// <summary>Gets the polynomial order of each memory.</summary>
    public int MemoryOrder => _memoryOrder;

    /// <summary>Gets the configured HiPPO measure.</summary>
    public string Measure => _measure;

    /// <summary>Gets the configured discretization method.</summary>
    public string Discretization => _discretization;

    private static void ValidateShapeMatch(Tensor<T> incoming, Tensor<T> existing, string parameterName)
    {
        if (incoming.Rank != existing.Rank || incoming.Length != existing.Length)
            throw new ArgumentException(
                $"Shape mismatch for {parameterName}: incoming rank={incoming.Rank} length={incoming.Length}, " +
                $"expected rank={existing.Rank} length={existing.Length}.");
        for (int dimension = 0; dimension < incoming.Rank; dimension++)
        {
            if (incoming.Shape[dimension] != existing.Shape[dimension])
                throw new ArgumentException(
                    $"Shape mismatch for {parameterName} at dimension {dimension}: " +
                    $"incoming={incoming.Shape[dimension]}, expected={existing.Shape[dimension]}.");
        }
    }

    /// <summary>
    /// Creates an original-paper HiPPO recurrent cell.
    /// </summary>
    /// <param name="hiddenSize">Hidden-state width. Paper default: 256.</param>
    /// <param name="inputSize">Input feature width.</param>
    /// <param name="memoryOrder">Polynomial order. Paper default resolves to hiddenSize.</param>
    /// <param name="memorySize">Number of independent memories. Paper default: 1.</param>
    /// <param name="measure">HiPPO measure: legs, legt, or lagt.</param>
    /// <param name="discretization">forward/euler, backward, bilinear, or zoh.</param>
    /// <param name="initialTime">Initial time index for scale-invariant LegS memory.</param>
    /// <param name="timeStep">LTI step; zero selects the paper default for the measure.</param>
    /// <param name="timescaleMin">Optional lower clamp for the effective step.</param>
    /// <param name="timescaleMax">Optional upper clamp for the effective step.</param>
    /// <param name="useGate">Whether to use the paper's standard sigmoid hidden-state gate.</param>
    public HippoMemoryCellLayer(
        [LayerState] int hiddenSize = 256,
        [LayerState] int inputSize = 1,
        [LayerState] int memoryOrder = -1,
        [LayerState] int memorySize = 1,
        [LayerState] string measure = "legs",
        [LayerState] string discretization = "bilinear",
        [LayerState] int initialTime = 0,
        [LayerState] double timeStep = 0.0,
        [LayerState] double timescaleMin = 0.0,
        [LayerState] double timescaleMax = double.PositiveInfinity,
        [LayerState] bool useGate = true,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { -1, inputSize }, new[] { -1, hiddenSize })
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (memoryOrder == 0 || memoryOrder < -1) throw new ArgumentOutOfRangeException(nameof(memoryOrder));
        if (memorySize <= 0) throw new ArgumentOutOfRangeException(nameof(memorySize));
        if (initialTime < 0) throw new ArgumentOutOfRangeException(nameof(initialTime));
        if (timeStep < 0 || double.IsNaN(timeStep) || double.IsInfinity(timeStep))
            throw new ArgumentOutOfRangeException(nameof(timeStep));
        if (timescaleMin < 0 || double.IsNaN(timescaleMin) || double.IsInfinity(timescaleMin))
            throw new ArgumentOutOfRangeException(nameof(timescaleMin));
        if (timescaleMax <= 0 || double.IsNaN(timescaleMax) || timescaleMax < timescaleMin)
            throw new ArgumentOutOfRangeException(nameof(timescaleMax));

        _hiddenSize = hiddenSize;
        _inputSize = inputSize;
        _memoryOrder = memoryOrder < 0 ? hiddenSize : memoryOrder;
        _memorySize = memorySize;
        _measure = NormalizeMeasure(measure);
        _discretization = NormalizeDiscretization(discretization);
        _initialTime = initialTime;
        _timeStep = timeStep > 0 ? timeStep : (_measure == "lagt" ? 1.0 : 0.01);
        _timescaleMin = timescaleMin;
        _timescaleMax = timescaleMax;
        _useGate = useGate;
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        (_a, _bRow) = CreateHippoOperator(_measure, _memoryOrder);
        _aTranspose = Engine.TensorPermute(_a, new[] { 1, 0 });

        _memoryWeights = new Tensor<T>(new[] { inputSize + hiddenSize, memorySize });
        _memoryBias = new Tensor<T>(new[] { memorySize });
        int hiddenInputSize = inputSize + memorySize * _memoryOrder;
        _hiddenWeights = new Tensor<T>(new[] { hiddenInputSize, hiddenSize });
        _hiddenBias = new Tensor<T>(new[] { hiddenSize });
        _gateWeights = new Tensor<T>(new[] { hiddenInputSize, hiddenSize });
        _gateBias = new Tensor<T>(new[] { hiddenSize });

        InitializeParameters(usePaperDefaults: initializationStrategy is null);
        RegisterParameters();

        if (_measure == "legs")
        {
            _ltiTransition = null;
            _ltiInput = null;
        }
        else
        {
            (_ltiTransition, _ltiInput) = BuildDiscreteTransition(ClampStep(_timeStep));
        }
    }

    private static string NormalizeMeasure(string value)
    {
        string normalized = (value ?? string.Empty).Trim().ToLowerInvariant();
        return normalized switch
        {
            "legs" => "legs",
            "legt" => "legt",
            "lagt" => "lagt",
            _ => throw new ArgumentException("HiPPO measure must be 'legs', 'legt', or 'lagt'.", nameof(value))
        };
    }

    private static string NormalizeDiscretization(string value)
    {
        string normalized = (value ?? string.Empty).Trim().ToLowerInvariant();
        return normalized switch
        {
            "euler" or "forward" or "forward_euler" or "forward_diff" => "forward",
            "backward" or "backward_euler" or "backward_diff" => "backward",
            "bilinear" or "tustin" or "trapezoidal" or "trapezoid" => "bilinear",
            "zoh" => "zoh",
            _ => throw new ArgumentException(
                "Discretization must be forward/euler, backward, bilinear, or zoh.", nameof(value))
        };
    }

    private (Tensor<T> A, Tensor<T> B) CreateHippoOperator(string measure, int order)
    {
        // The operator is a fixed constant built from fresh tensors, so it stays off the tape.
        using var noGrad = new NoGradScope<T>();
        var index = Engine.TensorArange<T>(0, order, 1.0);                                   // i
        var r = Engine.TensorSqrt(Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(index, NumOps.FromDouble(2.0)), NumOps.One));       // sqrt(2i + 1)
        var ones = new Tensor<T>(new[] { order, order });
        ones.Fill(NumOps.One);

        Tensor<T> a;
        Tensor<T> bVector;
        switch (measure)
        {
            case "legs":
                // A[i, j] = -r_i r_j below the diagonal, -(i + 1) on it, zero above; B[i] = r_i.
                bVector = r;
                a = Engine.TensorSubtract(
                    Engine.TensorDiag(Engine.TensorNegate(Engine.TensorAddScalar(index, NumOps.One))),
                    Engine.TensorTril(Engine.TensorOuter(r, r), -1));
                break;

            case "legt":
                // A[i, j] = -0.5 r_i r_j, with the sign flipped where i < j and i - j is odd; B[i] = 0.5 r_i.
                bVector = Engine.TensorMultiplyScalar(r, NumOps.FromDouble(0.5));
                var two = new Tensor<T>(new[] { order });
                two.Fill(NumOps.FromDouble(2.0));
                var parity = Engine.ScalarMinusTensor(NumOps.One,
                    Engine.TensorMultiplyScalar(Engine.TensorRemainder(index, two), NumOps.FromDouble(2.0)));  // (-1)^i
                // sign = 1 - [i < j] * (1 - (-1)^(i + j)), i.e. -1 exactly where i < j and i - j is odd.
                var sign = Engine.TensorSubtract(ones,
                    Engine.TensorTriu(Engine.TensorSubtract(ones, Engine.TensorOuter(parity, parity)), 1));
                a = Engine.TensorMultiplyScalar(
                    Engine.TensorMultiply(Engine.TensorOuter(r, r), sign), NumOps.FromDouble(-0.5));
                break;

            default: // lagt
                // A = I / 2 - tril(ones); B = ones.
                bVector = new Tensor<T>(new[] { order });
                bVector.Fill(NumOps.One);
                a = Engine.TensorSubtract(
                    Engine.TensorMultiplyScalar(Engine.TensorEye<T>(order), NumOps.FromDouble(0.5)),
                    Engine.TensorTril(ones, 0));
                break;
        }

        return (a, Engine.Reshape(bVector, new[] { 1, order }));
    }

    private void InitializeParameters(bool usePaperDefaults)
    {
        if (!usePaperDefaults)
        {
            InitializeLayerWeights(_memoryWeights, _memoryWeights.Shape[0], _memoryWeights.Shape[1]);
            InitializeLayerBiases(_memoryBias);
            InitializeLayerWeights(_hiddenWeights, _hiddenWeights.Shape[0], _hiddenWeights.Shape[1]);
            InitializeLayerBiases(_hiddenBias);
            InitializeLayerWeights(_gateWeights, _gateWeights.Shape[0], _gateWeights.Shape[1]);
            InitializeLayerBiases(_gateBias);
            return;
        }

        // Match the official HiPPO-RNN implementation:
        // W_uxh uses Kaiming-uniform with linear gain; W_hxm uses Xavier-normal;
        // the standard sigmoid gate keeps torch.nn.Linear's default uniform init.
        InitializeUniform(_memoryWeights, _memoryWeights.Shape[0], Math.Sqrt(3.0));
        InitializeUniform(_memoryBias, _memoryWeights.Shape[0]);
        InitializeLayerWeights(_hiddenWeights, _hiddenWeights.Shape[0], _hiddenWeights.Shape[1]);
        InitializeUniform(_hiddenBias, _hiddenWeights.Shape[0]);
        InitializeUniform(_gateWeights, _gateWeights.Shape[0]);
        InitializeUniform(_gateBias, _gateWeights.Shape[0]);
    }

    private void InitializeUniform(Tensor<T> tensor, int fanIn, double gain = 1.0)
    {
        double bound = gain / Math.Sqrt(Math.Max(1, fanIn));
        var data = tensor.DataVector.GetDataArray();
        for (int i = 0; i < tensor.Length; i++)
            data[i] = NumOps.FromDouble((Random.NextDouble() * 2.0 - 1.0) * bound);
    }

    private void RegisterParameters()
    {
        RegisterTrainableParameter(_memoryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_memoryBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_hiddenWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_hiddenBias, PersistentTensorRole.Biases);
        if (_useGate)
        {
            RegisterTrainableParameter(_gateWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_gateBias, PersistentTensorRole.Biases);
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Hand-written rather than generated, because the trailing axis CHANGES: this is a projection to
    /// the hidden width, so the axis roles alone would describe it as shape-preserving and be wrong.
    /// Derived from the tail of <see cref="ForwardTraced"/>:
    /// <c>outputShape = input.Shape.ToArray(); outputShape[^1] = _hiddenSize;</c>
    /// </para>
    /// <para>
    /// <c>Fixed(_hiddenSize)</c> is legitimate here in the way a hardcoded number would not be - the
    /// hidden width is a constructor argument kept in a field, and it is the same value the base
    /// constructor already publishes as the output shape (<c>base(new[] { -1, inputSize },
    /// new[] { -1, hiddenSize })</c>). The time axis is <c>Same</c> because the loop emits exactly one
    /// hidden state per step and stacks them on axis 1.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        // ForwardTraced rejects rank < 2 outright, and ranks above 3 have no distinct role available
        // for the extra leading axes, so both decline honestly.
        if (inputRank is not (2 or 3) || _hiddenSize <= 0) return null;

        var time = new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time));
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hiddenSize));

        return inputRank == 2
            ? new[] { time, features }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                time, features,
            };
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank < 2)
            throw new ArgumentException("HiPPO memory input must have a sequence and feature dimension.", nameof(input));
        if (input.Shape[input.Rank - 1] != _inputSize)
            throw new ArgumentException($"Expected input feature size {_inputSize}, got {input.Shape[input.Rank - 1]}.", nameof(input));

        int sequenceLength = input.Shape[input.Rank - 2];
        int batchSize = 1;
        for (int i = 0; i < input.Rank - 2; i++) batchSize *= input.Shape[i];
        var sequence = Engine.Reshape(input, new[] { batchSize, sequenceLength, _inputSize });

        var hidden = new Tensor<T>(new[] { batchSize, _hiddenSize });
        Tensor<T>? memory = null;
        var outputs = new Tensor<T>[sequenceLength];
        T minusOne = NumOps.FromDouble(-1.0);

        for (int step = 0; step < sequenceLength; step++)
        {
            var current = Engine.TensorSliceAxis(sequence, 1, step);
            var xh = Engine.TensorConcatenate(new[] { current, hidden }, axis: 1);
            var memoryInput = Engine.TensorMatMul(xh, _memoryWeights);
            memoryInput = Engine.TensorAdd(memoryInput, _memoryBias);
            memory = UpdateMemory(memory, memoryInput, step, batchSize);

            var flattenedMemory = Engine.Reshape(memory,
                new[] { batchSize, _memorySize * _memoryOrder });
            var xm = Engine.TensorConcatenate(new[] { current, flattenedMemory }, axis: 1);
            var candidate = Engine.TensorMatMul(xm, _hiddenWeights);
            candidate = Engine.TensorAdd(candidate, _hiddenBias);
            candidate = Engine.Tanh(candidate);

            if (_useGate)
            {
                var gate = Engine.TensorMatMul(xm, _gateWeights);
                gate = Engine.TensorAdd(gate, _gateBias);
                gate = Engine.Sigmoid(gate);
                var keep = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(gate, minusOne), NumOps.One);
                hidden = Engine.TensorAdd(
                    Engine.TensorMultiply(keep, hidden),
                    Engine.TensorMultiply(gate, candidate));
            }
            else
            {
                hidden = candidate;
            }

            outputs[step] = hidden;
        }

        var result = Engine.TensorStack(outputs, axis: 1);
        if (input.Rank == 2)
            return Engine.Reshape(result, new[] { sequenceLength, _hiddenSize });

        var outputShape = input.Shape.ToArray();
        outputShape[outputShape.Length - 1] = _hiddenSize;
        return Engine.Reshape(result, outputShape);
    }

    private Tensor<T> UpdateMemory(Tensor<T>? memory, Tensor<T> input, int step, int batchSize)
    {
        int scaledTime = step + _initialTime;
        if (_measure == "legs" && scaledTime == 0)
        {
            var first = Engine.Reshape(input, new[] { batchSize, _memorySize, 1 });
            if (_memoryOrder == 1) return first;
            var zeros = new Tensor<T>(new[] { batchSize, _memorySize, _memoryOrder - 1 });
            return Engine.TensorConcatenate(new[] { first, zeros }, axis: 2);
        }

        memory ??= new Tensor<T>(new[] { batchSize, _memorySize, _memoryOrder });
        var flatMemory = Engine.Reshape(memory, new[] { batchSize * _memorySize, _memoryOrder });
        var flatInput = Engine.Reshape(input, new[] { batchSize * _memorySize, 1 });

        if (_measure != "legs")
            return ApplyFixedTransition(flatMemory, flatInput, (_ltiTransition ?? throw new InvalidOperationException("_ltiTransition is null where it is required.")), (_ltiInput ?? throw new InvalidOperationException("_ltiInput is null where it is required.")), batchSize);

        double stepSize = ClampStep(1.0 / scaledTime);
        if (_discretization == "zoh")
        {
            if (!_legsZohCache.TryGetValue(scaledTime, out var transition))
            {
                double interval = ClampStep(Math.Log((scaledTime + 1.0) / scaledTime));
                transition = BuildZohTransition(interval);
                _legsZohCache[scaledTime] = transition;
            }
            return ApplyFixedTransition(flatMemory, flatInput, transition.A, transition.B, batchSize);
        }

        var memoryTimesA = Engine.TensorMatMul(flatMemory, _aTranspose);
        var inputTimesB = Engine.TensorMatMul(flatInput, _bRow);
        Tensor<T> updated;
        if (_discretization == "forward")
        {
            updated = Engine.TensorAdd(flatMemory,
                Engine.TensorMultiplyScalar(memoryTimesA, NumOps.FromDouble(stepSize)));
            updated = Engine.TensorAdd(updated,
                Engine.TensorMultiplyScalar(inputTimesB, NumOps.FromDouble(stepSize)));
        }
        else if (_discretization == "backward")
        {
            var rhs = Engine.TensorAdd(flatMemory,
                Engine.TensorMultiplyScalar(inputTimesB, NumOps.FromDouble(stepSize)));
            updated = SolveLegSLower(rhs, stepSize);
        }
        else
        {
            double halfStep = stepSize * 0.5;
            var rhs = Engine.TensorAdd(flatMemory,
                Engine.TensorMultiplyScalar(memoryTimesA, NumOps.FromDouble(halfStep)));
            rhs = Engine.TensorAdd(rhs,
                Engine.TensorMultiplyScalar(inputTimesB, NumOps.FromDouble(stepSize)));
            updated = SolveLegSLower(rhs, halfStep);
        }

        return Engine.Reshape(updated, new[] { batchSize, _memorySize, _memoryOrder });
    }

    private Tensor<T> ApplyFixedTransition(
        Tensor<T> memory, Tensor<T> input, Tensor<T> transition, Tensor<T> inputTransition, int batchSize)
    {
        var state = Engine.TensorMatMul(memory, transition);
        var injection = Engine.TensorMatMul(input, inputTransition);
        return Engine.Reshape(Engine.TensorAdd(state, injection),
            new[] { batchSize, _memorySize, _memoryOrder });
    }

    /// <summary>
    /// Solves x(I-scale*A)^T=rhs for the lower-triangular LegS operator and records a
    /// unary backward that solves the transposed system. Only the right-hand side is
    /// differentiable because the paper fixes A.
    /// </summary>
    private Tensor<T> SolveLegSLower(Tensor<T> rhs, double scale)
    {
        var result = SolveLegSLowerCore(rhs, scale, transpose: false);
        DifferentiableOps.RecordUnary(
            "HippoFixedLowerSolve",
            result,
            rhs,
            (gradOutput, inputs, _, _, engine, grads) =>
            {
                var gradInput = SolveLegSLowerCore(gradOutput, scale, transpose: true);
                DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
            });
        return result;
    }

    private Tensor<T> SolveLegSLowerCore(Tensor<T> rhs, double scale, bool transpose)
    {
        if (rhs.Rank != 2 || rhs.Shape[1] != _memoryOrder)
            throw new ArgumentException("HiPPO solve expects [batch*memory, memoryOrder].", nameof(rhs));

        // Each row x solves (I - scale*A) x = row (forward substitution on the lower-triangular LegS
        // operator), or (I - scale*A)^T x = row for the transposed system used by the backward pass.
        // The surrounding RecordUnary owns the tape entry, so the solve itself stays off the tape.
        using var noGrad = new NoGradScope<T>();
        var system = Engine.TensorSubtract(
            Engine.TensorEye<T>(_memoryOrder),
            Engine.TensorMultiplyScalar(_a, NumOps.FromDouble(scale)));
        // The linalg kernels read raw storage, so hand them contiguous operands.
        var rhsColumns = Engine.TensorTranspose(rhs).Contiguous();
        var solved = transpose
            ? SolveTriangularWide(Engine.TensorTranspose(system).Contiguous(), rhsColumns, upper: true)
            : SolveTriangularWide(system.Contiguous(), rhsColumns, upper: false);
        return Engine.TensorTranspose(solved).Contiguous();
    }

    private double ClampStep(double value) => Math.Max(_timescaleMin, Math.Min(_timescaleMax, value));

    private (Tensor<T> A, Tensor<T> B) BuildDiscreteTransition(double step)
    {
        if (_discretization == "zoh") return BuildZohTransition(step);

        // Fixed operator constants built from fresh tensors, so they stay off the tape.
        using var noGrad = new NoGradScope<T>();
        int n = _memoryOrder;
        var identity = Engine.TensorEye<T>(n);
        var bColumn = Engine.TensorTranspose(_bRow);                                         // [n, 1]
        Tensor<T> transition;
        Tensor<T> inputTransition;

        if (_discretization == "forward")
        {
            transition = Engine.TensorAdd(identity, Engine.TensorMultiplyScalar(_a, NumOps.FromDouble(step)));
            inputTransition = Engine.TensorMultiplyScalar(bColumn, NumOps.FromDouble(step));
        }
        else
        {
            // backward: (I - step*A)^-1 [I | step*B]; bilinear: (I - step/2*A)^-1 [I + step/2*A | step*B].
            double factor = _discretization == "bilinear" ? step * 0.5 : step;
            var scaledA = Engine.TensorMultiplyScalar(_a, NumOps.FromDouble(factor));
            var system = Engine.TensorSubtract(identity, scaledA);
            var stateRhs = _discretization == "bilinear" ? Engine.TensorAdd(identity, scaledA) : identity;
            var rhs = Engine.TensorConcatenate(
                new[] { stateRhs, Engine.TensorMultiplyScalar(bColumn, NumOps.FromDouble(step)) }, axis: 1);
            var (solution, info) = SolveSystemWide(system.Contiguous(), rhs.Contiguous());
            if (Engine.TensorCountNonzero(info) > 0)
                throw new InvalidOperationException("HiPPO discretization matrix is singular.");
            transition = Engine.TensorNarrow(solution, 1, 0, n);
            inputTransition = Engine.TensorNarrow(solution, 1, n, 1);
        }

        return (Engine.TensorTranspose(transition).Contiguous(), Engine.TensorTranspose(inputTransition).Contiguous());
    }

    private static (Tensor<T> Solution, Tensor<int> Info) SolveSystemWide(Tensor<T> system, Tensor<T> rhs)
    {
        if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
            return LinalgDispatch<T>.SolveEx(system, rhs);
        // The installed linalg kernels support float/double; retain low-precision layer support.
        using var noGrad = new NoGradScope<double>();
        var (solution, info) = LinalgDispatch<double>.SolveEx(system.Cast<double>(), rhs.Cast<double>());
        return (solution.Cast<T>(), info);
    }

    private static Tensor<T> SolveTriangularWide(Tensor<T> system, Tensor<T> rhs, bool upper)
    {
        if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
            return LinalgDispatch<T>.SolveTriangular(system, rhs, upper);
        using var noGrad = new NoGradScope<double>();
        return LinalgDispatch<double>.SolveTriangular(system.Cast<double>(), rhs.Cast<double>(), upper).Cast<T>();
    }

    private (Tensor<T> A, Tensor<T> B) BuildZohTransition(double interval)
    {
        // exp([[A, B], [0, 0]] * interval) = [[Ad, Bd], [0, 1]] (Van Loan), built off the tape.
        using var noGrad = new NoGradScope<T>();
        int n = _memoryOrder;
        T scale = NumOps.FromDouble(interval);
        var top = Engine.TensorConcatenate(new[]
        {
            Engine.TensorMultiplyScalar(_a, scale),
            Engine.TensorMultiplyScalar(Engine.TensorTranspose(_bRow), scale),
        }, axis: 1);                                                                         // [n, n + 1]
        var augmented = Engine.TensorConcatenate(new[] { top, new Tensor<T>(new[] { 1, n + 1 }) }, axis: 0);

        var exponential = MatrixExponential(augmented);
        var transition = Engine.TensorSlice(exponential, new[] { 0, 0 }, new[] { n, n });
        var inputTransition = Engine.TensorSlice(exponential, new[] { 0, n }, new[] { n, 1 });
        return (Engine.TensorTranspose(transition).Contiguous(), Engine.TensorTranspose(inputTransition).Contiguous());
    }

    // Pade approximant degrees and their backward-error bounds theta_m (Higham 2005, "The Scaling and
    // Squaring Method for the Matrix Exponential Revisited", Table 2.3), the scheme scipy.linalg.expm
    // follows. Coefficients b_k of the [m/m] Pade numerator are divided by b_0 when applied so every
    // value stays inside the range of low-precision T (b_0 of degree 13 is 6.5e16); P and Q share the
    // factor, so R = Q^-1 P is unchanged.
    private static readonly int[] PadeDegrees = { 3, 5, 7, 9, 13 };
    private static readonly double[] PadeThetas =
    {
        1.495585217958292e-2, 2.539398330063230e-1, 9.504178996162932e-1, 2.097847961257068e0, 5.371920351148152e0,
    };
    private static readonly double[][] PadeCoefficients =
    {
        new double[] { 120, 60, 12, 1 },
        new double[] { 30240, 15120, 3360, 420, 30, 1 },
        new double[] { 17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1 },
        new double[] { 17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1 },
        new double[]
        {
            64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000,
            10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1,
        },
    };

    /// <summary>
    /// Matrix exponential by scaling and squaring with a diagonal Pade approximant (Higham 2005, the
    /// algorithm of scipy.linalg.expm), in T through engine ops. The degree is the smallest m in
    /// {3, 5, 7, 9, 13} whose theta_m bounds the 1-norm; otherwise the matrix is scaled by 2^-s down to
    /// theta_13. Pade 13 needs about four fewer squarings than a Taylor series truncated at norm 0.5,
    /// and each squaring roughly doubles the rounding error, which is what made float drift.
    /// </summary>
    private Tensor<T> MatrixExponential(Tensor<T> matrix)
    {
        int n = matrix.Shape[0];
        var identity = Engine.TensorEye<T>(n);
        if (n == 0) return identity;

        // 1-norm (max column abs sum) as maxAbs * ||matrix / maxAbs||_1, so the column sums of a
        // wide low-precision matrix cannot overflow.
        double maxAbs = NumOps.ToDouble(Engine.TensorMaxValue(Engine.TensorAbs(matrix)));
        if (maxAbs == 0.0) return identity;
        var unit = Engine.TensorDivideScalar(matrix, NumOps.FromDouble(maxAbs));
        double norm = maxAbs * NumOps.ToDouble(
            Engine.TensorMaxValue(Engine.ReduceSum(Engine.TensorAbs(unit), new[] { 0 })));

        int degreeIndex = 0;
        while (degreeIndex < PadeDegrees.Length - 1 && norm > PadeThetas[degreeIndex]) degreeIndex++;
        int squarings = norm > PadeThetas[degreeIndex]
            ? Math.Max(0, (int)Math.Ceiling(Math.Log(norm / PadeThetas[degreeIndex], 2.0)))
            : 0;

        // Scale through the unit matrix: unit * (maxAbs / 2^s) keeps the factor representable in T.
        var scaled = Engine.TensorMultiplyScalar(unit, NumOps.FromDouble(maxAbs / Math.Pow(2.0, squarings)));
        var (u, v) = PadeTerms(scaled, identity, PadeDegrees[degreeIndex], PadeCoefficients[degreeIndex]);

        // (V - U) R = (V + U).
        var (result, info) = SolveSystemWide(
            Engine.TensorSubtract(v, u).Contiguous(), Engine.TensorAdd(v, u).Contiguous());
        if (Engine.TensorCountNonzero(info) > 0)
            throw new InvalidOperationException("HiPPO ZOH Pade denominator is singular.");

        for (int i = 0; i < squarings; i++) result = Engine.TensorMatMul(result, result);
        return result;
    }

    /// <summary>
    /// The odd (U) and even (V) parts of the [m/m] Pade numerator p(A) = V + U with coefficients
    /// normalized by b_0, evaluated as in Higham 2005, Algorithm 2.3.
    /// </summary>
    private (Tensor<T> U, Tensor<T> V) PadeTerms(Tensor<T> a, Tensor<T> identity, int degree, double[] b)
    {
        double b0 = b[0];
        Tensor<T> Scale(Tensor<T> m, int k) => Engine.TensorMultiplyScalar(m, NumOps.FromDouble(b[k] / b0));

        var a2 = Engine.TensorMatMul(a, a);
        if (degree == 13)
        {
            var a4 = Engine.TensorMatMul(a2, a2);
            var a6 = Engine.TensorMatMul(a4, a2);
            var oddInner = Engine.TensorMatMul(a6,
                Engine.TensorAdd(Engine.TensorAdd(Scale(a6, 13), Scale(a4, 11)), Scale(a2, 9)));
            oddInner = Engine.TensorAdd(
                Engine.TensorAdd(Engine.TensorAdd(oddInner, Scale(a6, 7)), Engine.TensorAdd(Scale(a4, 5), Scale(a2, 3))),
                Scale(identity, 1));
            var even = Engine.TensorMatMul(a6,
                Engine.TensorAdd(Engine.TensorAdd(Scale(a6, 12), Scale(a4, 10)), Scale(a2, 8)));
            even = Engine.TensorAdd(
                Engine.TensorAdd(Engine.TensorAdd(even, Scale(a6, 6)), Engine.TensorAdd(Scale(a4, 4), Scale(a2, 2))),
                identity);
            return (Engine.TensorMatMul(a, oddInner), even);
        }

        // Degrees 3..9: U = A * sum_j b_(2j+1) A^(2j), V = sum_j b_(2j) A^(2j), with b_0 / b_0 = 1.
        var oddSum = Scale(identity, 1);
        var evenSum = identity;
        var power = identity;
        for (int j = 1; 2 * j <= degree; j++)
        {
            power = Engine.TensorMatMul(power, a2);
            oddSum = Engine.TensorAdd(oddSum, Scale(power, (2 * j) + 1));
            evenSum = Engine.TensorAdd(evenSum, Scale(power, 2 * j));
        }

        return (Engine.TensorMatMul(a, oddSum), evenSum);
    }

    private IEnumerable<Tensor<T>> ParameterTensors()
    {
        yield return _memoryWeights;
        yield return _memoryBias;
        yield return _hiddenWeights;
        yield return _hiddenBias;
        if (_useGate)
        {
            yield return _gateWeights;
            yield return _gateBias;
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != ParameterCountHelper.ToFlatVectorSize(ParameterCount))
            throw new InvalidOperationException("HiPPO gradients are unavailable; run a backward pass first.");
        var parameters = GetParameters();
        SetParameters(Engine.Subtract(parameters, Engine.Multiply(gradients, learningRate)));
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        // State is local to each Forward call; only immutable ZOH transitions are cached.
    }

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        metadata["HiddenSize"] = _hiddenSize.ToString(culture);
        metadata["InputSize"] = _inputSize.ToString(culture);
        metadata["MemoryOrder"] = _memoryOrder.ToString(culture);
        metadata["MemorySize"] = _memorySize.ToString(culture);
        metadata["Measure"] = _measure;
        metadata["Discretization"] = _discretization;
        metadata["InitialTime"] = _initialTime.ToString(culture);
        metadata["TimeStep"] = _timeStep.ToString("R", culture);
        metadata["TimescaleMin"] = _timescaleMin.ToString("R", culture);
        metadata["TimescaleMax"] = _timescaleMax.ToString("R", culture);
        metadata["UseGate"] = _useGate.ToString(culture);
        return metadata;
    }
}
