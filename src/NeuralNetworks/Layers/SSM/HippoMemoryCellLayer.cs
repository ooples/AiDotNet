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
public class HippoMemoryCellLayer<T> : LayerBase<T>
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

    // Fixed HiPPO operator. These are buffers, not trainable parameters.
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
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _gateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _gateBias;

    private readonly Tensor<T>? _ltiTransition;
    private readonly Tensor<T>? _ltiInput;
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

    /// <inheritdoc />
    public override long ParameterCount =>
        _memoryWeights.Length + _memoryBias.Length +
        _hiddenWeights.Length + _hiddenBias.Length +
        (_useGate ? _gateWeights.Length + _gateBias.Length : 0);

    /// <inheritdoc />
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters() => _useGate
        ? new[] { _memoryWeights, _memoryBias, _hiddenWeights, _hiddenBias, _gateWeights, _gateBias }
        : new[] { _memoryWeights, _memoryBias, _hiddenWeights, _hiddenBias };

    /// <inheritdoc />
    /// <remarks>
    /// Parameter-buffer and copy-on-write paths replace tensor objects rather
    /// than copying values. Keep the fields consumed by <see cref="Forward"/>
    /// synchronized with those replacements.
    /// </remarks>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        int expected = _useGate ? 6 : 4;
        if (parameters.Count != expected)
            throw new ArgumentException($"Expected exactly {expected} HiPPO parameter tensors.", nameof(parameters));

        ValidateShapeMatch(parameters[0], _memoryWeights, nameof(_memoryWeights));
        ValidateShapeMatch(parameters[1], _memoryBias, nameof(_memoryBias));
        ValidateShapeMatch(parameters[2], _hiddenWeights, nameof(_hiddenWeights));
        ValidateShapeMatch(parameters[3], _hiddenBias, nameof(_hiddenBias));
        if (_useGate)
        {
            ValidateShapeMatch(parameters[4], _gateWeights, nameof(_gateWeights));
            ValidateShapeMatch(parameters[5], _gateBias, nameof(_gateBias));
        }

        _memoryWeights = parameters[0];
        _memoryBias = parameters[1];
        _hiddenWeights = parameters[2];
        _hiddenBias = parameters[3];
        if (_useGate)
        {
            _gateWeights = parameters[4];
            _gateBias = parameters[5];
        }
    }

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
        int hiddenSize = 256,
        int inputSize = 1,
        int memoryOrder = -1,
        int memorySize = 1,
        string measure = "legs",
        string discretization = "bilinear",
        int initialTime = 0,
        double timeStep = 0.0,
        double timescaleMin = 0.0,
        double timescaleMax = double.PositiveInfinity,
        bool useGate = true,
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
        var a = new Tensor<T>(new[] { order, order });
        var b = new Tensor<T>(new[] { 1, order });

        for (int i = 0; i < order; i++)
        {
            double ri = Math.Sqrt(2.0 * i + 1.0);
            switch (measure)
            {
                case "legs":
                    b[0, i] = NumOps.FromDouble(ri);
                    for (int j = 0; j <= i; j++)
                    {
                        double value = i == j
                            ? -(i + 1.0)
                            : -ri * Math.Sqrt(2.0 * j + 1.0);
                        a[i, j] = NumOps.FromDouble(value);
                    }
                    break;

                case "legt":
                    b[0, i] = NumOps.FromDouble(0.5 * ri);
                    for (int j = 0; j < order; j++)
                    {
                        double sign = i < j && ((i - j) & 1) != 0 ? -1.0 : 1.0;
                        a[i, j] = NumOps.FromDouble(-0.5 * ri * Math.Sqrt(2.0 * j + 1.0) * sign);
                    }
                    break;

                default: // lagt
                    b[0, i] = NumOps.One;
                    for (int j = 0; j <= i; j++)
                        a[i, j] = NumOps.FromDouble(i == j ? -0.5 : -1.0);
                    break;
            }
        }

        return (a, b);
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
    public override Tensor<T> Forward(Tensor<T> input)
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

        for (int step = 0; step < sequenceLength; step++)
        {
            var current = Engine.TensorSliceAxis(sequence, 1, step);
            var xh = Engine.TensorConcatenate(new[] { current, hidden }, axis: 1);
            var memoryInput = Engine.TensorMatMul(xh, _memoryWeights);
            memoryInput = Engine.TensorBroadcastAdd(memoryInput, _memoryBias);
            memory = UpdateMemory(memory, memoryInput, step, batchSize);

            var flattenedMemory = Engine.Reshape(memory,
                new[] { batchSize, _memorySize * _memoryOrder });
            var xm = Engine.TensorConcatenate(new[] { current, flattenedMemory }, axis: 1);
            var candidate = Engine.TensorMatMul(xm, _hiddenWeights);
            candidate = Engine.TensorBroadcastAdd(candidate, _hiddenBias);
            candidate = Engine.Tanh(candidate);

            if (_useGate)
            {
                var gate = Engine.TensorMatMul(xm, _gateWeights);
                gate = Engine.TensorBroadcastAdd(gate, _gateBias);
                gate = Engine.Sigmoid(gate);
                var keep = Engine.TensorAddScalar(
                    Engine.TensorMultiplyScalar(gate, NumOps.FromDouble(-1.0)), NumOps.One);
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
            return ApplyFixedTransition(flatMemory, flatInput, _ltiTransition!, _ltiInput!, batchSize);

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

        int rows = rhs.Shape[0];
        int n = _memoryOrder;
        T[] rhsData = rhs.IsContiguous && rhs._storageOffset == 0
            ? rhs.DataVector.GetDataArray()
            : rhs.GetFlattenedData();
        T[] aData = _a.DataVector.GetDataArray();
        var output = new Tensor<T>(new[] { rows, n });
        T[] outputData = output.DataVector.GetDataArray();

        for (int row = 0; row < rows; row++)
        {
            if (!transpose)
            {
                for (int i = 0; i < n; i++)
                {
                    double value = NumOps.ToDouble(rhsData[row * n + i]);
                    for (int j = 0; j < i; j++)
                        value += scale * NumOps.ToDouble(aData[i * n + j]) * NumOps.ToDouble(outputData[row * n + j]);
                    double diagonal = 1.0 - scale * NumOps.ToDouble(aData[i * n + i]);
                    outputData[row * n + i] = NumOps.FromDouble(value / diagonal);
                }
            }
            else
            {
                for (int i = n - 1; i >= 0; i--)
                {
                    double value = NumOps.ToDouble(rhsData[row * n + i]);
                    for (int j = i + 1; j < n; j++)
                        value += scale * NumOps.ToDouble(aData[j * n + i]) * NumOps.ToDouble(outputData[row * n + j]);
                    double diagonal = 1.0 - scale * NumOps.ToDouble(aData[i * n + i]);
                    outputData[row * n + i] = NumOps.FromDouble(value / diagonal);
                }
            }
        }

        return output;
    }

    private double ClampStep(double value) => Math.Max(_timescaleMin, Math.Min(_timescaleMax, value));

    private (Tensor<T> A, Tensor<T> B) BuildDiscreteTransition(double step)
    {
        if (_discretization == "zoh") return BuildZohTransition(step);

        int n = _memoryOrder;
        var a = ToDoubleMatrix(_a);
        var b = ToDoubleRow(_bRow);
        var identity = Identity(n);
        double[,] transition;
        double[] inputTransition;

        if (_discretization == "forward")
        {
            transition = Add(identity, Scale(a, step));
            inputTransition = Scale(b, step);
        }
        else
        {
            double factor = _discretization == "bilinear" ? step * 0.5 : step;
            var inverse = Invert(Subtract(identity, Scale(a, factor)));
            transition = _discretization == "bilinear"
                ? Multiply(inverse, Add(identity, Scale(a, factor)))
                : inverse;
            inputTransition = Multiply(inverse, Scale(b, step));
        }

        return (ToTensor(Transpose(transition)), ToRowTensor(inputTransition));
    }

    private (Tensor<T> A, Tensor<T> B) BuildZohTransition(double interval)
    {
        int n = _memoryOrder;
        var augmented = new double[n + 1, n + 1];
        var a = ToDoubleMatrix(_a);
        var b = ToDoubleRow(_bRow);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) augmented[i, j] = a[i, j] * interval;
            augmented[i, n] = b[i] * interval;
        }

        var exponential = MatrixExponential(augmented);
        var transition = new double[n, n];
        var inputTransition = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) transition[i, j] = exponential[i, j];
            inputTransition[i] = exponential[i, n];
        }
        return (ToTensor(Transpose(transition)), ToRowTensor(inputTransition));
    }

    private static double[,] MatrixExponential(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        double norm = 0;
        for (int i = 0; i < n; i++)
        {
            double row = 0;
            for (int j = 0; j < n; j++) row += Math.Abs(matrix[i, j]);
            norm = Math.Max(norm, row);
        }
        int squarings = norm <= 0.5 ? 0 : Math.Max(0, (int)Math.Ceiling(Math.Log(norm / 0.5, 2.0)));
        var scaled = Scale(matrix, 1.0 / Math.Pow(2.0, squarings));
        var result = Identity(n);
        var term = Identity(n);
        for (int k = 1; k <= 24; k++)
        {
            term = Scale(Multiply(term, scaled), 1.0 / k);
            result = Add(result, term);
        }
        for (int i = 0; i < squarings; i++) result = Multiply(result, result);
        return result;
    }

    private double[,] ToDoubleMatrix(Tensor<T> tensor)
    {
        int rows = tensor.Shape[0], columns = tensor.Shape[1];
        var result = new double[rows, columns];
        T[] data = tensor.DataVector.GetDataArray();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                result[i, j] = NumOps.ToDouble(data[i * columns + j]);
        return result;
    }

    private double[] ToDoubleRow(Tensor<T> tensor)
    {
        var result = new double[tensor.Shape[1]];
        T[] data = tensor.DataVector.GetDataArray();
        for (int i = 0; i < result.Length; i++) result[i] = NumOps.ToDouble(data[i]);
        return result;
    }

    private Tensor<T> ToTensor(double[,] values)
    {
        int rows = values.GetLength(0), columns = values.GetLength(1);
        var result = new Tensor<T>(new[] { rows, columns });
        T[] data = result.DataVector.GetDataArray();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                data[i * columns + j] = NumOps.FromDouble(values[i, j]);
        return result;
    }

    private Tensor<T> ToRowTensor(double[] values)
    {
        var result = new Tensor<T>(new[] { 1, values.Length });
        T[] data = result.DataVector.GetDataArray();
        for (int i = 0; i < values.Length; i++) data[i] = NumOps.FromDouble(values[i]);
        return result;
    }

    private static double[,] Identity(int n)
    {
        var result = new double[n, n];
        for (int i = 0; i < n; i++) result[i, i] = 1.0;
        return result;
    }

    private static double[,] Add(double[,] a, double[,] b)
    {
        int rows = a.GetLength(0), columns = a.GetLength(1);
        var result = new double[rows, columns];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) result[i, j] = a[i, j] + b[i, j];
        return result;
    }

    private static double[,] Subtract(double[,] a, double[,] b)
    {
        int rows = a.GetLength(0), columns = a.GetLength(1);
        var result = new double[rows, columns];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) result[i, j] = a[i, j] - b[i, j];
        return result;
    }

    private static double[,] Scale(double[,] matrix, double scale)
    {
        int rows = matrix.GetLength(0), columns = matrix.GetLength(1);
        var result = new double[rows, columns];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) result[i, j] = matrix[i, j] * scale;
        return result;
    }

    private static double[] Scale(double[] vector, double scale)
    {
        var result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++) result[i] = vector[i] * scale;
        return result;
    }

    private static double[,] Multiply(double[,] a, double[,] b)
    {
        int rows = a.GetLength(0), inner = a.GetLength(1), columns = b.GetLength(1);
        var result = new double[rows, columns];
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < inner; k++)
            {
                double aik = a[i, k];
                if (aik == 0) continue;
                for (int j = 0; j < columns; j++) result[i, j] += aik * b[k, j];
            }
        return result;
    }

    private static double[] Multiply(double[,] matrix, double[] vector)
    {
        int rows = matrix.GetLength(0), columns = matrix.GetLength(1);
        var result = new double[rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) result[i] += matrix[i, j] * vector[j];
        return result;
    }

    private static double[,] Transpose(double[,] matrix)
    {
        int rows = matrix.GetLength(0), columns = matrix.GetLength(1);
        var result = new double[columns, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) result[j, i] = matrix[i, j];
        return result;
    }

    private static double[,] Invert(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var augmented = new double[n, 2 * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) augmented[i, j] = matrix[i, j];
            augmented[i, n + i] = 1.0;
        }

        for (int column = 0; column < n; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < n; row++)
                if (Math.Abs(augmented[row, column]) > Math.Abs(augmented[pivot, column])) pivot = row;
            if (Math.Abs(augmented[pivot, column]) < 1e-14)
                throw new InvalidOperationException("HiPPO discretization matrix is singular.");
            if (pivot != column)
                for (int j = 0; j < 2 * n; j++)
                    (augmented[column, j], augmented[pivot, j]) = (augmented[pivot, j], augmented[column, j]);

            double diagonal = augmented[column, column];
            for (int j = 0; j < 2 * n; j++) augmented[column, j] /= diagonal;
            for (int row = 0; row < n; row++)
            {
                if (row == column) continue;
                double factor = augmented[row, column];
                if (factor == 0) continue;
                for (int j = 0; j < 2 * n; j++) augmented[row, j] -= factor * augmented[column, j];
            }
        }

        var result = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) result[i, j] = augmented[i, n + j];
        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var result = new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        int offset = 0;
        foreach (var tensor in ParameterTensors())
        {
            T[] data = tensor.DataVector.GetDataArray();
            for (int i = 0; i < tensor.Length; i++) result[offset++] = data[i];
        }
        return result;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int expected = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} HiPPO cell parameters, got {parameters.Length}.", nameof(parameters));
        int offset = 0;
        foreach (var tensor in ParameterTensors())
        {
            T[] data = tensor.DataVector.GetDataArray();
            for (int i = 0; i < tensor.Length; i++) data[i] = parameters[offset++];
        }
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
        for (int i = 0; i < parameters.Length; i++)
            parameters[i] = NumOps.Subtract(parameters[i], NumOps.Multiply(learningRate, gradients[i]));
        SetParameters(parameters);
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
