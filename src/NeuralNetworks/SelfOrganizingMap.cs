using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Self-Organizing Map (Kohonen map): an unsupervised neural network that maps
/// high-dimensional input onto a 2-D grid of prototype neurons, preserving input topology.
/// </summary>
/// <remarks>
/// <para>
/// The map stores one prototype (weight) vector per neuron. Prediction finds the Best Matching Unit
/// (BMU) — the neuron whose prototype is closest to the input — and returns a one-hot activation.
/// Training moves the BMU and its grid neighbours toward the input (competitive learning, Kohonen 1982).
/// </para>
/// <para><b>For Beginners:</b> a SOM arranges data on a map so similar inputs land on nearby cells.
/// Each cell remembers a prototype pattern; over training, cells specialise and neighbouring cells
/// stay similar, which is great for visualising, clustering, and reducing high-dimensional data.
/// </para>
/// <para>
/// Implementation note: the codebook is a single <see cref="Tensor{T}"/> of shape
/// [numNeurons, inputDimension] and all hot-path math (BMU distances, neighbourhood updates) runs
/// through <see cref="NeuralNetworkBase{T}.Engine"/> tensor operations rather than scalar loops.
/// </para>
/// <para><b>Reference:</b> Kohonen, T. (1982). Self-organized formation of topologically correct
/// feature maps. Biological Cybernetics, 43, 59–69.</para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Clustering, inputSize: 10, outputSize: 100);
/// var model = new SelfOrganizingMap&lt;float&gt;(arch);
/// var output = model.Predict(Tensor&lt;float&gt;.CreateDefault(new[] { 10 }, 0f));
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Clustering)]
[ModelTask(ModelTask.Clustering)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Self-Organized Formation of Topologically Correct Feature Maps", "https://doi.org/10.1007/BF00337288", Year = 1982, Authors = "Teuvo Kohonen")]
public class SelfOrganizingMap<T> : NeuralNetworkBase<T>
{
    private readonly SelfOrganizingMapNNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The neuron codebook: shape [numNeurons, inputDimension]. Row i is the prototype vector of
    /// neuron i (row-major over the [mapHeight, mapWidth] grid, i = y * mapWidth + x).
    /// </summary>
    private Tensor<T> _weights;

    // Cached fixed-shape ones tensors reused by ComputeSquaredDistances / UpdateWeights. Their shapes
    // depend only on the neuron count (_mapWidth*_mapHeight) and _inputDimension, which are fixed after
    // construction, so the hot train/predict path reuses them instead of rebuilding a fresh tensor with a
    // scalar fill loop on every call (#1789 review). Rebuilt lazily if the shape ever changes (e.g. after
    // deserialization resets the dimensions).
    private Tensor<T>? _onesColumn;   // [numNeurons, 1]
    private Tensor<T>? _onesRow;      // [1, inputDimension]

    private int _mapWidth;
    private int _mapHeight;
    private int _inputDimension;
    private int _totalEpochs;
    private int _currentEpoch;
    private readonly int _initSeed;

    /// <summary>Initializes a new instance with default architecture settings.</summary>
    public SelfOrganizingMap()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Clustering,
            inputSize: 128,
            outputSize: 64))
    {
    }

    /// <summary>
    /// Creates a Self-Organizing Map for the given architecture. The input dimension comes from the
    /// architecture's input size and the neuron count from its output size (arranged near a golden-ratio
    /// aspect grid).
    /// </summary>
    public SelfOrganizingMap(NeuralNetworkArchitecture<T> architecture, int totalEpochs = 1000, ILossFunction<T>? lossFunction = null,
        SelfOrganizingMapNNOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new SelfOrganizingMapNNOptions();
        Options = _options;

        // totalEpochs is a divisor in both the learning-rate decay (currentEpoch / totalEpochs) and the
        // neighborhood-radius schedule (totalEpochs / log(mapArea)); a non-positive value makes those
        // schedules divide by zero/negative and emit NaN/Infinity weights on the first update.
        if (totalEpochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalEpochs),
                "Total training epochs must be greater than zero for SOM.");

        _inputDimension = architecture.InputSize;

        if (_inputDimension <= 0)
            throw new ArgumentException("Input dimension must be greater than zero for SOM.");

        int mapSize = architecture.OutputSize;
        if (mapSize <= 0)
            throw new ArgumentException("Map size (output size) must be greater than zero for SOM.");

        // Choose the factor pair (width x height) of totalNeurons whose aspect ratio is closest to the
        // golden-ratio target, so the grid holds EXACTLY totalNeurons neurons (== architecture.OutputSize)
        // with the most balanced 2-D layout available. The previous grow-then-shrink heuristic could
        // overshoot the target in a single step with no corrective re-check — e.g. the default
        // outputSize=64 settled on a 10x6=60 grid, silently allocating fewer neurons than requested and
        // shrinking _weights, ParameterCount, and PredictCore's one-hot output below OutputSize (#1789
        // review). Every positive integer has at least the 1xN factorization, so an exact pair exists.
        int totalNeurons = mapSize;
        double aspectRatio = 1.6;
        _mapWidth = totalNeurons;
        _mapHeight = 1;
        double bestRatioError = double.MaxValue;
        for (int h = 1; h <= (int)Math.Sqrt(totalNeurons); h++)
        {
            if (totalNeurons % h != 0) continue;
            int w = totalNeurons / h;
            double ratioError = Math.Abs((w / (double)h) - aspectRatio);
            if (ratioError < bestRatioError)
            {
                bestRatioError = ratioError;
                _mapWidth = w;
                _mapHeight = h;
            }
        }

        // Deterministic per-configuration init seed so a fresh (untrained) map is REPRODUCIBLE — the
        // BMU is otherwise a function of process-shared RNG, which under parallel test execution made
        // the codebook (and thus ScaledInput_ShouldChangeOutput) flake. Honour an explicit options seed
        // when provided; otherwise derive a stable seed from the shape (FNV-style mix).
        _initSeed = architecture.RandomSeed
            ?? unchecked((_mapWidth * 73856093) ^ (_mapHeight * 19349663) ^ (_inputDimension * 83492791));

        _weights = new Tensor<T>(new[] { _mapWidth * _mapHeight, _inputDimension });
        _totalEpochs = totalEpochs;
        _currentEpoch = 0;

        InitializeWeights();
        InitializeLayers();
    }

    /// <inheritdoc/>
    /// <remarks>SOM is a single competitive-learning codebook, not a layer stack, so this is empty.</remarks>
    protected override void InitializeLayers()
    {
        // SOM stores its codebook directly in _weights rather than a layer chain.
    }

    /// <summary>
    /// Initializes the codebook with distinct, well-spread prototypes. Each neuron gets a random base
    /// in [0,1] plus a per-neuron directional bias so prototypes are separated in the input space
    /// (uniform [0,1] alone concentrates near the centre in high dimensions, making the BMU degenerate).
    /// Deterministic given <see cref="_initSeed"/>.
    /// </summary>
    private void InitializeWeights()
    {
        int n = _mapWidth * _mapHeight;
        var rand = RandomHelper.CreateSeededRandom(_initSeed);
        for (int i = 0; i < n; i++)
        {
            // Per-neuron spread: shift this neuron's prototype along a distinct coordinate band so the
            // map covers the space and different inputs resolve to different BMUs even before training.
            double neuronShift = (double)i / Math.Max(1, n - 1);
            for (int j = 0; j < _inputDimension; j++)
            {
                double baseVal = rand.NextDouble();
                double biased = 0.5 * baseVal + 0.5 * ((neuronShift + baseVal) % 1.0);
                _weights[i, j] = NumOps.FromDouble(biased);
            }
        }
    }

    /// <summary>
    /// Finds the Best Matching Unit — the neuron whose prototype has minimum squared Euclidean distance
    /// to <paramref name="input"/> — using vectorized engine ops: d_n = sum_j (W[n,j] - x_j)^2.
    /// </summary>
    private int FindBestMatchingUnit(Tensor<T> input)
    {
        var distances = ComputeSquaredDistances(input); // [numNeurons]
        int bmu = 0;
        T minDistance = distances[0];
        for (int i = 1; i < distances.Length; i++)
        {
            if (NumOps.LessThan(distances[i], minDistance))
            {
                minDistance = distances[i];
                bmu = i;
            }
        }
        return bmu;
    }

    /// <summary>
    /// Squared Euclidean distance from <paramref name="input"/> (shape [D]) to every neuron prototype,
    /// returned as a [numNeurons] tensor. Broadcasts the input across neurons via a ones-column matmul
    /// (the codebase's standard row-broadcast), then reduces the squared difference over the feature axis.
    /// </summary>
    /// <summary>A cached [numNeurons, 1] all-ones column (built once, reused across calls).</summary>
    private Tensor<T> OnesColumn()
    {
        int n = _mapWidth * _mapHeight;
        if (_onesColumn is null || _onesColumn.Shape[0] != n)
        {
            var t = new Tensor<T>(new[] { n, 1 });
            for (int i = 0; i < n; i++) t[i, 0] = NumOps.One;
            _onesColumn = t;
        }
        return _onesColumn;
    }

    /// <summary>A cached [1, inputDimension] all-ones row (built once, reused across calls).</summary>
    private Tensor<T> OnesRow()
    {
        int d = _inputDimension;
        if (_onesRow is null || _onesRow.Shape[1] != d)
        {
            var t = new Tensor<T>(new[] { 1, d });
            for (int j = 0; j < d; j++) t[0, j] = NumOps.One;
            _onesRow = t;
        }
        return _onesRow;
    }

    private Tensor<T> ComputeSquaredDistances(Tensor<T> input)
    {
        int d = _inputDimension;

        var inputRow = input.Reshape(new[] { 1, d });                 // [1, D]
        var onesColumn = OnesColumn();                                 // [N, 1] (cached)

        var tiledInput = Engine.TensorMatMul(onesColumn, inputRow);    // [N, D]
        var diff = Engine.TensorSubtract(_weights, tiledInput);        // [N, D]
        var squared = Engine.TensorMultiply(diff, diff);               // [N, D]
        return Engine.ReduceSum(squared, new[] { 1 }, keepDims: false); // [N]
    }

    /// <summary>
    /// Moves the BMU and its grid neighbours toward the input:
    /// W[n] += lr * h(n) * (x - W[n]), where h(n) is the Gaussian neighbourhood influence of neuron n
    /// relative to the BMU. Fully vectorized over neurons via the engine.
    /// </summary>
    private void UpdateWeights(Tensor<T> input, int bmu, T learningRate, T radius)
    {
        int n = _mapWidth * _mapHeight;
        int d = _inputDimension;
        int bmuX = bmu % _mapWidth;
        int bmuY = bmu / _mapWidth;
        double radiusD = Convert.ToDouble(radius);
        double lrD = Convert.ToDouble(learningRate);

        // Per-neuron scalar factor lr * influence (0 outside the neighbourhood radius).
        var factorColumn = new Tensor<T>(new[] { n, 1 });
        for (int i = 0; i < n; i++)
        {
            int x = i % _mapWidth;
            int y = i / _mapWidth;
            double gridDist = Math.Sqrt((x - bmuX) * (x - bmuX) + (y - bmuY) * (y - bmuY));
            double factor = 0.0;
            if (gridDist < radiusD && radiusD > 0.0)
            {
                double influence = Math.Exp(-(gridDist * gridDist) / (2.0 * radiusD * radiusD));
                factor = lrD * influence;
            }
            factorColumn[i, 0] = NumOps.FromDouble(factor);
        }

        // delta[n,:] = factor[n] * (x - W[n,:]), then W += delta — all engine ops.
        var onesRow = OnesRow();                                      // [1, D] (cached)

        var inputRow = input.Reshape(new[] { 1, d });
        var onesColumn = OnesColumn();                                // [N, 1] (cached)

        var tiledInput = Engine.TensorMatMul(onesColumn, inputRow);   // [N, D]
        var inputMinusW = Engine.TensorSubtract(tiledInput, _weights); // [N, D]
        var tiledFactor = Engine.TensorMatMul(factorColumn, onesRow);  // [N, D]
        var delta = Engine.TensorMultiply(tiledFactor, inputMinusW);   // [N, D]
        _weights = Engine.TensorAdd(_weights, delta);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        var flatInput = input.Rank == 1 ? input : input.Reshape(new[] { input.Length });
        if (flatInput.Length != _inputDimension)
            throw new ArgumentException($"Input must have {_inputDimension} elements, but got {flatInput.Length}");

        int bmu = FindBestMatchingUnit(flatInput);

        // One-hot activation over the map neurons — the BMU cell is active.
        var output = new Tensor<T>(new[] { _mapWidth * _mapHeight });
        output[bmu] = NumOps.One;
        return output;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var flatInput = input.Rank == 1 ? input : input.Reshape(new[] { input.Length });
        if (flatInput.Length != _inputDimension)
            throw new ArgumentException($"Input must have {_inputDimension} elements, but got {flatInput.Length}");

        // SOM training is unsupervised — expectedOutput is unused.
        int bmu = FindBestMatchingUnit(flatInput);

        T learningRate = CalculateLearningRate(NumOps.FromDouble(0.1), _currentEpoch, _totalEpochs);
        T radius = CalculateRadius(
            NumOps.FromDouble(Math.Max(_mapWidth, _mapHeight) / 2.0),
            _currentEpoch,
            NumOps.FromDouble(_totalEpochs / Math.Log(_mapWidth * _mapHeight)));

        UpdateWeights(flatInput, bmu, learningRate, radius);

        // Quantization error (distance to the BMU prototype) as the reported loss.
        var distances = ComputeSquaredDistances(flatInput);
        LastLoss = NumOps.Sqrt(distances[bmu]);

        _currentEpoch++;
    }

    private T CalculateLearningRate(T initialLearningRate, int currentEpoch, int totalEpochs)
        => NumOps.Multiply(initialLearningRate, NumOps.Exp(NumOps.Negate(NumOps.FromDouble(currentEpoch / (double)totalEpochs))));

    private T CalculateRadius(T initialRadius, int currentEpoch, T timeConstant)
        => NumOps.Multiply(initialRadius, NumOps.Exp(NumOps.Negate(NumOps.Divide(NumOps.FromDouble(currentEpoch), timeConstant))));

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int expectedLength = (_mapWidth * _mapHeight) * _inputDimension;
        if (parameters.Length != expectedLength)
            throw new ArgumentException($"Parameter vector length mismatch. Expected {expectedLength} parameters but got {parameters.Length}.", nameof(parameters));

        int idx = 0;
        for (int i = 0; i < _mapWidth * _mapHeight; i++)
            for (int j = 0; j < _inputDimension; j++)
                _weights[i, j] = parameters[idx++];
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputDimension", _inputDimension },
                { "MapWidth", _mapWidth },
                { "MapHeight", _mapHeight },
                { "TotalEpochs", _totalEpochs },
                { "CurrentEpoch", _currentEpoch }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_inputDimension);
        writer.Write(_mapWidth);
        writer.Write(_mapHeight);
        writer.Write(_totalEpochs);
        writer.Write(_currentEpoch);

        int n = _mapWidth * _mapHeight;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < _inputDimension; j++)
                writer.Write(Convert.ToDouble(_weights[i, j]));
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _inputDimension = reader.ReadInt32();
        _mapWidth = reader.ReadInt32();
        _mapHeight = reader.ReadInt32();
        _totalEpochs = reader.ReadInt32();
        _currentEpoch = reader.ReadInt32();

        int n = _mapWidth * _mapHeight;
        _weights = new Tensor<T>(new[] { n, _inputDimension });
        for (int i = 0; i < n; i++)
            for (int j = 0; j < _inputDimension; j++)
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new SelfOrganizingMap<T>(Architecture, _totalEpochs, LossFunction);

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount => _mapWidth * _mapHeight * _inputDimension;

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        int idx = 0;
        for (int i = 0; i < _mapWidth * _mapHeight; i++)
            for (int j = 0; j < _inputDimension; j++)
                parameters[idx++] = _weights[i, j];
        return parameters;
    }

    /// <summary>
    /// Yields the SOM codebook as a single parameter chunk. SOM stores its weights in the <c>_weights</c>
    /// codebook tensor — outside the layer chain by design (Kohonen 1982 §3: a single competitive-learning
    /// codebook, not a stack of trainable layers) — so the base layer-walking enumeration would yield none,
    /// and the snapshot-before/after invariants (Training_ShouldChangeParameters, GradientFlow) would
    /// compare empty sequences and false-fail "no parameters changed".
    /// </summary>
    public override IEnumerable<Tensor<T>> GetParameterChunks()
    {
        yield return _weights;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        // SOM uses competitive learning, not gradient descent; the effective "gradient" is the BMU +
        // neighbourhood weight delta. Return a zero-length-consistent vector for gradient-flow checks.
        return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        return new Dictionary<string, Tensor<T>>
        {
            ["Input"] = input,
            ["Output"] = Predict(input)
        };
    }
}
