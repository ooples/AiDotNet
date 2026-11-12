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
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(
///     InputType.OneDimensional, NeuralNetworkTaskType.Clustering, inputSize: 10, outputSize: 100);
///
/// var trainX = Tensor&lt;float&gt;.CreateRandom(8, 10);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(8, 100);
///
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new SelfOrganizingMap&lt;float&gt;(arch))
///     .Build(trainX, trainY);
///
/// var output = result.Predict(Tensor&lt;float&gt;.CreateDefault(new[] { 1, 10 }, 0f));
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
public partial class SelfOrganizingMap<T> : VectorModelLayoutBase<T>
{
    private readonly SelfOrganizingMapNNOptions _options;

    /// <summary>
    /// The neuron codebook: shape [numNeurons, inputDimension]. Row i is the prototype vector of
    /// neuron i (row-major over the [mapHeight, mapWidth] grid, i = y * mapWidth + x).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _weights;


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
    public SelfOrganizingMap(NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        SelfOrganizingMapNNOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        options ??= new SelfOrganizingMapNNOptions();
        _options = options;
        Options = _options;

        // options.TotalEpochs is a divisor in both the learning-rate decay (currentEpoch / options.TotalEpochs) and the
        // neighborhood-radius schedule (options.TotalEpochs / log(mapArea)); a non-positive value makes those
        // schedules divide by zero/negative and emit NaN/Infinity weights on the first update.
        if (options.TotalEpochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.TotalEpochs),
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
        _totalEpochs = options.TotalEpochs;
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

    /// <summary>The shared map over this model's codebook, with SOM_PAK's schedules over <c>TotalEpochs</c> steps.</summary>
    private SelfOrganizingMapCore<T> Map() => new(new SelfOrganizingMapConfiguration(
        _mapWidth, _mapHeight, _inputDimension, _totalEpochs, _options.InitialLearningRate, _options.InitialRadius), _weights);

    private int FindBestMatchingUnit(Tensor<T> input) => Map().BestMatchingUnit(input.AsSpan());

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // NO GPU LAYER-CHAIN FAST PATH HERE. The removed GPU fast path delegated to ForwardGpu, which
        // walks the LAYER CHAIN -- and this model deliberately has none: InitializeLayers is empty
        // because a SOM's parameters are its codebook in _weights, not a stack of layers. Walking an
        // empty chain returns the input verbatim, so with a DirectGpuTensorEngine (the default on
        // this machine) Predict returned its own input: 128 values straight through instead of a
        // one-hot activation over the 64 map neurons. The BMU was never computed and every
        // downstream consumer -- clustering, quantization error, the one-hot contract documented on
        // this class -- silently received a passthrough.
        //
        // This was invisible to ScaledInput_ShouldChangeOutput, which PASSES on the broken path: an
        // identity passthrough trivially changes when the input is scaled. The invariant was
        // satisfied for the wrong reason.
        //
        // The best-matching unit is the shared map's nearest codebook vector.
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
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var flatInput = input.Rank == 1 ? input : input.Reshape(new[] { input.Length });
        if (flatInput.Length != _inputDimension)
            throw new ArgumentException($"Input must have {_inputDimension} elements, but got {flatInput.Length}");

        // SOM training is unsupervised — expectedOutput is unused. One step of the shared map, which moves every neuron
        // toward the input by α(t)·h·(x − w); the direction h·(w − x) is the gradient it descended.
        var (bmu, direction) = Map().Step(flatInput.AsSpan(), _currentEpoch);
        PublishFlatParameterGradients(direction);

        // Quantization error (distance to the BMU prototype) as the reported loss.
        double error = 0;
        var prototype = _weights.AsSpan().Slice(bmu * _inputDimension, _inputDimension);
        for (int j = 0; j < _inputDimension; j++)
        {
            double diff = NumOps.ToDouble(flatInput[j]) - NumOps.ToDouble(prototype[j]);
            error += diff * diff;
        }
        LastLoss = NumOps.FromDouble(Math.Sqrt(error));
        _currentEpoch++;

        return true;
    }

    // UpdateParameters was overridden here to write the codebook back element by element, with its
    // own length check. The base distributes over the same declared tensor -- see
    // GetExtraTrainableTensors below -- and validates the length once, for every model.

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
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    /// <inheritdoc/>
    public override bool SupportsTraining => true;

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
