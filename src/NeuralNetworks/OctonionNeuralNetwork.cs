using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an Octonion-valued Neural Network for processing data in 8-dimensional hypercomplex space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// An Octonion Neural Network uses octonion algebra (8-dimensional non-associative division algebra)
/// for its computations. This provides richer representational capacity than real, complex, or
/// quaternion-valued networks, making it suitable for tasks requiring high-dimensional rotations
/// and transformations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Octonions are 8-dimensional numbers that extend complex numbers and quaternions.
/// While regular neural networks use simple numbers, octonion networks use these 8-dimensional numbers
/// which can capture more complex relationships in data. This is particularly useful for:
/// - 3D graphics and physics simulations
/// - Signal processing with multiple channels
/// - Tasks requiring rich rotational representations
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new OctonionNeuralNetworkOptions { InputSize = 8, HiddenSize = 64 };
/// var model = new OctonionNeuralNetwork&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 8 });
/// var output = model.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Deep Octonion Networks", "https://arxiv.org/abs/1903.08478", Year = 2019, Authors = "Jiasong Wu et al.")]
public partial class OctonionNeuralNetwork<T> : VectorModelLayoutBase<T>
{
    private readonly OctonionNeuralNetworkOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Initializes a new instance of the OctonionNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="optimizer">The optimization algorithm to use for training. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, MSE is used.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping during training.</param>
    /// <remarks>
    /// <para>
    /// Note: Input and output dimensions should be multiples of 8 to properly represent octonions.
    /// Each octonion has 8 real components (1 real + 7 imaginary).
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public OctonionNeuralNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 64,
            outputSize: 8))
    {
    }

    public OctonionNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = double.MaxValue,
        OctonionNeuralNetworkOptions? options = null)
        : base(
            architecture,
            lossFunction ?? (architecture.TaskType == Enums.NeuralNetworkTaskType.MultiClassClassification
                ? new CrossEntropyWithLogitsLoss<T>()
                : new MeanSquaredErrorLoss<T>()),
            maxGradNorm)
    {
        _options = options ?? new OctonionNeuralNetworkOptions();
        ValidateOptions(_options);
        Options = _options;
        _optimizer = optimizer ?? new NesterovAcceleratedGradientOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new NesterovAcceleratedGradientOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.InitialLearningRate,
                InitialMomentum = _options.Momentum,
                UseAdaptiveLearningRate = false,
                UseAdaptiveMomentum = false,
                EnableGradientClipping = false,
                LearningRateScheduler = new LambdaLRScheduler(
                    _options.InitialLearningRate,
                    epoch => epoch < _options.RampEpoch
                        ? 1.0
                        : epoch < _options.FirstDecayEpoch
                            ? 10.0
                            : epoch < _options.SecondDecayEpoch ? 1.0 : 0.1),
                SchedulerStepMode = SchedulerStepMode.StepPerEpoch
            });
        // Use the same loss function instance that was passed to base class
        _lossFunction = LossFunction;

        InitializeLayers();
    }

    private static void ValidateOptions(OctonionNeuralNetworkOptions options)
    {
        if (double.IsNaN(options.InitialLearningRate)
            || double.IsInfinity(options.InitialLearningRate)
            || options.InitialLearningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options.InitialLearningRate),
                options.InitialLearningRate,
                "InitialLearningRate must be finite and greater than zero.");
        }

        if (double.IsNaN(options.Momentum)
            || double.IsInfinity(options.Momentum)
            || options.Momentum < 0.0
            || options.Momentum >= 1.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options.Momentum),
                options.Momentum,
                "Momentum must be finite and in the range [0, 1).");
        }

        if (options.RampEpoch < 0
            || options.RampEpoch >= options.FirstDecayEpoch
            || options.FirstDecayEpoch >= options.SecondDecayEpoch)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options.RampEpoch),
                $"Epoch thresholds must satisfy 0 <= RampEpoch < FirstDecayEpoch < SecondDecayEpoch; "
                + $"received {options.RampEpoch}, {options.FirstDecayEpoch}, {options.SecondDecayEpoch}.");
        }
    }

    /// <summary>
    /// Initializes the layers of the octonion neural network based on the provided architecture.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Create default octonion layers based on architecture
            var inputShape = Architecture.GetInputShape();
            var outputShape = new[] { Architecture.OutputSize };
            var hiddenSizes = Architecture.GetHiddenLayerSizes();

            // Validate dimensions are divisible by 8 for octonion representation
            if (inputShape[0] % 8 != 0)
            {
                throw new ArgumentException(
                    $"Input dimension ({inputShape[0]}) must be divisible by 8 for octonion representation.",
                    nameof(Architecture));
            }
            if (outputShape[0] % 8 != 0)
            {
                throw new ArgumentException(
                    $"Output dimension ({outputShape[0]}) must be divisible by 8 for octonion representation.",
                    nameof(Architecture));
            }
            for (int i = 0; i < hiddenSizes.Length; i++)
            {
                if (hiddenSizes[i] % 8 != 0)
                {
                    throw new ArgumentException(
                        $"Hidden layer {i} dimension ({hiddenSizes[i]}) must be divisible by 8 for octonion representation.",
                        nameof(Architecture));
                }
            }

            // Input features divided by 8 for octonion representation
            int inputFeatures = inputShape[0] / 8;
            int outputFeatures = outputShape[0] / 8;

            if (hiddenSizes.Length == 0)
            {
                // Single layer: input -> output
                Layers.Add(new OctonionLinearLayer<T>(inputFeatures, outputFeatures));
            }
            else
            {
                // Input layer
                Layers.Add(new OctonionLinearLayer<T>(inputFeatures, hiddenSizes[0] / 8));

                // Hidden layers
                for (int i = 0; i < hiddenSizes.Length - 1; i++)
                {
                    Layers.Add(new OctonionLinearLayer<T>(hiddenSizes[i] / 8, hiddenSizes[i + 1] / 8));
                }

                // Output layer
                Layers.Add(new OctonionLinearLayer<T>(hiddenSizes[^1] / 8, outputFeatures));
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the octonion neural network for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var previousTrainingMode = IsTrainingMode;
        IsTrainingMode = false;

        try
        {
            TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
                nameof(OctonionNeuralNetwork<T>), "prediction");

            return Accelerate(input, () => Forward(input));
        }
        finally
        {
            IsTrainingMode = previousTrainingMode;
        }
    }

    /// <summary>
    /// Performs a forward pass through the network with the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;


        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(OctonionNeuralNetwork<T>), "forward pass");

        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Trains the octonion neural network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainWithTape(input, expectedOutput, _optimizer);
    }

    /// <summary>
    /// Retrieves metadata about the octonion neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "OctonionNeuralNetwork" },
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Architecture.GetOutputShape() },
                { "HiddenLayerSizes", Architecture.GetHiddenLayerSizes() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = SerializeForMetadata()
        };
    }





    /// <summary>
    /// Indicates whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Determines if a layer can serve as a valid input layer for this network.
    /// </summary>
    protected override bool IsValidInputLayer(ILayer<T> layer)
    {
        // Octonion layers are valid input layers for this network
        if (layer is OctonionLinearLayer<T>)
            return true;

        return base.IsValidInputLayer(layer);
    }

    /// <summary>
    /// Determines if a layer can serve as a valid output layer for this network.
    /// </summary>
    protected override bool IsValidOutputLayer(ILayer<T> layer)
    {
        // Octonion layers are valid output layers for this network
        if (layer is OctonionLinearLayer<T>)
            return true;

        return base.IsValidOutputLayer(layer);
    }
}
