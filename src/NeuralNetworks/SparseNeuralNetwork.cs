using AiDotNet.LearningRateSchedulers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Sparse Neural Network with efficient sparse weight matrices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A Sparse Neural Network uses sparse weight matrices where most values are zero.
/// This provides significant memory and computational savings for large networks,
/// especially when combined with network pruning techniques.
/// </para>
/// <para>
/// <b>For Beginners:</b> In a regular neural network, every neuron in one layer is connected
/// to every neuron in the next layer. In a sparse network, many of these connections are
/// removed (set to zero), keeping only the most important ones. This has several benefits:
/// - Uses less memory (only stores non-zero values)
/// - Runs faster (skips multiplications with zero)
/// - Can prevent overfitting (acts as regularization)
/// - Enables very large networks to fit in limited memory
///
/// Common use cases include:
/// - Network compression for mobile/edge deployment
/// - Recommender systems with sparse user-item matrices
/// - Graph neural networks with sparse adjacency matrices
/// - Pruned networks from neural architecture search
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 8, outputSize: 4);
/// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 784 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new SparseNeuralNetwork&lt;float&gt;(architecture))
///     .Build(trainX, trainY);
/// var output = result.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", "https://arxiv.org/abs/1803.03635")]
[PaperOptimizer(OptimizerKind.SgdMomentum, Momentum = 0.9, DecayRate = 0.1,
                Milestones = [80, 120], Schedule = LearningRateSchedulerType.MultiStep,
                ScheduleStepMode = SchedulerStepMode.StepPerEpoch,
                Source = "Frankle and Carbin 2019, Sec. 4: the convolutional CIFAR-10 experiments use "
                        + "momentum 0.9 and decrease the learning rate by a factor of 10 at epochs 80 "
                        + "and 120. The paper deliberately surveys several optimization strategies -- "
                        + "SGD, momentum and Adam -- so this records the setting of its convolutional "
                        + "experiments rather than a single choice for all of them, and states no rate, "
                        + "which is why the model keeps its own optimizer and is verified against this "
                        + "record.")]
public partial class SparseNeuralNetwork<T> : VectorModelLayoutBase<T>
{
    private readonly SparseNeuralNetworkOptions _options;

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
    /// The sparsity level (fraction of weights that are zero).
    /// </summary>
    private T _sparsity;

    /// <summary>
    /// Initializes a new instance of the SparseNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="optimizer">The optimization algorithm to use for training. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, MSE is used.</param>
    /// <remarks>
    /// <para>
    /// Higher sparsity values mean fewer connections and faster computation, but may reduce
    /// the network's capacity to learn complex patterns. A sparsity of 0.9 (90% zeros) is
    /// a good starting point for most applications.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public SparseNeuralNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 128,
            outputSize: 1))
    {
    }

    public SparseNeuralNetwork(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SparseNeuralNetworkOptions? options = null) : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), (options ??= new SparseNeuralNetworkOptions()).MaxGradNorm)
    {
        _options = options;
        Options = _options;

        if (options.Sparsity < 0 || options.Sparsity >= 1.0)
        {
            throw new ArgumentException("Sparsity must be in [0, 1).", nameof(options.Sparsity));
        }

        _sparsity = NumOps.FromDouble(options.Sparsity);
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this));
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the sparse neural network based on the provided architecture.
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
            var inputShape = Architecture.GetInputShape();
            var hiddenSizes = Architecture.GetHiddenLayerSizes();

            int inputFeatures = inputShape[0];
            int outputFeatures = Architecture.OutputSize;

            // Output layer uses identity activation regardless of hidden-layer
            // depth: SparseLinearLayer defaults to ReLU, which clamps negative
            // pre-activations to 0. On a regression head where ~50% of random
            // sparse-weighted sums are negative, that produces identical
            // (zero) outputs for distinct inputs — the network collapses
            // before training ever sees a gradient. Mocanu et al. (2018) and
            // every subsequent sparse-network paper (RigL, Top-KAST, …) use
            // identity for the regression output and ReLU only for hidden
            // layers. Mirror the convention here.
            var identity = new IdentityActivation<T>();

            if (hiddenSizes.Length == 0)
            {
                // Per Mocanu et al. (2018), sparse networks need hidden layers for
                // sparse-to-sparse connectivity. Single-layer sparse → dead ReLU neurons.
                int hiddenSize = Math.Max(32, (inputFeatures + outputFeatures) / 2);
                Layers.Add(new SparseLinearLayer<T>(inputFeatures, hiddenSize, NumOps.ToDouble(_sparsity)));
                Layers.Add(new SparseLinearLayer<T>(hiddenSize, outputFeatures, NumOps.ToDouble(_sparsity), identity));
            }
            else
            {
                Layers.Add(new SparseLinearLayer<T>(inputFeatures, hiddenSizes[0], NumOps.ToDouble(_sparsity)));

                for (int i = 0; i < hiddenSizes.Length - 1; i++)
                {
                    Layers.Add(new SparseLinearLayer<T>(hiddenSizes[i], hiddenSizes[i + 1], NumOps.ToDouble(_sparsity)));
                }

                Layers.Add(new SparseLinearLayer<T>(hiddenSizes[^1], outputFeatures, NumOps.ToDouble(_sparsity), identity));
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the sparse neural network for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        IsTrainingMode = false;

        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(SparseNeuralNetwork<T>), "prediction");

        var predictions = Accelerate(input, () => Forward(input));

        IsTrainingMode = true;

        return predictions;
    }

    /// <summary>
    /// Performs a forward pass through the network with the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass uses sparse matrix-vector multiplication (SpMV) for efficiency.
    /// Only non-zero weights are used in computation, significantly reducing the number
    /// of operations for highly sparse networks.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // Validate input shape before any processing (including GPU path)
        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(SparseNeuralNetwork<T>), "forward pass");

        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

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
    /// Retrieves metadata about the sparse neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "SparseNeuralNetwork" },
                { "Sparsity", NumOps.ToDouble(_sparsity) },
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Architecture.GetOutputShape() },
                { "HiddenLayerSizes", Architecture.GetHiddenLayerSizes() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "ParameterCount", GetParameterCount() }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <summary>
    /// Serializes sparse neural network-specific data to a binary writer.
    /// </summary>


    /// <summary>
    /// Deserializes sparse neural network-specific data from a binary reader.
    /// </summary>


    /// <summary>
    /// Indicates whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Determines if a layer can serve as a valid input layer for this network.
    /// </summary>
    protected override bool IsValidInputLayer(ILayer<T> layer)
    {
        // Sparse layers are valid input layers for this network
        if (layer is SparseLinearLayer<T>)
            return true;

        return base.IsValidInputLayer(layer);
    }

    /// <summary>
    /// Determines if a layer can serve as a valid output layer for this network.
    /// </summary>
    protected override bool IsValidOutputLayer(ILayer<T> layer)
    {
        // Sparse layers are valid output layers for this network
        if (layer is SparseLinearLayer<T>)
            return true;

        return base.IsValidOutputLayer(layer);
    }
}
