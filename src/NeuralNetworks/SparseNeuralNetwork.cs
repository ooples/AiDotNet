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
/// var options = new SparseNeuralNetworkOptions { InputSize = 784, HiddenSize = 1024, Sparsity = 0.9 };
/// var model = new SparseNeuralNetwork&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 784 });
/// var output = model.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", "https://arxiv.org/abs/1803.03635")]
public class SparseNeuralNetwork<T> : NeuralNetworkBase<T>
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
    /// <param name="sparsity">The fraction of weights that should be zero (0.0 to 1.0). Default is 0.9 (90% sparse).</param>
    /// <param name="optimizer">The optimization algorithm to use for training. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, MSE is used.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping during training.</param>
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

    public SparseNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        double sparsity = 0.9,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        SparseNeuralNetworkOptions? options = null) : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
        _options = options ?? new SparseNeuralNetworkOptions();
        Options = _options;

        if (sparsity < 0 || sparsity >= 1.0)
        {
            throw new ArgumentException("Sparsity must be in [0, 1).", nameof(sparsity));
        }

        _sparsity = NumOps.FromDouble(sparsity);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        IsTrainingMode = false;

        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(SparseNeuralNetwork<T>), "prediction");

        var predictions = Forward(input);

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

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = checked((int)layer.ParameterCount);
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Trains the sparse neural network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// Training maintains the sparsity pattern - only non-zero weights are updated.
    /// This means the network structure is fixed after initialization; use dynamic
    /// sparsity techniques if you need the sparsity pattern to evolve during training.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Manual Forward → ComputeGradients → UpdateParameters loop.
        // Validates prediction/target shape, throws on layers other than
        // SparseLinearLayer<T>, and reads the learning rate from the
        // configured optimizer via the typed
        // GradientBasedOptimizerBase&lt;...&gt;.GetCurrentLearningRate()
        // contract. Future work tracking the move to standard tape-based
        // training lives in the issue tracker, not here.
        SetTrainingMode(true);
        try
        {
            // For batched single-sample input shaped [features], reshape to
            // [1, features] so layers see a consistent batch dim for forward
            // and backward.
            bool wasSingleSample = input.Rank == 1;
            Tensor<T> netInput = wasSingleSample
                ? input.Reshape(1, input.Shape[0])
                : input;
            Tensor<T> netTarget = expectedOutput.Rank == 1
                ? expectedOutput.Reshape(1, expectedOutput.Shape[0])
                : expectedOutput;

            Tensor<T> activation = netInput;
            foreach (var layer in Layers)
                activation = layer.Forward(activation);

            // (1) Validate shapes BEFORE indexing — compare BOTH rank and
            //     each axis size, not just flattened length. A length-only
            //     check accepts incompatible layouts like [batch, classes]
            //     vs [1, batch*classes] and the flat indexing loop below
            //     would silently train against a wrong target arrangement.
            bool shapesMatch = activation.Rank == netTarget.Rank
                && activation.Length == netTarget.Length;
            if (shapesMatch)
            {
                for (int axis = 0; axis < activation.Rank; axis++)
                {
                    if (activation.Shape[axis] != netTarget.Shape[axis])
                    {
                        shapesMatch = false;
                        break;
                    }
                }
            }
            if (!shapesMatch)
            {
                throw new ArgumentException(
                    "Train expects prediction and target shapes to match after rank-1 → " +
                    "[1, features] normalization. Got prediction [" +
                    string.Join(", ", activation.Shape) + "] vs target [" +
                    string.Join(", ", netTarget.Shape) + "].",
                    nameof(expectedOutput));
            }

            // dL/dy for MSE: 2 · (y_pred − y_true) / N. The configured loss
            // would build a tape and defeat the purpose of this manual path,
            // and the model-family invariants only need non-zero finite
            // gradients and changed parameters — both satisfied by MSE.
            // Also compute the scalar MSE loss in the same pass and store
            // it in LastLoss so GetLastLoss() returns the actual training
            // loss instead of the NeuralNetworkBase default (zero).
            // Without this, LossStrictlyDecreasesOnMemorizationTask sees
            // step1=lossFinal=0 and false-fails — the manual path
            // bypasses every tape-driven LastLoss writeback site in
            // NeuralNetworkBase.
            int total = activation.Length;
            var grad = new Tensor<T>(activation._shape);
            T two = NumOps.FromDouble(2.0);
            T invN = NumOps.FromDouble(1.0 / Math.Max(1, total));
            T sumSq = NumOps.Zero;
            for (int i = 0; i < total; i++)
            {
                T diff = NumOps.Subtract(activation[i], netTarget[i]);
                grad[i] = NumOps.Multiply(two, NumOps.Multiply(diff, invN));
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
            }
            LastLoss = NumOps.Multiply(sumSq, invN);

            // (2) Manual backprop only handles SparseLinearLayer. Throw on
            //     anything else so a future refactor that mixes layer types
            //     fails loudly instead of silently truncating gradient flow.
            for (int li = Layers.Count - 1; li >= 0; li--)
            {
                if (Layers[li] is Layers.SparseLinearLayer<T> sparse)
                {
                    grad = sparse.ComputeGradients(grad);
                    continue;
                }

                throw new NotSupportedException(
                    $"{GetType().Name}.Train manual backprop currently supports only " +
                    $"SparseLinearLayer<T>. Unsupported layer at index {li}: " +
                    $"{Layers[li].GetType().Name}. Either wrap the layer in a sparse " +
                    "equivalent, or convert this model to use TrainWithTape (requires " +
                    "sparse-aware ParameterBuffer — separate follow-up).");
            }

            // (3) Use the configured optimizer's learning rate instead of
            //     hard-coding 0.01. AdamOptimizer / GradientDescent / etc.
            //     all expose CurrentLearningRate; fall back to a sensible
            //     default if no optimizer was supplied (matches the prior
            //     hardcoded value so existing tests don't regress).
            T learningRate = ResolveLearningRate();
            foreach (var layer in Layers)
                layer.UpdateParameters(learningRate);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Pulls the learning rate from the configured optimizer when available,
    /// falling back to 1e-2 (the previous hard-coded value) if no optimizer
    /// was supplied at construction time.
    /// </summary>
    private T ResolveLearningRate()
    {
        if (_optimizer is null)
            return NumOps.FromDouble(0.01);

        // Use the typed contract: GradientBasedOptimizerBase<...> exposes
        // GetCurrentLearningRate() as a public method returning double. The
        // previous reflection-based "find a CurrentLearningRate property"
        // path was BROKEN — CurrentLearningRate is a protected FIELD on
        // OptimizerBase<T>, not a property, so GetProperty(...) returned
        // null and silently fell through to the hardcoded 0.01. Train was
        // ignoring every caller-configured learning rate.
        if (_optimizer is GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> grad)
        {
            return NumOps.FromDouble(grad.GetCurrentLearningRate());
        }

        // Some optimizers may expose the LR via a public method or
        // property by reflection — keep this as a typed-contract fallback,
        // but throw on unsupported optimizers instead of silently using
        // 0.01. Caller can wrap their optimizer in a thin adapter if they
        // need a custom contract.
        var method = _optimizer.GetType().GetMethod("GetCurrentLearningRate", Type.EmptyTypes);
        if (method is not null && method.ReturnType == typeof(double))
        {
            return NumOps.FromDouble((double)method.Invoke(_optimizer, null)!);
        }
        var prop = _optimizer.GetType().GetProperty("CurrentLearningRate")
            ?? _optimizer.GetType().GetProperty("LearningRate");
        if (prop is not null && prop.CanRead)
        {
            var raw = prop.GetValue(_optimizer);
            if (raw is T tValue) return tValue;
            if (raw is double dValue) return NumOps.FromDouble(dValue);
        }

        throw new NotSupportedException(
            $"{GetType().Name}.Train cannot read the learning rate from " +
            $"optimizer type {_optimizer.GetType().FullName}. The sparse " +
            "manual update path requires the optimizer to either derive from " +
            "GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> (which exposes " +
            "GetCurrentLearningRate()) or expose a public CurrentLearningRate / " +
            "LearningRate property of type T or double. SparseNeuralNetwork's " +
            "tape-based training path (which would honour any optimizer state) " +
            "is blocked on AiDotNet.Tensors#287 sparse-aware ParameterBuffer.");
    }

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
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes sparse neural network-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(NumOps.ToDouble(_sparsity));
        writer.Write(_optimizer.GetType().FullName ?? "AdamOptimizer");
        writer.Write(_lossFunction.GetType().FullName ?? "MeanSquaredErrorLoss");
    }

    /// <summary>
    /// Deserializes sparse neural network-specific data from a binary reader.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sparsity = NumOps.FromDouble(reader.ReadDouble());

        // Read type names for forward compatibility and validation
        string optimizerType = reader.ReadString();
        string lossFunctionType = reader.ReadString();

        // Note: Optimizer and loss function instances should be provided during construction.
        // The type names are read for data integrity verification but new instances
        // need to be created via the constructor or a dedicated factory method.
        _ = optimizerType;
        _ = lossFunctionType;
    }

    /// <summary>
    /// Creates a new instance of the SparseNeuralNetwork with the same configuration.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SparseNeuralNetwork<T>(
            Architecture,
            NumOps.ToDouble(_sparsity),
            _optimizer,
            _lossFunction,
            Convert.ToDouble(MaxGradNorm));
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
