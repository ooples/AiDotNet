using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Feed-Forward Neural Network (FFNN) for processing data in a forward path.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A Feed-Forward Neural Network is the simplest type of artificial neural network, where connections
/// between nodes do not form a cycle. Information moves in only one direction -- forward -- from the input
/// nodes, through the hidden nodes (if any), and to the output nodes.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a Feed-Forward Neural Network like a series of information-processing
/// stages arranged in a line. Data flows only forward through these stages, never backward. Each stage
/// (or layer) processes the information and passes it to the next stage. This simple structure makes
/// FFNNs great for many common tasks like classification (deciding which category something belongs to)
/// or regression (predicting a numerical value).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new FeedForwardNeuralNetworkOptions { InputSize = 10, HiddenLayers = new[] { 64, 32 }, OutputSize = 2 };
/// var model = new FeedForwardNeuralNetwork&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 10 });
/// var output = model.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Learning Internal Representations by Error Propagation", "https://doi.org/10.21236/ADA164453")]
public class FeedForwardNeuralNetwork<T> : NeuralNetworkBase<T>
{
    private readonly FeedForwardNeuralNetworkOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This function measures how well the network is performing and guides the learning process.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as the scorekeeper for the network. It tells the network
    /// how far off its predictions are from the correct answers.
    /// </para>
    /// </remarks>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer determines how the network's internal values are adjusted based on the calculated error.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like the network's learning strategy. It decides how to adjust
    /// the network's settings to improve its performance over time.
    /// </para>
    /// </remarks>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Initializes a new instance of the FeedForwardNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="optimizer">The optimization algorithm to use for training. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, an appropriate loss function is selected based on the task type.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping during training.</param>
    /// <remarks>
    /// <para>
    /// Feed-Forward Neural Networks can work with various input dimensions and are typically used for
    /// classification and regression tasks with structured data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When creating a Feed-Forward Neural Network, you need to provide a blueprint (architecture)
    /// that defines the structure of your network. This constructor sets up the network based on that blueprint.
    /// It also prepares the learning strategy (optimizer) and the way to measure mistakes (loss function).
    /// If you don't specify these, it chooses reasonable defaults based on the type of task you're trying to solve.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public FeedForwardNeuralNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 128,
            outputSize: 1))
    {
    }

    public FeedForwardNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        FeedForwardNeuralNetworkOptions? options = null) : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new FeedForwardNeuralNetworkOptions();
        Options = _options;
        // Default to AMSGrad-mode Adam (Reddi, Kale, Kumar 2018). Standard
        // Adam's bias-corrected m̂ / √v̂ ratio doesn't decay fast enough
        // after gradient convergence, so on fixed-input regression tasks
        // it drifts the parameters away from a tight optimum over long
        // training runs (MoreData_ShouldNotDegrade invariant). AMSGrad's
        // running v̂_max guarantees the denominator can only grow,
        // bounding the drift to negligible levels. Issue #1332 cluster 6.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { UseAMSGrad = true });

        // Select appropriate loss function based on task type if not provided
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default feed-forward layers if none are specified.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the different processing stages (layers) of your
    /// neural network. If you've specified custom layers in your architecture, it will use those.
    /// If not, it will create a standard set of layers commonly used for feed-forward networks,
    /// with the right number of neurons at each stage based on your architecture settings.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultFeedForwardLayers(Architecture));
        }
    }

    /// <summary>
    /// Makes a prediction using the feed-forward neural network for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network to generate a prediction.
    /// Unlike the vector-based Predict method, this takes a tensor directly as input.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method makes predictions using the trained neural network.
    /// It accepts input data in tensor format (multi-dimensional arrays), processes it through
    /// all the network's layers, and returns the prediction as a tensor. This is useful when
    /// working with data that naturally has multiple dimensions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Ensure the network is in inference mode
        IsTrainingMode = false;

        ValidateInputShape(input, "prediction");

        // Fused fast path: a pure stack of dense layers with fused-eligible scalar
        // activations runs as ONE IEngine.MlpForward call instead of a per-layer
        // tape/dispatch walk — the kernel the AiDotNet.Tensors MLP micro-benchmarks
        // beat PyTorch-CPU on. Falls back to the generic Forward for any layer the
        // kernel can't represent (non-dense, vector activation, unmapped activation,
        // mixed hidden activations) or if the kernel declines (e.g. active tape).
        if (TryFusedDensePredict(input, out var fused))
        {
            IsTrainingMode = true;
            return fused;
        }

        var predictions = Accelerate(input, () => Forward(input));

        IsTrainingMode = true;

        return predictions;
    }

    /// <summary>
    /// Attempts the fused multi-layer-perceptron inference kernel for a pure
    /// dense+activation stack. Returns false (and the caller uses the generic
    /// per-layer <see cref="Forward"/>) whenever the stack isn't representable by
    /// <c>IEngine.MlpForward</c>. Activation→kernel mapping is open/closed: each
    /// activation that has an exact fused equivalent implements
    /// <see cref="ActivationFunctions.Fused.IFusedActivation"/>; there is no switch.
    /// </summary>
    private bool TryFusedDensePredict(Tensor<T> input, out Tensor<T> output)
    {
        output = Tensor<T>.Empty();
        var layers = Layers;
        if (layers is null || layers.Count == 0) return false;

        var weights = new List<Tensor<T>>(layers.Count);
        var biases = new List<Tensor<T>?>(layers.Count);
        var hiddenActivation = Tensors.Engines.FusedActivationType.None;
        bool hiddenActivationSet = false;
        var outputActivation = Tensors.Engines.FusedActivationType.None;

        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is not Layers.DenseLayer<T> dense) return false;
            // Vector activations aren't covered by the scalar fused kernels.
            if (dense.VectorActivation is not null) return false;

            Tensors.Engines.FusedActivationType act;
            if (dense.ScalarActivation is null)
                act = Tensors.Engines.FusedActivationType.None;
            else if (dense.ScalarActivation is ActivationFunctions.Fused.IFusedActivation fused
                     && fused.TryGetFusedActivation(out var ft))
                act = ft;
            else
                return false; // activation has no exact fused equivalent (or custom param)

            if (i == layers.Count - 1)
            {
                outputActivation = act;
            }
            else if (!hiddenActivationSet)
            {
                hiddenActivation = act;
                hiddenActivationSet = true;
            }
            else if (hiddenActivation != act)
            {
                // MlpForward applies a single hidden activation to every non-final
                // layer; a stack with mixed hidden activations can't use it.
                return false;
            }

            // DenseLayer initializes its weights lazily on first Forward. On a
            // fresh network's very first inference the weights are still the [0,0]
            // sentinel, which MlpForward would reject. Bail to the generic Forward
            // (which runs the lazy shape resolution); subsequent inferences, with
            // weights materialized, take the fused path.
            var w = dense.GetWeights();
            if (w.Rank != 2 || w.Shape[0] == 0 || w.Shape[1] == 0) return false;
            weights.Add(w);
            biases.Add(dense.GetBiases());
        }

        try
        {
            // Flagship compiled-inference tier (float only): replay a cached, self-tuned
            // CompiledMlp plan (array-based, near-zero per-call allocation, per-layer
            // managed-vs-native kernel selection) instead of the Tensor-based MlpForward.
            // At the kernel level CompiledMlp.Run beats torch.compile (~0.11 ms vs
            // ~0.22 ms at bs1 on the AIsEval MLP); MlpForward's per-call Tensor/dispatch
            // overhead is what loses. Falls through to MlpForward when ineligible.
            if (typeof(T) == typeof(float)
                && TryCompiledMlpPredict(input, weights, biases, hiddenActivation, outputActivation, out output))
            {
                return true;
            }

            output = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.MlpForward(
                input, weights, biases, hiddenActivation, outputActivation);
            return true;
        }
        catch (InvalidOperationException)
        {
            // MlpForward is forward-only and throws under an active GradientTape;
            // fall back to the generic path rather than failing the prediction.
            output = Tensor<T>.Empty();
            return false;
        }
    }

    // ── Compiled-inference plan cache (float pure-dense fast path) ────────────
    private AiDotNet.Tensors.Engines.Compilation.CompiledMlp? _compiledMlpPlan;
    private float[][]? _compiledMlpWeightRefs;   // backing arrays the plan was built from
    private int _compiledMlpMaxBatch;

    /// <summary>
    /// Runs the pure-dense float stack through the cached <c>CompiledMlp</c> plan.
    /// Returns false when the input shape isn't a contiguous rank-1/2 batch the plan
    /// can replay (caller then uses <c>MlpForward</c>). The plan is (re)built when it's
    /// absent, the batch exceeds the buffers it was sized for, or any layer's weight
    /// backing array was reallocated — the same frozen-weights-during-inference contract
    /// as the MlpForward path (which likewise relies on SgemmWithCachedB's
    /// identity-keyed pack), plus a reallocation guard the cached plan requires.
    /// </summary>
    private bool TryCompiledMlpPredict(
        Tensor<T> input,
        List<Tensor<T>> weights,
        List<Tensor<T>?> biases,
        Tensors.Engines.FusedActivationType hiddenActivation,
        Tensors.Engines.FusedActivationType outputActivation,
        out Tensor<T> output)
    {
        output = Tensor<T>.Empty();

        // Only contiguous rank-1 ([features]) or rank-2 ([batch, features]) inputs map
        // to the plan's [batch, features] array contract.
        if (!input.IsContiguous || input.Rank < 1 || input.Rank > 2) return false;
        int batch = input.Rank == 2 ? input.Shape[0] : 1;
        int inFeatures = input.Shape[input.Rank - 1];
        if (batch < 1) return false;

        int layerCount = weights.Count;
        var wRefs = new float[layerCount][];
        for (int i = 0; i < layerCount; i++)
            wRefs[i] = (float[])(object)weights[i].GetDataArray();
        if (wRefs[0].Length < (long)inFeatures * weights[0].Shape[1]) return false;

        bool rebuild = _compiledMlpPlan is null
            || batch > _compiledMlpMaxBatch
            || _compiledMlpWeightRefs is null
            || _compiledMlpWeightRefs.Length != layerCount;
        if (!rebuild)
        {
            for (int i = 0; i < layerCount; i++)
                if (!ReferenceEquals(_compiledMlpWeightRefs![i], wRefs[i])) { rebuild = true; break; }
        }

        if (rebuild)
        {
            var inF = new int[layerCount];
            var outF = new int[layerCount];
            var bArrs = new float[]?[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                inF[i] = weights[i].Shape[0];
                outF[i] = weights[i].Shape[1];
                var b = biases[i];
                bArrs[i] = b is null ? null : (float[])(object)b.GetDataArray();
            }
            // Size buffers for at least this batch; grow (never shrink) so cycling
            // batch sizes doesn't thrash. maxBatch caps the ping-pong scratch.
            int maxBatch = Math.Max(batch, _compiledMlpMaxBatch);
            _compiledMlpPlan = Tensors.Engines.Compilation.CompiledMlp.Create(
                wRefs, bArrs, inF, outF, hiddenActivation, outputActivation, maxBatch);
            _compiledMlpWeightRefs = wRefs;
            _compiledMlpMaxBatch = maxBatch;
        }

        var plan = _compiledMlpPlan!;
        if (inFeatures != plan.InputFeatures) return false;

        var inputArr = (float[])(object)input.GetDataArray();
        // Array lengths are int in C#; compute the output element count in a checked int so an
        // oversized batch×features product throws OverflowException rather than silently wrapping
        // (the prior (long) length forced an int-narrowing conversion at the array creation site).
        var outArr = new float[checked(batch * plan.OutputFeatures)];
        plan.Run(inputArr, batch, outArr);

        var resultShape = input.Rank == 2 ? new[] { batch, plan.OutputFeatures } : new[] { plan.OutputFeatures };
        output = (Tensor<T>)(object)new Tensor<float>(outArr, resultShape);
        return true;
    }

    /// <summary>
    /// Accepts either an unbatched input matching <c>Architecture.GetInputShape()</c>
    /// or a batched input <c>[B, ...expectedShape]</c> whose trailing dims match
    /// each axis exactly. Throws via <see cref="TensorValidator"/> otherwise.
    /// </summary>
    private void ValidateInputShape(Tensor<T> input, string operationName)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        var expectedShape = Architecture.GetInputShape();
        if (input.Rank == expectedShape.Length + 1)
        {
            bool axisMatch = true;
            for (int i = 0; i < expectedShape.Length; i++)
            {
                if (input.Shape[i + 1] != expectedShape[i])
                {
                    axisMatch = false;
                    break;
                }
            }
            if (axisMatch) return;
        }
        TensorValidator.ValidateShape(input, expectedShape,
            nameof(FeedForwardNeuralNetwork<T>), operationName);
    }

    /// <summary>
    /// Performs a forward pass through the network with the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <exception cref="TensorShapeMismatchException">Thrown when the input shape doesn't match the expected input shape.</exception>
    /// <remarks>
    /// <para>
    /// The forward pass sequentially processes the input through each layer of the network.
    /// This is the core operation for making predictions with the neural network.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method takes your input data and passes it through each layer
    /// of the neural network in sequence. Think of it like an assembly line where each station (layer)
    /// processes the data and passes it to the next station. The final output contains the network's prediction.
    /// This is the engine that powers the prediction process.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        ValidateInputShape(input, "forward pass");

        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // CPU path: each layer processes input and may download results
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
    /// <remarks>
    /// <para>
    /// This method distributes the parameters to each layer based on their parameter count.
    /// It's typically called during training after calculating parameter updates.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After the backward pass calculates how to improve the network,
    /// this method actually applies those improvements. It takes a list of updated settings
    /// (parameters) and distributes them to each layer in the network. This method is
    /// called repeatedly during training to gradually improve the network's accuracy.
    /// </para>
    /// </remarks>
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
    /// Trains the feed-forward neural network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method performs one training iteration, including forward pass, loss calculation,
    /// backward pass, and parameter update.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network learns. You show it some input data and
    /// tell it what the correct output should be. The network makes a guess, compares it to
    /// the correct answer, and then adjusts its internal settings to do better next time.
    /// This process is repeated many times with different examples to train the network.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainWithTape(input, expectedOutput, _optimizer);
    }

    /// <summary>
    /// Retrieves metadata about the feed-forward neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method collects and returns various pieces of information about the network's structure and configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like getting a summary of the network's blueprint. It tells you
    /// how many layers it has, what types of layers they are, and other important details about how
    /// the network is set up. This can be useful for documentation or debugging purposes.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
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
    /// Serializes feed-forward neural network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific parameters and state of the feed-forward neural network to a binary stream.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like saving the network's current state to a file. It records all
    /// the important information about the network so you can reload it later exactly as it is now.
    /// This is useful when you want to save a trained model for later use.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Serialize optimizer and loss function interfaces
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);
        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Deserialize and restore optimizer
        var optimizer = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader);
        if (optimizer != null)
        {
            _optimizer = optimizer;
        }

        // Deserialize and restore loss function
        var lossFunction = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader);
        if (lossFunction != null)
        {
            _lossFunction = lossFunction;
        }
    }

    /// <summary>
    /// Creates a new instance of the FeedForwardNeuralNetwork with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new FeedForwardNeuralNetwork instance with the same architecture, optimizer, and loss function as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the FeedForwardNeuralNetwork with the same architecture, optimizer, and loss function
    /// as the current instance. This is useful for model cloning, ensemble methods, or cross-validation scenarios where
    /// multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method creates a fresh copy of the neural network's blueprint.
    /// 
    /// When you need multiple versions of the same type of neural network with identical settings:
    /// - This method creates a new, empty network with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new network has the same structure but no trained data
    /// - This is useful for techniques that need multiple models, like ensemble methods
    /// 
    /// For example, when testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new FeedForwardNeuralNetwork<T>(
            Architecture,
            _optimizer,
            _lossFunction,
            Convert.ToDouble(MaxGradNorm));
    }

    /// <summary>
    /// Indicates whether this network supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Feed-forward neural networks support training through backpropagation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This indicates that the network can learn from data.
    /// Feed-forward networks are designed to be trained, so this property returns true.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;
}
