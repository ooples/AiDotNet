namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Mixture-of-Experts (MoE) neural network that routes inputs through multiple specialist networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A Mixture-of-Experts neural network employs multiple expert networks and a gating mechanism to
/// route inputs to the most appropriate experts. This architecture enables:
/// - Increased model capacity without proportional compute cost (sparse activation)
/// - Specialization of different experts on different aspects of the problem
/// - Improved scalability for large-scale problems
/// </para>
/// <para>
/// The architecture consists of:
/// - Multiple expert networks (can be feed-forward, convolutional, etc.)
/// - A gating/routing network that learns to select appropriate experts
/// - Optional load balancing loss to ensure all experts are utilized
/// </para>
/// <para>
/// <b>For Beginners:</b> Mixture-of-Experts is like having a team of specialists rather than one generalist.
///
/// Imagine you're running a hospital:
/// - Instead of one doctor handling everything, you have specialists (cardiologist, neurologist, etc.)
/// - A triage system (gating network) decides which specialist(s) should see each patient
/// - Each specialist only handles cases they're best suited for
///
/// In a MoE neural network:
/// - Multiple "expert" networks specialize in different patterns in your data
/// - A "gating network" learns to route each input to the best expert(s)
/// - Only a few experts process each input (sparse activation), making it efficient
/// - The final prediction combines the outputs from the selected experts
///
/// This model automatically implements IFullModel, allowing it to work with AiModelBuilder
/// just like any other neural network in AiDotNet.
/// </para>
/// <para>
/// <b>Key Features:</b>
/// <list type="bullet">
/// <item><description>Configurable number of expert networks</description></item>
/// <item><description>Top-K sparse routing for computational efficiency</description></item>
/// <item><description>Automatic load balancing to prevent expert collapse</description></item>
/// <item><description>Integration with AiModelBuilder for easy training</description></item>
/// <item><description>Full support for serialization and deserialization</description></item>
/// </list>
/// </para>
/// </remarks>
public class MixtureOfExpertsNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Configuration options for the MoE network.
    /// </summary>
    private readonly MixtureOfExpertsOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The core Mixture-of-Experts layer.
    /// </summary>
    private MixtureOfExpertsLayer<T>? _moeLayer;

    /// <summary>
    /// Initializes a new instance of the MixtureOfExpertsNeuralNetwork class.
    /// </summary>
    /// <param name="options">Configuration options for the Mixture-of-Experts model.</param>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="optimizer">The optimization algorithm to use for training. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, an appropriate loss function is selected based on the task type.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping during training.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a Mixture-of-Experts neural network based on the provided options and architecture.
    /// The options control MoE-specific parameters like number of experts, Top-K routing, and load balancing.
    /// The architecture defines the overall network structure including input/output dimensions and task type.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When creating a Mixture-of-Experts model, you provide two types of configuration:
    ///
    /// 1. Options (MixtureOfExpertsOptions): MoE-specific settings
    ///    - How many expert networks to create
    ///    - How many experts to activate per input (Top-K)
    ///    - Expert dimensions and architecture
    ///    - Load balancing settings
    ///
    /// 2. Architecture (NeuralNetworkArchitecture): General network settings
    ///    - What type of task (classification, regression, etc.)
    ///    - Input and output dimensions
    ///    - Any additional layers beyond the MoE layer
    ///
    /// If you don't specify an optimizer or loss function, the model will choose sensible defaults
    /// based on your task type (e.g., CrossEntropy for classification, MSE for regression).
    ///
    /// The model automatically integrates with AiModelBuilder, so you can train it
    /// using the standard AiDotNet pattern without any special handling.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Create options for 8 experts with Top-2 routing
    /// var options = new MixtureOfExpertsOptions&lt;float&gt;
    /// {
    ///     NumExperts = 8,
    ///     TopK = 2,
    ///     InputDim = 128,
    ///     OutputDim = 128,
    ///     UseLoadBalancing = true,
    ///     LoadBalancingWeight = 0.01
    /// };
    ///
    /// // Create architecture for classification
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
    ///     inputType: InputType.OneDimensional,
    ///     taskType: NeuralNetworkTaskType.MultiClassClassification,
    ///     inputSize: 128,
    ///     outputSize: 10
    /// );
    ///
    /// // Create the model
    /// var model = new MixtureOfExpertsNeuralNetwork&lt;float&gt;(options, architecture);
    ///
    /// // Use with AiModelBuilder (standard pattern)
    /// var builder = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;();
    /// var result = builder.ConfigureModel(model).Build(trainingData, trainingLabels);
    /// </code>
    /// </example>
    public MixtureOfExpertsNeuralNetwork(
        MixtureOfExpertsOptions<T> options,
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        _options.Validate();

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture and options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates the Mixture-of-Experts layer based on the configuration options,
    /// and adds any additional layers specified in the architecture.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the expert networks and gating mechanism.
    ///
    /// The initialization process:
    /// 1. Creates expert networks based on your options (number, dimensions, etc.)
    /// 2. Creates the gating/routing network that learns to select experts
    /// 3. Adds any additional layers you specified in the architecture
    ///
    /// You don't need to call this manually - it's automatically called during construction.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        // If custom layers are provided in the architecture, use them
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        // Otherwise, create MoE layer based on options
        _moeLayer = new MixtureOfExpertsBuilder<T>()
            .WithExperts(_options.NumExperts)
            .WithDimensions(_options.InputDim, _options.OutputDim)
            .WithTopK(_options.TopK)
            .WithLoadBalancing(_options.UseLoadBalancing, _options.LoadBalancingWeight)
            .WithHiddenExpansion(_options.HiddenExpansion)
            .Build();

        Layers.Add(_moeLayer);

        // Add output layer if needed
        if (_options.OutputDim != Architecture.OutputSize)
        {
            var outputLayer = new DenseLayer<T>(
                _options.OutputDim,
                Architecture.OutputSize,
                NeuralNetworkHelper<T>.GetDefaultActivationFunction(Architecture.TaskType)
            );
            Layers.Add(outputLayer);
        }
    }

    /// <summary>
    /// Makes a prediction using the Mixture-of-Experts network for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network to generate a prediction.
    /// The gating network determines which experts to activate, and only those experts
    /// process the input for efficiency.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method makes predictions using the trained Mixture-of-Experts model.
    ///
    /// What happens during prediction:
    /// 1. The input goes to the gating network
    /// 2. The gating network selects the best experts for this input (Top-K)
    /// 3. Only the selected experts process the input
    /// 4. Expert outputs are combined using learned weights
    /// 5. The final prediction is returned
    ///
    /// This sparse activation (only using some experts) makes MoE much faster than
    /// running all experts for every input, while maintaining high quality predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Ensure the network is in inference mode
        IsTrainingMode = false;

        // Validate input shape
        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(MixtureOfExpertsNeuralNetwork<T>), "prediction");

        // Perform forward pass
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
    /// The forward pass sequentially processes the input through each layer of the network,
    /// with the MoE layer using sparse expert activation based on the gating network's decisions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method processes input through all the network's layers.
    ///
    /// For MoE networks:
    /// 1. Input goes through the MoE layer (which internally routes to experts)
    /// 2. Then through any additional layers you added
    /// 3. Final output is returned
    ///
    /// Think of it like an assembly line where each station (layer) processes
    /// the data and passes it to the next station.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(MixtureOfExpertsNeuralNetwork<T>), "forward pass");

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
    /// Performs a backward pass through the network to calculate gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the network's output.</param>
    /// <returns>The gradient of the loss with respect to the network's input.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass propagates gradients backward through each layer, including the
    /// MoE layer which distributes gradients to the experts that were activated during the forward pass.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network learns from mistakes.
    ///
    /// After making a prediction, we calculate how wrong it was (the error).
    /// This method works backward through the network, calculating how each part
    /// contributed to the error. This information is used to improve the network.
    ///
    /// For MoE networks:
    /// - Gradients flow back through the output layers
    /// - Then through the MoE layer to the activated experts
    /// - The gating network also learns which experts to select
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        Tensor<T> gradient = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        return gradient;
    }

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters to each layer, including all expert networks
    /// within the MoE layer and the gating network.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After calculating how to improve the network, this method
    /// applies those improvements.
    ///
    /// It distributes updated settings (parameters) to:
    /// - All expert networks
    /// - The gating network
    /// - Any additional layers
    ///
    /// This is called repeatedly during training to gradually improve accuracy.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Trains the Mixture-of-Experts network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method performs one training iteration, including:
    /// - Forward pass through the network
    /// - Primary loss calculation (task-specific loss)
    /// - Auxiliary loss calculation (load balancing loss)
    /// - Backward pass (backpropagation)
    /// - Parameter update via optimizer
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the MoE network learns.
    ///
    /// Training process:
    /// 1. Show the network some input data and the correct output
    /// 2. Network makes a prediction using selected experts
    /// 3. Calculate two types of errors:
    ///    - Task error: How wrong the prediction was
    ///    - Balance error: Whether experts are being used evenly
    /// 4. Adjust the network to reduce both errors
    /// 5. Repeat many times with different examples
    ///
    /// The load balancing ensures all experts contribute and don't collapse to
    /// using just one or two experts.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        IsTrainingMode = true;

        // Forward pass to get prediction
        var prediction = Forward(input);

        // Calculate primary loss
        var primaryLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        // Calculate auxiliary losses from layers that support them (e.g., load balancing loss)
        T auxiliaryLoss = NumOps.Zero;
        foreach (var auxLayer in Layers.OfType<IAuxiliaryLossLayer<T>>().Where(l => l.UseAuxiliaryLoss))
        {
            var layerAuxLoss = auxLayer.ComputeAuxiliaryLoss();
            var weightedAuxLoss = NumOps.Multiply(layerAuxLoss, auxLayer.AuxiliaryLossWeight);
            auxiliaryLoss = NumOps.Add(auxiliaryLoss, weightedAuxLoss);
        }

        // Combine primary and auxiliary losses
        LastLoss = NumOps.Add(primaryLoss, auxiliaryLoss);

        // Calculate output gradient (derivative of loss with respect to network output)
        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = Tensor<T>.FromVector(outputGradient);

        // Backpropagation
        Backward(outputGradientTensor);

        // Update parameters using the optimizer
        _optimizer.UpdateParameters(Layers);

        IsTrainingMode = false;
    }

    /// <summary>
    /// Retrieves metadata about the Mixture-of-Experts neural network model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method collects and returns various pieces of information about the network's structure,
    /// including MoE-specific details like number of experts and routing strategy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This provides a summary of your MoE network's configuration.
    ///
    /// The metadata includes:
    /// - How many expert networks you have
    /// - How many experts are activated per input (Top-K)
    /// - Expert dimensions and architecture details
    /// - Load balancing settings
    /// - Overall network structure
    ///
    /// This is useful for documentation, debugging, or understanding model differences.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.MixtureOfExperts,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Architecture.GetOutputShape() },
                { "NumExperts", _options.NumExperts },
                { "TopK", _options.TopK },
                { "ExpertInputDim", _options.InputDim },
                { "ExpertOutputDim", _options.OutputDim },
                { "HiddenExpansion", _options.HiddenExpansion },
                { "UseLoadBalancing", _options.UseLoadBalancing },
                { "LoadBalancingWeight", _options.LoadBalancingWeight },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Mixture-of-Experts network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the MoE-specific configuration and state to a binary stream,
    /// allowing the model to be saved and loaded later.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This saves your trained MoE model to a file.
    ///
    /// It records:
    /// - All expert network weights
    /// - Gating network weights
    /// - Configuration settings
    /// - Optimizer and loss function types
    ///
    /// This allows you to:
    /// - Save a trained model for later use
    /// - Share models with others
    /// - Deploy models to production
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write MoE options
        writer.Write(_options.NumExperts);
        writer.Write(_options.TopK);
        writer.Write(_options.InputDim);
        writer.Write(_options.OutputDim);
        writer.Write(_options.HiddenExpansion);
        writer.Write(_options.UseLoadBalancing);
        writer.Write(_options.LoadBalancingWeight);

        // Write optimizer type
        writer.Write(_optimizer.GetType().FullName ?? "AdamOptimizer");

        // Write loss function type
        writer.Write(_lossFunction.GetType().FullName ?? "MeanSquaredErrorLoss");
    }

    /// <summary>
    /// Deserializes Mixture-of-Experts network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the MoE-specific configuration and state from a binary stream,
    /// restoring a previously saved model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This loads a previously saved MoE model from a file.
    ///
    /// It restores:
    /// - All expert network weights
    /// - Gating network weights
    /// - Configuration settings
    /// - Optimizer and loss function types
    ///
    /// The loaded model is ready to use for predictions without retraining.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read MoE options
        _options.NumExperts = reader.ReadInt32();
        _options.TopK = reader.ReadInt32();
        _options.InputDim = reader.ReadInt32();
        _options.OutputDim = reader.ReadInt32();
        _options.HiddenExpansion = reader.ReadInt32();
        _options.UseLoadBalancing = reader.ReadBoolean();
        _options.LoadBalancingWeight = reader.ReadDouble();

        // Read optimizer type (not used after reading)
        reader.ReadString();

        // Read loss function type (not used after reading)
        reader.ReadString();
    }

    /// <summary>
    /// Creates a new instance of the MixtureOfExpertsNeuralNetwork with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new MixtureOfExpertsNeuralNetwork instance with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance with the same architecture, options, optimizer, and loss function
    /// as the current instance. This is useful for model cloning, ensemble methods, or cross-validation scenarios.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a fresh copy of your MoE network's blueprint.
    ///
    /// The new network:
    /// - Has the same number and configuration of experts
    /// - Uses the same routing strategy (Top-K)
    /// - Has the same load balancing settings
    /// - BUT has newly initialized weights (no learned data)
    ///
    /// Use cases:
    /// - Testing the same model architecture on different data
    /// - Creating ensemble models (multiple models voting on predictions)
    /// - Cross-validation (training and testing on different data splits)
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create a clone of the options to ensure the new instance has independent configuration
        var clonedOptions = new MixtureOfExpertsOptions<T>
        {
            NumExperts = _options.NumExperts,
            TopK = _options.TopK,
            InputDim = _options.InputDim,
            OutputDim = _options.OutputDim,
            HiddenExpansion = _options.HiddenExpansion,
            UseLoadBalancing = _options.UseLoadBalancing,
            LoadBalancingWeight = _options.LoadBalancingWeight,
            RandomSeed = _options.RandomSeed
        };

        // Pass null for optimizer to create a fresh optimizer instance for the clone
        return new MixtureOfExpertsNeuralNetwork<T>(
            clonedOptions,
            Architecture,
            null,  // Let constructor create new optimizer
            _lossFunction,
            Convert.ToDouble(MaxGradNorm));
    }

    /// <summary>
    /// Indicates whether this network supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mixture-of-Experts networks fully support training through backpropagation,
    /// including gradient flow to both the expert networks and the gating network.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This indicates that the MoE network can learn from data.
    ///
    /// The network learns:
    /// - How to route inputs to appropriate experts (gating network)
    /// - How each expert should process its specialized inputs (expert networks)
    /// - How to balance usage across all experts (load balancing)
    ///
    /// This property returns true, meaning the network is designed to be trained.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;
}
