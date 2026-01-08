namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Radial Basis Function Network, which is a type of neural network that uses radial basis functions as activation functions.
/// </summary>
/// <remarks>
/// <para>
/// A Radial Basis Function Network (RBFN) is a specialized type of neural network that uses radial basis functions as
/// activation functions. Unlike traditional neural networks, RBFNs typically have only one hidden layer with a non-linear
/// radial basis function, followed by a linear output layer. This architecture makes them particularly effective for
/// function approximation, classification, and systems control problems.
/// </para>
/// <para><b>For Beginners:</b> A Radial Basis Function Network is a special type of neural network designed to learn patterns differently than standard networks.
/// 
/// Think of it like a weather prediction system:
/// - Traditional neural networks might look at many factors and gradually learn weather patterns
/// - An RBFN instead has "weather experts" (the radial basis functions) who each specialize in a specific weather pattern
/// - When new data comes in, each expert reports how similar the current conditions are to their specialty
/// - The network combines these similarity reports to make a prediction
/// 
/// For example, one expert might specialize in "sunny summer days," another in "rainy spring mornings," and so on.
/// When given new weather data, they each report how close it matches their expertise, and the network uses these
/// reports to predict the weather.
/// 
/// RBFNs are particularly good at:
/// - Learning complex patterns quickly
/// - Function approximation (finding the shape of unknown mathematical functions)
/// - Classification problems (determining which category something belongs to)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RadialBasisFunctionNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the size of the input layer (number of input features).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The input size determines the dimensionality of the input space that the network can process.
    /// It represents the number of features or variables that the network takes as input.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many different measurements or features the network can accept.
    /// 
    /// For example, if predicting house prices:
    /// - InputSize = 3 means the network looks at 3 features (like square footage, number of bedrooms, and age)
    /// - InputSize = 10 would mean the network considers 10 different features about each house
    /// 
    /// Think of it as the number of different pieces of information the network considers when making predictions.
    /// </para>
    /// </remarks>
    private int _inputSize { get; set; }

    /// <summary>
    /// Gets or sets the size of the hidden layer (number of radial basis functions).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The hidden size determines the number of radial basis functions (RBFs) in the hidden layer.
    /// Each RBF acts as a localized receptor that responds to a specific region of the input space.
    /// A larger hidden size allows the network to model more complex functions but increases computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many "experts" or "pattern recognizers" the network has.
    /// 
    /// Continuing our weather prediction example:
    /// - HiddenSize = 5 means the network has 5 different weather experts
    /// - HiddenSize = 100 means it has 100 experts, each specializing in a more specific pattern
    /// 
    /// More experts (larger hidden size) means:
    /// - The network can recognize more detailed patterns
    /// - It can make more precise predictions
    /// - But it requires more computational power and may risk "memorizing" rather than "learning"
    /// </para>
    /// </remarks>
    private int _hiddenSize { get; set; }

    /// <summary>
    /// Gets or sets the size of the output layer (number of output values).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The output size determines the dimensionality of the output space that the network produces.
    /// It represents the number of values or categories that the network predicts as output.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many different values or answers the network gives.
    /// 
    /// For example:
    /// - OutputSize = 1 might mean a single prediction (like a house price)
    /// - OutputSize = 3 might mean predicting three different values (like temperature, humidity, and wind speed)
    /// - OutputSize = 10 might mean classifying into ten categories (like digits 0-9)
    /// 
    /// Think of it as the number of answers the network provides for each input it receives.
    /// </para>
    /// </remarks>
    private int _outputSize { get; set; }

    /// <summary>
    /// Gets or sets the radial basis function used in the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The radial basis function defines how the hidden layer neurons respond to input data.
    /// It typically measures the distance between the input and a center point, producing higher output
    /// for inputs closer to the center. Different types of radial basis functions (like Gaussian,
    /// multiquadric, or inverse quadratic) have different mathematical properties and can be suitable
    /// for different types of problems.
    /// </para>
    /// <para><b>For Beginners:</b> This is the mathematical formula that each "expert" uses to determine similarity.
    /// 
    /// Think of it as the method each expert uses to decide how closely the current situation matches their specialty:
    /// - A Gaussian RBF (the most common) works like a "bell curve" - giving high similarity for very close matches, 
    ///   with similarity dropping off quickly as differences increase
    /// - Other types of RBFs might drop off more slowly or have different shapes
    /// 
    /// For example, with a Gaussian RBF, an expert in "sunny summer days" would give:
    /// - High similarity for hot, clear days in July
    /// - Medium similarity for warm, partly cloudy days in June
    /// - Very low similarity for cold, snowy days in January
    /// </para>
    /// </remarks>
    private IRadialBasisFunction<T> _radialBasisFunction { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="RadialBasisFunctionNetwork{T}"/> class with the specified architecture and radial basis function.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the RBFN.</param>
    /// <param name="radialBasisFunction">The radial basis function to use. If null, a Gaussian RBF is used by default.</param>
    /// <exception cref="ArgumentException">Thrown when the architecture does not provide valid input shape or output size.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Radial Basis Function Network with the specified architecture and radial basis function.
    /// It extracts necessary parameters from the architecture, such as input shape, output size, and hidden layer size.
    /// If the architecture does not explicitly define a hidden layer size, a default of 64 is used. The default radial
    /// basis function is Gaussian if none is provided.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Radial Basis Function Network with its basic components.
    /// 
    /// When creating a new RBFN:
    /// - architecture: Defines the overall structure of the neural network
    /// - radialBasisFunction: Sets the type of similarity formula the experts use (default is Gaussian)
    /// 
    /// The constructor figures out:
    /// - How many inputs the network will accept (_inputSize)
    /// - How many experts to create (_hiddenSize, default is 64 if not specified)
    /// - How many outputs the network will produce (_outputSize)
    /// 
    /// Then it initializes the layers of the network accordingly.
    /// </para>
    /// </remarks>
    public RadialBasisFunctionNetwork(NeuralNetworkArchitecture<T> architecture, IRadialBasisFunction<T>? radialBasisFunction = null, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        // Get the input shape and output size from the architecture
        var inputShape = architecture.GetInputShape();
        int outputSize = architecture.OutputSize;

        // For RBF networks, we need to determine the hidden layer size
        // If the architecture has custom layers defined, we can extract it from there
        // Otherwise, we'll use a default or specified value
        int hiddenSize;
        if (architecture.Layers != null && architecture.Layers.Count >= 2)
        {
            // Extract hidden size from the architecture's layers if available
            hiddenSize = architecture.Layers[1].GetOutputShape()[0];
        }
        else
        {
            hiddenSize = 64; // Default to 64 if not specified
        }

        // Validate the network structure
        if (inputShape == null || inputShape.Length == 0 || outputSize <= 0)
        {
            throw new ArgumentException("RBFN requires valid input shape and output size specifications.");
        }

        // Set the properties
        _inputSize = inputShape[0];
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;

        // Default to Gaussian RBF if not specified
        _radialBasisFunction = radialBasisFunction ?? new GaussianRBF<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Radial Basis Function Network. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration optimized for RBF networks
    /// is created. In a typical RBFN, this involves creating a hidden layer with radial basis functions and
    /// an output layer with linear activation.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for RBF networks are created automatically
    /// - The system checks that any custom layers will work properly with the RBFN
    /// 
    /// A typical RBFN has just two main layers:
    /// 1. A hidden layer with radial basis functions (our "experts")
    /// 2. An output layer that combines the experts' outputs (usually with a simple linear function)
    /// 
    /// This simple structure is what makes RBFNs both powerful and efficient for certain types of problems.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultRBFNetworkLayers(Architecture));
        }
    }

    /// <summary>
    /// Updates the parameters of the radial basis function network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the radial basis function network based on the provided parameter
    /// updates. The parameters vector is divided into segments corresponding to each layer's parameter count,
    /// and each segment is applied to its respective layer. In an RBFN, these parameters typically include the centers
    /// and widths of the RBFs in the hidden layer, and the weights in the output layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the RBFN makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// For an RBFN, these adjustments might include:
    /// - Updating what patterns each "expert" specializes in (their centers)
    /// - Changing how strictly each expert defines "similarity" (their widths)
    /// - Adjusting how much influence each expert has on the final prediction (output weights)
    /// 
    /// This process allows the RBFN to improve its performance over time by refining its experts
    /// and how it combines their opinions.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the current state of the Radial Basis Function Network.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor after passing through all layers of the network.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the input tensor is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the input tensor has incorrect dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network, transforming the input data through each layer
    /// to produce a final prediction. It includes input validation to ensure the provided tensor matches the
    /// expected input shape of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the network makes predictions based on new data.
    /// 
    /// The prediction process:
    /// 1. Checks if the input data is valid and has the correct shape
    /// 2. Passes the input through each layer of the network
    /// 3. Each layer transforms the data in some way
    /// 4. The final layer produces the network's prediction
    /// 
    /// Think of it like a factory assembly line:
    /// - The input data enters the first station (layer)
    /// - Each station processes the data and passes it to the next
    /// - The last station outputs the final product (prediction)
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Validate input
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input tensor cannot be null.");
        }

        var inputShape = input.Shape;
        var expectedShape = Architecture.GetInputShape();

        // Ensure input has correct shape
        if (inputShape.Length != expectedShape.Length)
        {
            throw new ArgumentException($"Input tensor has wrong number of dimensions. Expected {expectedShape.Length}, got {inputShape.Length}.");
        }

        // Check if dimensions match
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (inputShape[i] != expectedShape[i])
            {
                throw new ArgumentException($"Input dimension mismatch at index {i}. Expected {expectedShape[i]}, got {inputShape[i]}.");
            }
        }

        // Forward pass through each layer in the network
        Tensor<T> currentOutput = input;
        foreach (var layer in Layers)
        {
            currentOutput = layer.Forward(currentOutput);
        }

        return currentOutput;
    }

    /// <summary>
    /// Trains the Radial Basis Function Network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor used for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <exception cref="ArgumentNullException">Thrown when either input or expectedOutput is null.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the training process for the RBFN. It performs a forward pass, calculates the error
    /// between the network's prediction and the expected output, and then backpropagates this error to adjust
    /// the network's parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the network learns from examples.
    /// 
    /// The training process:
    /// 1. Takes an input and its correct answer (expected output)
    /// 2. Makes a prediction using the current network state
    /// 3. Compares the prediction to the correct answer to calculate the error
    /// 4. Uses this error to adjust the network's internal settings (backpropagation)
    /// 
    /// It's like a student solving math problems:
    /// - The student (network) tries to solve a problem (make a prediction)
    /// - They check their answer against the correct one
    /// - If they're wrong, they figure out why and adjust their approach
    /// - Over time, they get better at solving similar problems
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input tensor cannot be null.");
        }

        if (expectedOutput == null)
        {
            throw new ArgumentNullException(nameof(expectedOutput), "Expected output tensor cannot be null.");
        }

        // Forward pass with memory for backpropagation
        Vector<T> outputVector = ForwardWithMemory(input).ToVector();

        // Calculate error/loss
        Vector<T> expectedOutputVector = expectedOutput.ToVector();
        Vector<T> errorVector = outputVector.Subtract(expectedOutputVector);

        // Calculate and set the loss using the loss function
        LastLoss = LossFunction.CalculateLoss(outputVector, expectedOutputVector);

        // Backpropagate error through the network
        Backpropagate(Tensor<T>.FromVector(errorVector));
    }

    /// <summary>
    /// Gets metadata about the Radial Basis Function Network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata that describes the RBFN, including its type, architecture details,
    /// and other relevant information. This metadata can be useful for model management, documentation,
    /// and versioning.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary of your network's setup and characteristics.
    /// 
    /// The metadata includes:
    /// - The type of model (Radial Basis Function Network)
    /// - How many inputs, hidden neurons, and outputs the network has
    /// - What type of radial basis function is being used
    /// - A description of the network's structure
    /// 
    /// This is useful for:
    /// - Keeping track of different versions of your model
    /// - Comparing different network configurations
    /// - Documenting your model's setup for future reference
    /// 
    /// Think of it like a spec sheet for a car, listing all its important features and capabilities.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetworkRegression,
            FeatureCount = _inputSize,
            Description = $"RBFN with {_inputSize} inputs, {_hiddenSize} RBF neurons, and {_outputSize} outputs",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputSize", _inputSize },
                { "HiddenSize", _hiddenSize },
                { "OutputSize", _outputSize },
                { "RadialBasisFunction", _radialBasisFunction.GetType().Name }
            },
            ModelData = this.Serialize()
        };

        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data for the Radial Basis Function Network.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific configuration and state of the RBFN to a binary stream.
    /// It includes the network's structural parameters and the type of radial basis function used.
    /// This data is essential for later reconstruction of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the unique settings of your RBFN.
    /// 
    /// It writes:
    /// - The number of inputs, hidden neurons, and outputs
    /// - Information about the specific type of radial basis function used
    /// 
    /// Saving these details allows you to recreate the exact same network structure later.
    /// It's like writing down a recipe so you can make the same dish again in the future.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write RBFN-specific data
        writer.Write(_inputSize);
        writer.Write(_hiddenSize);
        writer.Write(_outputSize);

        SerializationHelper<T>.SerializeInterface(writer, _radialBasisFunction);
    }

    /// <summary>
    /// Deserializes network-specific data for the Radial Basis Function Network.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the specific configuration and state of the RBFN from a binary stream.
    /// It reconstructs the network's structural parameters and radial basis function type to match
    /// the state of the network when it was serialized.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the unique settings of your RBFN.
    /// 
    /// It reads:
    /// - The number of inputs, hidden neurons, and outputs
    /// - Information about the specific type of radial basis function used
    /// 
    /// Loading these details allows you to recreate the exact same network structure that was previously saved.
    /// It's like following a recipe to recreate a dish exactly as it was made before.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read RBFN-specific data
        _inputSize = reader.ReadInt32();
        _hiddenSize = reader.ReadInt32();
        _outputSize = reader.ReadInt32();

        // Read and set the radial basis function if a custom one was used
        _radialBasisFunction = DeserializationHelper.DeserializeInterface<IRadialBasisFunction<T>>(reader) ?? new GaussianRBF<T>();
    }

    /// <summary>
    /// Creates a new instance of the radial basis function network with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="RadialBasisFunctionNetwork{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new radial basis function network that has the same configuration as the current instance.
    /// It's used for model persistence, cloning, and transferring the model's configuration to new instances.
    /// The new instance will have the same architecture and radial basis function type as the original,
    /// but will not share parameter values unless they are explicitly copied after creation.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like creating a blueprint copy of your network that can be used to:
    /// - Save your model's settings
    /// - Create a new identical model
    /// - Transfer your model's configuration to another system
    /// 
    /// This is useful when you want to:
    /// - Create multiple similar radial basis function networks
    /// - Save a model's configuration for later use
    /// - Reset a model while keeping its settings
    /// 
    /// Note that while the settings are copied, the learned parameters (like the centers of the "experts" 
    /// and the output weights) are not automatically transferred, so the new instance will need training 
    /// or parameter copying to match the performance of the original.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create a new instance with the cloned architecture and RBF
        return new RadialBasisFunctionNetwork<T>(Architecture, _radialBasisFunction, LossFunction);
    }
}
