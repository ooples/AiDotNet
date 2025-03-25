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
    public RadialBasisFunctionNetwork(NeuralNetworkArchitecture<T> architecture, IRadialBasisFunction<T>? radialBasisFunction = null) : base(architecture)
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
        _inputSize = inputShape[0]; // Assuming 1D input for simplicity
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
    /// Processes the input through the radial basis function network to produce a prediction.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector after processing through the network.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the Radial Basis Function Network. It processes the input
    /// through each layer of the network in sequence, transforming it according to the operations defined
    /// in each layer. For an RBFN, this typically involves measuring the similarity of the input to each
    /// RBF center in the hidden layer, and then linearly combining these similarities in the output layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method is how the RBFN processes information and makes predictions.
    /// 
    /// During the prediction process:
    /// - The input data enters the network
    /// - In the hidden layer, each "expert" (RBF) measures how similar the input is to their specialty
    /// - In the output layer, these similarity scores are combined using weights to produce the final prediction
    /// 
    /// For example, if predicting tomorrow's temperature:
    /// 1. Input data about today's weather is fed into the network
    /// 2. Each expert reports a similarity score (how similar is today to the patterns they recognize?)
    /// 3. The output layer combines these scores, giving more weight to experts who have been reliable in the past
    /// 4. The result is the predicted temperature
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
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
    /// Saves the state of the Radial Basis Function Network to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to save the state to.</param>
    /// <exception cref="ArgumentNullException">Thrown if the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer serialization fails or if the RBF function type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the entire state of the Radial Basis Function Network, including all layers and the
    /// type of radial basis function used. It writes the number of layers, the type and state of each layer, and
    /// the type of the radial basis function to the provided binary writer.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the entire state of the RBFN to a file.
    /// 
    /// When serializing:
    /// - All the network's layers are saved (their types and internal values)
    /// - The type of similarity function (RBF) being used is saved
    /// - The saved file can later be used to restore the exact same network state
    /// 
    /// This is useful for:
    /// - Saving a trained model to use later
    /// - Sharing a model with others
    /// - Creating backups during long training processes
    /// 
    /// Think of it like taking a complete snapshot of the network that can be restored later.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Encountered a null layer during serialization.");

            string? fullName = layer.GetType().FullName;
            if (string.IsNullOrEmpty(fullName))
                throw new InvalidOperationException($"Unable to get full name for layer type {layer.GetType()}");

            writer.Write(fullName);
            layer.Serialize(writer);
        }

        // Serialize the RBF function type
        writer.Write(_radialBasisFunction.GetType().FullName ?? throw new InvalidOperationException("Unable to get full name for RBF function type"));
    }

    /// <summary>
    /// Loads the state of the Radial Basis Function Network from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to load the state from.</param>
    /// <exception cref="ArgumentNullException">Thrown if the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown if layer deserialization fails or if the RBF function type cannot be instantiated.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the state of the Radial Basis Function Network from a binary reader. It reads
    /// the number of layers, recreates each layer based on its type, deserializes the layer state, and finally
    /// recreates the radial basis function based on its type.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved RBFN state from a file.
    /// 
    /// When deserializing:
    /// - The number and types of layers are read from the file
    /// - Each layer is recreated and its state is restored
    /// - The type of similarity function (RBF) is read and recreated
    /// 
    /// This allows you to:
    /// - Load a previously trained model
    /// - Continue using or training a model from where you left off
    /// - Use models created by others
    /// 
    /// Think of it like restoring a complete snapshot of the network that was saved earlier.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            if (string.IsNullOrEmpty(layerTypeName))
                throw new InvalidOperationException("Encountered an empty layer type name during deserialization.");

            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");

            if (!typeof(ILayer<T>).IsAssignableFrom(layerType))
                throw new InvalidOperationException($"Type {layerTypeName} does not implement ILayer<T>");

            object? instance = Activator.CreateInstance(layerType);
            if (instance == null)
                throw new InvalidOperationException($"Failed to create an instance of {layerTypeName}");

            var layer = (ILayer<T>)instance;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }

        // Deserialize the RBF function type
        string rbfTypeName = reader.ReadString();
        Type? rbfType = Type.GetType(rbfTypeName);
        if (rbfType == null)
            throw new InvalidOperationException($"Cannot find type {rbfTypeName}");

        if (!typeof(IRadialBasisFunction<T>).IsAssignableFrom(rbfType))
            throw new InvalidOperationException($"Type {rbfTypeName} does not implement IRadialBasisFunction<T>");

        object? rbfInstance = Activator.CreateInstance(rbfType);
        if (rbfInstance == null)
            throw new InvalidOperationException($"Failed to create an instance of {rbfTypeName}");

        _radialBasisFunction = (IRadialBasisFunction<T>)rbfInstance;
    }
}