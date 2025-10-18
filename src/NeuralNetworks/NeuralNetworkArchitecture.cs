namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Defines the structure and configuration of a neural network, including its layers, input/output dimensions, and task-specific properties.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The NeuralNetworkArchitecture class serves as a blueprint for constructing neural networks with specific configurations.
/// It handles the validation of input dimensions, layer compatibility, and provides methods for retrieving information about
/// the network's structure. This architecture can be used to create various types of neural networks with different input 
/// dimensionalities and layer arrangements.
/// </para>
/// <para><b>For Beginners:</b> Think of NeuralNetworkArchitecture as the blueprint for building a neural network.
/// 
/// Just like an architect's blueprint for a building specifies:
/// - How many floors the building will have
/// - The size and purpose of each room
/// - How rooms connect to each other
/// 
/// The NeuralNetworkArchitecture defines:
/// - What kind of data your network will process (like images or text)
/// - How many layers your network will have
/// - How many neurons are in each layer
/// - How the layers connect to process your data
/// 
/// Before you can build a neural network, you need this blueprint to ensure all the parts
/// will fit together correctly. It helps prevent errors like trying to feed image data into
/// a network designed for text, or having layers that don't match up in size.
/// </para>
/// </remarks>
public class NeuralNetworkArchitecture<T>
{
    /// <summary>
    /// Gets the optional list of predefined layers for the neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property allows you to explicitly define the layers that will make up the neural network.
    /// If not provided, default layers will be created based on other architecture parameters when
    /// the neural network is initialized.
    /// </para>
    /// <para><b>For Beginners:</b> This is where you can specify exact layers for your network.
    /// 
    /// Think of this as customizing the rooms in your building:
    /// - Instead of using standard room designs, you specify exactly what you want
    /// - You control the exact size, type, and connections of each layer
    /// - This gives you precise control over how your network processes data
    /// 
    /// If you leave this empty (null), the system will create standard layers
    /// based on your other settings, like creating standard rooms based on
    /// the overall building design.
    /// </para>
    /// </remarks>
    public List<ILayer<T>>? Layers { get; }

    /// <summary>
    /// Our version of lazy initialization to be used with setting architecture settings after they are cached
    /// </summary>
    public bool IsInitialized { get; private set; }

    /// <summary>
    /// Gets a value indicating whether the input height will be determined at runtime.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When this property is true, it indicates that the input height (number of samples)
    /// will be determined at runtime rather than being fixed during architecture initialization.
    /// This is common for classification and regression tasks where the number of samples can vary.
    /// </para>
    /// <para><b>For Beginners:</b> This indicates that your network can handle any number of examples.
    /// 
    /// For normal machine learning workflows, you don't know ahead of time exactly how many
    /// examples (samples) your network will process. This property tells the system that
    /// the number of samples will be provided when actually using the network, not when
    /// designing it.
    /// 
    /// Think of it like a recipe that says "serves 4" vs "serves any number of people".
    /// This property, when true, indicates your network can adapt to any number of input samples.
    /// </para>
    /// </remarks>
    public bool IsDynamicSampleCount { get; }

    /// <summary>
    /// Gets the type of input the neural network is designed to handle.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property defines the dimensionality of the input data that the neural network is designed to process.
    /// Options include OneDimensional (for vector data), TwoDimensional (for matrix data like images),
    /// and ThreeDimensional (for volumetric data like color images or video).
    /// </para>
    /// <para><b>For Beginners:</b> This specifies what shape of data your network will process.
    /// 
    /// Neural networks can handle different types of data:
    /// - OneDimensional: Simple lists of values (like a customer's age, income, etc.)
    /// - TwoDimensional: Grid-like data (like a grayscale image)
    /// - ThreeDimensional: Cube-like data (like a color image with red, green, blue channels)
    /// 
    /// This is important because the network needs to know how to interpret the input.
    /// For example, in a color image, pixels that are next to each other horizontally,
    /// vertically, or across color channels have different kinds of relationships.
    /// </para>
    /// </remarks>
    public InputType InputType { get; private set; }

    /// <summary>
    /// Gets or sets the size of the input vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For one-dimensional inputs, this specifies the number of input features.
    /// For multi-dimensional inputs, this represents the total number of input elements
    /// (calculated as InputHeight * InputWidth * InputDepth).
    /// </para>
    /// <para><b>For Beginners:</b> This is the total number of input values your network receives.
    /// 
    /// For example:
    /// - For a list of 10 customer attributes: InputSize = 10
    /// - For a 28×28 grayscale image: InputSize = 784 (28×28)
    /// - For a 32×32 color image: InputSize = 3072 (32×32×3 color channels)
    /// 
    /// This tells the network how many "inputs" to expect. Think of it like how many
    /// separate pieces of information your network will consider at once.
    /// </para>
    /// </remarks>
    public int InputSize { get; private set; }

    /// <summary>
    /// Gets the size of the output vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property specifies the dimensionality of the network's output. It represents the number of
    /// output neurons in the final layer. For classification tasks, this typically equals the number of classes.
    /// For regression tasks, this equals the number of values to predict.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many values your network will output.
    /// 
    /// For example:
    /// - For classifying 10 digits (0-9): OutputSize = 10
    /// - For predicting a single value (like house price): OutputSize = 1
    /// - For predicting multiple values (like x,y coordinates): OutputSize = 2
    /// 
    /// Think of this as how many answers your network gives at once.
    /// </para>
    /// </remarks>
    public int OutputSize { get; private set; }

    /// <summary>
    /// Gets the height dimension for 2D or 3D inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For two-dimensional or three-dimensional inputs, this property specifies the height of the input.
    /// For example, for image data, this would be the height in pixels.
    /// </para>
    /// <para><b>For Beginners:</b> For grid-like data (like images), this is the number of rows.
    /// 
    /// For example:
    /// - For a 28×28 image: InputHeight = 28
    /// 
    /// This is only used when working with multi-dimensional data like images.
    /// For simple lists of values, you'd use InputSize instead.
    /// </para>
    /// </remarks>
    public int InputHeight { get; private set; }

    /// <summary>
    /// Gets the width dimension for 2D or 3D inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For two-dimensional or three-dimensional inputs, this property specifies the width of the input.
    /// For example, for image data, this would be the width in pixels.
    /// </para>
    /// <para><b>For Beginners:</b> For grid-like data (like images), this is the number of columns.
    /// 
    /// For example:
    /// - For a 28×28 image: InputWidth = 28
    /// 
    /// This is only used when working with multi-dimensional data like images.
    /// For simple lists of values, you'd use InputSize instead.
    /// </para>
    /// </remarks>
    public int InputWidth { get; private set; }

    /// <summary>
    /// Gets the depth dimension for 3D inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For three-dimensional inputs, this property specifies the depth of the input.
    /// For example, for color image data, this would typically be 3 (for RGB channels).
    /// </para>
    /// <para><b>For Beginners:</b> For 3D data (like color images), this is the number of channels or layers.
    /// 
    /// For example:
    /// - For a color RGB image: InputDepth = 3 (red, green, blue channels)
    /// - For a grayscale image: InputDepth = 1
    /// 
    /// This is only used when working with three-dimensional data.
    /// For simpler data types, it's usually set to 1 and doesn't affect the network.
    /// </para>
    /// </remarks>
    public int InputDepth { get; private set; }

    /// <summary>
    /// Gets the type of task the neural network is designed to perform.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property specifies the type of task that the neural network is intended to solve,
    /// such as classification (assigning inputs to discrete categories) or regression (predicting continuous values).
    /// The task type affects the default network configuration, particularly the output layer and activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This defines what kind of problem your network is solving.
    /// 
    /// Common task types include:
    /// - Classification: Sorting inputs into categories (like "dog" or "cat" for images)
    /// - Regression: Predicting a number value (like house prices)
    /// - Sequence Generation: Creating sequences (like text or music)
    /// 
    /// The task type helps determine how your network should be structured and trained.
    /// For example, a classification network typically ends with a Softmax activation
    /// to output probabilities for each category, while a regression network
    /// might end with a linear activation to output any numerical value.
    /// </para>
    /// </remarks>
    public NeuralNetworkTaskType TaskType { get; private set; }

    /// <summary>
    /// Gets the complexity level of the neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property defines the general complexity of the network architecture, which affects the default number of layers
    /// and neurons when automatically generating the network structure. Options typically include Simple, Medium, and Complex.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how powerful and complex your network will be.
    /// 
    /// Think of this like choosing a building size:
    /// - Simple: A small network with few layers and neurons (fast but less powerful)
    /// - Medium: A balanced network (good for many common tasks)
    /// - Complex: A large network with many layers and neurons (powerful but slower to train)
    /// 
    /// Simpler networks train faster and need less data, but may not learn very complex patterns.
    /// Complex networks can learn sophisticated patterns but need more data and computing power.
    /// 
    /// When starting out, Medium complexity is often a good choice.
    /// </para>
    /// </remarks>
    public NetworkComplexity Complexity { get; }


    /// <summary>
    /// Gets the dimensionality of the input (1, 2, or 3).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This computed property returns the number of dimensions in the input data based on the InputType.
    /// It returns 1 for OneDimensional, 2 for TwoDimensional, and 3 for ThreeDimensional.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many dimensions your input data has.
    /// 
    /// It's calculated automatically based on your InputType:
    /// - 1: For simple lists of values (like customer attributes)
    /// - 2: For grid data (like grayscale images)
    /// - 3: For volumetric data (like color images)
    /// 
    /// This information helps the network properly process the structure of your data.
    /// </para>
    /// </remarks>
    public int InputDimension => 
        InputType == InputType.OneDimensional ? 1 :
        InputType == InputType.TwoDimensional ? 2 : 3;

    /// <summary>
    /// Gets the calculated total size of the input based on dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This computed property calculates the total number of input elements based on the input dimensions.
    /// For one-dimensional inputs, it returns InputSize. For multi-dimensional inputs, it calculates
    /// the product of the dimensions (InputHeight * InputWidth * InputDepth).
    /// </para>
    /// <para><b>For Beginners:</b> This calculates the total number of input values based on your dimensions.
    /// 
    /// It's automatically calculated depending on your input type:
    /// - For 1D data: Just returns your InputSize
    /// - For 2D data: Calculates InputHeight × InputWidth
    /// - For 3D data: Calculates InputHeight × InputWidth × InputDepth
    /// 
    /// For example, a 28×28 image has 784 total pixels, so CalculatedInputSize would be 784.
    /// 
    /// This helps ensure all your dimension settings are consistent with each other.
    /// </para>
    /// </remarks>
    public int CalculatedInputSize =>
        InputType switch
        {
            InputType.OneDimensional => InputSize > 0 ? InputSize : throw new InvalidOperationException("InputSize must be set for OneDimensional input."),
            InputType.TwoDimensional => IsDynamicSampleCount && InputHeight == 0 ? InputWidth : InputHeight * InputWidth,
            InputType.ThreeDimensional => InputHeight * InputWidth * InputDepth,
            _ => throw new InvalidOperationException("Invalid InputDimensionality"),
        };

    /// <summary>
    /// Determines whether the network should return the full sequence or just the final output.
    /// </summary>
    /// <returns>True if full sequence should be returned; otherwise, false.</returns>
    public bool ShouldReturnFullSequence { get; }

    /// <summary>
    /// A cache name to be used to look for a custom input cache
    /// </summary>
    public string CacheName { get; private set; }

    /// <summary>
    /// Gets or sets the loss function to be used for training this neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The loss function measures how far the network's predictions are from the actual values.
    /// Different tasks require different loss functions (e.g., cross-entropy for classification,
    /// mean squared error for regression).
    /// </para>
    /// <para><b>For Beginners:</b> The loss function tells the network how wrong it is.
    /// Think of it like a scoring system that measures the difference between what the network
    /// predicted and what the answer should be. The network tries to minimize this "wrongness"
    /// during training.
    /// </para>
    /// </remarks>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer used to update the network's parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The optimizer determines how the network's weights are updated based on the gradients
    /// computed from the loss function. Common optimizers include SGD, Adam, and RMSProp.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer is the strategy used to improve the network.
    /// After measuring how wrong the network is (using the loss function), the optimizer
    /// decides exactly how to adjust the network's parameters to make it more accurate.
    /// Different optimizers use different strategies to find the best adjustments.
    /// </para>
    /// </remarks>
    public IOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer { get; set; }

    /// <summary>
    /// Gets a value indicating whether this architecture instance is a placeholder.
    /// </summary>
    /// <value>
    /// <c>true</c> if this instance is a placeholder with minimal validation; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// A placeholder architecture is a special type of neural network architecture that bypasses
    /// strict dimension validation. It's primarily used for serialization, default model creation,
    /// or when an architecture needs to be initialized before all dimensions are known.
    /// </para>
    /// <para>
    /// Placeholder architectures typically have zero or undefined dimensions and are expected to
    /// be properly configured before actual training begins. They allow for model templates to be
    /// created and stored without requiring complete specifications.
    /// </para>
    /// <para><b>For Beginners:</b> Think of a placeholder as an "incomplete blueprint" for your neural network.
    /// 
    /// A placeholder is useful when:
    /// - You need to create a default or empty model
    /// - You want to save a model structure without all the details filled in
    /// - You're creating a model programmatically and will set the dimensions later
    /// 
    /// When IsPlaceholder is true, the system won't complain about missing dimensions
    /// or other validation issues that would normally cause errors. This gives you more
    /// flexibility, but you'll need to ensure the model is properly configured before
    /// using it for actual training.
    /// </para>
    /// </remarks>
    public bool IsPlaceholder { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNetworkArchitecture{T}"/> class with the specified parameters.
    /// </summary>
    /// <param name="inputType">The type of input data (one-dimensional, two-dimensional, or three-dimensional).</param>
    /// <param name="taskType">The type of task the neural network will perform (classification, regression, etc.).</param>
    /// <param name="complexity">The complexity level of the neural network. Default is Medium.</param>
    /// <param name="layers">Optional predefined layers for the neural network. Default is null.</param>
    /// <param name="shouldReturnFullSequence">Indicates whether sequence models should return full sequence or just the final output. Default is false.</param>
    /// <param name="isDynamicSampleCount">Indicates whether the network can handle varying sample counts. Default is false.</param>
    /// <param name="isPlaceholder">Disables validation and should only be used in special circumstances like when a default model value is necessary. Default is false.</param>
    /// <exception cref="ArgumentException">Thrown when the input dimensions are invalid or inconsistent, unless the architecture is created as a placeholder.</exception>
    /// <remarks>
    /// <para>
    /// This constructor initializes a neural network architecture with the specified parameters and validates
    /// that the input dimensions are consistent and appropriate for the selected input type. It also checks
    /// that any provided layers are compatible with the input and output dimensions.
    /// </para>
    /// <para>
    /// For placeholder models (where all dimensions are zero), validation will be more lenient to allow
    /// for serialization and default model creation. Full validation occurs during actual model training.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the blueprint for your neural network with your chosen settings.
    /// 
    /// When creating a neural network architecture, you specify:
    /// 
    /// 1. What kind of input data you're using:
    ///    - One-dimensional for lists of values
    ///    - Two-dimensional for grid data like grayscale images
    ///    - Three-dimensional for volumetric data like color images
    /// 
    /// 2. What task you're solving:
    ///    - Classification (sorting into categories)
    ///    - Regression (predicting numerical values)
    ///    - Other specialized tasks
    /// 
    /// 3. Other settings like:
    ///    - Complexity (how powerful the network should be)
    ///    - Input dimensions (size, height, width, depth)
    ///    - Output size (how many values to predict)
    ///    - Optional custom layers
    ///    - Whether to return full sequences (for time series or text data)
    ///    - Whether the network can handle varying batch sizes
    /// 
    /// The constructor checks that all your settings make sense together,
    /// unless you're creating a placeholder model where zero dimensions are allowed.
    /// </para>
    /// </remarks>
    public NeuralNetworkArchitecture(
        NetworkComplexity complexity = NetworkComplexity.Medium,
        NeuralNetworkTaskType? taskType = null,
        bool shouldReturnFullSequence = false,
        List<ILayer<T>>? layers = null,
        bool isDynamicSampleCount = true,
        bool isPlaceholder = false,
        string? cacheName = null)
    {
        // Set only the parameters that don't depend on data dimensions
        Complexity = complexity;
        ShouldReturnFullSequence = shouldReturnFullSequence;
        Layers = layers;
        IsInitialized = false;  // Mark this as a placeholder until dimensions are set

        // Set other properties to default/placeholder values
        InputType = InputType.OneDimensional;  // Will be updated later
        TaskType = taskType ?? NeuralNetworkTaskType.Custom;  // Default to regression if not specified
        InputSize = 0;
        InputHeight = 0;
        InputWidth = 0;
        InputDepth = 1;
        OutputSize = 0;
        IsDynamicSampleCount = true;
        IsPlaceholder = isPlaceholder;  // Allow placeholder behavior
        IsDynamicSampleCount = isDynamicSampleCount;  // Allow dynamic sample count
        CacheName = cacheName ?? string.Empty;  // Default to empty string if not provided
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNetworkArchitecture{T}"/> class for regression tasks.
    /// </summary>
    /// <param name="inputFeatures">The number of input features (columns in the input matrix).</param>
    /// <param name="outputSize">The number of output values to predict (typically 1 for simple regression).</param>
    /// <param name="complexity">The complexity level of the neural network. Default is Medium.</param>
    /// <remarks>
    /// <para>
    /// This constructor provides a simplified way to create a neural network architecture specifically for regression tasks.
    /// It automatically sets the appropriate input type to TwoDimensional (for a matrix of samples × features) and
    /// sets the task type to Regression.
    /// </para>
    /// <para><b>For Beginners:</b> This is a simpler way to create a neural network for predicting numerical values.
    /// 
    /// Use this constructor when:
    /// - Your input is a matrix where each row is a sample and each column is a feature
    ///   (like house size, number of bedrooms, etc. for multiple houses)
    /// - You want to predict one or more numerical values (like house prices)
    /// 
    /// For example, to create a network that predicts house prices based on 5 features:
    /// 
    /// ```csharp
    /// var architecture = new NeuralNetworkArchitecture<double>(
    ///     inputFeatures: 5,  // 5 features (size, bedrooms, etc.)
    ///     outputSize: 1      // 1 output (price)
    /// );
    /// ```
    /// 
    /// The network will automatically create appropriate layers based on the complexity setting.
    /// You don't need to specify the number of samples - the network will handle any number of samples.
    /// </para>
    /// </remarks>
    public NeuralNetworkArchitecture(
        int inputFeatures,
        int outputSize,
        NetworkComplexity complexity = NetworkComplexity.Medium)
        : this(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: complexity,
            inputHeight: 0,  // This will be determined by the number of samples at runtime
            inputWidth: inputFeatures,
            outputSize: outputSize)
    {
        // The base constructor handles validation and setup
        // We don't need to create custom layers here as the neural network will create
        // appropriate default layers based on the architecture parameters
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNetworkArchitecture{T}"/> class for classification tasks.
    /// </summary>
    /// <param name="inputFeatures">The number of input features (columns in the input matrix).</param>
    /// <param name="numClasses">The number of classes to classify into.</param>
    /// <param name="isMultiClass">Whether this is a multi-class classification (true) or binary classification (false). Default is true.</param>
    /// <param name="complexity">The complexity level of the neural network. Default is Medium.</param>
    /// <remarks>
    /// <para>
    /// This constructor provides a simplified way to create a neural network architecture specifically for classification tasks.
    /// It automatically sets the appropriate input type to TwoDimensional (for a matrix of samples × features) and
    /// sets the task type to either MultiClassClassification or BinaryClassification based on the isMultiClass parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This is a simpler way to create a neural network for classifying data into categories.
    /// 
    /// Use this constructor when:
    /// - Your input is a matrix where each row is a sample and each column is a feature
    ///   (like petal length, petal width, etc. for multiple flowers)
    /// - You want to classify each sample into one of several categories (like flower species)
    /// 
    /// For example, to create a network that classifies flowers into 3 species based on 4 features:
    /// 
    /// ```csharp
    /// var architecture = new NeuralNetworkArchitecture<double>(
    ///     inputFeatures: 4,  // 4 features (petal length, width, etc.)
    ///     numClasses: 3      // 3 classes (setosa, versicolor, virginica)
    /// );
    /// ```
    /// 
    /// The network will automatically create appropriate layers based on the complexity setting.
    /// You don't need to specify the number of samples - the network will handle any number of samples.
    /// </para>
    /// </remarks>
    public NeuralNetworkArchitecture(
        int inputFeatures,
        int numClasses,
        bool isMultiClass = true,
        NetworkComplexity complexity = NetworkComplexity.Medium)
        : this(
            inputType: InputType.TwoDimensional,
            taskType: isMultiClass ? NeuralNetworkTaskType.MultiClassClassification : NeuralNetworkTaskType.BinaryClassification,
            complexity: complexity,
            inputHeight: 0,  // This will be determined by the number of samples at runtime
            inputWidth: inputFeatures,
            outputSize: isMultiClass ? numClasses : 1)  // For binary classification, we only need 1 output (probability)
    {
        // The base constructor handles validation and setup
    }

    /// <summary>
    /// Gets the sizes of the hidden layers in the neural network.
    /// </summary>
    /// <returns>An array containing the size of each hidden layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates and returns the number of neurons in each hidden layer of the neural network.
    /// A hidden layer is any layer between the input and output layers. If no layers are defined or if there
    /// are fewer than 3 layers (meaning no hidden layers), an empty array is returned.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you how many neurons are in each hidden layer.
    /// 
    /// Hidden layers are the middle layers in your network:
    /// - They sit between the input layer (which receives your data)
    /// - And the output layer (which produces the final prediction)
    /// - They're where most of the pattern recognition happens
    /// 
    /// For example, if your network has structure [784, 128, 64, 10]:
    /// - Input layer: 784 neurons
    /// - First hidden layer: 128 neurons
    /// - Second hidden layer: 64 neurons
    /// - Output layer: 10 neurons
    /// 
    /// This method would return [128, 64], the sizes of just the hidden layers.
    /// 
    /// This is useful for understanding or visualizing your network's structure.
    /// </para>
    /// </remarks>
    public int[] GetHiddenLayerSizes()
    {
        if (Layers == null || Layers.Count <= 1)
        {
            return [];
        }

        var hiddenLayerSizes = new List<int>();
        for (int i = 1; i < Layers.Count - 1; i++)
        {
            var outputShape = Layers[i].GetOutputShape();
            hiddenLayerSizes.Add(outputShape.Aggregate(1, (a, b) => a * b));
        }

        return [.. hiddenLayerSizes];
    }

    /// <summary>
    /// Gets the shape of the input as an array of dimensions.
    /// </summary>
    /// <returns>An array representing the input shape.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the shape of the input data as an array of dimensions. The format depends on the InputType:
    /// - For OneDimensional: [InputSize]
    /// - For TwoDimensional: [InputHeight, InputWidth]
    /// - For ThreeDimensional: [InputDepth, InputHeight, InputWidth]
    /// </para>
    /// <para><b>For Beginners:</b> This tells you the exact shape of your input data.
    /// 
    /// Different types of data have different shapes:
    /// - 1D data: Returns [size] - like [10] for 10 features
    /// - 2D data: Returns [height, width] - like [28, 28] for a 28×28 image
    /// - 3D data: Returns [depth, height, width] - like [3, 32, 32] for a color image
    /// 
    /// This shape information is important when:
    /// - Preparing your data for the network
    /// - Designing compatible layers
    /// - Debugging shape-related errors
    /// 
    /// Many neural network errors happen because data shapes don't match up correctly,
    /// so this method helps ensure your network is properly configured.
    /// </para>
    /// </remarks>
    public int[] GetInputShape()
    {
        return InputType switch
        {
            InputType.OneDimensional => [InputSize],
            InputType.TwoDimensional => [InputHeight, InputWidth],
            InputType.ThreeDimensional => [InputDepth, InputHeight, InputWidth],
            _ => throw new InvalidOperationException("Invalid InputDimensionality"),
        };
    }

    /// <summary>
    /// Gets the shape of the output as an array of dimensions.
    /// </summary>
    /// <returns>An array representing the output shape.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the shape of the output from the neural network. If layers are defined,
    /// it returns the output shape of the final layer. If no layers are defined, it returns the same
    /// shape as the input, since the output would be unchanged.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you the shape of your network's output.
    /// 
    /// For most common networks:
    /// - Classification networks: Returns [number_of_classes]
    /// - Regression networks: Returns [number_of_values_to_predict]
    /// 
    /// For example:
    /// - A network classifying digits 0-9 would have output shape [10]
    /// - A network predicting x,y coordinates would have output shape [2]
    /// 
    /// If you haven't defined any layers yet, this returns the same shape as
    /// your input (since with no layers, input flows straight to output).
    /// 
    /// This helps you understand what shape of data to expect from your network.
    /// </para>
    /// </remarks>
    public int[] GetOutputShape()
    {
        if (Layers == null || Layers.Count == 0)
        {
            return GetInputShape(); // If no layers, output shape is the same as input shape
        }

        return Layers[Layers.Count - 1].GetOutputShape();
    }

    /// <summary>
    /// Calculates the total size of the output.
    /// </summary>
    /// <returns>The total number of output elements.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the total number of elements in the output by multiplying all dimensions
    /// of the output shape. For example, if the output shape is [10, 10], the total output size would be 100.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates the total number of values in your network's output.
    /// 
    /// It multiplies all the dimensions of your output shape:
    /// - For a shape [10] (like 10 classes): Total is 10
    /// - For a shape [5, 5] (like a 5×5 grid): Total is 25
    /// 
    /// Most common networks have simple outputs:
    /// - Classification: Equal to the number of categories
    /// - Regression: Equal to the number of values being predicted
    /// 
    /// But some specialized networks might output matrices or tensors,
    /// in which case this method helps calculate the total number of output values.
    /// </para>
    /// </remarks>
    public int CalculateOutputSize()
    {
        var outputShape = GetOutputShape();
        int result = 1;

        for (int i = 0; i < outputShape.Length; i++)
        {
            result *= outputShape[i];
        }

        return result;
    }

    /// <summary>
    /// Gets the size of each layer in the neural network.
    /// </summary>
    /// <returns>An array containing the size of each layer, including input and output layers.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an array containing the number of neurons in each layer of the neural network,
    /// starting with the input layer and ending with the output layer. The size of each layer is calculated
    /// as the product of all dimensions in the layer's output shape.
    /// </para>
    /// <para><b>For Beginners:</b> This gives you the number of neurons in each layer of your network.
    /// 
    /// It returns a list of sizes for all layers, including:
    /// - The input layer (first value)
    /// - All hidden layers (middle values)
    /// - The output layer (last value)
    /// 
    /// For example, a network for classifying 28×28 images into 10 categories
    /// might return: [784, 128, 64, 10]
    /// - 784: Input layer (28×28 pixels)
    /// - 128: First hidden layer
    /// - 64: Second hidden layer
    /// - 10: Output layer (10 categories)
    /// 
    /// This is useful for visualizing your network structure or debugging
    /// to make sure your layers are the sizes you expect.
    /// </para>
    /// </remarks>
    public int[] GetLayerSizes()
    {
        if (Layers == null || Layers.Count == 0)
        {
            return [CalculatedInputSize];
        }

        var layerSizes = new List<int> { CalculatedInputSize };
        foreach (var layer in Layers)
        {
            layerSizes.Add(layer.GetOutputShape().Aggregate(1, (a, b) => a * b));
        }

        return [.. layerSizes];
    }

    /// <summary>
    /// Validates the input dimensions to ensure they are consistent and appropriate for the selected input type.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when input dimensions are invalid or inconsistent.</exception>
    /// <exception cref="InvalidOperationException">Thrown when an invalid input dimension is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method validates that the input dimensions provided are consistent with the specified InputType.
    /// It checks that appropriate dimensions are provided for each input type and that any calculated sizes
    /// match the explicitly provided InputSize. It also validates that any provided layers are compatible with
    /// the input dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This makes sure all your dimension settings make sense together.
    /// 
    /// This method performs important checks like:
    /// - For 1D data: Ensuring InputSize is provided and positive
    /// - For 2D data: Ensuring InputHeight and InputWidth are positive
    /// - For 3D data: Ensuring InputHeight, InputWidth, and InputDepth are all positive
    /// 
    /// It also checks that if you provide both InputSize and other dimension parameters,
    /// they're consistent with each other. For example, if you set:
    /// - InputSize = 25
    /// - InputHeight = 5
    /// - InputWidth = 5
    /// 
    /// These are consistent because 5×5=25. But if you set InputSize=30, it would
    /// throw an error because 5×5≠30.
    /// 
    /// This prevents many common errors when setting up neural networks.
    /// </para>
    /// </remarks>
    private void ValidateInputDimensions()
    {
        int calculatedSize = InputType switch
        {
            InputType.OneDimensional => InputSize,
            InputType.TwoDimensional => InputHeight * InputWidth,
            InputType.ThreeDimensional => InputHeight * InputWidth * InputDepth,
            _ => throw new InvalidOperationException("Invalid InputDimensionality"),
        };

        switch (InputType)
        {
            case InputType.OneDimensional:
                if (InputSize <= 0)
                {
                    throw new ArgumentException("InputSize must be greater than 0 for OneDimensional input.");
                }
                if (InputHeight != 0 || InputWidth != 0 || InputDepth != 1)
                {
                    throw new ArgumentException("InputHeight, InputWidth, and InputDepth should not be set for OneDimensional input.");
                }
                break;

            case InputType.TwoDimensional:
                // Special handling for dynamic sample count
                if (IsDynamicSampleCount && InputHeight == 0 && InputWidth > 0)
                {
                    // This is fine - height (samples) will be determined at runtime
                    // Using a placeholder value of 1 for calculated size
                    calculatedSize = InputWidth;
                }
                else if (InputHeight <= 0 || InputWidth <= 0)
                {
                    throw new ArgumentException("Both InputHeight and InputWidth must be greater than 0 for TwoDimensional input.");
                }

                if (InputDepth != 1)
                {
                    throw new ArgumentException("InputDepth should be 1 for TwoDimensional input.");
                }
                break;

            case InputType.ThreeDimensional:
                if (InputHeight <= 0 || InputWidth <= 0 || InputDepth <= 0)
                {
                    throw new ArgumentException("InputHeight, InputWidth, and InputDepth must all be greater than 0 for ThreeDimensional input.");
                }
                break;

            default:
                throw new ArgumentException("Invalid InputDimensionality specified.");
        }

        // Special case for dynamic sample count where InputSize might be unknown
        if (IsDynamicSampleCount && InputType == InputType.TwoDimensional && InputHeight == 0)
        {
            // If InputSize is specified, it should match InputWidth
            if (InputSize > 0 && InputSize != InputWidth)
            {
                throw new ArgumentException($"For dynamic sample count with TwoDimensional input, InputSize ({InputSize}) should match InputWidth ({InputWidth}).");
            }

            // Set InputSize to InputWidth as a placeholder
            InputSize = InputWidth;
        }
        else if (InputSize > 0 && InputSize != calculatedSize)
        {
            throw new ArgumentException($"Provided InputSize ({InputSize}) does not match the calculated size based on dimensions ({calculatedSize}). For {InputType} input, use either InputSize or the appropriate dimension parameters, not both.");
        }
        else if (InputSize == 0)
        {
            // If InputSize wasn't provided, set it to the calculated size
            InputSize = calculatedSize;
        }

        // Validate layers if provided
        if (Layers != null && Layers.Count > 0)
        {
            var firstLayer = Layers[0];
            int firstLayerInputSize = firstLayer.GetInputShape().Aggregate(1, (a, b) => a * b);

            // Special case for dynamic sample count
            if (IsDynamicSampleCount && InputType == InputType.TwoDimensional)
            {
                // For dynamic sample count, the first layer's input size should match the InputWidth
                if (firstLayerInputSize != InputWidth)
                {
                    throw new ArgumentException($"For dynamic sample count, the first layer's input size ({firstLayerInputSize}) must match the input width ({InputWidth}).");
                }
            }
            else if (firstLayerInputSize != InputSize)
            {
                throw new ArgumentException($"The first layer's input size ({firstLayerInputSize}) must match the input size ({InputSize}).");
            }
        }
    }
}