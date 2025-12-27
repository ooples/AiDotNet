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
    public InputType InputType { get; }

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
    public int OutputSize { get; }

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
    public int InputHeight { get; }

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
    public int InputWidth { get; }

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
    public int InputDepth { get; }

    /// <summary>
    /// Gets the dimensionality of image embeddings for multimodal networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property specifies the output dimension of the image encoder in multimodal architectures
    /// like CLIP. Common values are 768 (ViT-B/32) or 1024 (ViT-L/14). This dimension represents
    /// the size of the feature vector produced by the vision transformer or CNN backbone.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the "description" vector that represents an image.
    ///
    /// When a multimodal model (like CLIP) processes an image:
    /// - The image encoder converts the image into a vector of numbers
    /// - This vector captures the "meaning" or "content" of the image
    /// - ImageEmbeddingDim specifies how many numbers are in this vector
    ///
    /// For example:
    /// - ImageEmbeddingDim = 768 means images become 768-dimensional vectors
    /// - Larger dimensions can capture more detail but need more computation
    ///
    /// This is only used for multimodal networks that process both images and text.
    /// For standard image classifiers, use InputHeight/InputWidth instead.
    /// </para>
    /// </remarks>
    public int ImageEmbeddingDim { get; }

    /// <summary>
    /// Gets the dimensionality of text embeddings for multimodal networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property specifies the output dimension of the text encoder in multimodal architectures
    /// like CLIP. Common values are 512 or 768. This dimension represents the size of the feature
    /// vector produced by the text transformer.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the "description" vector that represents text.
    ///
    /// When a multimodal model (like CLIP) processes text:
    /// - The text encoder converts words/sentences into a vector of numbers
    /// - This vector captures the "meaning" or "semantics" of the text
    /// - TextEmbeddingDim specifies how many numbers are in this vector
    ///
    /// For example:
    /// - TextEmbeddingDim = 512 means text becomes 512-dimensional vectors
    /// - This vector can then be compared with image embeddings
    ///
    /// For CLIP-style models, text and image embeddings are projected to the same space,
    /// allowing direct comparison between text descriptions and images.
    /// </para>
    /// </remarks>
    public int TextEmbeddingDim { get; }

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
    public NeuralNetworkTaskType TaskType { get; }

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
    /// Gets or sets a value indicating whether all layers in this architecture should use automatic differentiation by default.
    /// </summary>
    /// <value>
    /// <c>true</c> if layers should use autodiff by default; <c>false</c> for manual backward implementations. Default is <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property sets the default autodiff mode for all layers created with this architecture.
    /// Individual layers can still override this setting via their <c>UseAutodiff</c> property.
    /// Manual backward passes are typically faster but require explicit gradient code, while autodiff
    /// is more flexible for custom or experimental layers.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how gradient computation works for all layers in the network.
    ///
    /// When building a network from this architecture:
    /// - <b>false (default):</b> All layers use fast, hand-optimized gradient code
    /// - <b>true:</b> All layers use automatic differentiation for gradients
    ///
    /// Most users should leave this as false (default) for best performance. Set to true only for:
    /// - Research and experimentation with novel architectures
    /// - Networks with many custom layers that have complex gradients
    /// - When you want to verify gradient correctness during development
    ///
    /// <b>Note:</b> Individual layers can override this setting. This just sets the default.
    /// </para>
    /// </remarks>
    public bool UseAutodiff { get; set; } = false;

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
            InputType.TwoDimensional => InputHeight * InputWidth,
            InputType.ThreeDimensional => InputHeight * InputWidth * InputDepth,
            _ => throw new InvalidOperationException("Invalid InputDimensionality"),
        };

    /// <summary>
    /// Determines whether the network should return the full sequence or just the final output.
    /// </summary>
    /// <returns>True if full sequence should be returned; otherwise, false.</returns>
    public bool ShouldReturnFullSequence { get; }

    /// <summary>
    /// Gets a value indicating whether the architecture has been initialized.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property tracks whether the neural network architecture has been properly initialized
    /// with all necessary data and configurations. An uninitialized architecture may need to load
    /// cached data or perform other initialization steps before the network can be used.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the network architecture is ready to use.
    ///
    /// Think of this like a checklist before starting:
    /// - false: The architecture is created but not fully set up yet
    /// - true: Everything is ready and the network can be used
    ///
    /// This is useful because sometimes a network needs to load previously saved data
    /// or perform setup steps before it can start training or making predictions.
    /// </para>
    /// </remarks>
    public bool IsInitialized { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNetworkArchitecture{T}"/> class with the specified parameters.
    /// </summary>
    /// <param name="inputType">The type of input data (one-dimensional, two-dimensional, or three-dimensional).</param>
    /// <param name="taskType">The type of task the neural network will perform (classification, regression, etc.).</param>
    /// <param name="complexity">The complexity level of the neural network. Default is Medium.</param>
    /// <param name="inputSize">The size of the input vector (for one-dimensional input). Default is 0.</param>
    /// <param name="inputHeight">The height of the input (for two/three-dimensional input). Default is 0.</param>
    /// <param name="inputWidth">The width of the input (for two/three-dimensional input). Default is 0.</param>
    /// <param name="inputDepth">The depth of the input (for three-dimensional input). Default is 1.</param>
    /// <param name="outputSize">The size of the output vector. Default is 0.</param>
    /// <param name="layers">Optional predefined layers for the neural network. Default is null.</param>
    /// <param name="rbmLayers">Optional RBM layers for pre-training. Default is null.</param>
    /// <param name="imageEmbeddingDim">The dimensionality of image embeddings for multimodal networks. Default is 0 (not multimodal).</param>
    /// <param name="textEmbeddingDim">The dimensionality of text embeddings for multimodal networks. Default is 0 (not multimodal).</param>
    /// <exception cref="ArgumentException">Thrown when the input dimensions are invalid or inconsistent.</exception>
    /// <remarks>
    /// <para>
    /// This constructor initializes a neural network architecture with the specified parameters and validates
    /// that the input dimensions are consistent and appropriate for the selected input type. It also checks
    /// that any provided layers are compatible with the input and output dimensions.
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
    /// 
    /// The constructor checks that all your settings make sense together.
    /// For example, it will catch errors like trying to use both InputSize=100
    /// and InputHeight=10, InputWidth=20 (which would imply InputSize=200).
    /// </para>
    /// </remarks>
    [JsonConstructor]
    public NeuralNetworkArchitecture(
        InputType inputType,
        NeuralNetworkTaskType taskType,
        NetworkComplexity complexity = NetworkComplexity.Medium,
        int inputSize = 0,
        int inputHeight = 0,
        int inputWidth = 0,
        int inputDepth = 1,
        int outputSize = 0,
        List<ILayer<T>>? layers = null,
        bool shouldReturnFullSequence = false,
        int imageEmbeddingDim = 0,
        int textEmbeddingDim = 0)
    {
        InputType = inputType;
        TaskType = taskType;
        Complexity = complexity;
        InputSize = inputSize;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        InputDepth = inputDepth;
        ShouldReturnFullSequence = shouldReturnFullSequence;
        Layers = layers;
        ImageEmbeddingDim = imageEmbeddingDim;
        TextEmbeddingDim = textEmbeddingDim;
        OutputSize = outputSize;

        ValidateInputDimensions();
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
            return Array.Empty<int>();
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
    /// Initializes the architecture from cached data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the neural network architecture using previously cached or saved data.
    /// It marks the architecture as initialized once the process is complete. This is useful when
    /// loading a pre-trained network or resuming training from a checkpoint.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the architecture to use saved information.
    ///
    /// Think of this like:
    /// - Loading a saved game - you want to continue from where you left off
    /// - Restoring a workspace - bringing back your previous setup
    /// - Rehydrating freeze-dried food - adding back what was removed to make it usable again
    ///
    /// When you train a neural network, you might save its state and come back to it later.
    /// This method helps restore that saved state so the network can continue working.
    ///
    /// After calling this method, IsInitialized will be set to true, indicating the
    /// architecture is ready for use.
    /// </para>
    /// </remarks>
    public void InitializeFromCachedData()
    {
        // Mark the architecture as initialized
        // In a more complete implementation, this would load cached configuration data
        // such as layer weights, biases, and other parameters from storage
        IsInitialized = true;
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
                if (InputHeight <= 0 || InputWidth <= 0)
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

        if (InputSize > 0 && InputSize != calculatedSize)
        {
            throw new ArgumentException($"Provided InputSize ({InputSize}) does not match the calculated size based on dimensions ({calculatedSize}). For {InputType} input, use either InputSize or the appropriate dimension parameters, not both.");
        }

        // If InputSize wasn't provided, set it to the calculated size
        if (InputSize == 0)
        {
            InputSize = calculatedSize;
        }

        // Validate layers if provided
        if (Layers != null && Layers.Count > 0)
        {
            var firstLayer = Layers[0];
            int firstLayerInputSize = firstLayer.GetInputShape().Aggregate(1, (a, b) => a * b);

            if (firstLayerInputSize != InputSize)
            {
                throw new ArgumentException($"The first layer's input size ({firstLayerInputSize}) must match the input size ({InputSize}).");
            }
        }
    }
}
