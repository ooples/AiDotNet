namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a spatial transformer layer that enables spatial manipulations of data via a learnable transformation.
/// </summary>
/// <remarks>
/// <para>
/// A spatial transformer layer performs learnable geometric transformations on input feature maps. It consists of
/// three main components: a localization network that predicts transformation parameters, a grid generator that
/// creates a sampling grid, and a sampler that applies the transformation using bilinear interpolation. This allows
/// the network to automatically learn invariance to translation, scale, rotation, and more general warping.
/// </para>
/// <para><b>For Beginners:</b> This layer helps a neural network focus on the important parts of an image by learning to transform it.
/// 
/// Think of it like having a smart camera that can:
/// - Zoom in on the important objects
/// - Rotate images to make them easier to recognize
/// - Crop out distractions
/// - Fix distortions or perspective problems
/// 
/// Benefits include:
/// - Automatic learning of the best transformation for the task
/// - Improved recognition of objects regardless of their position or orientation
/// - Better handling of distorted or warped inputs
/// 
/// For example, when recognizing handwritten digits, a spatial transformer might learn to straighten
/// tilted digits or zoom in on the digit, making it easier for the rest of the network to classify.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SpatialTransformerLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Weights for the first layer of the localization network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix contains the weights for the first fully connected layer of the localization network,
    /// which predicts the transformation parameters. Its dimensions are [inputHeight * inputWidth, 32].
    /// </para>
    /// <para><b>For Beginners:</b> These are the adjustable values for the first part of the "smart camera".
    /// 
    /// The localization network is like a mini neural network within this layer that decides
    /// how to transform the input. This matrix stores the connection strengths in the first layer
    /// of this mini-network. The network looks at the input and starts figuring out what transformation
    /// would be most helpful.
    /// </para>
    /// </remarks>
    private Matrix<T> _localizationWeights1 = default!;

    /// <summary>
    /// Biases for the first layer of the localization network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the bias values for the first fully connected layer of the localization network.
    /// It has a length of 32.
    /// </para>
    /// <para><b>For Beginners:</b> These are additional adjustable values for the first part of the "smart camera".
    /// 
    /// Biases help the network learn more complex patterns by shifting the activation values.
    /// They work together with the weights to determine how the network responds to different inputs.
    /// </para>
    /// </remarks>
    private Vector<T> _localizationBias1 = default!;

    /// <summary>
    /// Weights for the second layer of the localization network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix contains the weights for the second fully connected layer of the localization network,
    /// which produces the final transformation parameters. Its dimensions are [32, 6], where 6 represents
    /// the parameters of a 2D affine transformation (2x3 matrix).
    /// </para>
    /// <para><b>For Beginners:</b> These are the adjustable values for the second part of the "smart camera".
    /// 
    /// This second set of weights takes the initial processing from the first layer and
    /// refines it into the exact transformation parameters. The 6 output values represent
    /// how to scale, rotate, translate, and shear the input image.
    /// </para>
    /// </remarks>
    private Matrix<T> _localizationWeights2 = default!;

    /// <summary>
    /// Biases for the second layer of the localization network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the bias values for the second fully connected layer of the localization network.
    /// It has a length of 6, corresponding to the 6 parameters of a 2D affine transformation.
    /// </para>
    /// <para><b>For Beginners:</b> These are additional adjustable values for the second part of the "smart camera".
    /// 
    /// These biases are initialized in a special way so that the layer starts by doing no transformation
    /// (identity transform). This helps the network start with a neutral behavior and gradually
    /// learn more complex transformations as needed.
    /// </para>
    /// </remarks>
    private Vector<T> _localizationBias2 = default!;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the input to the layer during the forward pass, which is needed during the backward pass
    /// to compute gradients. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the layer's short-term memory of what input it received.
    /// 
    /// During training, the layer needs to remember what input it processed so that it can
    /// properly calculate how to improve. This temporary storage is cleared between batches
    /// or when you explicitly reset the layer.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the output from the layer during the forward pass, which is needed during the backward pass
    /// to compute gradients. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is the layer's memory of what output it produced.
    /// 
    /// The layer needs to remember what output it generated so that during training it can
    /// understand how changes to the output affect the overall network performance.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Stores the transformation matrix from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the 2x3 transformation matrix computed during the forward pass, which is needed
    /// during the backward pass to compute gradients. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the exact transformation that was applied to the input.
    /// 
    /// The transformation matrix contains the specific rotation, scaling, translation, and other
    /// operations that were applied to the input. Keeping track of this is crucial for the
    /// backward pass, when the layer needs to understand exactly how it processed the data.
    /// </para>
    /// </remarks>
    private Matrix<T>? _lastTransformationMatrix;

    /// <summary>
    /// Gradient of the loss with respect to the weights of the first localization layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient (rate of change) of the loss function with respect to each parameter in
    /// the weights of the first localization layer. These gradients are computed during the backward pass and
    /// used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how each value in the first layer's weights should be changed
    /// to improve the network's performance.
    /// 
    /// During training, this gradient tells the layer how to adjust its first set of weights
    /// to make the overall network more accurate. Larger gradient values indicate weights
    /// that need more adjustment.
    /// </para>
    /// </remarks>
    private Matrix<T>? _localizationWeights1Gradient;

    /// <summary>
    /// Gradient of the loss with respect to the biases of the first localization layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient (rate of change) of the loss function with respect to each bias parameter in
    /// the first localization layer. These gradients are computed during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how each bias value in the first layer should be changed
    /// to improve the network's performance.
    /// 
    /// Similar to the weight gradients, these values tell the layer how to adjust the bias terms
    /// in the first layer during training. They help fine-tune the network's behavior.
    /// </para>
    /// </remarks>
    private Vector<T>? _localizationBias1Gradient;

    /// <summary>
    /// Gradient of the loss with respect to the weights of the second localization layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient (rate of change) of the loss function with respect to each parameter in
    /// the weights of the second localization layer. These gradients are computed during the backward pass and
    /// used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how each value in the second layer's weights should be changed
    /// to improve the network's performance.
    /// 
    /// These gradients specifically target the weights in the second layer, which directly influence
    /// the transformation parameters. They are crucial for helping the network learn the right kinds
    /// of transformations for the task.
    /// </para>
    /// </remarks>
    private Matrix<T>? _localizationWeights2Gradient;

    /// <summary>
    /// Gradient of the loss with respect to the biases of the second localization layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient (rate of change) of the loss function with respect to each bias parameter in
    /// the second localization layer. These gradients are computed during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how each bias value in the second layer should be changed
    /// to improve the network's performance.
    /// 
    /// These gradients help adjust the biases that directly affect the transformation parameters.
    /// Since these biases are initialized to represent an identity transformation (no change),
    /// the gradients show how to move away from that neutral state toward more helpful transformations.
    /// </para>
    /// </remarks>
    private Vector<T>? _localizationBias2Gradient;

    /// <summary>
    /// The height of the input feature map.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies the height dimension of the input feature map.
    /// </para>
    /// <para><b>For Beginners:</b> This is the height (number of rows) in the input image or feature map.
    /// 
    /// The layer needs to know the exact dimensions of the input to properly calculate the
    /// transformation and sampling operations.
    /// </para>
    /// </remarks>
    private readonly int _inputHeight;

    /// <summary>
    /// The width of the input feature map.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies the width dimension of the input feature map.
    /// </para>
    /// <para><b>For Beginners:</b> This is the width (number of columns) in the input image or feature map.
    /// 
    /// Together with the height, this defines the full dimensions of the input data that will
    /// be transformed by this layer.
    /// </para>
    /// </remarks>
    private readonly int _inputWidth;

    /// <summary>
    /// The height of the output feature map.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies the height dimension of the output feature map after transformation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the height of the transformed output image or feature map.
    /// 
    /// The output dimensions can be different from the input dimensions, allowing the layer
    /// to resize the feature map as part of the transformation.
    /// </para>
    /// </remarks>
    private readonly int _outputHeight;

    /// <summary>
    /// The width of the output feature map.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies the width dimension of the output feature map after transformation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the width of the transformed output image or feature map.
    /// 
    /// Like the output height, this can be different from the input width, giving the layer
    /// flexibility in how it transforms the data.
    /// </para>
    /// </remarks>
    private readonly int _outputWidth;

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Always returns <c>true</c> as spatial transformer layers have trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the spatial transformer layer can be trained. The layer contains trainable parameters
    /// in the localization network that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer contains numbers (parameters) that can be adjusted during training
    /// - It will improve its performance as it sees more examples
    /// - It participates in the learning process of the neural network
    /// 
    /// The spatial transformer will gradually learn the best transformations for the task at hand.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="SpatialTransformerLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputHeight">The height of the input feature map.</param>
    /// <param name="inputWidth">The width of the input feature map.</param>
    /// <param name="outputHeight">The height of the output feature map.</param>
    /// <param name="outputWidth">The width of the output feature map.</param>
    /// <param name="activationFunction">The activation function to apply in the localization network. Defaults to Tanh if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a spatial transformer layer with the specified input and output dimensions and a scalar activation
    /// function for the localization network. It initializes the weights and biases of the localization network.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new spatial transformer layer with basic settings.
    /// 
    /// When creating a spatial transformer, you need to specify:
    /// - inputHeight and inputWidth: The dimensions of the data going into the layer
    /// - outputHeight and outputWidth: The dimensions you want after transformation
    /// - activationFunction: The function applied to neuron outputs in the localization network
    /// 
    /// The constructor automatically sets up the localization network that will learn
    /// how to transform the data. By default, it uses the tanh activation function,
    /// which works well for predicting transformation parameters.
    /// </para>
    /// </remarks>
    public SpatialTransformerLayer(int inputHeight, int inputWidth, int outputHeight, int outputWidth, IActivationFunction<T>? activationFunction = null)
        : base([inputHeight, inputWidth], [outputHeight, outputWidth], activationFunction ?? new TanhActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;

        // Initialize localization network weights and biases
        _localizationWeights1 = new Matrix<T>(inputHeight * inputWidth, 32);
        _localizationBias1 = new Vector<T>(32);
        _localizationWeights2 = new Matrix<T>(32, 6);
        _localizationBias2 = new Vector<T>(6);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SpatialTransformerLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputHeight">The height of the input feature map.</param>
    /// <param name="inputWidth">The width of the input feature map.</param>
    /// <param name="outputHeight">The height of the output feature map.</param>
    /// <param name="outputWidth">The width of the output feature map.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply in the localization network. Defaults to Tanh if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a spatial transformer layer with the specified input and output dimensions and a vector activation
    /// function for the localization network. It initializes the weights and biases of the localization network.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new spatial transformer layer with advanced settings.
    /// 
    /// This is similar to the basic constructor, but with one key difference:
    /// - It uses a vector activation function, which works on groups of numbers at once
    /// - This can capture relationships between different elements in the output
    /// 
    /// This constructor is for advanced users who need more sophisticated activation patterns
    /// for their neural networks. For most cases, the basic constructor is sufficient.
    /// </para>
    /// </remarks>
    public SpatialTransformerLayer(int inputHeight, int inputWidth, int outputHeight, int outputWidth, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([inputHeight, inputWidth], [outputHeight, outputWidth], vectorActivationFunction ?? new TanhActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;

        // Initialize localization network weights and biases
        _localizationWeights1 = new Matrix<T>(inputHeight * inputWidth, 32);
        _localizationBias1 = new Vector<T>(32);
        _localizationWeights2 = new Matrix<T>(32, 6);
        _localizationBias2 = new Vector<T>(6);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases of the localization network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights of the localization network using a scaled random initialization,
    /// and initializes the biases. The biases of the second layer are set to represent an identity transformation
    /// (no transformation) as a starting point.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the initial values for the layer's learnable parameters.
    /// 
    /// The initialization process:
    /// - Sets up the weights with small random values using a special scaling formula
    /// - Sets all biases in the first layer to zero
    /// - Sets the biases in the second layer to represent an identity transformation
    /// 
    /// The identity transformation is like telling the layer "start by doing nothing to the input."
    /// This gives the network a good starting point, and it can learn more complex transformations
    /// as training progresses.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_localizationWeights1.Rows + _localizationWeights1.Columns)));
        InitializeMatrix(_localizationWeights1, scale);
        InitializeMatrix(_localizationWeights2, scale);

        for (int i = 0; i < _localizationBias1.Length; i++)
        {
            _localizationBias1[i] = NumOps.Zero;
        }

        // Initialize the localization bias2 to represent identity transformation
        _localizationBias2[0] = NumOps.One;
        _localizationBias2[4] = NumOps.One;
    }

    /// <summary>
    /// Initializes a matrix with scaled random values.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the given matrix with random values between -0.5 and 0.5, scaled by the provided scale factor.
    /// This type of initialization helps prevent large initial values that could cause instability during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a matrix with appropriate random starting values.
    /// 
    /// It works by:
    /// - Generating random numbers between -0.5 and 0.5
    /// - Multiplying each by a scale factor to get the right size
    /// - Setting each element in the matrix to these scaled random values
    /// 
    /// Having good starting values helps the network learn more effectively from the beginning.
    /// </para>
    /// </remarks>
    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <summary>
    /// Performs the forward pass of the spatial transformer layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after spatial transformation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the spatial transformer layer. It first uses the localization
    /// network to predict the transformation parameters, then generates a sampling grid in the output space,
    /// and finally samples the input according to the grid to produce the transformed output.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes data through the spatial transformer.
    /// 
    /// The forward pass happens in three steps:
    /// 1. Localization Network: Analyzes the input and decides what transformation to apply
    ///    - Predicts 6 parameters that define how to transform the input (rotation, scaling, etc.)
    ///
    /// 2. Grid Generator: Creates a grid of sampling points in the output space
    ///    - Determines where each output pixel should come from in the input
    ///
    /// 3. Sampler: Applies the transformation by sampling the input at the calculated positions
    ///    - Uses bilinear interpolation to smoothly sample between pixels
    ///    - Produces the final transformed output
    ///
    /// The method also saves the input, output, and transformation matrix for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        // Localization network
        var flattenedInput = input.Reshape(batchSize, _inputHeight * _inputWidth);
        var localization1 = flattenedInput.Multiply(_localizationWeights1).Add(_localizationBias1);
        localization1 = ApplyActivation(localization1);
        var transformationParams = localization1.Multiply(_localizationWeights2).Add(_localizationBias2);

        // Convert transformation parameters to 2x3 transformation matrices
        _lastTransformationMatrix = ConvertToTransformationMatrix(transformationParams);

        // Grid generator
        var outputGrid = GenerateOutputGrid();

        // Sampler
        var output = SampleInputImage(input, outputGrid, _lastTransformationMatrix);

        _lastOutput = output;
        return _lastOutput;
    }

    /// <summary>
    /// Converts the transformation parameters to a 2x3 transformation matrix.
    /// </summary>
    /// <param name="transformationParams">The 6 parameters predicted by the localization network.</param>
    /// <returns>A 2x3 affine transformation matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the 6 parameters predicted by the localization network into a 2x3 affine transformation matrix.
    /// It applies constraints to prevent extreme transformations and ensures numerical stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the predicted parameters into a format that can be used for transformation.
    /// 
    /// The 6 parameters represent:
    /// - Scaling and rotation (4 parameters)
    /// - Translation (2 parameters)
    /// 
    /// The method:
    /// - Converts these parameters into a 2×3 matrix format that can be used for transformation
    /// - Applies limits to prevent extreme transformations that might distort the image too much
    /// - Ensures that small parameter values result in transformations close to identity (no change)
    /// 
    /// Think of it like converting a set of instructions (rotate by 30 degrees, scale by 1.5, etc.)
    /// into a specific formula that can be applied to each pixel.
    /// </para>
    /// </remarks>
    private Matrix<T> ConvertToTransformationMatrix(Tensor<T> transformationParams)
    {
        if (transformationParams.Shape[0] != 1 || transformationParams.Shape[1] != 6)
        {
            throw new ArgumentException("Transformation parameters should be a 1x6 tensor.");
        }

        var matrix = new Matrix<T>(2, 3);

        // Extract the parameters
        T theta11 = transformationParams[0, 0];
        T theta12 = transformationParams[0, 1];
        T theta13 = transformationParams[0, 2];
        T theta21 = transformationParams[0, 3];
        T theta22 = transformationParams[0, 4];
        T theta23 = transformationParams[0, 5];

        // Apply constraints to prevent extreme transformations
        T scale = NumOps.FromDouble(0.1); // Adjust this value to control the scale of transformations
    
        // Limit scaling and shearing
        theta11 = MathHelper.Tanh(NumOps.Multiply(theta11, scale));
        theta12 = MathHelper.Tanh(NumOps.Multiply(theta12, scale));
        theta21 = MathHelper.Tanh(NumOps.Multiply(theta21, scale));
        theta22 = MathHelper.Tanh(NumOps.Multiply(theta22, scale));

        // Limit translation
        theta13 = MathHelper.Tanh(NumOps.Multiply(theta13, scale));
        theta23 = MathHelper.Tanh(NumOps.Multiply(theta23, scale));

        // Ensure the transformation is close to identity if parameters are small
        T epsilon = NumOps.FromDouble(1e-5);
        theta11 = NumOps.Add(theta11, NumOps.One);
        theta22 = NumOps.Add(theta22, NumOps.One);

        // Construct the transformation matrix
        matrix[0, 0] = theta11;
        matrix[0, 1] = theta12;
        matrix[0, 2] = theta13;
        matrix[1, 0] = theta21;
        matrix[1, 1] = theta22;
        matrix[1, 2] = theta23;

        return matrix;
    }

    /// <summary>
    /// Generates a grid of sampling coordinates in the output space.
    /// </summary>
    /// <returns>A tensor containing the (x, y) coordinates for each position in the output grid.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a regular grid of coordinates in the output space, with values normalized to the range [-1, 1].
    /// These coordinates will be transformed using the transformation matrix to determine the sampling locations in the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a map of where each output pixel should be in the final result.
    /// 
    /// The grid generator:
    /// - Creates a grid of coordinates covering the entire output space
    /// - Normalizes these coordinates to the range [-1, 1], where (0,0) is the center
    /// - This normalized range makes the math for transformations cleaner
    /// 
    /// For example, in a 3x3 output:
    /// - The top-left point would be (-1, -1)
    /// - The center point would be (0, 0)
    /// - The bottom-right point would be (1, 1)
    /// 
    /// This regular grid will later be transformed to sample from the right places in the input.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateOutputGrid()
    {
        // Generate a grid of (x, y) coordinates for the output
        var grid = new Tensor<T>([_outputHeight, _outputWidth, 2]);
        for (int i = 0; i < _outputHeight; i++)
        {
            for (int j = 0; j < _outputWidth; j++)
            {
                grid[i, j, 0] = NumOps.FromDouble((double)j / (_outputWidth - 1) * 2 - 1);
                grid[i, j, 1] = NumOps.FromDouble((double)i / (_outputHeight - 1) * 2 - 1);
            }
        }

        return grid;
    }

    /// <summary>
    /// Samples the input image according to the transformed coordinates.
    /// </summary>
    /// <param name="input">The input tensor to sample from.</param>
    /// <param name="outputGrid">The grid of coordinates in the output space.</param>
    /// <param name="transformationMatrix">The 2x3 affine transformation matrix.</param>
    /// <returns>The sampled output tensor after transformation.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the spatial transformation by sampling the input at locations determined by transforming
    /// the output grid coordinates using the transformation matrix. It uses bilinear interpolation to sample between
    /// discrete pixel locations for smoother results.
    /// </para>
    /// <para><b>For Beginners:</b> This method actually performs the transformation by sampling from the right input locations.
    /// 
    /// For each output pixel:
    /// 1. The method transforms its coordinates to find where in the input image it should come from
    /// 2. These transformed coordinates usually fall between exact pixel locations
    /// 3. Bilinear interpolation is used to smoothly blend the four nearest input pixels
    /// 4. The result is a smoothly transformed output without jagged edges
    /// 
    /// For example, if the transformation is a 45-degree rotation, each output pixel's value
    /// would be calculated by sampling from the appropriate rotated position in the input,
    /// using interpolation to handle positions between pixels.
    /// </para>
    /// </remarks>
    private Tensor<T> SampleInputImage(Tensor<T> input, Tensor<T> outputGrid, Matrix<T> transformationMatrix)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[3];
        var output = new Tensor<T>([batchSize, _outputHeight, _outputWidth, channels]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int y = 0; y < _outputHeight; y++)
            {
                for (int x = 0; x < _outputWidth; x++)
                {
                    // Apply transformation to output coordinates
                    T srcX = NumOps.Add(
                        NumOps.Add(
                            NumOps.Multiply(transformationMatrix[0, 0], outputGrid[y, x, 0]),
                            NumOps.Multiply(transformationMatrix[0, 1], outputGrid[y, x, 1])
                        ),
                        transformationMatrix[0, 2]
                    );
                    T srcY = NumOps.Add(
                        NumOps.Add(
                            NumOps.Multiply(transformationMatrix[1, 0], outputGrid[y, x, 0]),
                            NumOps.Multiply(transformationMatrix[1, 1], outputGrid[y, x, 1])
                        ),
                        transformationMatrix[1, 2]
                    );

                    // Convert to input image coordinates
                    srcX = NumOps.Multiply(NumOps.Add(srcX, NumOps.One), NumOps.Divide(NumOps.FromDouble(_inputWidth - 1), NumOps.FromDouble(2)));
                    srcY = NumOps.Multiply(NumOps.Add(srcY, NumOps.One), NumOps.Divide(NumOps.FromDouble(_inputHeight - 1), NumOps.FromDouble(2)));

                    // Compute the four nearest neighbor coordinates
                    int x0 = (int)Math.Floor(Convert.ToDouble(srcX));
                    int x1 = Math.Min(x0 + 1, _inputWidth - 1);
                    int y0 = (int)Math.Floor(Convert.ToDouble(srcY));
                    int y1 = Math.Min(y0 + 1, _inputHeight - 1);

                    // Compute interpolation weights
                    T wx1 = NumOps.Subtract(srcX, NumOps.FromDouble(x0));
                    T wx0 = NumOps.Subtract(NumOps.One, wx1);
                    T wy1 = NumOps.Subtract(srcY, NumOps.FromDouble(y0));
                    T wy0 = NumOps.Subtract(NumOps.One, wy1);

                    // Perform bilinear interpolation for each channel
                    for (int c = 0; c < channels; c++)
                    {
                        T v00 = input[b, y0, x0, c];
                        T v01 = input[b, y0, x1, c];
                        T v10 = input[b, y1, x0, c];
                        T v11 = input[b, y1, x1, c];

                        T interpolated = NumOps.Add(
                            NumOps.Add(
                                NumOps.Multiply(NumOps.Multiply(v00, wx0), wy0),
                                NumOps.Multiply(NumOps.Multiply(v01, wx1), wy0)
                            ),
                            NumOps.Add(
                                NumOps.Multiply(NumOps.Multiply(v10, wx0), wy1),
                                NumOps.Multiply(NumOps.Multiply(v11, wx1), wy1)
                            )
                        );

                        output[b, y, x, c] = interpolated;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the spatial transformer layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the spatial transformer layer, which is used during training to propagate
    /// error gradients back through the network. It computes gradients for the localization network parameters and returns
    /// the gradient with respect to the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// and parameters should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The method throws an error if the forward pass hasn't been called first
    /// 2. It computes gradients for:
    ///    - The input tensor (how the input should change)
    ///    - The localization network parameters (how the transformation should change)
    /// 3. This involves backpropagating through three components:
    ///    - The sampler (how changes in output affect the sampling process)
    ///    - The grid generator (how changes in sampling affect the grid coordinates)
    ///    - The localization network (how changes in grid coordinates affect the transformation parameters)
    /// 
    /// The backward pass is complex because it must calculate how small changes in the transformation
    /// parameters affect the final output. This allows the network to learn the optimal transformation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastTransformationMatrix == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int channels = _lastInput.Shape[3];

        // Initialize gradients
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _localizationWeights1Gradient = new Matrix<T>(_localizationWeights1.Rows, _localizationWeights1.Columns);
        _localizationBias1Gradient = new Vector<T>(_localizationBias1.Length);
        _localizationWeights2Gradient = new Matrix<T>(_localizationWeights2.Rows, _localizationWeights2.Columns);
        _localizationBias2Gradient = new Vector<T>(_localizationBias2.Length);

        // Generate output grid
        var outputGrid = GenerateOutputGrid();

        for (int b = 0; b < batchSize; b++)
        {
            // Backward pass through the sampler
            var samplerGradient = BackwardSampler(outputGradient.GetSlice(b), _lastInput.GetSlice(b), outputGrid, _lastTransformationMatrix);

            // Backward pass through the grid generator
            var gridGeneratorGradient = BackwardGridGenerator(samplerGradient, _lastTransformationMatrix);

            // Backward pass through the localization network
            BackwardLocalizationNetwork(gridGeneratorGradient, _lastInput.GetSlice(b));

            // Accumulate input gradient
            for (int y = 0; y < _inputHeight; y++)
            {
                for (int x = 0; x < _inputWidth; x++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        inputGradient[b, y, x, c] = samplerGradient[y, x, c];
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the sampler operations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output of the layer.</param>
    /// <param name="input">The input tensor for the current batch.</param>
    /// <param name="outputGrid">The grid of coordinates in the output space.</param>
    /// <param name="transformationMatrix">The transformation matrix used in the forward pass.</param>
    /// <returns>The gradient of the loss with respect to the sampler operations.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the gradient of the loss with respect to the sampler operations by distributing
    /// the output gradient back to the input locations according to the transformation. It uses the same
    /// bilinear interpolation weights as in the forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes in the output affect the sampling process.
    /// 
    /// During the backward pass through the sampler:
    /// - The method takes gradients from the output and distributes them back to the input locations
    /// - It uses the same transformation and interpolation weights as the forward pass, but in reverse
    /// - Each output gradient is distributed to the four nearest input pixels that contributed to it
    /// - This shows how changes in the input would affect the output after transformation
    /// 
    /// This step is crucial for calculating how the input should change to improve the overall result.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardSampler(Tensor<T> outputGradient, Tensor<T> input, Tensor<T> outputGrid, Matrix<T> transformationMatrix)
    {
        var samplerGradient = new Tensor<T>(input.Shape);

        for (int y = 0; y < _outputHeight; y++)
        {
            for (int x = 0; x < _outputWidth; x++)
            {
                // Apply transformation to output coordinates
                T srcX = NumOps.Add(
                    NumOps.Add(
                        NumOps.Multiply(transformationMatrix[0, 0], outputGrid[y, x, 0]),
                        NumOps.Multiply(transformationMatrix[0, 1], outputGrid[y, x, 1])
                    ),
                    transformationMatrix[0, 2]
                );
                T srcY = NumOps.Add(
                    NumOps.Add(
                        NumOps.Multiply(transformationMatrix[1, 0], outputGrid[y, x, 0]),
                        NumOps.Multiply(transformationMatrix[1, 1], outputGrid[y, x, 1])
                    ),
                    transformationMatrix[1, 2]
                );

                // Convert to input image coordinates
                srcX = NumOps.Multiply(NumOps.Add(srcX, NumOps.One), NumOps.Divide(NumOps.FromDouble(_inputWidth - 1), NumOps.FromDouble(2)));
                srcY = NumOps.Multiply(NumOps.Add(srcY, NumOps.One), NumOps.Divide(NumOps.FromDouble(_inputHeight - 1), NumOps.FromDouble(2)));

                // Compute the four nearest neighbor coordinates
                int x0 = (int)Math.Floor(Convert.ToDouble(srcX));
                int x1 = Math.Min(x0 + 1, _inputWidth - 1);
                int y0 = (int)Math.Floor(Convert.ToDouble(srcY));
                int y1 = Math.Min(y0 + 1, _inputHeight - 1);

                // Compute interpolation weights
                T wx1 = NumOps.Subtract(srcX, NumOps.FromDouble(x0));
                T wx0 = NumOps.Subtract(NumOps.One, wx1);
                T wy1 = NumOps.Subtract(srcY, NumOps.FromDouble(y0));
                T wy0 = NumOps.Subtract(NumOps.One, wy1);

                // Distribute gradients to the four nearest neighbors
                for (int c = 0; c < input.Shape[3]; c++)
                {
                    T gradValue = outputGradient[y, x, c];
                    samplerGradient[y0, x0, c] = NumOps.Add(samplerGradient[y0, x0, c], NumOps.Multiply(NumOps.Multiply(gradValue, wx0), wy0));
                    samplerGradient[y0, x1, c] = NumOps.Add(samplerGradient[y0, x1, c], NumOps.Multiply(NumOps.Multiply(gradValue, wx1), wy0));
                    samplerGradient[y1, x0, c] = NumOps.Add(samplerGradient[y1, x0, c], NumOps.Multiply(NumOps.Multiply(gradValue, wx0), wy1));
                    samplerGradient[y1, x1, c] = NumOps.Add(samplerGradient[y1, x1, c], NumOps.Multiply(NumOps.Multiply(gradValue, wx1), wy1));
                }
            }
        }

        return samplerGradient;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the grid generator operations.
    /// </summary>
    /// <param name="samplerGradient">The gradient of the loss with respect to the sampler operations.</param>
    /// <param name="transformationMatrix">The transformation matrix used in the forward pass.</param>
    /// <returns>The gradient of the loss with respect to the transformation matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the gradient of the loss with respect to the grid generator operations, which includes
    /// the gradient with respect to the transformation matrix. It calculates how changes in the transformation
    /// parameters affect the output through the sampling process.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes in the grid coordinates affect the overall result.
    /// 
    /// During the backward pass through the grid generator:
    /// - The method takes gradients from the sampler and calculates how they relate to the transformation parameters
    /// - It accumulates gradients for each element in the transformation matrix
    /// - This shows how changing each transformation parameter would affect the final output
    /// 
    /// This step is important for understanding how to adjust the transformation to improve the result.
    /// </para>
    /// </remarks>
    private Matrix<T> BackwardGridGenerator(Tensor<T> samplerGradient, Matrix<T> transformationMatrix)
    {
        var gridGeneratorGradient = new Matrix<T>(2, 3);

        for (int y = 0; y < _outputHeight; y++)
        {
            for (int x = 0; x < _outputWidth; x++)
            {
                T gradX = samplerGradient[y, x, 0];
                T gradY = samplerGradient[y, x, 1];

                gridGeneratorGradient[0, 0] = NumOps.Add(gridGeneratorGradient[0, 0], NumOps.Multiply(gradX, NumOps.FromDouble((double)x / (_outputWidth - 1) * 2 - 1)));
                gridGeneratorGradient[0, 1] = NumOps.Add(gridGeneratorGradient[0, 1], NumOps.Multiply(gradX, NumOps.FromDouble((double)y / (_outputHeight - 1) * 2 - 1)));
                gridGeneratorGradient[0, 2] = NumOps.Add(gridGeneratorGradient[0, 2], gradX);
                gridGeneratorGradient[1, 0] = NumOps.Add(gridGeneratorGradient[1, 0], NumOps.Multiply(gradY, NumOps.FromDouble((double)x / (_outputWidth - 1) * 2 - 1)));
                gridGeneratorGradient[1, 1] = NumOps.Add(gridGeneratorGradient[1, 1], NumOps.Multiply(gradY, NumOps.FromDouble((double)y / (_outputHeight - 1) * 2 - 1)));
                gridGeneratorGradient[1, 2] = NumOps.Add(gridGeneratorGradient[1, 2], gradY);
            }
        }

        return gridGeneratorGradient;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the localization network parameters.
    /// </summary>
    /// <param name="gridGeneratorGradient">The gradient of the loss with respect to the transformation matrix.</param>
    /// <param name="input">The input tensor for the current batch.</param>
    /// <remarks>
    /// <para>
    /// This method computes the gradient of the loss with respect to the weights and biases of the localization network.
    /// It uses the chain rule to propagate the gradient from the transformation matrix back through the localization network.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the localization network's parameters should change.
    /// 
    /// During the backward pass through the localization network:
    /// - The method takes gradients from the grid generator and propagates them back through the network
    /// - It computes how each weight and bias in the localization network affects the transformation
    /// - These gradients are stored for later use when updating the parameters
    /// 
    /// This step is crucial for learning the optimal transformation parameters for the task.
    /// It shows how to adjust the "smart camera" weights to improve the overall performance.
    /// </para>
    /// </remarks>
    private void BackwardLocalizationNetwork(Matrix<T> gridGeneratorGradient, Tensor<T> input)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
    
        // Flatten input
        var flattenedInput = input.Reshape([input.Shape[0], _inputHeight * _inputWidth * input.Shape[3]]);

        // Backward pass through the second layer of localization network
        var dL_dTheta = gridGeneratorGradient.Reshape(gridGeneratorGradient.Rows, 6);
        var dL_dL1 = dL_dTheta.Multiply(_localizationWeights2.Transpose());

        _localizationWeights2Gradient ??= new Matrix<T>(_localizationWeights2.Rows, _localizationWeights2.Columns);
        _localizationWeights2Gradient = _localizationWeights2Gradient.Add(
            flattenedInput.ToMatrix().Transpose().Multiply(dL_dL1));

        _localizationBias2Gradient ??= new Vector<T>(_localizationBias2.Length);
        for (int i = 0; i < dL_dTheta.Rows; i++)
        {
            _localizationBias2Gradient = _localizationBias2Gradient.Add(dL_dTheta.GetRow(i));
        }

        // Backward pass through the activation function
        var z1 = flattenedInput.ToMatrix().Multiply(_localizationWeights1).AddColumn(_localizationBias1);
        var dL_dZ1 = ApplyActivationGradient(dL_dL1, z1);

        // Backward pass through the first layer of localization network
        _localizationWeights1Gradient ??= new Matrix<T>(_localizationWeights1.Rows, _localizationWeights1.Columns);
        _localizationWeights1Gradient = _localizationWeights1Gradient.Add(
            flattenedInput.Transpose([1, 0]).ToMatrix().Multiply(dL_dZ1));

        _localizationBias1Gradient ??= new Vector<T>(_localizationBias1.Length);
        for (int i = 0; i < dL_dZ1.Rows; i++)
        {
            _localizationBias1Gradient = _localizationBias1Gradient.Add(dL_dZ1.GetRow(i));
        }
    }

    /// <summary>
    /// Applies the activation function gradient to a matrix.
    /// </summary>
    /// <param name="upstream">The upstream gradient coming from the next layer.</param>
    /// <param name="z">The input to the activation function in the forward pass.</param>
    /// <returns>The gradient after applying the activation function gradient.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the gradient of the activation function to each element of the matrix. It implements
    /// the backpropagation through the activation function by computing how changes in the activation output
    /// affect the activation input.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes in the output of an activation function
    /// relate to changes in its input.
    /// 
    /// During backward propagation:
    /// - The activation function gradient shows how sensitive the output is to small changes in the input
    /// - This method applies this gradient to each value in the matrix
    /// - The result shows how to adjust the inputs to the activation function to improve the output
    /// 
    /// This is a crucial step in the backpropagation algorithm that allows neural networks to learn.
    /// </para>
    /// </remarks>
   private Matrix<T> ApplyActivationGradient(Matrix<T> upstream, Matrix<T> z)
    {
        var gradient = new Matrix<T>(z.Rows, z.Columns);

        for (int i = 0; i < z.Rows; i++)
        {
            Vector<T> rowZ = z.GetRow(i);
            Vector<T> rowUpstream = upstream.GetRow(i);
            Vector<T> rowGradient = ApplyActivationDerivative(rowZ, rowUpstream);
            gradient.SetRow(i, rowGradient);
        }

        return gradient;
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the localization network based on the gradients calculated during
    /// the backward pass. The learning rate controls the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's learnable values during training.
    /// 
    /// When updating parameters:
    /// - Each weight and bias is adjusted in the direction that reduces the error
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates lead to more stable but slower learning
    /// - Larger learning rates can learn faster but might become unstable
    /// 
    /// The update process:
    /// - Subtract the gradient (multiplied by the learning rate) from each parameter
    /// - This moves the parameters in the direction that reduces the error
    /// - Over many updates, the layer learns the optimal transformation for the task
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_localizationWeights1Gradient == null || _localizationBias1Gradient == null ||
            _localizationWeights2Gradient == null || _localizationBias2Gradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _localizationWeights1 = _localizationWeights1.Subtract(_localizationWeights1Gradient.Multiply(learningRate));
        _localizationBias1 = _localizationBias1.Subtract(_localizationBias1Gradient.Multiply(learningRate));
        _localizationWeights2 = _localizationWeights2.Subtract(_localizationWeights2Gradient.Multiply(learningRate));
        _localizationBias2 = _localizationBias2.Subtract(_localizationBias2Gradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters of the layer (localization network weights and biases) and combines them
    /// into a single vector. This is useful for optimization algorithms that operate on all parameters at once,
    /// or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer into a single list.
    /// 
    /// The parameters:
    /// - Are the weights and biases of the localization network
    /// - Are converted from matrices and vectors to a single long list (vector)
    /// - Can be used to save the state of the layer or apply optimization techniques
    /// 
    /// This method is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _localizationWeights1.Rows * _localizationWeights1.Columns +
                          _localizationBias1.Length +
                          _localizationWeights2.Rows * _localizationWeights2.Columns +
                          _localizationBias2.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy localization weights1
        for (int i = 0; i < _localizationWeights1.Rows; i++)
        {
            for (int j = 0; j < _localizationWeights1.Columns; j++)
            {
                parameters[index++] = _localizationWeights1[i, j];
            }
        }
    
        // Copy localization bias1
        for (int i = 0; i < _localizationBias1.Length; i++)
        {
            parameters[index++] = _localizationBias1[i];
        }
    
        // Copy localization weights2
        for (int i = 0; i < _localizationWeights2.Rows; i++)
        {
            for (int j = 0; j < _localizationWeights2.Columns; j++)
            {
                parameters[index++] = _localizationWeights2[i, j];
            }
        }
    
        // Copy localization bias2
        for (int i = 0; i < _localizationBias2.Length; i++)
        {
            parameters[index++] = _localizationBias2[i];
        }
    
        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters of the layer (localization network weights and biases) from a single vector.
    /// It expects the vector to contain the parameters in the same order as they are retrieved by GetParameters().
    /// This is useful for loading saved model weights or for implementing optimization algorithms that operate
    /// on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer from a single list.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with exactly the right number of values
    /// - The values are distributed back into the weights and biases matrices and vectors
    /// - The order must match how they were stored in GetParameters()
    /// 
    /// This method is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _localizationWeights1.Rows * _localizationWeights1.Columns +
                          _localizationBias1.Length +
                          _localizationWeights2.Rows * _localizationWeights2.Columns +
                          _localizationBias2.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set localization weights1
        for (int i = 0; i < _localizationWeights1.Rows; i++)
        {
            for (int j = 0; j < _localizationWeights1.Columns; j++)
            {
                _localizationWeights1[i, j] = parameters[index++];
            }
        }
    
        // Set localization bias1
        for (int i = 0; i < _localizationBias1.Length; i++)
        {
            _localizationBias1[i] = parameters[index++];
        }
    
        // Set localization weights2
        for (int i = 0; i < _localizationWeights2.Rows; i++)
        {
            for (int j = 0; j < _localizationWeights2.Columns; j++)
            {
                _localizationWeights2[i, j] = parameters[index++];
            }
        }
    
        // Set localization bias2
        for (int i = 0; i < _localizationBias2.Length; i++)
        {
            _localizationBias2[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the spatial transformer layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the spatial transformer layer, including the cached inputs, outputs,
    /// transformation matrix, and gradients. This is useful when starting to process a new batch or when implementing
    /// stateful networks that need to be reset between sequences.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs from previous passes are cleared
    /// - The transformation matrix is cleared
    /// - All gradients are cleared
    /// 
    /// This is important for:
    /// - Processing a new batch of unrelated data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Think of it like clearing your workspace before starting a new project -
    /// it ensures that old information doesn't interfere with new processing.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastTransformationMatrix = null;
        _localizationWeights1Gradient = null;
        _localizationBias1Gradient = null;
        _localizationWeights2Gradient = null;
        _localizationBias2Gradient = null;
    }
}