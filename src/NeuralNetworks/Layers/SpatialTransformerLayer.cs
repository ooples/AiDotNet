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
public class SpatialTransformerLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets a value indicating whether auxiliary loss is enabled for this layer.
    /// </summary>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the auxiliary loss contribution.
    /// </summary>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Stores the last computed transformation regularization loss for diagnostic purposes.
    /// </summary>
    private T _lastTransformationLoss;

    /// <summary>
    /// Weights for the first layer of the localization network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the weights for the first fully connected layer of the localization network,
    /// which predicts the transformation parameters. Its shape is [inputHeight * inputWidth, 32].
    /// </para>
    /// <para><b>For Beginners:</b> These are the adjustable values for the first part of the "smart camera".
    ///
    /// The localization network is like a mini neural network within this layer that decides
    /// how to transform the input. This tensor stores the connection strengths in the first layer
    /// of this mini-network. The network looks at the input and starts figuring out what transformation
    /// would be most helpful.
    /// </para>
    /// </remarks>
    private Tensor<T> _localizationWeights1;

    /// <summary>
    /// Biases for the first layer of the localization network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the bias values for the first fully connected layer of the localization network.
    /// It has shape [32].
    /// </para>
    /// <para><b>For Beginners:</b> These are additional adjustable values for the first part of the "smart camera".
    ///
    /// Biases help the network learn more complex patterns by shifting the activation values.
    /// They work together with the weights to determine how the network responds to different inputs.
    /// </para>
    /// </remarks>
    private Tensor<T> _localizationBias1;

    /// <summary>
    /// Weights for the second layer of the localization network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the weights for the second fully connected layer of the localization network,
    /// which produces the final transformation parameters. Its shape is [32, 6], where 6 represents
    /// the parameters of a 2D affine transformation (2x3 tensor).
    /// </para>
    /// <para><b>For Beginners:</b> These are the adjustable values for the second part of the "smart camera".
    ///
    /// This second set of weights takes the initial processing from the first layer and
    /// refines it into the exact transformation parameters. The 6 output values represent
    /// how to scale, rotate, translate, and shear the input image.
    /// </para>
    /// </remarks>
    private Tensor<T> _localizationWeights2;

    /// <summary>
    /// Biases for the second layer of the localization network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the bias values for the second fully connected layer of the localization network.
    /// It has shape [6], corresponding to the 6 parameters of a 2D affine transformation.
    /// </para>
    /// <para><b>For Beginners:</b> These are additional adjustable values for the second part of the "smart camera".
    ///
    /// These biases are initialized in a special way so that the layer starts by doing no transformation
    /// (identity transform). This helps the network start with a neutral behavior and gradually
    /// learn more complex transformations as needed.
    /// </para>
    /// </remarks>
    private Tensor<T> _localizationBias2;

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
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stores whether the original input included a channel dimension.
    /// </summary>
    private bool _inputHadChannel;

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
    /// This field caches the 2x3 transformation tensor computed during the forward pass, which is needed
    /// during the backward pass to compute gradients. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the exact transformation that was applied to the input.
    ///
    /// The transformation tensor contains the specific rotation, scaling, translation, and other
    /// operations that were applied to the input. Keeping track of this is crucial for the
    /// backward pass, when the layer needs to understand exactly how it processed the data.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastTransformationMatrix;

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
    private Tensor<T>? _localizationWeights1Gradient;

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
    private Tensor<T>? _localizationBias1Gradient;

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
    private Tensor<T>? _localizationWeights2Gradient;

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
    private Tensor<T>? _localizationBias2Gradient;
    private Tensor<T>? _lastFlattenedInput;
    private Tensor<T>? _lastLocalization1;

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
        // Initialize auxiliary loss fields first so compiler knows they're set
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastTransformationLoss = NumOps.Zero;

        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;

        // Initialize localization network weights and biases as Tensor<T>
        _localizationWeights1 = new Tensor<T>([inputHeight * inputWidth, 32]);
        _localizationBias1 = new Tensor<T>([32]);
        _localizationWeights2 = new Tensor<T>([32, 6]);
        _localizationBias2 = new Tensor<T>([6]);

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
        // Initialize auxiliary loss fields first so compiler knows they're set
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastTransformationLoss = NumOps.Zero;

        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;

        // Initialize localization network weights and biases as Tensor<T>
        _localizationWeights1 = new Tensor<T>([inputHeight * inputWidth, 32]);
        _localizationBias1 = new Tensor<T>([32]);
        _localizationWeights2 = new Tensor<T>([32, 6]);
        _localizationBias2 = new Tensor<T>([6]);

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
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_localizationWeights1.Shape[0] + _localizationWeights1.Shape[1])));
        InitializeTensor(_localizationWeights1, scale);
        InitializeTensor(_localizationWeights2, scale);

        // Initialize bias1 to zero
        _localizationBias1.Fill(NumOps.Zero);

        // Initialize the localization bias2 to represent identity transformation
        _localizationBias2.Fill(NumOps.Zero);
        _localizationBias2[0] = NumOps.One;
        _localizationBias2[4] = NumOps.One;
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the given tensor with random values between -0.5 and 0.5, scaled by the provided scale factor.
    /// This type of initialization helps prevent large initial values that could cause instability during training.
    /// </para>
    /// </remarks>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        // Vectorized initialization using Engine operations
        int totalElements = tensor.Length;
        var randomTensor = Tensor<T>.CreateRandom(totalElements, 1); // [0, 1]
        var half = NumOps.FromDouble(0.5);

        // Create tensor filled with 0.5 for subtraction
        var halfTensor = new Tensor<T>([totalElements]);
        halfTensor.Fill(half);

        // Transform to [-0.5, 0.5] * scale using Engine ops
        var centeredTensor = Engine.TensorSubtract(randomTensor.Reshape([totalElements]), halfTensor);
        var scaledTensor = Engine.TensorMultiplyScalar(centeredTensor, scale);

        // Copy back to original tensor (preserving shape)
        var reshapedResult = scaledTensor.Reshape(tensor.Shape);
        Array.Copy(reshapedResult.ToArray(), tensor.ToArray(), totalElements);
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
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;
        if (rank < 2)
            throw new ArgumentException("SpatialTransformerLayer expects at least 2D input [height, width].", nameof(input));

        _inputHadChannel = false;
        int channelCount = 1;
        int heightIndex = rank - 2;
        int widthIndex = rank - 1;

        if (rank >= 4 && input.Shape[^1] == 1 && input.Shape[^2] == _inputWidth && input.Shape[^3] == _inputHeight)
        {
            _inputHadChannel = true;
            heightIndex = rank - 3;
            widthIndex = rank - 2;
            channelCount = input.Shape[^1];
        }

        int inputHeight = input.Shape[heightIndex];
        int inputWidth = input.Shape[widthIndex];
        if (inputHeight != _inputHeight || inputWidth != _inputWidth)
            throw new ArgumentException($"Expected input spatial dims [{_inputHeight}, {_inputWidth}] but got [{inputHeight}, {inputWidth}].", nameof(input));

        if (channelCount != 1)
            throw new ArgumentException("SpatialTransformerLayer currently supports single-channel inputs.", nameof(input));

        int batchDims = rank - (_inputHadChannel ? 3 : 2);
        int flatBatch = 1;
        for (int d = 0; d < batchDims; d++)
            flatBatch *= input.Shape[d];

        Tensor<T> inputSpatial = _inputHadChannel
            ? input.Reshape([flatBatch, _inputHeight, _inputWidth, channelCount])
            : input.Reshape([flatBatch, _inputHeight, _inputWidth]);

        Tensor<T> inputNHWC = _inputHadChannel
            ? inputSpatial
            : inputSpatial.Reshape([flatBatch, _inputHeight, _inputWidth, 1]);

        _lastInput = inputNHWC;

        var flattenedInput = _inputHadChannel
            ? inputNHWC.Reshape([flatBatch, _inputHeight * _inputWidth * channelCount])
            : inputSpatial.Reshape([flatBatch, _inputHeight * _inputWidth]);
        _lastFlattenedInput = flattenedInput;

        // First layer: localization1 = flattenedInput @ _localizationWeights1 + _localizationBias1
        var localization1 = Engine.TensorMatMul(flattenedInput, _localizationWeights1);
        var bias1Expanded = _localizationBias1.Reshape([1, _localizationBias1.Shape[0]]);
        localization1 = Engine.TensorBroadcastAdd(localization1, bias1Expanded);
        localization1 = ApplyActivation(localization1);
        _lastLocalization1 = localization1;

        // Second layer: transformationParams = localization1 @ _localizationWeights2 + _localizationBias2
        var transformationParams = Engine.TensorMatMul(localization1, _localizationWeights2);
        var bias2Expanded = _localizationBias2.Reshape([1, _localizationBias2.Shape[0]]);
        transformationParams = Engine.TensorBroadcastAdd(transformationParams, bias2Expanded);

        _lastTransformationMatrix = ConvertToTransformationMatrix(transformationParams);

        var theta = _lastTransformationMatrix;
        var grid = Engine.AffineGrid(theta, _outputHeight, _outputWidth);
        var output = Engine.GridSample(inputNHWC, grid);

        _lastOutput = output;

        if (_inputHadChannel)
        {
            var outShape = new int[batchDims + 3];
            for (int d = 0; d < batchDims; d++)
                outShape[d] = _originalInputShape[d];
            outShape[batchDims] = _outputHeight;
            outShape[batchDims + 1] = _outputWidth;
            outShape[batchDims + 2] = channelCount;
            return output.Reshape(outShape);
        }

        var outputNoChannel = output.Reshape([flatBatch, _outputHeight, _outputWidth]);
        if (batchDims == 0)
        {
            return outputNoChannel.Reshape([_outputHeight, _outputWidth]);
        }

        var outputShape = new int[batchDims + 2];
        for (int d = 0; d < batchDims; d++)
            outputShape[d] = _originalInputShape[d];
        outputShape[batchDims] = _outputHeight;
        outputShape[batchDims + 1] = _outputWidth;

        return outputNoChannel.Reshape(outputShape);
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
    private Tensor<T> ConvertToTransformationMatrix(Tensor<T> transformationParams)
    {
        if (transformationParams.Shape.Length != 2 || transformationParams.Shape[1] != 6)
        {
            throw new ArgumentException("Transformation parameters should be a [batch, 6] tensor.");
        }

        int batchSize = transformationParams.Shape[0];
        var tensor = new Tensor<T>([batchSize, 2, 3]);

        T scale = NumOps.FromDouble(0.1);
        for (int b = 0; b < batchSize; b++)
        {
            T theta11 = transformationParams[b, 0];
            T theta12 = transformationParams[b, 1];
            T theta13 = transformationParams[b, 2];
            T theta21 = transformationParams[b, 3];
            T theta22 = transformationParams[b, 4];
            T theta23 = transformationParams[b, 5];

            theta11 = MathHelper.Tanh(NumOps.Multiply(theta11, scale));
            theta12 = MathHelper.Tanh(NumOps.Multiply(theta12, scale));
            theta21 = MathHelper.Tanh(NumOps.Multiply(theta21, scale));
            theta22 = MathHelper.Tanh(NumOps.Multiply(theta22, scale));

            theta13 = MathHelper.Tanh(NumOps.Multiply(theta13, scale));
            theta23 = MathHelper.Tanh(NumOps.Multiply(theta23, scale));

            theta11 = NumOps.Add(theta11, NumOps.One);
            theta22 = NumOps.Add(theta22, NumOps.One);

            tensor[b, 0, 0] = theta11;
            tensor[b, 0, 1] = theta12;
            tensor[b, 0, 2] = theta13;
            tensor[b, 1, 0] = theta21;
            tensor[b, 1, 1] = theta22;
            tensor[b, 1, 2] = theta23;
        }

        return tensor;
    }


    // Removed GenerateOutputGrid (unused)

    // Removed SampleInputImage (unused)

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
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastTransformationMatrix == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        // Build theta tensor replicated across batch using Engine ops
        var thetaExpanded = Engine.TensorExpandDims(_lastTransformationMatrix, 0); // [1, 2, 3]
        var theta = Engine.TensorTile(thetaExpanded, [batchSize, 1, 1]); // [batch, 2, 3]

        // Autodiff graph for sampling backward
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "st_input", requiresGradient: true);
        var thetaNode = Autodiff.TensorOperations<T>.Variable(theta, "st_theta", requiresGradient: true);
        var gridNode = Autodiff.TensorOperations<T>.AffineGrid(thetaNode, _outputHeight, _outputWidth);
        var outputNode = Autodiff.TensorOperations<T>.GridSample(inputNode, gridNode);
        outputNode.Gradient = outputGradient;
        outputNode.Backward();

        // Propagate theta gradient into localization network
        if (thetaNode.Gradient != null)
        {
            var thetaGradFlat = thetaNode.Gradient.Reshape([batchSize, 6]);

            // Gradients for second layer
            var lastLoc1 = _lastLocalization1!;
            var loc1T = Engine.TensorTranspose(lastLoc1);
            _localizationWeights2Gradient = Engine.TensorMatMul(loc1T, thetaGradFlat);
            _localizationBias2Gradient = Engine.ReduceSum(thetaGradFlat, new[] { 0 }, keepDims: false);

            // Backprop to localization1
            var weights2T = Engine.TensorTranspose(_localizationWeights2);
            var loc1Back = Engine.TensorMatMul(thetaGradFlat, weights2T);

            // Activation derivative for localization1
            Tensor<T> activationDeriv;
            if (VectorActivation != null)
            {
                activationDeriv = VectorActivation.Derivative(lastLoc1);
            }
            else if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T>)
            {
                var act = ScalarActivation;
                activationDeriv = lastLoc1.Transform((x, _) => act.Derivative(x));
            }
            else
            {
                activationDeriv = new Tensor<T>(lastLoc1.Shape);
                activationDeriv.Fill(NumOps.One);
            }

            var loc1Grad = Engine.TensorMultiply(loc1Back, activationDeriv);

            // Gradients for first layer
            var lastFlat = _lastFlattenedInput!;
            var flatInputT = Engine.TensorTranspose(lastFlat);
            _localizationWeights1Gradient = Engine.TensorMatMul(flatInputT, loc1Grad);
            _localizationBias1Gradient = Engine.ReduceSum(loc1Grad, new[] { 0 }, keepDims: false);

            // Input gradient contribution from localization network
            var weights1T = Engine.TensorTranspose(_localizationWeights1);
            var inputLocGradFlat = Engine.TensorMatMul(loc1Grad, weights1T);
            var inputLocGrad = inputLocGradFlat.Reshape(_lastInput.Shape);

            if (inputNode.Gradient != null)
            {
                inputNode.Gradient = Engine.TensorAdd(inputNode.Gradient, inputLocGrad);
            }
        }

        var inputGradient = inputNode.Gradient ?? throw new InvalidOperationException("Spatial transformer backward failed.");

        // Restore gradient to original input shape
        if (_originalInputShape != null && _originalInputShape.Length != 2)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation via AffineGrid and GridSample operations.
    /// The localization network gradients are computed using standard matrix operations,
    /// while the spatial transformation uses the specialized autodiff operations.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastTransformationMatrix == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        // For now, SpatialTransformer with full localization network autodiff is complex
        // We use the specialized AffineGrid and GridSample operations for the transformation part
        // but handle the localization network manually to avoid excessive complexity

        // Convert transformation matrix to tensor [batch, 2, 3] using Engine ops
        // Note: Current implementation uses single transformation matrix for simplicity
        var thetaExpanded = Engine.TensorExpandDims(_lastTransformationMatrix, 0); // [1, 2, 3]
        var thetaTensor = Engine.TensorTile(thetaExpanded, [batchSize, 1, 1]); // [batch, 2, 3]

        // Create computation nodes
        var inputNode = Autodiff.TensorOperations<T>.Variable(
            _lastInput,
            "input",
            requiresGradient: true);

        var thetaNode = Autodiff.TensorOperations<T>.Variable(
            thetaTensor,
            "theta",
            requiresGradient: true);

        // Apply AffineGrid to generate sampling grid
        var gridNode = Autodiff.TensorOperations<T>.AffineGrid(
            thetaNode,
            _outputHeight,
            _outputWidth);

        // Apply GridSample to sample from input
        var outputNode = Autodiff.TensorOperations<T>.GridSample(
            inputNode,
            gridNode);

        // Set the output gradient
        outputNode.Gradient = outputGradient;

        // Perform backward pass with inlined topological sort
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

        while (stack.Count > 0)
        {
            var (currentNode, processed) = stack.Pop();
            if (visited.Contains(currentNode)) continue;

            if (processed)
            {
                visited.Add(currentNode);
                topoOrder.Add(currentNode);
            }
            else
            {
                stack.Push((currentNode, true));
                foreach (var parent in currentNode.Parents)
                {
                    if (!visited.Contains(parent))
                    {
                        stack.Push((parent, false));
                    }
                }
            }
        }

        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // For the localization network parameters, we need to backpropagate from theta gradients
        // This requires converting theta gradients back and computing localization network gradients
        if (thetaNode.Gradient != null)
        {
            // Initialize parameter gradients as Tensor<T>
            _localizationWeights1Gradient = new Tensor<T>([_localizationWeights1.Shape[0], _localizationWeights1.Shape[1]]);
            _localizationWeights1Gradient.Fill(NumOps.Zero);
            _localizationBias1Gradient = new Tensor<T>([_localizationBias1.Shape[0]]);
            _localizationBias1Gradient.Fill(NumOps.Zero);
            _localizationWeights2Gradient = new Tensor<T>([_localizationWeights2.Shape[0], _localizationWeights2.Shape[1]]);
            _localizationWeights2Gradient.Fill(NumOps.Zero);
            _localizationBias2Gradient = new Tensor<T>([_localizationBias2.Shape[0]]);
            _localizationBias2Gradient.Fill(NumOps.Zero);

            // Extract theta gradient (averaging across batch for simplicity in this implementation)
            // Average theta gradient across batch using Engine ops
            // Reshape [batch, 2, 3] -> [batch, 6], then mean
            var thetaGradBatch = thetaNode.Gradient.Reshape([batchSize, 6]);
            var thetaGrad = Engine.ReduceMean(thetaGradBatch, [0], keepDims: true); // [1, 6]

            // Backpropagate through localization network using Engine operations
            // This is a simplified version - full implementation would process each batch item
            var flattenedInput = _lastInput.Reshape([batchSize, _inputHeight * _inputWidth]);
            var localization1 = Engine.TensorMatMul(flattenedInput, _localizationWeights1);
            var bias1Expanded = _localizationBias1.Reshape([1, _localizationBias1.Shape[0]]);
            localization1 = Engine.TensorBroadcastAdd(localization1, bias1Expanded);

            // Gradient for localization bias2: copy from thetaGrad
            Engine.TensorCopy(thetaGrad.Reshape([6]), _localizationBias2Gradient);

            // Gradient for localization weights2 using outer product
            // loc1[0] is [32], thetaGrad[0] is [6] -> outer product gives [32, 6]
            var loc1First = localization1.Slice(0, 0, 1).Reshape([32]); // First batch item [32]
            var thetaGradFlat = thetaGrad.Reshape([6]);
            _localizationWeights2Gradient = Engine.TensorOuterProduct(loc1First, thetaGradFlat);
        }

        // Return input gradient
        var inputGradient = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");

        // Restore gradient to original input shape
        if (_originalInputShape != null && _originalInputShape.Length != 2)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    // Removed BackwardSampler (unused)

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
    private Tensor<T> BackwardGridGenerator(Tensor<T> samplerGradient, Tensor<T> transformationMatrix)
    {
        // Fully vectorized implementation using Engine operations
        // samplerGradient shape: [outputHeight, outputWidth, 2]

        // Extract gradX and gradY channels using TensorSliceAxis
        var gradX2D = Engine.TensorSliceAxis(samplerGradient, 2, 0); // [H, W]
        var gradY2D = Engine.TensorSliceAxis(samplerGradient, 2, 1); // [H, W]

        // Flatten for subsequent operations
        var flatGradX = gradX2D.Reshape([_outputHeight * _outputWidth]);
        var flatGradY = gradY2D.Reshape([_outputHeight * _outputWidth]);

        // Create normalized coordinate grids [-1, 1] using TensorLinspace and TensorMeshgrid
        var xRange = Engine.TensorLinspace(NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0), _outputWidth);
        var yRange = Engine.TensorLinspace(NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0), _outputHeight);
        var (xGrid, yGrid) = Engine.TensorMeshgrid(xRange, yRange);

        // Flatten coordinates
        var xCoords = xGrid.Reshape([_outputHeight * _outputWidth]);
        var yCoords = yGrid.Reshape([_outputHeight * _outputWidth]);

        // Compute gradient contributions using Engine operations
        // grad[0,0] = sum(gradX * xCoords)
        // grad[0,1] = sum(gradX * yCoords)
        // grad[0,2] = sum(gradX)
        // grad[1,0] = sum(gradY * xCoords)
        // grad[1,1] = sum(gradY * yCoords)
        // grad[1,2] = sum(gradY)

        var gradX_xCoords = Engine.TensorMultiply(flatGradX, xCoords);
        var gradX_yCoords = Engine.TensorMultiply(flatGradX, yCoords);
        var gradY_xCoords = Engine.TensorMultiply(flatGradY, xCoords);
        var gradY_yCoords = Engine.TensorMultiply(flatGradY, yCoords);

        // Sum all elements using ReduceSum
        var grad00 = Engine.ReduceSum(gradX_xCoords, [0], keepDims: false);
        var grad01 = Engine.ReduceSum(gradX_yCoords, [0], keepDims: false);
        var grad02 = Engine.ReduceSum(flatGradX, [0], keepDims: false);
        var grad10 = Engine.ReduceSum(gradY_xCoords, [0], keepDims: false);
        var grad11 = Engine.ReduceSum(gradY_yCoords, [0], keepDims: false);
        var grad12 = Engine.ReduceSum(flatGradY, [0], keepDims: false);

        // Build the result tensor
        var gridGeneratorGradient = new Tensor<T>([2, 3]);
        gridGeneratorGradient[0, 0] = grad00.GetFlat(0);
        gridGeneratorGradient[0, 1] = grad01.GetFlat(0);
        gridGeneratorGradient[0, 2] = grad02.GetFlat(0);
        gridGeneratorGradient[1, 0] = grad10.GetFlat(0);
        gridGeneratorGradient[1, 1] = grad11.GetFlat(0);
        gridGeneratorGradient[1, 2] = grad12.GetFlat(0);

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
    private void BackwardLocalizationNetwork(Tensor<T> gridGeneratorGradient, Tensor<T> input)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        // Flatten input
        var flattenedInput = input.Reshape([input.Shape[0], _inputHeight * _inputWidth * input.Shape[3]]);

        // Backward pass through the second layer of localization network
        // gridGeneratorGradient is [2, 3], reshape to [1, 6] for dL_dTheta
        var dL_dTheta = gridGeneratorGradient.Reshape([1, 6]);

        // dL_dL1 = dL_dTheta @ _localizationWeights2.T
        var weights2T = Engine.TensorTranspose(_localizationWeights2);
        var dL_dL1 = Engine.TensorMatMul(dL_dTheta, weights2T);

        // Update weights2 gradient: _localizationWeights2Gradient += flattenedInput.T @ dL_dTheta
        // flattenedInput is [batchSize, inputSize], dL_dTheta is [1, 6]
        // For simplicity, we use the first sample
        var flatInputT = Engine.TensorTranspose(flattenedInput);
        var w2GradUpdate = Engine.TensorMatMul(flatInputT, dL_dTheta);
        // Add gradient update to weights2 gradient using Engine ops
        // Only take relevant part [32, 6]
        var w2UpdateSliced = Engine.TensorSlice(w2GradUpdate, [0, 0], [_localizationWeights2.Shape[0], _localizationWeights2.Shape[1]]);
        _localizationWeights2Gradient = Engine.TensorAdd(_localizationWeights2Gradient!, w2UpdateSliced);

        // Update bias2 gradient: sum over rows of dL_dTheta -> squeeze first row
        var bias2Update = Engine.TensorSqueeze(dL_dTheta, 0); // [6]
        _localizationBias2Gradient = Engine.TensorAdd(_localizationBias2Gradient!, bias2Update);

        // Backward pass through the activation function
        // z1 = flattenedInput @ _localizationWeights1 + bias1
        var z1 = Engine.TensorMatMul(flattenedInput, _localizationWeights1);
        var bias1Expanded = _localizationBias1.Reshape([1, _localizationBias1.Shape[0]]);
        z1 = Engine.TensorAdd(z1, bias1Expanded);

        // Apply activation gradient
        var dL_dZ1 = ApplyActivationGradientTensor(dL_dL1, z1);

        // Backward pass through the first layer of localization network
        // _localizationWeights1Gradient += flattenedInput.T @ dL_dZ1
        var w1GradUpdate = Engine.TensorMatMul(flatInputT, dL_dZ1);
        // Add gradient update using Engine ops - slice to match shape
        var w1UpdateSliced = Engine.TensorSlice(w1GradUpdate, [0, 0], [_localizationWeights1.Shape[0], _localizationWeights1.Shape[1]]);
        _localizationWeights1Gradient = Engine.TensorAdd(_localizationWeights1Gradient!, w1UpdateSliced);

        // Update bias1 gradient: sum over batch dimension (axis 0) of dL_dZ1
        var bias1Update = Engine.ReduceSum(dL_dZ1, [0], keepDims: false); // [32]
        _localizationBias1Gradient = Engine.TensorAdd(_localizationBias1Gradient!, bias1Update);
    }

    /// <summary>
    /// Applies the activation function gradient to a tensor.
    /// </summary>
    /// <param name="upstream">The upstream gradient coming from the next layer.</param>
    /// <param name="z">The input to the activation function in the forward pass.</param>
    /// <returns>The gradient after applying the activation function gradient.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the gradient of the activation function to each element of the tensor. It implements
    /// the backpropagation through the activation function by computing how changes in the activation output
    /// affect the activation input.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes in the output of an activation function
    /// relate to changes in its input.
    ///
    /// During backward propagation:
    /// - The activation function gradient shows how sensitive the output is to small changes in the input
    /// - This method applies this gradient to each value in the tensor
    /// - The result shows how to adjust the inputs to the activation function to improve the output
    ///
    /// This is a crucial step in the backpropagation algorithm that allows neural networks to learn.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyActivationGradientTensor(Tensor<T> upstream, Tensor<T> z)
    {
        // Use the tensor-based ApplyActivationDerivative from LayerBase
        // This handles both scalar and vector activation functions efficiently
        return ApplyActivationDerivative(z, upstream);
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

        // Update using Engine operations: w = w - lr * gradient
        var scaledW1Grad = Engine.TensorMultiplyScalar(_localizationWeights1Gradient, learningRate);
        _localizationWeights1 = Engine.TensorSubtract(_localizationWeights1, scaledW1Grad);

        var scaledB1Grad = Engine.TensorMultiplyScalar(_localizationBias1Gradient, learningRate);
        _localizationBias1 = Engine.TensorSubtract(_localizationBias1, scaledB1Grad);

        var scaledW2Grad = Engine.TensorMultiplyScalar(_localizationWeights2Gradient, learningRate);
        _localizationWeights2 = Engine.TensorSubtract(_localizationWeights2, scaledW2Grad);

        var scaledB2Grad = Engine.TensorMultiplyScalar(_localizationBias2Gradient, learningRate);
        _localizationBias2 = Engine.TensorSubtract(_localizationBias2, scaledB2Grad);
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
        // Use Vector<T>.Concatenate for efficient parameter collection
        var flatW1 = new Vector<T>(_localizationWeights1.ToArray());
        var flatB1 = new Vector<T>(_localizationBias1.ToArray());
        var flatW2 = new Vector<T>(_localizationWeights2.ToArray());
        var flatB2 = new Vector<T>(_localizationBias2.ToArray());

        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(flatW1, flatB1),
            Vector<T>.Concatenate(flatW2, flatB2));
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
        int w1Size = _localizationWeights1.Shape[0] * _localizationWeights1.Shape[1];
        int b1Size = _localizationBias1.Shape[0];
        int w2Size = _localizationWeights2.Shape[0] * _localizationWeights2.Shape[1];
        int b2Size = _localizationBias2.Shape[0];
        int totalParams = w1Size + b1Size + w2Size + b2Size;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set localization weights1 using Tensor<T>.FromVector with slice
        var w1Params = parameters.Slice(index, w1Size);
        index += w1Size;
        _localizationWeights1 = Tensor<T>.FromVector(w1Params).Reshape([_localizationWeights1.Shape[0], _localizationWeights1.Shape[1]]);

        // Set localization bias1
        var b1Params = parameters.Slice(index, b1Size);
        index += b1Size;
        _localizationBias1 = Tensor<T>.FromVector(b1Params);

        // Set localization weights2
        var w2Params = parameters.Slice(index, w2Size);
        index += w2Size;
        _localizationWeights2 = Tensor<T>.FromVector(w2Params).Reshape([_localizationWeights2.Shape[0], _localizationWeights2.Shape[1]]);

        // Set localization bias2
        var b2Params = parameters.Slice(index, b2Size);
        _localizationBias2 = Tensor<T>.FromVector(b2Params);
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

    /// <summary>
    /// Computes the auxiliary loss for this layer based on transformation regularization.
    /// </summary>
    /// <returns>The computed auxiliary loss value.</returns>
    public T ComputeAuxiliaryLoss()
    {
        // Placeholder - full implementation would regularize transformation parameters
        // to prevent extreme transformations
        _lastTransformationLoss = NumOps.Zero;
        return _lastTransformationLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the auxiliary loss computation.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about the auxiliary loss.</returns>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalTransformationLoss", $"{_lastTransformationLoss}" },
            { "TransformationWeight", $"{AuxiliaryLossWeight}" },
            { "UseTransformationLoss", UseAuxiliaryLoss.ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_localizationWeights1 == null || _localizationBias1 == null ||
            _localizationWeights2 == null || _localizationBias2 == null)
            throw new InvalidOperationException("Layer not initialized. Call Initialize() first.");

        // Create input node
        var symbolicInput = new Tensor<T>(InputShape);
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Localization network: 2-layer fully connected network
        // Layer 1: Flatten input and apply first fully connected layer
        int batchSize = InputShape[0];
        var flattenedShape = new int[] { batchSize, _inputHeight * _inputWidth };
        var flattenedInput = Autodiff.TensorOperations<T>.Reshape(inputNode, flattenedShape);

        // Use tensors directly (already Tensor<T>)
        var weights1Node = Autodiff.TensorOperations<T>.Constant(_localizationWeights1, "localization_weights1");
        var bias1Node = Autodiff.TensorOperations<T>.Constant(_localizationBias1, "localization_bias1");

        // First layer: MatMul + Add + Activation
        var localization1 = Autodiff.TensorOperations<T>.MatrixMultiply(flattenedInput, weights1Node);
        localization1 = Autodiff.TensorOperations<T>.Add(localization1, bias1Node);

        // Apply activation function
        if (ScalarActivation != null && ScalarActivation.SupportsJitCompilation)
            localization1 = ScalarActivation.ApplyToGraph(localization1);
        else if (VectorActivation != null && VectorActivation.SupportsJitCompilation)
            localization1 = VectorActivation.ApplyToGraph(localization1);
        else
            localization1 = Autodiff.TensorOperations<T>.Tanh(localization1);

        // Layer 2: Second fully connected layer to get transformation parameters
        var weights2Node = Autodiff.TensorOperations<T>.Constant(_localizationWeights2, "localization_weights2");
        var bias2Node = Autodiff.TensorOperations<T>.Constant(_localizationBias2, "localization_bias2");

        var transformationParams = Autodiff.TensorOperations<T>.MatrixMultiply(localization1, weights2Node);
        transformationParams = Autodiff.TensorOperations<T>.Add(transformationParams, bias2Node);

        // Reshape transformation parameters to [batch, 2, 3] for affine transformation matrix
        var thetaShape = new int[] { batchSize, 2, 3 };
        var thetaNode = Autodiff.TensorOperations<T>.Reshape(transformationParams, thetaShape);

        // Generate sampling grid using AffineGrid
        var gridNode = Autodiff.TensorOperations<T>.AffineGrid(thetaNode, _outputHeight, _outputWidth);

        // Sample from input using GridSample
        var outputNode = Autodiff.TensorOperations<T>.GridSample(inputNode, gridNode);

        return outputNode;
    }

    public override bool SupportsJitCompilation => _localizationWeights1 != null && _localizationBias1 != null &&
                                                     _localizationWeights2 != null && _localizationBias2 != null;

}
