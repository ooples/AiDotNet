namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Locally Connected layer which applies different filters to different regions of the input, unlike a convolutional layer which shares filters.
/// </summary>
/// <remarks>
/// <para>
/// The Locally Connected layer is similar to a convolutional layer in that it applies filters to local regions 
/// of the input, but differs in that it uses different filter weights for each spatial location. This increases
/// the number of parameters and the expressiveness of the model, but reduces generalization capabilities.
/// It's useful when the patterns in different regions of the input are inherently different, such as in
/// face recognition where different parts of a face have different characteristics.
/// </para>
/// <para><b>For Beginners:</b> This layer is like a specialized convolutional layer where each region gets its own unique filter.
/// 
/// Think of a Locally Connected layer like having specialized detectors for different regions:
/// - In a regular convolutional layer, the same filter slides across the entire input
/// - In a locally connected layer, each position has its own unique filter
/// - This means the layer can learn location-specific features
/// 
/// For example, in face recognition:
/// - A convolutional layer would use the same detector for eyes, whether looking at the top-left or bottom-right
/// - A locally connected layer would use different detectors depending on where it's looking
/// 
/// This specialization increases the model's power but:
/// - Requires more parameters
/// - May not generalize as well to new examples
/// - Is more computationally intensive
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LocallyConnectedLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight tensors for the locally connected filters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the filter weights, with a separate set of weights for each spatial location.
    /// The shape is [outputHeight, outputWidth, outputChannels, kernelSize, kernelSize, inputChannels],
    /// which captures the fact that there's a unique kernel for each output position.
    /// </para>
    /// <para><b>For Beginners:</b> These are the learnable filter values specific to each location.
    /// 
    /// The weights tensor:
    /// - Contains the filter values for each position
    /// - Is 6-dimensional to capture all the necessary information
    /// - Has different filters for each (x,y) position in the output
    /// 
    /// The 6 dimensions are:
    /// 1. Output height position
    /// 2. Output width position
    /// 3. Output channel
    /// 4. Kernel height position
    /// 5. Kernel width position
    /// 6. Input channel
    /// 
    /// This complex structure allows each position to have its own specialized filter.
    /// </para>
    /// </remarks>
    private Tensor<T> _weights = default!;

    /// <summary>
    /// The bias values for each output channel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the bias values that are added to each output channel after the filter
    /// application. These are shared across spatial locations but different for each output channel.
    /// </para>
    /// <para><b>For Beginners:</b> These are additional learnable values added to each output channel.
    /// 
    /// The biases:
    /// - Are added to each output after applying the filters
    /// - Help the network learn by providing an adjustable baseline
    /// - Are the same for a given channel across all spatial positions
    /// 
    /// They're like a "starting point" that the network can adjust during learning.
    /// </para>
    /// </remarks>
    private Vector<T> _biases = default!;

    /// <summary>
    /// Stores the input tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the gradients for the weights calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _weightGradients;

    /// <summary>
    /// Stores the gradients for the biases calculated during the backward pass.
    /// </summary>
    private Vector<T>? _biasGradients;

    /// <summary>
    /// The height of the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the height dimension of the input tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This is the height (number of rows) of the input data.
    /// 
    /// For example, if processing 28x28 images, this would be 28.
    /// </para>
    /// </remarks>
    private readonly int _inputHeight;

    /// <summary>
    /// The width of the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the width dimension of the input tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This is the width (number of columns) of the input data.
    /// 
    /// For example, if processing 28x28 images, this would be 28.
    /// </para>
    /// </remarks>
    private readonly int _inputWidth;

    /// <summary>
    /// The number of channels in the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the number of channels in the input tensor. For RGB images, this would be 3.
    /// For grayscale images, this would be 1. For feature maps from previous layers, this could be any number.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of feature channels in the input data.
    /// 
    /// - For color images: 3 channels (Red, Green, Blue)
    /// - For grayscale images: 1 channel
    /// - For hidden layers: However many features the previous layer produced
    /// </para>
    /// </remarks>
    private readonly int _inputChannels;

    /// <summary>
    /// The height of the output tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the height dimension of the output tensor, calculated as (inputHeight - kernelSize) / stride + 1.
    /// </para>
    /// <para><b>For Beginners:</b> This is the height of the output after applying the filters.
    /// 
    /// It depends on:
    /// - The input height
    /// - The kernel size 
    /// - The stride (how many pixels the filter moves each step)
    /// 
    /// It's calculated as: (inputHeight - kernelSize) / stride + 1
    /// </para>
    /// </remarks>
    private readonly int _outputHeight;

    /// <summary>
    /// The width of the output tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the width dimension of the output tensor, calculated as (inputWidth - kernelSize) / stride + 1.
    /// </para>
    /// <para><b>For Beginners:</b> This is the width of the output after applying the filters.
    /// 
    /// It depends on:
    /// - The input width
    /// - The kernel size 
    /// - The stride (how many pixels the filter moves each step)
    /// 
    /// It's calculated as: (inputWidth - kernelSize) / stride + 1
    /// </para>
    /// </remarks>
    private readonly int _outputWidth;

    /// <summary>
    /// The number of channels in the output tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the number of channels in the output tensor, which is the number of different
    /// filters applied at each spatial location.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of features the layer will produce.
    /// 
    /// Each output channel:
    /// - Represents a different feature or pattern the layer is looking for
    /// - Is created by a separate filter
    /// - Helps the network learn more complex representations
    /// 
    /// The more output channels, the more patterns the layer can detect,
    /// but also the more parameters it needs to learn.
    /// </para>
    /// </remarks>
    private readonly int _outputChannels;

    /// <summary>
    /// The size of the kernel (filter) in both height and width dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the size of the square kernel (filter) applied to the input. The kernel
    /// has dimensions kernelSize x kernelSize.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of each filter window that processes the input.
    /// 
    /// The kernel size:
    /// - Determines how large an area each filter covers
    /// - Is the same for both height and width (square filter)
    /// - Affects how local or global the detected features are
    /// 
    /// Common values are 3x3, 5x5, or 7x7. Larger kernels can detect more global patterns
    /// but require more parameters.
    /// </para>
    /// </remarks>
    private readonly int _kernelSize;

    /// <summary>
    /// The stride (step size) of the kernel when moving across the input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents how many pixels the kernel moves in each step when scanning the input.
    /// A stride of 1 means the kernel moves one pixel at a time, while a stride of 2 means it skips
    /// every other pixel.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many pixels the filter moves each step.
    /// 
    /// The stride:
    /// - Controls how much overlap there is between filter applications
    /// - Affects the size of the output (larger stride = smaller output)
    /// - Can help reduce computation by processing fewer positions
    /// 
    /// A stride of 1 means the filter moves one pixel at a time (maximum overlap).
    /// A stride of 2 means the filter jumps two pixels each time (less overlap, smaller output).
    /// </para>
    /// </remarks>
    private readonly int _stride;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> because this layer has trainable parameters (weights and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The LocallyConnectedLayer always returns true because it contains trainable weights and biases.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has parameters that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// The Locally Connected layer always supports training because it has weights 
    /// and biases that are learned during training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="LocallyConnectedLayer{T}"/> class with the specified dimensions, kernel parameters, and element-wise activation function.
    /// </summary>
    /// <param name="inputHeight">The height of the input tensor.</param>
    /// <param name="inputWidth">The width of the input tensor.</param>
    /// <param name="inputChannels">The number of channels in the input tensor.</param>
    /// <param name="outputChannels">The number of channels in the output tensor.</param>
    /// <param name="kernelSize">The size of the kernel (filter) in both height and width dimensions.</param>
    /// <param name="stride">The stride (step size) of the kernel when moving across the input.</param>
    /// <param name="activationFunction">The activation function to apply after the locally connected operation. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Locally Connected layer with the specified dimensions, kernel parameters,
    /// and element-wise activation function. It initializes the weights and biases and calculates the output
    /// dimensions based on the input dimensions, kernel size, and stride.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new locally connected layer with standard activation function.
    /// 
    /// When creating this layer, you specify:
    /// - inputHeight, inputWidth: The dimensions of your input data
    /// - inputChannels: How many channels your input data has
    /// - outputChannels: How many different features you want the layer to detect
    /// - kernelSize: The size of each filter window (e.g., 3 for a 3x3 filter)
    /// - stride: How many pixels the filter moves each step
    /// - activationFunction: What function to apply to the output (default is ReLU)
    /// 
    /// For example, to process 28x28 grayscale images with 16 output features, 3x3 filters,
    /// and a stride of 1, you would use: inputHeight=28, inputWidth=28, inputChannels=1,
    /// outputChannels=16, kernelSize=3, stride=1.
    /// </para>
    /// </remarks>
    public LocallyConnectedLayer(
        int inputHeight, 
        int inputWidth, 
        int inputChannels, 
        int outputChannels, 
        int kernelSize, 
        int stride, 
        IActivationFunction<T>? activationFunction = null)
        : base(
            [inputHeight, inputWidth, inputChannels], 
            [
                (inputHeight - kernelSize) / stride + 1, 
                (inputWidth - kernelSize) / stride + 1, 
                outputChannels
            ], 
            activationFunction ?? new ReLUActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _outputHeight = (inputHeight - kernelSize) / stride + 1;
        _outputWidth = (inputWidth - kernelSize) / stride + 1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;

        // Initialize weights and biases
        _weights = new Tensor<T>([_outputHeight, _outputWidth, _outputChannels, _kernelSize, _kernelSize, _inputChannels]);
        _biases = new Vector<T>(_outputChannels);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LocallyConnectedLayer{T}"/> class with the specified dimensions, kernel parameters, and vector activation function.
    /// </summary>
    /// <param name="inputHeight">The height of the input tensor.</param>
    /// <param name="inputWidth">The width of the input tensor.</param>
    /// <param name="inputChannels">The number of channels in the input tensor.</param>
    /// <param name="outputChannels">The number of channels in the output tensor.</param>
    /// <param name="kernelSize">The size of the kernel (filter) in both height and width dimensions.</param>
    /// <param name="stride">The stride (step size) of the kernel when moving across the input.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after the locally connected operation. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Locally Connected layer with the specified dimensions, kernel parameters,
    /// and vector activation function. Vector<double> activation functions operate on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new locally connected layer with an advanced vector-based activation.
    /// 
    /// Vector<double> activation functions:
    /// - Process entire groups of numbers together, not just one at a time
    /// - Can capture relationships between different features
    /// - May be more powerful for complex patterns
    /// 
    /// Otherwise, this constructor works just like the standard one, setting up the layer with:
    /// - The specified dimensions and parameters
    /// - Proper calculation of output dimensions
    /// - Initialization of weights and biases
    /// </para>
    /// </remarks>
    public LocallyConnectedLayer(
        int inputHeight, 
        int inputWidth, 
        int inputChannels, 
        int outputChannels, 
        int kernelSize, 
        int stride, 
        IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(
            [inputHeight, inputWidth, inputChannels], 
            [
                (inputHeight - kernelSize) / stride + 1, 
                (inputWidth - kernelSize) / stride + 1, 
                outputChannels
            ], 
            vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _outputHeight = (inputHeight - kernelSize) / stride + 1;
        _outputWidth = (inputWidth - kernelSize) / stride + 1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;

        // Initialize weights and biases
        _weights = new Tensor<T>([_outputHeight, _outputWidth, _outputChannels, _kernelSize, _kernelSize, _inputChannels]);
        _biases = new Vector<T>(_outputChannels);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights using Xavier initialization, which scales the random values
    /// based on the number of input and output connections. The biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for the layer's weights and biases.
    /// 
    /// For weights:
    /// - Values are randomized with Xavier initialization
    /// - This helps prevent the signals from growing too large or too small
    /// - Different weights for each position, output channel, and filter position
    /// 
    /// For biases:
    /// - All values start at zero
    /// - They will adjust during training to fit the data better
    /// 
    /// Good initialization is important because it affects how quickly and how well the network learns.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // Xavier initialization for weights
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_kernelSize * _kernelSize * _inputChannels + _outputChannels)));
    
        for (int h = 0; h < _outputHeight; h++)
        {
            for (int w = 0; w < _outputWidth; w++)
            {
                for (int oc = 0; oc < _outputChannels; oc++)
                {
                    for (int kh = 0; kh < _kernelSize; kh++)
                    {
                        for (int kw = 0; kw < _kernelSize; kw++)
                        {
                            for (int ic = 0; ic < _inputChannels; ic++)
                            {
                                _weights[h, w, oc, kh, kw, ic] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                            }
                        }
                    }
                }
            }
        }

        // Initialize biases to zero
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Performs the forward pass of the locally connected layer.
    /// </summary>
    /// <param name="input">The input tensor to process. Shape should be [batchSize, inputHeight, inputWidth, inputChannels].</param>
    /// <returns>The output tensor after applying the locally connected operation and activation. Shape will be [batchSize, outputHeight, outputWidth, outputChannels].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the locally connected layer. It applies different filters
    /// to each spatial location of the input, followed by adding biases and applying the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the locally connected filters.
    /// 
    /// During the forward pass:
    /// 1. For each position in the output:
    ///    - Apply a unique filter to the corresponding region of the input
    ///    - Sum up the results of element-wise multiplications
    ///    - Add the bias for the output channel
    /// 2. Apply the activation function to add non-linearity
    /// 
    /// This process is similar to a convolution, but instead of re-using the same filter for all
    /// positions, each position has its own specialized filter.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        var output = new Tensor<T>([batchSize, _outputHeight, _outputWidth, _outputChannels]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < _outputHeight; h++)
            {
                for (int w = 0; w < _outputWidth; w++)
                {
                    for (int oc = 0; oc < _outputChannels; oc++)
                    {
                        T sum = NumOps.Zero;
                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                for (int ic = 0; ic < _inputChannels; ic++)
                                {
                                    int ih = h * _stride + kh;
                                    int iw = w * _stride + kw;
                                    sum = NumOps.Add(sum, NumOps.Multiply(input[b, ih, iw, ic], _weights[h, w, oc, kh, kw, ic]));
                                }
                            }
                        }
                        sum = NumOps.Add(sum, _biases[oc]);
                        output[b, h, w, oc] = sum;
                    }
                }
            }
        }

        // Apply activation function
        return ApplyActivation(output);
    }

    /// <summary>
    /// Performs the backward pass of the locally connected layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the locally connected layer, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradients for the weights and
    /// biases, and returns the gradient with respect to the input for further backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// and parameters should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The layer receives information about how its output contributed to errors
    /// 2. It calculates how the weights and biases should change to reduce errors
    /// 3. It calculates how the input should change, which will be used by earlier layers
    /// 
    /// This process involves:
    /// - Applying the derivative of the activation function
    /// - Computing gradients for each unique filter
    /// - Computing gradients for biases
    /// - Computing how the input should change
    /// 
    /// The method will throw an error if you try to run it before performing a forward pass.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _weightGradients = new Tensor<T>(_weights.Shape);
        _biasGradients = new Vector<T>(_biases.Length);

        // Apply activation derivative
        outputGradient = ApplyActivationDerivative(_lastInput, outputGradient);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < _outputHeight; h++)
            {
                for (int w = 0; w < _outputWidth; w++)
                {
                    for (int oc = 0; oc < _outputChannels; oc++)
                    {
                        T gradOutput = outputGradient[b, h, w, oc];
                        _biasGradients[oc] = NumOps.Add(_biasGradients[oc], gradOutput);

                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                for (int ic = 0; ic < _inputChannels; ic++)
                                {
                                    int ih = h * _stride + kh;
                                    int iw = w * _stride + kw;
                                    T inputValue = _lastInput[b, ih, iw, ic];
                                    _weightGradients[h, w, oc, kh, kw, ic] = NumOps.Add(_weightGradients[h, w, oc, kh, kw, ic], NumOps.Multiply(gradOutput, inputValue));
                                    inputGradient[b, ih, iw, ic] = NumOps.Add(inputGradient[b, ih, iw, ic], NumOps.Multiply(gradOutput, _weights[h, w, oc, kh, kw, ic]));
                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called before UpdateParameters.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the layer based on the gradients calculated during
    /// the backward pass. The learning rate controls the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - All weights and biases are adjusted to reduce prediction errors
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time, gradually improving
    /// its ability to extract useful features from the input.
    /// 
    /// The method will throw an error if you try to run it before performing a backward pass.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightGradients == null || _biasGradients == null)
        {
            throw new InvalidOperationException("UpdateParameters called before Backward. Gradients are null.");
        }

        // Update weights
        for (int h = 0; h < _outputHeight; h++)
        {
            for (int w = 0; w < _outputWidth; w++)
            {
                for (int oc = 0; oc < _outputChannels; oc++)
                {
                    for (int kh = 0; kh < _kernelSize; kh++)
                    {
                        for (int kw = 0; kw < _kernelSize; kw++)
                        {
                            for (int ic = 0; ic < _inputChannels; ic++)
                            {
                                T update = NumOps.Multiply(learningRate, _weightGradients[h, w, oc, kh, kw, ic]);
                                _weights[h, w, oc, kh, kw, ic] = NumOps.Subtract(_weights[h, w, oc, kh, kw, ic], update);
                            }
                        }
                    }
                }
            }
        }

        // Update biases
        for (int oc = 0; oc < _outputChannels; oc++)
        {
            T update = NumOps.Multiply(learningRate, _biasGradients[oc]);
            _biases[oc] = NumOps.Subtract(_biases[oc], update);
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) and combines them into a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving and loading
    /// model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include all the unique filter weights (which can be very many!) and biases
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// For locally connected layers, this vector can be very large due to the
    /// unique filters for each spatial location.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _weights.Length + _biases.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy weights parameters
        for (int h = 0; h < _outputHeight; h++)
        {
            for (int w = 0; w < _outputWidth; w++)
            {
                for (int oc = 0; oc < _outputChannels; oc++)
                {
                    for (int kh = 0; kh < _kernelSize; kh++)
                    {
                        for (int kw = 0; kw < _kernelSize; kw++)
                        {
                            for (int ic = 0; ic < _inputChannels; ic++)
                            {
                                parameters[index++] = _weights[h, w, oc, kh, kw, ic];
                            }
                        }
                    }
                }
            }
        }

        // Copy bias parameters
        for (int oc = 0; oc < _outputChannels; oc++)
        {
            parameters[index++] = _biases[oc];
        }

        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all the weights and biases of the layer from a single vector of parameters.
    /// The vector must have the correct length to match the total number of parameters in the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The values are distributed to all the weights and biases in the correct order
    /// - Throws an error if the input doesn't match the expected number of parameters
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Setting specific parameter values for testing
    /// 
    /// For locally connected layers, this vector needs to be very large to account for
    /// all the unique filters at each spatial location.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _weights.Length + _biases.Length;
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;
        // Set weights parameters
        for (int h = 0; h < _outputHeight; h++)
        {
            for (int w = 0; w < _outputWidth; w++)
            {
                for (int oc = 0; oc < _outputChannels; oc++)
                {
                    for (int kh = 0; kh < _kernelSize; kh++)
                    {
                        for (int kw = 0; kw < _kernelSize; kw++)
                        {
                            for (int ic = 0; ic < _inputChannels; ic++)
                            {
                                _weights[h, w, oc, kh, kw, ic] = parameters[index++];
                            }
                        }
                    }
                }
            }
        }

        // Set bias parameters
        for (int oc = 0; oc < _outputChannels; oc++)
        {
            _biases[oc] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from forward and backward passes.
    /// This includes the last input tensor and the weight and bias gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input from the last forward pass is cleared
    /// - All gradient information from the last backward pass is cleared
    /// - The layer is ready for new data without being influenced by previous data
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// It helps ensure that each training or prediction batch is processed independently.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _weightGradients = null;
        _biasGradients = null;
    }
}