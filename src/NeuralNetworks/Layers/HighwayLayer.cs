namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Highway Neural Network layer that allows information to flow unchanged through the network.
/// </summary>
/// <remarks>
/// <para>
/// A Highway Layer enables networks to effectively train even when they are very deep by introducing
/// "gating units" which learn to selectively pass or transform information. Unlike regular feed-forward
/// layers, highway layers have two "lanes": the transform lane that processes input data and the bypass
/// lane that passes information unchanged. The balance between these two lanes is controlled by a learned
/// gating mechanism.
/// </para>
/// <para><b>For Beginners:</b> This layer helps solve a common problem in deep neural networks: difficulty in training very deep networks.
/// 
/// Think of the Highway Layer like a road with two lanes:
/// - The "transform lane" processes the data like a regular neural network layer
/// - The "bypass lane" lets information pass through unchanged
/// - A "gate" controls how much information flows through each lane
/// 
/// For example, when processing an image, the gate might let basic features like edges pass through
/// directly while sending more complex features through the transform lane for further processing.
/// 
/// This helps the network train more effectively because important information can flow more easily
/// through many layers without being lost or distorted.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HighwayLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight matrix used to transform the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix contains the learnable parameters that transform the input features. The dimensions are
    /// [inputDimension, inputDimension] because highway layers maintain the same dimensionality for input and output.
    /// </para>
    /// <para><b>For Beginners:</b> These weights determine how the input data is transformed in the transform lane.
    /// 
    /// Think of these weights as:
    /// - Filters that extract and combine information from the input
    /// - Learnable parameters that are adjusted during training
    /// - The "processing" part of the highway layer
    /// 
    /// During training, these weights are adjusted to better recognize important patterns in your data.
    /// </para>
    /// </remarks>
    private Matrix<T> _transformWeights = default!;

    /// <summary>
    /// The bias vector added to the transformed input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the learnable bias parameters that are added to the transformed input. Adding a bias
    /// allows the layer to shift the activation function's output.
    /// </para>
    /// <para><b>For Beginners:</b> The transform bias is like a "default value" or "starting point" for each feature.
    /// 
    /// It helps the layer by:
    /// - Allowing outputs to be non-zero even when inputs are zero
    /// - Giving the model flexibility to fit data better
    /// - Providing an adjustable "baseline" for the transformation
    /// 
    /// It's like setting the initial position before fine-tuning.
    /// </para>
    /// </remarks>
    private Vector<T> _transformBias = default!;

    /// <summary>
    /// The weight matrix used to compute the gate values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix contains the learnable parameters that determine how much of the transformed output versus the
    /// original input should be used for each feature. The dimensions are [inputDimension, inputDimension].
    /// </para>
    /// <para><b>For Beginners:</b> These weights control the "traffic signals" of the highway layer.
    /// 
    /// The gate weights:
    /// - Determine which lane (transform or bypass) each piece of information should take
    /// - Learn which features are better left unchanged and which need transformation
    /// - Act as the decision-making mechanism of the highway layer
    /// 
    /// During training, these weights learn the optimal balance between preserving and transforming the input.
    /// </para>
    /// </remarks>
    private Matrix<T> _gateWeights = default!;

    /// <summary>
    /// The bias vector added to the gate computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the learnable bias parameters that are added to the gate computation. Initially biased
    /// negative to allow more information to flow through the transform lane during early training.
    /// </para>
    /// <para><b>For Beginners:</b> The gate bias controls the default behavior of the gates.
    /// 
    /// It helps the layer by:
    /// - Setting an initial preference for one lane over the other
    /// - Usually starts negative to favor the transform lane during early training
    /// - Gets adjusted during training to find the optimal balance
    /// 
    /// Think of it as setting the default position of the "traffic signals" before the network learns
    /// the best settings for each specific input pattern.
    /// </para>
    /// </remarks>
    private Vector<T> _gateBias = default!;

    /// <summary>
    /// Stores the input tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the output tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Stores the transformed output tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastTransformOutput;

    /// <summary>
    /// Stores the gate output tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastGateOutput;

    /// <summary>
    /// Stores the gradients for the transform weights calculated during the backward pass.
    /// </summary>
    private Matrix<T>? _transformWeightsGradient;

    /// <summary>
    /// Stores the gradients for the transform bias calculated during the backward pass.
    /// </summary>
    private Vector<T>? _transformBiasGradient;

    /// <summary>
    /// Stores the gradients for the gate weights calculated during the backward pass.
    /// </summary>
    private Matrix<T>? _gateWeightsGradient;

    /// <summary>
    /// Stores the gradients for the gate bias calculated during the backward pass.
    /// </summary>
    private Vector<T>? _gateBiasGradient;

    /// <summary>
    /// The element-wise activation function applied to the transform output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the transformed input. Typically, a tanh activation function
    /// is used, which outputs values between -1 and 1, adding non-linearity to the transformation.
    /// </para>
    /// <para><b>For Beginners:</b> This function shapes the transformed data.
    /// 
    /// The activation function:
    /// - Adds non-linearity, helping the network learn complex patterns
    /// - Usually tanh (hyperbolic tangent), which outputs values between -1 and 1
    /// - Helps keep the values in a manageable range
    /// 
    /// Think of it as a way to normalize and shape the information after transformation.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _transformActivation;

    /// <summary>
    /// The element-wise activation function applied to the gate output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This activation function is applied to the gate values. Typically, a sigmoid activation function
    /// is used, which outputs values between 0 and 1, allowing the gates to act as "carry" versus "transform" controllers.
    /// </para>
    /// <para><b>For Beginners:</b> This function shapes the gate values into percentages.
    /// 
    /// The gate activation:
    /// - Usually sigmoid, which outputs values between 0 and 1
    /// - Turns gate values into percentages (0% to 100%)
    /// - 0 means "use only the bypass lane", 1 means "use only the transform lane"
    /// 
    /// This ensures the gates work like mixing controls, blending the transformed and bypassed information.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _gateActivation;

    /// <summary>
    /// The vector activation function applied to the transform output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector activation function is applied to the entire transform output vector at once rather than
    /// element-wise. This can capture dependencies between different elements of the output.
    /// </para>
    /// <para><b>For Beginners:</b> A more advanced way to shape transformed data.
    /// 
    /// A vector activation:
    /// - Works on the entire vector at once, not just individual values
    /// - Can capture relationships between different features
    /// - Potentially more powerful than element-wise activations
    /// 
    /// This is useful for capturing complex patterns that depend on multiple features interacting together.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _vectorTransformActivation;

    /// <summary>
    /// The vector activation function applied to the gate output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector activation function is applied to the entire gate output vector at once rather than
    /// element-wise. This can capture dependencies between different elements of the gates.
    /// </para>
    /// <para><b>For Beginners:</b> A more advanced way to shape gate values.
    /// 
    /// A vector gate activation:
    /// - Works on all gates together, not just individual gates
    /// - Can make decisions based on relationships between features
    /// - Potentially allows for more sophisticated gating mechanisms
    /// 
    /// This allows the gates to consider how different features relate to each other
    /// when deciding what information to pass through each lane.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _vectorGateActivation;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> because this layer has trainable parameters (weights and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The HighwayLayer always returns true because it contains trainable weights and biases.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its internal values during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// The Highway layer always supports training because it has weights and biases that can be updated.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="HighwayLayer{T}"/> class with the specified dimensions and element-wise activation functions.
    /// </summary>
    /// <param name="inputDimension">The dimension of the input and output vectors.</param>
    /// <param name="transformActivation">The activation function for the transform path. Defaults to tanh if not specified.</param>
    /// <param name="gateActivation">The activation function for the gate values. Defaults to sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Highway layer with the specified dimension and element-wise activation functions.
    /// The weights are initialized randomly with a scale factor, and the transform biases are initialized to zero while
    /// the gate biases are initialized to negative values to allow more information flow through the transform path initially.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Highway layer with standard activation functions.
    /// 
    /// When creating a Highway layer, you specify:
    /// - inputDimension: How many features your data has (same for input and output)
    /// - transformActivation: How to shape transformed data (default is tanh, values between -1 and 1)
    /// - gateActivation: How to control the gates (default is sigmoid, values between 0 and 1)
    /// 
    /// The layer automatically initializes with:
    /// - Random weights for both transform and gate paths
    /// - Zero biases for the transform path
    /// - Negative biases for the gates (initially favoring the transform path)
    /// </para>
    /// </remarks>
    public HighwayLayer(int inputDimension, IActivationFunction<T>? transformActivation = null, IActivationFunction<T>? gateActivation = null)
        : base([inputDimension], [inputDimension], transformActivation ?? new TanhActivation<T>())
    {
        _transformWeights = new Matrix<T>(inputDimension, inputDimension);
        _transformBias = new Vector<T>(inputDimension);
        _gateWeights = new Matrix<T>(inputDimension, inputDimension);
        _gateBias = new Vector<T>(inputDimension);

        _transformActivation = transformActivation ?? new TanhActivation<T>();
        _gateActivation = gateActivation ?? new SigmoidActivation<T>();

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="HighwayLayer{T}"/> class with the specified dimensions and vector activation functions.
    /// </summary>
    /// <param name="inputDimension">The dimension of the input and output vectors.</param>
    /// <param name="transformActivation">The vector activation function for the transform path. Defaults to tanh if not specified.</param>
    /// <param name="gateActivation">The vector activation function for the gate values. Defaults to sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Highway layer with the specified dimension and vector activation functions.
    /// Vector<double> activation functions operate on entire vectors rather than individual elements, which can capture
    /// dependencies between different elements of the vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Highway layer with more advanced vector-based activation functions.
    /// 
    /// Vector<double> activation functions:
    /// - Process entire groups of numbers together, not just one at a time
    /// - Can capture relationships between different features
    /// - May be more powerful for complex patterns
    /// 
    /// This constructor is useful when you need the layer to understand how different
    /// features interact with each other, rather than treating each feature independently.
    /// </para>
    /// </remarks>
    public HighwayLayer(int inputDimension, IVectorActivationFunction<T>? transformActivation = null, IVectorActivationFunction<T>? gateActivation = null)
        : base([inputDimension], [inputDimension], transformActivation ?? new TanhActivation<T>())
    {
        _transformWeights = new Matrix<T>(inputDimension, inputDimension);
        _transformBias = new Vector<T>(inputDimension);
        _gateWeights = new Matrix<T>(inputDimension, inputDimension);
        _gateBias = new Vector<T>(inputDimension);

        _vectorTransformActivation = transformActivation ?? new TanhActivation<T>();
        _vectorGateActivation = gateActivation ?? new SigmoidActivation<T>();

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights using a scaled random initialization scheme, which helps with 
    /// training stability. The transform biases are initialized to zero, while the gate biases are 
    /// initialized to negative values to allow more information flow through the transform path initially.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for the layer's weights and biases.
    /// 
    /// For weights:
    /// - Values are randomized to break symmetry (prevent all neurons from learning the same thing)
    /// - The scale factor helps prevent the signals from growing too large or too small
    /// - Both transform and gate weights are initialized the same way
    /// 
    /// For biases:
    /// - Transform biases start at zero
    /// - Gate biases start at -1, which causes the gates to initially favor the transform path
    ///   (allowing the network to learn useful transformations before deciding what to bypass)
    /// 
    /// These initialization choices help the network train more effectively from the beginning.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_transformWeights.Rows + _transformWeights.Columns)));
        InitializeMatrix(_transformWeights, scale);
        InitializeMatrix(_gateWeights, scale);

        for (int i = 0; i < _transformBias.Length; i++)
        {
            _transformBias[i] = NumOps.Zero;
            _gateBias[i] = NumOps.FromDouble(-1.0); // Initialize gate bias to negative values to allow more information flow initially
        }
    }

    /// <summary>
    /// Initializes a matrix with scaled random values.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the provided matrix with random values between -0.5 and 0.5, scaled by the provided scale factor.
    /// This type of initialization helps with training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a matrix with random starting values for weights.
    /// 
    /// The method:
    /// - Generates random numbers between -0.5 and 0.5
    /// - Multiplies them by a scale factor to control their size
    /// - Fills each position in the matrix with these scaled random values
    /// 
    /// Good initialization is important because it affects how quickly and how well the network learns.
    /// The scale factor is calculated based on the size of the layer to help maintain stable gradients
    /// during training.
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
    /// Performs the forward pass of the highway layer.
    /// </summary>
    /// <param name="input">The input tensor to process. Shape should be [batchSize, inputDimension].</param>
    /// <returns>The output tensor with shape [batchSize, inputDimension].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the highway layer according to the formula:
    /// output = gate * transform_output + (1 - gate) * input.
    /// The gate values control how much of the transformed output versus the original input is used for each feature.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the highway layer.
    /// 
    /// During the forward pass:
    /// 1. The transform path processes the input data using weights and activation
    /// 2. The gate controller computes values between 0 and 1 for each feature
    /// 3. The final output mixes the original input and transformed data according to the gate values
    /// 
    /// For example, if a gate value is 0.7, the output will be 70% from the transform path
    /// and 30% directly from the input. This allows the layer to learn which features
    /// should be transformed and which should pass through unchanged.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int inputDimension = input.Shape[1];

        var transformOutput = input.Multiply(_transformWeights).Add(_transformBias);
        transformOutput = ApplyActivation(transformOutput, _transformActivation, _vectorTransformActivation);
        _lastTransformOutput = transformOutput;

        var gateOutput = input.Multiply(_gateWeights).Add(_gateBias);
        gateOutput = ApplyActivation(gateOutput, _gateActivation, _vectorGateActivation);
        _lastGateOutput = gateOutput;

        var output = gateOutput.ElementwiseMultiply(transformOutput)
            .Add(input.ElementwiseMultiply(gateOutput.ElementwiseSubtract(Tensor<T>.CreateDefault(gateOutput.Shape, NumOps.One))));

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Applies the appropriate activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <param name="scalarActivation">The element-wise activation function to apply, if specified.</param>
    /// <param name="vectorActivation">The vector activation function to apply, if specified.</param>
    /// <returns>The activated tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no activation function is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method applies either the element-wise activation function or the vector activation function to the input tensor.
    /// If a vector activation function is provided, it is preferred over the element-wise activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the shaping function to the values.
    /// 
    /// The method:
    /// - First checks if a vector activation function is available (processes all features together)
    /// - If not, uses the element-wise activation function (processes each feature independently)
    /// - Throws an error if neither is available
    /// 
    /// This flexibility allows the layer to use either simple element-wise activations
    /// or more advanced vector-based activations depending on what was specified when creating the layer.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyActivation(Tensor<T> input, IActivationFunction<T>? scalarActivation, IVectorActivationFunction<T>? vectorActivation)
    {
        if (vectorActivation != null)
        {
            return vectorActivation.Activate(input);
        }
        else if (scalarActivation != null)
        {
            return input.Transform((x, _) => scalarActivation.Activate(x));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    /// <summary>
    /// Performs the backward pass of the highway layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the highway layer, which is used during training to propagate
    /// error gradients back through the network. It calculates the gradients for all the weights and biases,
    /// and returns the gradient with respect to the input for further backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// and parameters should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The layer receives information about how its output should change to reduce the overall error
    /// 2. It calculates how the gate values should change to better control the mix of transformed vs. bypassed data
    /// 3. It calculates how the transform parameters should change to better process the input
    /// 4. It determines how the input should change, which will be used by earlier layers
    /// 
    /// This process involves complex calculations that essentially run the layer's logic in reverse,
    /// figuring out how each component contributed to errors in the output.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastTransformOutput == null || _lastGateOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputDimension = _lastInput.Shape[1];

        var gateGradient = outputGradient.ElementwiseMultiply(_lastTransformOutput.ElementwiseSubtract(_lastInput));
        gateGradient = ApplyActivationDerivative(gateGradient, _lastGateOutput, _gateActivation, _vectorGateActivation);

        var transformGradient = outputGradient.ElementwiseMultiply(_lastGateOutput);
        transformGradient = ApplyActivationDerivative(transformGradient, _lastTransformOutput, _transformActivation, _vectorTransformActivation);

        _gateWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(gateGradient).ToMatrix();
        _gateBiasGradient = gateGradient.Sum([0]).ToVector();

        _transformWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(transformGradient).ToMatrix();
        _transformBiasGradient = transformGradient.Sum([0]).ToVector();

        var inputGradient = gateGradient.Multiply(_gateWeights.Transpose())
            .Add(transformGradient.Multiply(_transformWeights.Transpose()))
            .Add(outputGradient.ElementwiseMultiply(_lastGateOutput.ElementwiseSubtract(Tensor<T>.CreateDefault(_lastGateOutput.Shape, NumOps.One))));

        return inputGradient;
    }

    /// <summary>
    /// Applies the derivative of the appropriate activation function to the input tensor.
    /// </summary>
    /// <param name="gradient">The gradient tensor to modify.</param>
    /// <param name="lastOutput">The output from the last forward pass.</param>
    /// <param name="scalarActivation">The element-wise activation function to apply the derivative of, if specified.</param>
    /// <param name="vectorActivation">The vector activation function to apply the derivative of, if specified.</param>
    /// <returns>The gradient tensor with the activation derivative applied.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no activation function is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of either the element-wise activation function or the vector activation function
    /// to the gradient tensor. This is used during the backward pass to calculate gradients for the layer's parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how sensitive the activation function is to changes.
    /// 
    /// During backpropagation (learning):
    /// - We need to know how much a small change in input affects the output
    /// - This method calculates that sensitivity using the derivative of the activation function
    /// - It applies this to the incoming gradient to properly scale the effect of parameter changes
    /// 
    /// Think of it as measuring the "slope" of the activation function at each point,
    /// which tells us how to adjust the parameters to improve the network's performance.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyActivationDerivative(Tensor<T> gradient, Tensor<T> lastOutput, IActivationFunction<T>? scalarActivation, IVectorActivationFunction<T>? vectorActivation)
    {
        if (vectorActivation != null)
        {
            return gradient.ElementwiseMultiply(vectorActivation.Derivative(lastOutput));
        }
        else if (scalarActivation != null)
        {
            return gradient.ElementwiseMultiply(lastOutput.Transform((x, _) => scalarActivation.Derivative(x)));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called before UpdateParameters.</exception>
    /// <remarks>
    /// <para>
    /// This method updates all the weight matrices and bias vectors of the highway layer based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter updates.
    /// This is typically called after the backward pass during training.
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
    /// its ability to decide what information should be transformed and what should pass through unchanged.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_transformWeightsGradient == null || _transformBiasGradient == null || 
            _gateWeightsGradient == null || _gateBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _transformWeights = _transformWeights.Subtract(_transformWeightsGradient.Multiply(learningRate));
        _transformBias = _transformBias.Subtract(_transformBiasGradient.Multiply(learningRate));
        _gateWeights = _gateWeights.Subtract(_gateWeightsGradient.Multiply(learningRate));
        _gateBias = _gateBias.Subtract(_gateBiasGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) and combines them into a single vector.
    /// The parameters are arranged in the following order: transform weights, transform biases, gate weights, gate biases.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include weights and biases from both transform and gate paths
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _transformWeights.Rows * _transformWeights.Columns + 
                          _transformBias.Length + 
                          _gateWeights.Rows * _gateWeights.Columns + 
                          _gateBias.Length;

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy transform weights parameters
        for (int i = 0; i < _transformWeights.Rows; i++)
        {
            for (int j = 0; j < _transformWeights.Columns; j++)
            {
                parameters[index++] = _transformWeights[i, j];
            }
        }

        // Copy transform bias parameters
        for (int i = 0; i < _transformBias.Length; i++)
        {
            parameters[index++] = _transformBias[i];
        }

        // Copy gate weights parameters
        for (int i = 0; i < _gateWeights.Rows; i++)
        {
            for (int j = 0; j < _gateWeights.Columns; j++)
            {
                parameters[index++] = _gateWeights[i, j];
            }
        }

        // Copy gate bias parameters
        for (int i = 0; i < _gateBias.Length; i++)
        {
            parameters[index++] = _gateBias[i];
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
    /// This method sets all the weight matrices and bias vectors of the highway layer from a single vector of parameters.
    /// The parameters should be arranged in the following order: transform weights, transform biases, gate weights, gate biases.
    /// This is useful for loading saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The parameters must be in the right order: transform weights, transform biases, gate weights, gate biases
    /// - This maintains the same structure used by GetParameters
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedLength = _transformWeights.Rows * _transformWeights.Columns + 
                             _transformBias.Length + 
                             _gateWeights.Rows * _gateWeights.Columns + 
                             _gateBias.Length;

        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set transform weights parameters
        for (int i = 0; i < _transformWeights.Rows; i++)
        {
            for (int j = 0; j < _transformWeights.Columns; j++)
            {
                _transformWeights[i, j] = parameters[index++];
            }
        }

        // Set transform bias parameters
        for (int i = 0; i < _transformBias.Length; i++)
        {
            _transformBias[i] = parameters[index++];
        }

        // Set gate weights parameters
        for (int i = 0; i < _gateWeights.Rows; i++)
        {
            for (int j = 0; j < _gateWeights.Columns; j++)
            {
                _gateWeights[i, j] = parameters[index++];
            }
        }

        // Set gate bias parameters
        for (int i = 0; i < _gateBias.Length; i++)
        {
            _gateBias[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from forward and backward passes.
    /// This includes the last input, output, transform output, gate output, and all gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - All stored information about previous inputs and outputs is removed
    /// - All calculated gradients are cleared
    /// - The layer is ready for new data without being influenced by previous data
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// For example, if you've processed one batch of images and want to start with a new batch,
    /// you should reset the state to prevent the new processing from being influenced by the previous batch.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastTransformOutput = null;
        _lastGateOutput = null;
        _transformWeightsGradient = null;
        _transformBiasGradient = null;
        _gateWeightsGradient = null;
        _gateBiasGradient = null;
    }
}