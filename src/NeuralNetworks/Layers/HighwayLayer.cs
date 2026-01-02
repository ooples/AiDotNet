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
public class HighwayLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
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
    /// Stores the last computed gate balance loss for diagnostic purposes.
    /// </summary>
    private T _lastGateBalanceLoss;

    /// <summary>
    /// The weight tensor used to transform the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the learnable parameters that transform the input features. The dimensions are
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
    private Tensor<T> _transformWeights;

    /// <summary>
    /// The bias tensor added to the transformed input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the learnable bias parameters that are added to the transformed input. Adding a bias
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
    private Tensor<T> _transformBias;

    /// <summary>
    /// The weight tensor used to compute the gate values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the learnable parameters that determine how much of the transformed output versus the
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
    private Tensor<T> _gateWeights;

    /// <summary>
    /// The bias tensor added to the gate computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the learnable bias parameters that are added to the gate computation. Initially biased
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
    private Tensor<T> _gateBias;

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
    /// Stores the pre-activation transform values from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastTransformPreActivation;

    /// <summary>
    /// Stores the pre-activation gate values from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastGatePreActivation;

    /// <summary>
    /// Stores the gradients for the transform weights calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _transformWeightsGradient;

    /// <summary>
    /// Stores the gradients for the transform bias calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _transformBiasGradient;

    /// <summary>
    /// Stores the gradients for the gate weights calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _gateWeightsGradient;

    /// <summary>
    /// Stores the gradients for the gate bias calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _gateBiasGradient;

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
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <value>
    /// The sum of elements in all weight and bias tensors (transform weights, transform bias, gate weights, gate bias).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the total count of learnable parameters across all four parameter tensors:
    /// transform weights, transform biases, gate weights, and gate biases.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many numbers the layer can adjust during training.
    /// For a Highway layer with 100 input/output dimensions, you would have:
    /// - 10,000 transform weights (100 x 100)
    /// - 100 transform biases
    /// - 10,000 gate weights (100 x 100)
    /// - 100 gate biases
    /// - Total: 20,200 parameters
    /// </para>
    /// </remarks>
    public override int ParameterCount =>
        _transformWeights.Length + _transformBias.Length + _gateWeights.Length + _gateBias.Length;

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
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastGateBalanceLoss = NumOps.Zero;

        _transformWeights = new Tensor<T>([inputDimension, inputDimension]);
        _transformBias = new Tensor<T>([inputDimension]);
        _gateWeights = new Tensor<T>([inputDimension, inputDimension]);
        _gateBias = new Tensor<T>([inputDimension]);

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
    /// Vector activation functions operate on entire vectors rather than individual elements, which can capture
    /// dependencies between different elements of the vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Highway layer with more advanced vector-based activation functions.
    /// 
    /// Vector activation functions:
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
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastGateBalanceLoss = NumOps.Zero;

        _transformWeights = new Tensor<T>([inputDimension, inputDimension]);
        _transformBias = new Tensor<T>([inputDimension]);
        _gateWeights = new Tensor<T>([inputDimension, inputDimension]);
        _gateBias = new Tensor<T>([inputDimension]);

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
        int inputDimension = _transformWeights.Shape[0];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (inputDimension + inputDimension)));
        InitializeTensor(_transformWeights, scale);
        InitializeTensor(_gateWeights, scale);

        for (int i = 0; i < _transformBias.Length; i++)
        {
            _transformBias[i] = NumOps.Zero;
            _gateBias[i] = NumOps.FromDouble(-1.0); // Initialize gate bias to negative values to allow more information flow initially
        }
    }

    /// <summary>
    /// Initializes a 2D tensor with scaled random values.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the provided tensor with random values between -0.5 and 0.5, scaled by the provided scale factor.
    /// This type of initialization helps with training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a tensor with random starting values for weights.
    ///
    /// The method:
    /// - Generates random numbers between -0.5 and 0.5
    /// - Multiplies them by a scale factor to control their size
    /// - Fills each position in the tensor with these scaled random values
    ///
    /// Good initialization is important because it affects how quickly and how well the network learns.
    /// The scale factor is calculated based on the size of the layer to help maintain stable gradients
    /// during training.
    /// </para>
    /// </remarks>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                tensor[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
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

        // Transform path: transform = activation(input @ weights + bias)
        var transformLinear = input.MatrixMultiply(_transformWeights);
        var transformWithBias = Engine.TensorBroadcastAdd(transformLinear, _transformBias);
        _lastTransformPreActivation = transformWithBias; // Store pre-activation for backward pass
        var transformOutput = ApplyActivation(transformWithBias, _transformActivation, _vectorTransformActivation);
        _lastTransformOutput = transformOutput;

        // Gate path: gate = sigmoid(input @ weights + bias)
        var gateLinear = input.MatrixMultiply(_gateWeights);
        var gateWithBias = Engine.TensorBroadcastAdd(gateLinear, _gateBias);
        _lastGatePreActivation = gateWithBias; // Store pre-activation for backward pass
        var gateOutput = ApplyActivation(gateWithBias, _gateActivation, _vectorGateActivation);
        _lastGateOutput = gateOutput;

        // Highway output: output = gate * transform + (1 - gate) * input
        // Rewritten as: output = gate * (transform - input) + input
        var transformMinusInput = Engine.TensorSubtract(transformOutput, input);
        var gatedDiff = Engine.TensorMultiply(gateOutput, transformMinusInput);
        var output = Engine.TensorAdd(gatedDiff, input);

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
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }


    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. It's slower than the
    /// manual implementation but can be useful for:
    /// - Verifying gradient correctness
    /// - Rapid prototyping with custom modifications
    /// - Research and experimentation
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Create input variable node
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Create variable nodes for weights and biases with gradient tracking
        var transformWeightsNode = Autodiff.TensorOperations<T>.Variable(_transformWeights, "transform_weights", requiresGradient: true);
        var transformBiasNode = Autodiff.TensorOperations<T>.Variable(_transformBias, "transform_bias", requiresGradient: true);
        var gateWeightsNode = Autodiff.TensorOperations<T>.Variable(_gateWeights, "gate_weights", requiresGradient: true);
        var gateBiasNode = Autodiff.TensorOperations<T>.Variable(_gateBias, "gate_bias", requiresGradient: true);

        // Step 1: Compute transform path: transform = activation(input @ weights + bias)
        var transformLinear = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, transformWeightsNode);
        var transformWithBias = Autodiff.TensorOperations<T>.Add(transformLinear, transformBiasNode);

        // Apply transform activation (typically Tanh)
        Autodiff.ComputationNode<T> transformOutput;
        if (_transformActivation != null && _transformActivation.SupportsJitCompilation)
        {
            transformOutput = _transformActivation.ApplyToGraph(transformWithBias);
        }
        else
        {
            // Default to Tanh if no activation specified
            transformOutput = Autodiff.TensorOperations<T>.Tanh(transformWithBias);
        }

        // Step 2: Compute gate path: gate = sigmoid(input @ weights + bias)
        var gateLinear = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, gateWeightsNode);
        var gateWithBias = Autodiff.TensorOperations<T>.Add(gateLinear, gateBiasNode);

        // Apply gate activation (typically Sigmoid)
        Autodiff.ComputationNode<T> gateOutput;
        if (_gateActivation != null && _gateActivation.SupportsJitCompilation)
        {
            gateOutput = _gateActivation.ApplyToGraph(gateWithBias);
        }
        else
        {
            // Default to Sigmoid if no activation specified
            gateOutput = Autodiff.TensorOperations<T>.Sigmoid(gateWithBias);
        }

        // Step 3: Compute highway output: output = gate * transform + (1 - gate) * input
        // Rewritten as: output = gate * (transform - input) + input
        var transformMinusInput = Autodiff.TensorOperations<T>.Subtract(transformOutput, inputNode);
        var gatedDiff = Autodiff.TensorOperations<T>.ElementwiseMultiply(gateOutput, transformMinusInput);
        var outputNode = Autodiff.TensorOperations<T>.Add(gatedDiff, inputNode);

        // Set the output gradient
        outputNode.Gradient = outputGradient;

        // Production-grade: Inline topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract parameter gradients
        if (transformWeightsNode.Gradient != null)
            _transformWeightsGradient = transformWeightsNode.Gradient;
        if (transformBiasNode.Gradient != null)
            _transformBiasGradient = transformBiasNode.Gradient;
        if (gateWeightsNode.Gradient != null)
            _gateWeightsGradient = gateWeightsNode.Gradient;
        if (gateBiasNode.Gradient != null)
            _gateBiasGradient = gateBiasNode.Gradient;

        // Extract and return the input gradient
        if (inputNode.Gradient == null)
            throw new InvalidOperationException("Gradient computation failed in automatic differentiation.");

        return inputNode.Gradient;
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastTransformOutput == null || _lastGateOutput == null ||
            _lastTransformPreActivation == null || _lastGatePreActivation == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // dL/d(transform - input) = dL/dOutput * gate
        // dL/dgate = dL/dOutput * (transform - input)
        var transformMinusInput = Engine.TensorSubtract(_lastTransformOutput, _lastInput);
        var gateGradient = Engine.TensorMultiply(outputGradient, transformMinusInput);
        // Use pre-activation values for derivative computation (activation functions expect pre-activation inputs)
        gateGradient = ApplyActivationDerivative(gateGradient, _lastGatePreActivation, _gateActivation, _vectorGateActivation);

        var transformGradient = Engine.TensorMultiply(outputGradient, _lastGateOutput);
        // Use pre-activation values for derivative computation
        transformGradient = ApplyActivationDerivative(transformGradient, _lastTransformPreActivation, _transformActivation, _vectorTransformActivation);

        // Compute weight gradients: dW = input^T @ gradient
        var inputT = _lastInput.Transpose([1, 0]);
        _gateWeightsGradient = inputT.MatrixMultiply(gateGradient);
        _gateBiasGradient = gateGradient.Sum([0]);

        _transformWeightsGradient = inputT.MatrixMultiply(transformGradient);
        _transformBiasGradient = transformGradient.Sum([0]);

        // Compute input gradient: dL/dInput = dL/dGateLinear @ W_gate^T + dL/dTransformLinear @ W_transform^T + dL/dOutput * (1 - gate)
        var gateWeightsT = _gateWeights.Transpose([1, 0]);
        var transformWeightsT = _transformWeights.Transpose([1, 0]);

        var inputGradFromGate = gateGradient.MatrixMultiply(gateWeightsT);
        var inputGradFromTransform = transformGradient.MatrixMultiply(transformWeightsT);

        // (1 - gate) contribution
        var ones = Tensor<T>.CreateDefault(_lastGateOutput.Shape, NumOps.One);
        var oneMinusGate = Engine.TensorSubtract(ones, _lastGateOutput);
        var bypassGradient = Engine.TensorMultiply(outputGradient, oneMinusGate);

        var inputGradient = Engine.TensorAdd(inputGradFromGate, inputGradFromTransform);
        inputGradient = Engine.TensorAdd(inputGradient, bypassGradient);

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

        // Use Engine operations for parameter updates
        var scaledTransformWeightsGrad = Engine.TensorMultiplyScalar(_transformWeightsGradient, learningRate);
        var scaledTransformBiasGrad = Engine.TensorMultiplyScalar(_transformBiasGradient, learningRate);
        var scaledGateWeightsGrad = Engine.TensorMultiplyScalar(_gateWeightsGradient, learningRate);
        var scaledGateBiasGrad = Engine.TensorMultiplyScalar(_gateBiasGradient, learningRate);

        _transformWeights = Engine.TensorSubtract(_transformWeights, scaledTransformWeightsGrad);
        _transformBias = Engine.TensorSubtract(_transformBias, scaledTransformBiasGrad);
        _gateWeights = Engine.TensorSubtract(_gateWeights, scaledGateWeightsGrad);
        _gateBias = Engine.TensorSubtract(_gateBias, scaledGateBiasGrad);
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
        return Vector<T>.Concatenate(
            new Vector<T>(_transformWeights.ToArray()),
            new Vector<T>(_transformBias.ToArray()),
            new Vector<T>(_gateWeights.ToArray()),
            new Vector<T>(_gateBias.ToArray()));
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
        int transformWeightsSize = _transformWeights.Shape[0] * _transformWeights.Shape[1];
        int gateWeightsSize = _gateWeights.Shape[0] * _gateWeights.Shape[1];
        int expectedLength = transformWeightsSize + _transformBias.Length +
                             gateWeightsSize + _gateBias.Length;

        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, but got {parameters.Length}");
        }

        int index = 0;

        _transformWeights = new Tensor<T>(_transformWeights.Shape, parameters.Slice(index, transformWeightsSize));
        index += transformWeightsSize;

        _transformBias = new Tensor<T>(_transformBias.Shape, parameters.Slice(index, _transformBias.Length));
        index += _transformBias.Length;

        _gateWeights = new Tensor<T>(_gateWeights.Shape, parameters.Slice(index, gateWeightsSize));
        index += gateWeightsSize;

        _gateBias = new Tensor<T>(_gateBias.Shape, parameters.Slice(index, _gateBias.Length));
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
        _lastTransformPreActivation = null;
        _lastGatePreActivation = null;
        _transformWeightsGradient = null;
        _transformBiasGradient = null;
        _gateWeightsGradient = null;
        _gateBiasGradient = null;
    }

    /// <summary>
    /// Computes the auxiliary loss for this layer based on gate balance regularization.
    /// </summary>
    /// <returns>The computed auxiliary loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes a gate-balance regularization loss that encourages the gates to maintain
    /// a balanced value around 0.5, preventing degenerate gating where all gates collapse to 0 or 1.
    /// The loss is computed as the squared deviation of the mean gate value from 0.5, averaged across
    /// all dimensions and batch samples.
    /// </para>
    /// <para><b>For Beginners:</b> This prevents the highway layer from "cheating" by always using
    /// only one lane (transform or bypass). By penalizing gates that drift too far from 0.5, we ensure
    /// the network learns to use both lanes effectively, making the highway mechanism meaningful.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastGateOutput == null)
        {
            _lastGateBalanceLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Compute mean gate value across batch and dimensions
        int batchSize = _lastGateOutput.Shape[0];
        int inputDimension = _lastGateOutput.Shape[1];
        int totalElements = batchSize * inputDimension;

        T sum = NumOps.Zero;
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < inputDimension; d++)
            {
                T gateValue = _lastGateOutput[new int[] { b, d }];
                sum = NumOps.Add(sum, gateValue);
            }
        }

        T meanGate = NumOps.Divide(sum, NumOps.FromDouble(totalElements));

        // Compute loss = (mean_gate - 0.5)^2 to encourage balanced gating
        T targetGate = NumOps.FromDouble(0.5);
        T deviation = NumOps.Subtract(meanGate, targetGate);
        T rawLoss = NumOps.Multiply(deviation, deviation);

        // Store unweighted loss for diagnostics
        _lastGateBalanceLoss = rawLoss;

        // Apply auxiliary loss weight and return weighted loss
        T weightedLoss = NumOps.Multiply(rawLoss, AuxiliaryLossWeight);
        return weightedLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the auxiliary loss computation.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about the auxiliary loss.</returns>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalGateBalanceLoss", _lastGateBalanceLoss?.ToString() ?? "0" },
            { "GateBalanceWeight", AuxiliaryLossWeight?.ToString() ?? "0.01" },
            { "UseGateBalance", UseAuxiliaryLoss.ToString() }
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

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> when weights are initialized and activation functions support JIT.
    /// </value>
    /// <remarks>
    /// <para>
    /// Highway layers support JIT compilation when:
    /// - Transform and gate weights are initialized
    /// - The transform activation function (typically Tanh) supports JIT
    /// - The gate activation function (typically Sigmoid) supports JIT
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        _transformWeights != null && _transformBias != null &&
        _gateWeights != null && _gateBias != null &&
        (_transformActivation?.SupportsJitCompilation ?? _vectorTransformActivation != null) &&
        (_gateActivation?.SupportsJitCompilation ?? _vectorGateActivation != null);

    /// <summary>
    /// Exports the highway layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the gated highway output.</returns>
    /// <remarks>
    /// <para>
    /// The highway layer computation graph implements:
    /// output = gate * transform(input) + (1 - gate) * input
    ///
    /// Where:
    /// - transform = activation(input @ transformWeights + transformBias)
    /// - gate = sigmoid(input @ gateWeights + gateBias)
    /// </para>
    /// <para><b>For Beginners:</b> This creates an optimized version of the highway layer.
    /// The gate controls how much information flows through the transform path vs. the bypass path.
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (_transformWeights == null || _transformBias == null ||
            _gateWeights == null || _gateBias == null)
            throw new InvalidOperationException("Weights and biases not initialized.");

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input node with batch dimension
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "highway_input");
        inputNodes.Add(inputNode);

        // Create variable nodes for weights and biases with gradient tracking
        var transformWeightsNode = Autodiff.TensorOperations<T>.Variable(_transformWeights, "transform_weights", requiresGradient: true);
        var transformBiasNode = Autodiff.TensorOperations<T>.Variable(_transformBias, "transform_bias", requiresGradient: true);
        var gateWeightsNode = Autodiff.TensorOperations<T>.Variable(_gateWeights, "gate_weights", requiresGradient: true);
        var gateBiasNode = Autodiff.TensorOperations<T>.Variable(_gateBias, "gate_bias", requiresGradient: true);

        // Step 1: Compute transform path: transform = activation(input @ weights + bias)
        var transformLinear = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, transformWeightsNode);
        var transformWithBias = Autodiff.TensorOperations<T>.Add(transformLinear, transformBiasNode);

        // Apply transform activation (typically Tanh)
        Autodiff.ComputationNode<T> transformOutput;
        if (_transformActivation != null && _transformActivation.SupportsJitCompilation)
        {
            transformOutput = _transformActivation.ApplyToGraph(transformWithBias);
        }
        else
        {
            // Default to Tanh if no activation specified
            transformOutput = Autodiff.TensorOperations<T>.Tanh(transformWithBias);
        }

        // Step 2: Compute gate path: gate = sigmoid(input @ weights + bias)
        var gateLinear = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, gateWeightsNode);
        var gateWithBias = Autodiff.TensorOperations<T>.Add(gateLinear, gateBiasNode);

        // Apply gate activation (typically Sigmoid)
        Autodiff.ComputationNode<T> gateOutput;
        if (_gateActivation != null && _gateActivation.SupportsJitCompilation)
        {
            gateOutput = _gateActivation.ApplyToGraph(gateWithBias);
        }
        else
        {
            // Default to Sigmoid if no activation specified
            gateOutput = Autodiff.TensorOperations<T>.Sigmoid(gateWithBias);
        }

        // Step 3: Compute highway output: output = gate * transform + (1 - gate) * input
        // Rewrite as: output = gate * transform + input - gate * input
        //           = gate * (transform - input) + input
        var transformMinusInput = Autodiff.TensorOperations<T>.Subtract(transformOutput, inputNode);
        var gatedDiff = Autodiff.TensorOperations<T>.ElementwiseMultiply(gateOutput, transformMinusInput);
        var output = Autodiff.TensorOperations<T>.Add(gatedDiff, inputNode);

        return output;
    }
}
