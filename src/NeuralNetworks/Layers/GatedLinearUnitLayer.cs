namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Gated Linear Unit (GLU) layer in a neural network that combines linear transformation with multiplicative gating.
/// </summary>
/// <remarks>
/// <para>
/// A Gated Linear Unit (GLU) is a neural network layer that combines linear transformations with a gating mechanism.
/// It applies two parallel linear transformations to the input: one produces a linear output, and the other produces
/// a gate that controls how much of the linear output passes through. The final output is the element-wise product
/// of the linear output and the activated gate. GLUs were introduced to help with vanishing gradient problems in
/// deep networks and have been particularly effective in natural language processing and sequence modeling tasks.
/// </para>
/// <para><b>For Beginners:</b> A Gated Linear Unit is like a smart filter that controls how much information flows through.
/// 
/// Imagine water flowing through a pipe with an adjustable valve:
/// - The water is the input data
/// - One part of the layer (linear part) processes the water
/// - Another part (gate) controls how much processed water flows through
/// - Together they decide "what information is important to keep"
/// 
/// For example, in language processing:
/// - The linear transformation might extract features from words
/// - The gate might decide which features are relevant to the current context
/// - Their combination helps the network focus on important information
/// 
/// GLUs are particularly good at:
/// - Controlling information flow through the network
/// - Helping gradients flow during training (preventing vanishing gradients)
/// - Allowing the network to selectively use information
/// 
/// This selectivity is valuable in many tasks, especially those involving sequences
/// like text or time-series data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GatedLinearUnitLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight matrix for the linear transformation path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the learnable weights for the linear transformation part of the GLU.
    /// The shape is [outputDimension, inputDimension], where each row represents the weights
    /// for one output neuron in the linear path.
    /// </para>
    /// <para><b>For Beginners:</b> These weights determine how input data is transformed before gating.
    /// 
    /// The linear weights work like in a standard neural network layer:
    /// - They transform the input data into a new representation
    /// - Each output value is a weighted sum of all inputs
    /// - During training, these weights adjust to extract useful features
    /// 
    /// These weights focus on transforming the data without considering
    /// which parts are important to keep or filter out (that's the gate's job).
    /// </para>
    /// </remarks>
    private Matrix<T> _linearWeights = default!;

    /// <summary>
    /// The weight matrix for the gating path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the learnable weights for the gating transformation part of the GLU.
    /// The shape is [outputDimension, inputDimension], where each row represents the weights
    /// for one output neuron in the gating path.
    /// </para>
    /// <para><b>For Beginners:</b> These weights determine how each input influences the gates.
    /// 
    /// The gate weights control what information is important:
    /// - They transform input data into control signals (gates)
    /// - These gates will determine how much information passes through
    /// - During training, these weights learn to recognize important patterns
    /// 
    /// Think of these weights as learning when to open or close the valve
    /// for different types of input information.
    /// </para>
    /// </remarks>
    private Matrix<T> _gateWeights = default!;

    /// <summary>
    /// The bias vector for the linear transformation path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the learnable bias terms for the linear transformation part of the GLU.
    /// The biases are added to the weighted sum of inputs in the linear path before being gated.
    /// </para>
    /// <para><b>For Beginners:</b> These biases are default or starting values for the linear path.
    /// 
    /// Linear biases work like in a standard neural network layer:
    /// - They provide an adjustable baseline for each output
    /// - They're added after the weighted sum but before gating
    /// - During training, they adjust to help produce better features
    /// 
    /// Each output neuron in the linear path has its own bias value.
    /// </para>
    /// </remarks>
    private Vector<T> _linearBias = default!;

    /// <summary>
    /// The bias vector for the gating path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the learnable bias terms for the gating part of the GLU.
    /// The biases are added to the weighted sum of inputs in the gating path before
    /// applying the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> These biases affect how open or closed the gates are by default.
    /// 
    /// Gate biases control the default state of each gate:
    /// - Positive values make gates tend to be more open
    /// - Negative values make gates tend to be more closed
    /// - They're adjusted during training to find optimal default settings
    /// 
    /// For example, with sigmoid activation:
    /// - A large negative bias makes the gate mostly closed by default
    /// - A large positive bias makes the gate mostly open by default
    /// - A bias near zero lets the gate be more responsive to the input
    /// </para>
    /// </remarks>
    private Vector<T> _gateBias = default!;

    /// <summary>
    /// The input tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the input received during the last forward pass. It is necessary for computing
    /// gradients during the backward pass (backpropagation).
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what input data was processed most recently.
    /// 
    /// During training:
    /// - The layer needs to remember what input values it processed
    /// - This helps when calculating how to improve the weights and biases
    /// - It's like keeping your work when solving a math problem
    /// 
    /// This value is automatically cleared between training batches to save memory.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The output tensor from the linear path of the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the output from the linear transformation (before gating) during the last forward pass.
    /// It is used during backpropagation to compute gradients for the gating path.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers the values produced by the linear path before gating.
    /// 
    /// During training:
    /// - The layer needs to know what values were being gated
    /// - This helps calculate how the gate affected the final output
    /// - It's essential for computing how to improve both paths
    /// 
    /// This intermediate result is saved because it's needed when computing
    /// gradients during the backward pass.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastLinearOutput;

    /// <summary>
    /// The output tensor from the gating path of the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the output from the gating path (after activation) during the last forward pass.
    /// It is used during backpropagation to compute gradients for the linear path and the gate activation.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers how open or closed each gate was in the latest calculation.
    /// 
    /// During training:
    /// - The layer needs to know what gate values were applied
    /// - This helps calculate how the gates affected the output
    /// - It's needed for computing gradients for both paths
    /// 
    /// These gate values (typically between 0 and 1) determined how much
    /// of each linear output value passed through to the final output.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastGateOutput;

    /// <summary>
    /// The gradients for the linear weights, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the gradients of the loss with respect to each linear weight.
    /// These gradients are used to update the linear weights during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each linear weight value.
    /// 
    /// During training:
    /// - The network calculates how each linear weight contributed to errors
    /// - Gradients show both direction and amount to change each weight
    /// - Larger gradients mean bigger adjustments are needed
    /// 
    /// These gradients help the linear path learn to produce better features
    /// that, when gated appropriately, lead to better final outputs.
    /// </para>
    /// </remarks>
    private Matrix<T>? _linearWeightsGradient;

    /// <summary>
    /// The gradients for the gate weights, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the gradients of the loss with respect to each gate weight.
    /// These gradients are used to update the gate weights during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each gate weight value.
    /// 
    /// During training:
    /// - The network calculates how each gate weight contributed to errors
    /// - Gradients show how to change weights to make gates work better
    /// - They help the gates learn to identify important information
    /// 
    /// These gradients help the gating mechanism learn when to allow
    /// information through and when to block it for better results.
    /// </para>
    /// </remarks>
    private Matrix<T>? _gateWeightsGradient;

    /// <summary>
    /// The gradients for the linear biases, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the gradients of the loss with respect to each linear bias.
    /// These gradients are used to update the linear biases during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each linear bias value.
    /// 
    /// During training:
    /// - The network calculates how each linear bias contributed to errors
    /// - Gradients guide adjustments to improve performance
    /// - They help fine-tune the baseline of each feature
    /// 
    /// These gradients help the linear path produce better default values
    /// before gating is applied.
    /// </para>
    /// </remarks>
    private Vector<T>? _linearBiasGradient;

    /// <summary>
    /// The gradients for the gate biases, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the gradients of the loss with respect to each gate bias.
    /// These gradients are used to update the gate biases during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each gate bias value.
    /// 
    /// During training:
    /// - The network calculates how each gate bias contributed to errors
    /// - Gradients guide adjustments to improve gating behavior
    /// - They help fine-tune the default openness of each gate
    /// 
    /// These gradients help the gates learn optimal default settings
    /// for controlling information flow.
    /// </para>
    /// </remarks>
    private Vector<T>? _gateBiasGradient;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because GLU layers have trainable parameters (weights and biases for both paths).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the GLU layer supports training through backpropagation.
    /// The layer has trainable parameters (weights and biases for both linear and gating paths)
    /// that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer adjusts its weights and biases during training
    /// - It improves its performance as it sees more data
    /// - It has parameters for both the linear and gating paths that adapt
    /// 
    /// GLU layers are powerful learning components because they can learn
    /// both what features to extract and which ones are important in context.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="GatedLinearUnitLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputDimension">The number of input features.</param>
    /// <param name="outputDimension">The number of output features.</param>
    /// <param name="gateActivation">The activation function to apply to the gating mechanism. Defaults to Sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new GLU layer with the specified input and output dimensions and
    /// gate activation function. The weights for both paths are initialized with small random values,
    /// and the biases are initialized to zero. The activation function operates on individual scalar values
    /// in the gate output tensor. The default gate activation is sigmoid, which produces values between 0 and 1
    /// that act as gates controlling how much of the linear output passes through.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the GLU layer with the dimensions you need and the activation for the gate.
    /// 
    /// When creating a GLU layer, you need to specify:
    /// - Input dimension: How many values are coming into the layer
    /// - Output dimension: How many values you want the layer to produce
    /// - Gate activation: What function controls the gate values (default: sigmoid)
    /// 
    /// For example:
    /// ```csharp
    /// // Create a GLU layer with 512 inputs, 256 outputs, and sigmoid gating
    /// var gluLayer = new GatedLinearUnitLayer<float>(512, 256);
    /// 
    /// // Create a GLU layer with custom activation for the gate
    /// var customGluLayer = new GatedLinearUnitLayer<float>(100, 50, new TanhActivation<float>());
    /// ```
    /// 
    /// The sigmoid activation (default) produces gates between 0 and 1, where:
    /// - 0 means "block this completely"
    /// - 1 means "let this pass completely"
    /// - Values between allow partial information flow
    /// 
    /// Other activations can be used for specialized gating behavior.
    /// </para>
    /// </remarks>
    public GatedLinearUnitLayer(int inputDimension, int outputDimension, IActivationFunction<T>? gateActivation = null)
        : base([inputDimension], [outputDimension], gateActivation ?? new SigmoidActivation<T>())
    {
        _linearWeights = new Matrix<T>(outputDimension, inputDimension);
        _gateWeights = new Matrix<T>(outputDimension, inputDimension);
        _linearBias = new Vector<T>(outputDimension);
        _gateBias = new Vector<T>(outputDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GatedLinearUnitLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputDimension">The number of input features.</param>
    /// <param name="outputDimension">The number of output features.</param>
    /// <param name="gateActivation">The vector activation function to apply to the gating mechanism. Defaults to Sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new GLU layer with the specified input and output dimensions and
    /// vector gate activation function. The weights for both paths are initialized with small random values,
    /// and the biases are initialized to zero. Unlike the other constructor, this one accepts a vector activation
    /// function that operates on entire vectors rather than individual scalar values.
    /// </para>
    /// <para><b>For Beginners:</b> This is an alternative setup that uses a different kind of activation function for the gate.
    /// 
    /// This constructor is almost identical to the first one, but with one key difference:
    /// - Regular activation: processes each gate value separately
    /// - Vector<double> activation: processes the entire gate vector together
    /// 
    /// Vector<double> activations might be useful for specialized gating where
    /// gate values should influence each other. For most common use cases,
    /// the standard constructor with sigmoid activation works well.
    /// 
    /// The default is still sigmoid activation, which is usually the best
    /// choice for GLU layers because its 0-1 range makes it ideal for gating.
    /// </para>
    /// </remarks>
    public GatedLinearUnitLayer(int inputDimension, int outputDimension, IVectorActivationFunction<T>? gateActivation = null)
        : base([inputDimension], [outputDimension], gateActivation ?? new SigmoidActivation<T>())
    {
        _linearWeights = new Matrix<T>(outputDimension, inputDimension);
        _gateWeights = new Matrix<T>(outputDimension, inputDimension);
        _linearBias = new Vector<T>(outputDimension);
        _gateBias = new Vector<T>(outputDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases with appropriate values for effective training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights for both the linear and gating paths using a scaling factor
    /// based on the dimensions of the weight matrices. This helps with training convergence by setting
    /// initial values to an appropriate scale. All biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the initial values for weights and biases before training.
    /// 
    /// For good training:
    /// - Weights need to start with small random values
    /// - These values are carefully scaled based on layer size
    /// - Too large or too small values can make training difficult
    /// 
    /// The method:
    /// - Calculates an appropriate scale for the random values
    /// - Initializes both linear and gate weights with this scale
    /// - Sets all biases to zero as a starting point
    /// 
    /// Good initialization helps the network start training effectively.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_linearWeights.Rows + _linearWeights.Columns)));
        InitializeMatrix(_linearWeights, scale);
        InitializeMatrix(_gateWeights, scale);

        for (int i = 0; i < _linearBias.Length; i++)
        {
            _linearBias[i] = NumOps.Zero;
            _gateBias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Helper method to initialize a matrix with scaled random values.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This helper method initializes a matrix with random values scaled by the provided factor.
    /// The random values are centered around zero and scaled to help with training convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This fills a weight matrix with appropriate random starting values.
    /// 
    /// The method:
    /// - Goes through each position in the matrix
    /// - Assigns a random value centered around zero (between -0.5 and 0.5)
    /// - Scales that value by the provided factor
    /// 
    /// This scaling ensures the initial weights are in an appropriate range
    /// that will allow effective gradient-based learning.
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
    /// Performs the forward pass of the GLU layer.
    /// </summary>
    /// <param name="input">The input tensor to process. Shape: [batchSize, inputDimension].</param>
    /// <returns>The output tensor after gated linear transformation. Shape: [batchSize, outputDimension].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the GLU layer. It performs two parallel linear transformations
    /// on the input: one for the linear path and one for the gating path. The gating path output is passed through
    /// an activation function (typically sigmoid), and then the two outputs are multiplied element-wise. This gating
    /// mechanism allows the layer to selectively pass information through.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer processes input data through both paths.
    /// 
    /// The forward pass works in these steps:
    /// 1. Linear Path: Transform the input using linear weights and biases
    ///    - This creates features that might be useful
    /// 2. Gate Path: Transform the input using gate weights and biases
    ///    - This determines how important each feature is
    /// 3. Apply activation to the gate values (typically sigmoid)
    ///    - Converts gate values to be between 0 and 1
    /// 4. Multiply the linear output by the activated gate values
    ///    - This lets important features pass through and blocks others
    /// 
    /// The result is that the layer can learn both:
    /// - What features to extract (linear path)
    /// - Which features are important in each context (gate path)
    /// 
    /// This selective focus helps the network learn more effectively.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int inputDimension = input.Shape[1];

        var linearOutput = input.Multiply(_linearWeights).Add(_linearBias);
        var gateOutput = input.Multiply(_gateWeights).Add(_gateBias);

        _lastLinearOutput = linearOutput;
        _lastGateOutput = ApplyActivation(gateOutput);

        var output = _lastLinearOutput.ElementwiseMultiply(_lastGateOutput);

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the GLU layer to compute gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient tensor from the next layer. Shape: [batchSize, outputDimension].</param>
    /// <returns>The gradient tensor to be passed to the previous layer. Shape: [batchSize, inputDimension].</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass (backpropagation) of the GLU layer. It computes the gradients
    /// of the loss with respect to the layer's weights, biases, and inputs. The computation accounts for
    /// the two paths (linear and gating) and their interaction through element-wise multiplication.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer learns from its mistakes during training.
    /// 
    /// The backward pass is more complex in GLU layers because of the two paths:
    /// 
    /// 1. First, compute gradients for both paths:
    ///    - Linear path gradient: outputGradient × gate values
    ///    - Gate path gradient: outputGradient × linear output
    /// 
    /// 2. For the gate path, apply the activation derivative
    ///    - This accounts for how the activation affected the gates
    /// 
    /// 3. Compute gradients for all parameters:
    ///    - Linear weights: Based on input and linear gradient
    ///    - Gate weights: Based on input and gate gradient
    ///    - Linear biases: Sum of linear gradients
    ///    - Gate biases: Sum of gate gradients
    /// 
    /// 4. Compute gradient for the input (to pass to previous layer):
    ///    - Combine contributions from both paths
    /// 
    /// This process ensures that both paths learn appropriately
    /// based on their contribution to the final output.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastLinearOutput == null || _lastGateOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var linearGradient = outputGradient.ElementwiseMultiply(_lastGateOutput);
        var gateGradient = outputGradient.ElementwiseMultiply(_lastLinearOutput);

        gateGradient = ApplyActivationDerivative(_lastGateOutput, gateGradient);

        _linearWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(linearGradient).ToMatrix();
        _gateWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(gateGradient).ToMatrix();

        _linearBiasGradient = linearGradient.Sum([0]).ToVector();
        _gateBiasGradient = gateGradient.Sum([0]).ToVector();

        var inputGradient = linearGradient.Multiply(_linearWeights.Transpose())
                            .Add(gateGradient.Multiply(_gateWeights.Transpose()));

        return inputGradient;
    }

    /// <summary>
    /// Updates the weights and biases for both paths using the calculated gradients and the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates all trainable parameters of the GLU layer based on the gradients calculated during
    /// the backward pass. The parameters include weights and biases for both the linear and gating paths.
    /// The learning rate determines the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method changes the weights and biases to improve future predictions.
    /// 
    /// After calculating how each parameter should change:
    /// - All parameters are adjusted in the direction that reduces errors
    /// - The learning rate controls how big these adjustments are
    /// 
    /// The updates apply to all four sets of parameters:
    /// 1. Linear weights: For better feature extraction
    /// 2. Gate weights: For better selection of important features
    /// 3. Linear biases: For better baseline feature values
    /// 4. Gate biases: For better default gate openness
    /// 
    /// Each parameter moves a small step in the direction that improves performance.
    /// The minus sign means we move in the opposite direction of the gradient
    /// to minimize error.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_linearWeightsGradient == null || _gateWeightsGradient == null || 
            _linearBiasGradient == null || _gateBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _linearWeights = _linearWeights.Subtract(_linearWeightsGradient.Multiply(learningRate));
        _gateWeights = _gateWeights.Subtract(_gateWeightsGradient.Multiply(learningRate));
        _linearBias = _linearBias.Subtract(_linearBiasGradient.Multiply(learningRate));
        _gateBias = _gateBias.Subtract(_gateBiasGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters of the GLU layer as a single vector. The parameters
    /// include weights and biases for both the linear and gating paths. The order is: linear weights, gate weights,
    /// linear biases, gate biases.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the layer's learnable values into a single list.
    /// 
    /// The parameters include four sets of values:
    /// 1. Linear weights: Main transformation parameters
    /// 2. Gate weights: Selection mechanism parameters
    /// 3. Linear biases: Baseline adjustments for features
    /// 4. Gate biases: Default settings for gates
    /// 
    /// All these values are collected in a specific order into a single vector.
    /// This combined list is useful for:
    /// - Saving a trained model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques
    /// 
    /// For a layer with 100 inputs and 50 outputs, this would return:
    /// - 5,000 linear weight parameters (100 × 50)
    /// - 5,000 gate weight parameters (100 × 50)
    /// - 50 linear bias parameters
    /// - 50 gate bias parameters
    /// - Totaling 10,100 parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _linearWeights.Rows * _linearWeights.Columns + 
                          _gateWeights.Rows * _gateWeights.Columns + 
                          _linearBias.Length + _gateBias.Length;

        var parameters = new Vector<T>(totalParams);

        int index = 0;

        // Copy linear weights parameters
        for (int i = 0; i < _linearWeights.Rows; i++)
        {
            for (int j = 0; j < _linearWeights.Columns; j++)
            {
                parameters[index++] = _linearWeights[i, j];
            }
        }

        // Copy gate weights parameters
        for (int i = 0; i < _gateWeights.Rows; i++)
        {
            for (int j = 0; j < _gateWeights.Columns; j++)
            {
                parameters[index++] = _gateWeights[i, j];
            }
        }

        // Copy linear bias parameters
        for (int i = 0; i < _linearBias.Length; i++)
        {
            parameters[index++] = _linearBias[i];
        }

        // Copy gate bias parameters
        for (int i = 0; i < _gateBias.Length; i++)
        {
            parameters[index++] = _gateBias[i];
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
    /// This method sets all trainable parameters of the GLU layer from a single vector. The parameters
    /// should be in the same order as produced by GetParameters: linear weights, gate weights,
    /// linear biases, gate biases.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's learnable values from a provided list.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the exact right length
    /// - The values are distributed to the correct parameters in order
    /// - They must follow the same order used in GetParameters
    /// 
    /// This method is useful for:
    /// - Restoring a saved model
    /// - Loading pre-trained parameters
    /// - Testing specific parameter configurations
    /// 
    /// The method verifies that the vector contains exactly the right number
    /// of parameters before applying them.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedLength = _linearWeights.Rows * _linearWeights.Columns + 
                             _gateWeights.Rows * _gateWeights.Columns + 
                             _linearBias.Length + _gateBias.Length;

        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set linear weights parameters
        for (int i = 0; i < _linearWeights.Rows; i++)
        {
            for (int j = 0; j < _linearWeights.Columns; j++)
            {
                _linearWeights[i, j] = parameters[index++];
            }
        }

        // Set gate weights parameters
        for (int i = 0; i < _gateWeights.Rows; i++)
        {
            for (int j = 0; j < _gateWeights.Columns; j++)
            {
                _gateWeights[i, j] = parameters[index++];
            }
        }

        // Set linear bias parameters
        for (int i = 0; i < _linearBias.Length; i++)
        {
            _linearBias[i] = parameters[index++];
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
    /// This method resets the internal state of the GLU layer by clearing all cached values from forward
    /// and backward passes. This includes inputs, intermediate outputs, and gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input is cleared
    /// - The saved linear and gate outputs are cleared
    /// - All calculated gradients are cleared
    /// - The layer forgets previous calculations it performed
    /// 
    /// This is typically called:
    /// - Between training batches to free up memory
    /// - When switching from training to evaluation mode
    /// - When starting to process completely new data
    /// 
    /// It's like wiping a whiteboard clean before starting a new calculation.
    /// Note that this doesn't affect the learned weights and biases, just the
    /// temporary working data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastLinearOutput = null;
        _lastGateOutput = null;
        _linearWeightsGradient = null;
        _gateWeightsGradient = null;
        _linearBiasGradient = null;
        _gateBiasGradient = null;
    }
}