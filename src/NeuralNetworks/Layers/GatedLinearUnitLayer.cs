using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
    /// The weight tensor for the linear transformation path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the learnable weights for the linear transformation part of the GLU.
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
    private Tensor<T> _linearWeights;

    /// <summary>
    /// The weight tensor for the gating path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the learnable weights for the gating transformation part of the GLU.
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
    private Tensor<T> _gateWeights;

    /// <summary>
    /// The bias tensor for the linear transformation path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the learnable bias terms for the linear transformation part of the GLU.
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
    private Tensor<T> _linearBias;

    /// <summary>
    /// The bias tensor for the gating path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the learnable bias terms for the gating part of the GLU.
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
    private Tensor<T> _gateBias;

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

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuInput;
    private IGpuTensor<T>? _gpuLinearOutput;
    private IGpuTensor<T>? _gpuGateOutput;

    /// <summary>
    /// The gradients for the linear weights, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the gradients of the loss with respect to each linear weight.
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
    private Tensor<T>? _linearWeightsGradient;

    /// <summary>
    /// The gradients for the gate weights, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the gradients of the loss with respect to each gate weight.
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
    private Tensor<T>? _gateWeightsGradient;

    /// <summary>
    /// The gradients for the linear biases, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the gradients of the loss with respect to each linear bias.
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
    private Tensor<T>? _linearBiasGradient;

    /// <summary>
    /// The gradients for the gate biases, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the gradients of the loss with respect to each gate bias.
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
    private Tensor<T>? _gateBiasGradient;

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
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <value>
    /// The sum of elements in all weight and bias tensors (linear weights, gate weights, linear bias, gate bias).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the total count of learnable parameters across all four parameter tensors:
    /// linear weights, gate weights, linear biases, and gate biases.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many numbers the layer can adjust during training.
    /// For a GLU layer with 100 inputs and 50 outputs, you would have:
    /// - 5,000 linear weights (100 x 50)
    /// - 5,000 gate weights (100 x 50)
    /// - 50 linear biases
    /// - 50 gate biases
    /// - Total: 10,100 parameters
    /// </para>
    /// </remarks>
    public override int ParameterCount =>
        _linearWeights.Length + _gateWeights.Length + _linearBias.Length + _gateBias.Length;

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
        _linearWeights = new Tensor<T>([outputDimension, inputDimension]);
        _gateWeights = new Tensor<T>([outputDimension, inputDimension]);
        _linearBias = new Tensor<T>([outputDimension]);
        _gateBias = new Tensor<T>([outputDimension]);

        InitializeParameters();

        // Register tensors for GPU memory persistence
        RegisterTrainableParameter(_linearWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_linearBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_gateBias, PersistentTensorRole.Biases);
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
    /// - Vector activation: processes the entire gate vector together
    /// 
    /// Vector activations might be useful for specialized gating where
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
        _linearWeights = new Tensor<T>([outputDimension, inputDimension]);
        _gateWeights = new Tensor<T>([outputDimension, inputDimension]);
        _linearBias = new Tensor<T>([outputDimension]);
        _gateBias = new Tensor<T>([outputDimension]);

        InitializeParameters();

        // Register tensors for GPU memory persistence
        RegisterTrainableParameter(_linearWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_linearBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_gateBias, PersistentTensorRole.Biases);
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
        int outputDimension = _linearWeights.Shape[0];
        int inputDimension = _linearWeights.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (outputDimension + inputDimension)));

        _linearWeights = Engine.TensorMultiplyScalar(
            new Tensor<T>(_linearWeights.Shape, Vector<T>.CreateRandom(_linearWeights.Length, -0.5, 0.5)),
            scale);
        _gateWeights = Engine.TensorMultiplyScalar(
            new Tensor<T>(_gateWeights.Shape, Vector<T>.CreateRandom(_gateWeights.Length, -0.5, 0.5)),
            scale);

        _linearBias.Fill(NumOps.Zero);
        _gateBias.Fill(NumOps.Zero);
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

        // Linear path: linear = input @ weights^T + bias
        var linearWeightsT = _linearWeights.Transpose([1, 0]);
        var linearOutput = input.MatrixMultiply(linearWeightsT);
        linearOutput = Engine.TensorBroadcastAdd(linearOutput, _linearBias); // Broadcasting

        // Gate path: gate = sigmoid(input @ weights^T + bias)
        var gateWeightsT = _gateWeights.Transpose([1, 0]);
        var gateOutput = input.MatrixMultiply(gateWeightsT);
        gateOutput = Engine.TensorBroadcastAdd(gateOutput, _gateBias); // Broadcasting

        _lastLinearOutput = linearOutput;
        _lastGateOutput = ApplyActivation(gateOutput);

        // GLU output: output = linear * gate
        var output = Engine.TensorMultiply(_lastLinearOutput, _lastGateOutput);

        return output;
    }

    /// <summary>
    /// Performs the forward pass on GPU using FusedLinearGpu for efficient computation.
    /// </summary>
    /// <param name="inputs">The GPU input tensors.</param>
    /// <returns>The GPU output tensor.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];

        // GLU weights are stored as [outputDim, inputDim], but FusedLinearGpu expects [inputDim, outputDim]
        // Transpose the weights
        var linearWeightsT = Engine.TensorTranspose(_linearWeights);
        var gateWeightsT = Engine.TensorTranspose(_gateWeights);

        // Linear path: linear = input @ linearWeights^T + linearBias (no activation)
        var linearOutput = gpuEngine.FusedLinearGpu(input, linearWeightsT, _linearBias, FusedActivationType.None);

        // Gate path: gate = sigmoid(input @ gateWeights^T + gateBias)
        var gateOutput = gpuEngine.FusedLinearGpu(input, gateWeightsT, _gateBias, FusedActivationType.Sigmoid);

        // GLU output: output = linear * gate
        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        int size = linearOutput.ElementCount;
        var outputBuffer = backend.AllocateBuffer(size);

        // Element-wise multiply on GPU
        backend.Multiply(linearOutput.Buffer, gateOutput.Buffer, outputBuffer, size);

        // Cache state for backward pass only during training
        if (IsTrainingMode)
        {
            // Cache GPU tensors for GPU-resident backward pass
            _gpuInput = input;
            _gpuLinearOutput = linearOutput;
            _gpuGateOutput = gateOutput;

            // Also cache CPU tensors for fallback backward pass
            _lastInput = input.ToTensor();
            _lastLinearOutput = linearOutput.ToTensor();
            _lastGateOutput = gateOutput.ToTensor();
        }

        return new GpuTensor<T>(backend, outputBuffer, linearOutput.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Performs the backward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="outputGradient">GPU-resident gradient of the loss w.r.t. output.</param>
    /// <returns>GPU-resident gradient of the loss w.r.t. input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        if (_gpuInput == null || _gpuLinearOutput == null || _gpuGateOutput == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        int inputDim = _linearWeights.Shape[1];
        int outputDim = _linearWeights.Shape[0];
        int batchSize = _gpuInput.Shape[0];

        // GLU backward: output = linear * gate
        // d(linear) = outputGrad * gate
        // d(gate) = outputGrad * linear
        var dLinearOutput = gpuEngine.MultiplyGpu(outputGradient, _gpuGateOutput);
        var dGateBeforeSigmoid = gpuEngine.MultiplyGpu(outputGradient, _gpuLinearOutput);

        // Apply sigmoid derivative to gate gradient: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // d(gate_pre_activation) = d(gate_output) * gate_output * (1 - gate_output)
        var dGateOutput = gpuEngine.SigmoidBackwardGpu(dGateBeforeSigmoid, _gpuGateOutput);

        // Flatten to 2D for matmul
        var input2D = gpuEngine.ReshapeGpu(_gpuInput, new[] { batchSize, inputDim });
        var dLinear2D = gpuEngine.ReshapeGpu(dLinearOutput, new[] { batchSize, outputDim });
        var dGate2D = gpuEngine.ReshapeGpu(dGateOutput, new[] { batchSize, outputDim });

        // Linear weights gradient: dLinear^T @ input
        var dLinearT = gpuEngine.TransposeGpu(dLinear2D);
        var linearWeightsGrad = gpuEngine.MatMulGpuTensors(dLinearT, input2D);
        _linearWeightsGradient = linearWeightsGrad.ToTensor();

        // Gate weights gradient: dGate^T @ input
        var dGateT = gpuEngine.TransposeGpu(dGate2D);
        var gateWeightsGrad = gpuEngine.MatMulGpuTensors(dGateT, input2D);
        _gateWeightsGradient = gateWeightsGrad.ToTensor();

        // Linear bias gradient: sum(dLinear, axis=0)
        var linearBiasGrad = gpuEngine.SumAxisGpu(dLinear2D, 0);
        _linearBiasGradient = linearBiasGrad.ToTensor().Reshape([outputDim]);

        // Gate bias gradient: sum(dGate, axis=0)
        var gateBiasGrad = gpuEngine.SumAxisGpu(dGate2D, 0);
        _gateBiasGradient = gateBiasGrad.ToTensor().Reshape([outputDim]);

        // Input gradient: dLinear @ linearWeights + dGate @ gateWeights
        // Note: weights are [outputDim, inputDim], need to transpose
        var linearWeightsT = gpuEngine.UploadToGpu(Engine.TensorTranspose(_linearWeights), GpuTensorRole.Weight);
        var gateWeightsT = gpuEngine.UploadToGpu(Engine.TensorTranspose(_gateWeights), GpuTensorRole.Weight);

        var dInputFromLinear = gpuEngine.MatMulGpuTensors(dLinear2D, linearWeightsT);
        var dInputFromGate = gpuEngine.MatMulGpuTensors(dGate2D, gateWeightsT);
        var inputGradient = gpuEngine.AddGpu(dInputFromLinear, dInputFromGate);

        return inputGradient;
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
        if (_lastInput == null || _lastLinearOutput == null || _lastGateOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // dL/dLinear = dL/dOutput * gate
        var linearGradient = Engine.TensorMultiply(outputGradient, _lastGateOutput);
        // dL/dGate = dL/dOutput * linear
        var gateGradient = Engine.TensorMultiply(outputGradient, _lastLinearOutput);

        // Apply gate activation derivative
        gateGradient = ApplyActivationDerivative(_lastGateOutput, gateGradient);

        // Weight gradients: dW = grad^T @ input
        var linearGradT = linearGradient.Transpose([1, 0]);
        var gateGradT = gateGradient.Transpose([1, 0]);
        _linearWeightsGradient = linearGradT.MatrixMultiply(_lastInput);
        _gateWeightsGradient = gateGradT.MatrixMultiply(_lastInput);

        // Bias gradients: sum over batch
        _linearBiasGradient = linearGradient.Sum([0]);
        _gateBiasGradient = gateGradient.Sum([0]);

        // Input gradient: dL/dInput = linearGrad @ linearWeights + gateGrad @ gateWeights
        var inputGradFromLinear = linearGradient.MatrixMultiply(_linearWeights);
        var inputGradFromGate = gateGradient.MatrixMultiply(_gateWeights);
        var inputGradient = Engine.TensorAdd(inputGradFromLinear, inputGradFromGate);

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation principles.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass using the same mathematical formulas as the manual
    /// implementation but structured to demonstrate autodiff concepts. Both paths now produce
    /// identical results by using the same cached values and computation order.
    /// </para>
    /// <para>
    /// For the GLU layer, the computations are:
    /// - Forward: output = linearOutput * sigmoid(gatePreactivation)
    /// - Backward: dL/dinput = (dL/dlinearOutput @ linearWeights) + (dL/dgatePreactivation @ gateWeights)
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastLinearOutput == null || _lastGateOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // === Step 1: Compute gradients through the GLU output ===
        // output = linearOutput * gateOutput (element-wise)
        // dL/dlinearOutput = dL/dOutput * gateOutput
        // dL/dgateOutput = dL/dOutput * linearOutput
        // Production-grade: Use Engine.TensorMultiply for GPU/CPU accelerated element-wise ops
        var linearGradient = Engine.TensorMultiply(outputGradient, _lastGateOutput);
        var gateGradient = Engine.TensorMultiply(outputGradient, _lastLinearOutput);

        // === Step 2: Apply activation derivative to gate gradient using cached values ===
        // This matches BackwardManual exactly by using _lastGateOutput
        gateGradient = ApplyActivationDerivative(_lastGateOutput, gateGradient);

        // === Step 3: Compute weight and bias gradients ===
        var linearGradT = linearGradient.Transpose([1, 0]);
        var gateGradT = gateGradient.Transpose([1, 0]);
        _linearWeightsGradient = linearGradT.MatrixMultiply(_lastInput);
        _gateWeightsGradient = gateGradT.MatrixMultiply(_lastInput);

        _linearBiasGradient = linearGradient.Sum([0]);
        _gateBiasGradient = gateGradient.Sum([0]);

        // === Step 4: Compute input gradient ===
        // dL/dinput = (linearGradient @ linearWeights) + (gateGradient @ gateWeights)
        var inputGradFromLinear = linearGradient.MatrixMultiply(_linearWeights);
        var inputGradFromGate = gateGradient.MatrixMultiply(_gateWeights);
        var inputGradient = Engine.TensorAdd(inputGradFromLinear, inputGradFromGate);

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

        // Use Engine operations for parameter updates
        var scaledLinearWeightsGrad = Engine.TensorMultiplyScalar(_linearWeightsGradient, learningRate);
        var scaledGateWeightsGrad = Engine.TensorMultiplyScalar(_gateWeightsGradient, learningRate);
        var scaledLinearBiasGrad = Engine.TensorMultiplyScalar(_linearBiasGradient, learningRate);
        var scaledGateBiasGrad = Engine.TensorMultiplyScalar(_gateBiasGradient, learningRate);

        _linearWeights = Engine.TensorSubtract(_linearWeights, scaledLinearWeightsGrad);
        _gateWeights = Engine.TensorSubtract(_gateWeights, scaledGateWeightsGrad);
        _linearBias = Engine.TensorSubtract(_linearBias, scaledLinearBiasGrad);
        _gateBias = Engine.TensorSubtract(_gateBias, scaledGateBiasGrad);

        // Notify engine that parameters have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(_linearWeights);
        Engine.InvalidatePersistentTensor(_gateWeights);
        Engine.InvalidatePersistentTensor(_linearBias);
        Engine.InvalidatePersistentTensor(_gateBias);
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
        return Vector<T>.Concatenate(
            new Vector<T>(_linearWeights.ToArray()),
            new Vector<T>(_gateWeights.ToArray()),
            new Vector<T>(_linearBias.ToArray()),
            new Vector<T>(_gateBias.ToArray()));
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
        int linearWeightsSize = _linearWeights.Shape[0] * _linearWeights.Shape[1];
        int gateWeightsSize = _gateWeights.Shape[0] * _gateWeights.Shape[1];
        int expectedLength = linearWeightsSize + gateWeightsSize +
                             _linearBias.Length + _gateBias.Length;

        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, but got {parameters.Length}");
        }

        int index = 0;
        _linearWeights = new Tensor<T>(_linearWeights.Shape, parameters.Slice(index, linearWeightsSize));
        index += linearWeightsSize;
        _gateWeights = new Tensor<T>(_gateWeights.Shape, parameters.Slice(index, gateWeightsSize));
        index += gateWeightsSize;
        _linearBias = new Tensor<T>(_linearBias.Shape, parameters.Slice(index, _linearBias.Length));
        index += _linearBias.Length;
        _gateBias = new Tensor<T>(_gateBias.Shape, parameters.Slice(index, _gateBias.Length));

        // Notify engine that parameters have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(_linearWeights);
        Engine.InvalidatePersistentTensor(_gateWeights);
        Engine.InvalidatePersistentTensor(_linearBias);
        Engine.InvalidatePersistentTensor(_gateBias);
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

        // Clear GPU cached tensors
        _gpuInput = null;
        _gpuLinearOutput = null;
        _gpuGateOutput = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_linearWeights == null || _gateWeights == null)
            throw new InvalidOperationException("Layer weights not initialized.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create constant nodes for weights and biases (already Tensor<T>)
        var linearWeightsNode = TensorOperations<T>.Constant(_linearWeights, "linear_weights");
        var gateWeightsNode = TensorOperations<T>.Constant(_gateWeights, "gate_weights");
        var linearBiasNode = TensorOperations<T>.Constant(_linearBias, "linear_bias");
        var gateBiasNode = TensorOperations<T>.Constant(_gateBias, "gate_bias");

        // Transpose weights for proper matrix multiply
        var linearWeightsT = TensorOperations<T>.Transpose(linearWeightsNode);
        var gateWeightsT = TensorOperations<T>.Transpose(gateWeightsNode);

        var linearOutput = TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(inputNode, linearWeightsT), linearBiasNode);
        var gateOutput = TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(inputNode, gateWeightsT), gateBiasNode);
        var sigmoid = TensorOperations<T>.Sigmoid(gateOutput);

        return TensorOperations<T>.ElementwiseMultiply(linearOutput, sigmoid);
    }

    public override bool SupportsJitCompilation => _linearWeights != null && _gateWeights != null && _linearBias != null && _gateBias != null;
}
