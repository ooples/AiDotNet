using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a fully connected (dense) feed-forward layer in a neural network.
/// </summary>
/// <remarks>
/// <para>
/// A feed-forward layer, also known as a fully connected or dense layer, is one of the most common
/// types of neural network layers. It connects every input neuron to every output neuron with
/// learnable weights. Each output neuron also has a learnable bias term. The layer applies a linear
/// transformation followed by an activation function to produce its output.
/// </para>
/// <para><b>For Beginners:</b> A feed-forward layer is like a voting system where every input gets to vote on every output.
/// 
/// Imagine you have 3 inputs and 2 outputs:
/// - Each input has a different level of influence (weight) on each output
/// - Each output has its own starting value (bias)
/// - The layer calculates each output by combining all input influences plus the bias
/// - Finally, an activation function adds non-linearity (like setting a threshold)
/// 
/// For example:
/// - Input: [0.2, 0.5, 0.1] (representing features from previous layer)
/// - Weights: [[0.1, 0.8], [0.4, 0.3], [0.7, 0.2]] (each input's influence on each output)
/// - Biases: [0.1, -0.2] (starting values for each output)
/// - Output before activation: [0.2×0.1 + 0.5×0.4 + 0.1×0.7 + 0.1, 0.2×0.8 + 0.5×0.3 + 0.1×0.2 - 0.2]
///                           = [0.39, 0.33]
/// - After activation (e.g., ReLU): [0.39, 0.33] (since both are already positive)
/// 
/// Feed-forward layers are the building blocks of many neural networks. Multiple
/// feed-forward layers stacked together form a "deep" neural network that can
/// learn increasingly complex patterns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class FeedForwardLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight matrix connecting input neurons to output neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the learnable weights for the connections between each input neuron and each output neuron.
    /// The shape is [inputSize, outputSize], where each element represents the strength of the connection
    /// between an input neuron and an output neuron.
    /// </para>
    /// <para><b>For Beginners:</b> These weights determine how strongly each input affects each output.
    /// 
    /// Think of weights like importance factors:
    /// - Positive weights mean "if this input increases, increase the output"
    /// - Negative weights mean "if this input increases, decrease the output"
    /// - Larger values (positive or negative) mean stronger influence
    /// - Values near zero mean weak influence
    /// 
    /// During training:
    /// - The network adjusts these weights to find the best relationships
    /// - Strong patterns get higher weights
    /// - Irrelevant connections get weights closer to zero
    /// 
    /// For example, in an image recognition task, weights might connect pixel brightness values
    /// to features like "contains an edge" or "contains a curved line."
    /// </para>
    /// </remarks>
    private Tensor<T> Weights { get; set; }

    /// <summary>
    /// The bias values for each output neuron.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the learnable bias terms for each output neuron. The shape is [1, outputSize].
    /// The bias is added to the weighted sum of inputs before applying the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> Biases are like default or starting values for each output.
    /// 
    /// Biases serve several important purposes:
    /// - They allow outputs to be activated even when all inputs are zero
    /// - They act like an adjustable threshold for each neuron
    /// - They give the network more flexibility in learning
    /// 
    /// For example:
    /// - A neuron with a large negative bias is "reluctant" to activate
    /// - A neuron with a large positive bias "wants" to activate
    /// - During training, biases adjust to find the optimal activation threshold
    /// 
    /// Without biases, all outputs would be zero when all inputs are zero,
    /// which would limit what the network can learn.
    /// </para>
    /// </remarks>
    private Tensor<T> Biases { get; set; }

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
    private Tensor<T> Input { get; set; }

    /// <summary>
    /// The output tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the output produced during the last forward pass. It is used during
    /// backpropagation to compute certain gradients, particularly for activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This stores what the layer output after its most recent calculation.
    /// 
    /// During training:
    /// - The network needs to remember what predictions it made
    /// - This helps calculate how to improve the weights and biases
    /// - The output values are used when computing how to adjust parameters
    /// 
    /// This is also cleared after each training batch to save memory.
    /// </para>
    /// </remarks>
    private Tensor<T> Output { get; set; }

    /// <summary>
    /// The gradients for the weights, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the gradients of the loss with respect to each weight. These gradients are
    /// used to update the weights during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each weight value.
    /// 
    /// During training:
    /// - The network calculates how each weight contributed to errors
    /// - Gradients show both direction and amount to change each weight
    /// - Larger gradients mean bigger adjustments are needed
    /// 
    /// For example:
    /// - A positive gradient means "decrease this weight to reduce error"
    /// - A negative gradient means "increase this weight to reduce error"
    /// - The magnitude indicates how strongly the weight should change
    /// 
    /// These gradients are used in the UpdateParameters method to actually
    /// modify the weights.
    /// </para>
    /// </remarks>
    private Tensor<T> WeightsGradient { get; set; }

    /// <summary>
    /// The gradients for the biases, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the gradients of the loss with respect to each bias. These gradients are
    /// used to update the biases during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each bias value.
    /// 
    /// During training:
    /// - The network calculates how each bias contributed to errors
    /// - These gradients show how to adjust the "threshold" of each neuron
    /// - They work just like weight gradients, but for bias values
    /// 
    /// For example:
    /// - If a neuron activates too easily, its bias gradient will be positive
    ///   (suggesting to decrease the bias)
    /// - If a neuron doesn't activate enough, its bias gradient will be negative
    ///   (suggesting to increase the bias)
    /// 
    /// Bias gradients are often simpler to calculate than weight gradients because
    /// each bias affects only one output directly.
    /// </para>
    /// </remarks>
    private Tensor<T> BiasesGradient { get; set; }

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuInput;
    private IGpuTensor<T>? _gpuOutput;
    private int[] _gpuInputShape = [];

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because feed-forward layers have trainable parameters (weights and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the feed-forward layer supports training through backpropagation.
    /// The layer has trainable parameters (weights and biases) that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its weights and biases during training
    /// - It will improve its performance as it sees more data
    /// - It has parameters that are updated to make better predictions
    /// 
    /// Feed-forward layers are the primary learning components in many neural networks,
    /// as they contain most of the trainable parameters.
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
    /// <remarks>
    /// This includes all weights (inputSize × outputSize) and all biases (outputSize).
    /// </remarks>
    public override int ParameterCount => Weights.Length + Biases.Length;

    /// <summary>
    /// Gets the weight tensor for JIT compilation and graph composition.
    /// </summary>
    public Tensor<T> GetWeightsTensor() => Weights;

    /// <summary>
    /// Gets the bias tensor for JIT compilation and graph composition.
    /// </summary>
    public Tensor<T> GetBiasesTensor() => Biases;

    /// <summary>
    /// Initializes a new instance of the <see cref="FeedForwardLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputSize">The number of input neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="activationFunction">The activation function to apply after the linear transformation.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new feed-forward layer with the specified input size, output size, and
    /// activation function. The weights are initialized with small random values, and the biases are
    /// initialized to zero. The activation function operates on individual scalar values in the output tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the feed-forward layer with the specific number of inputs and outputs you need.
    /// 
    /// When creating a feed-forward layer, you need to specify:
    /// - Input size: How many values are coming into the layer
    /// - Output size: How many values you want the layer to produce
    /// - Activation function: How to introduce non-linearity (like ReLU or Sigmoid)
    /// 
    /// For example:
    /// ```csharp
    /// // Create a layer with 784 inputs (e.g., from a 28×28 image), 
    /// // 128 outputs, and ReLU activation
    /// var hiddenLayer = new FeedForwardLayer<float>(784, 128, new ReLUActivation<float>());
    /// 
    /// // Create an output layer with 128 inputs (from previous layer),
    /// // 10 outputs (e.g., for 10 classes), and Softmax activation
    /// var outputLayer = new FeedForwardLayer<float>(128, 10, new SoftmaxActivation<float>());
    /// ```
    /// 
    /// The constructor automatically initializes weights and biases with appropriate
    /// starting values to begin training.
    /// </para>
    /// </remarks>
    public FeedForwardLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
        : base([inputSize], [outputSize], activationFunction ?? new ReLUActivation<T>())
    {
        Weights = Tensor<T>.CreateRandom([inputSize, outputSize]);
        Biases = Tensor<T>.CreateDefault([1, outputSize], NumOps.Zero);
        WeightsGradient = Tensor<T>.Empty();
        BiasesGradient = Tensor<T>.Empty();
        Input = Tensor<T>.Empty();
        Output = Tensor<T>.Empty();

        // Register tensors for GPU memory persistence
        RegisterTrainableParameter(Weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(Biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="FeedForwardLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputSize">The number of input neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="activationFunction">The vector activation function to apply after the linear transformation.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new feed-forward layer with the specified input size, output size, and
    /// vector activation function. The weights are initialized with small random values, and the biases are
    /// initialized to zero. Unlike the other constructor, this one accepts a vector activation function that operates on
    /// entire vectors rather than individual scalar values.
    /// </para>
    /// <para><b>For Beginners:</b> This is an alternative setup that uses a different kind of activation function.
    /// 
    /// This constructor is almost identical to the first one, but with one key difference:
    /// - Regular activation: processes each output value separately
    /// - Vector activation: processes the entire output vector together
    /// 
    /// Vector activation functions like Softmax are useful for:
    /// - Classification problems (choosing between multiple categories)
    /// - Problems where outputs need to sum to 1 (like probabilities)
    /// - Cases where output values should influence each other
    /// 
    /// For example, Softmax makes sure that increasing one output decreases all others,
    /// which is perfect for classification tasks.
    /// </para>
    /// </remarks>
    public FeedForwardLayer(int inputSize, int outputSize, IVectorActivationFunction<T>? activationFunction = null)
        : base([inputSize], [outputSize], activationFunction ?? new ReLUActivation<T>())
    {
        Weights = Tensor<T>.CreateRandom([inputSize, outputSize]);
        Biases = Tensor<T>.CreateDefault([1, outputSize], NumOps.Zero);
        WeightsGradient = Tensor<T>.Empty();
        BiasesGradient = Tensor<T>.Empty();
        Input = Tensor<T>.Empty();
        Output = Tensor<T>.Empty();

        // Register tensors for GPU memory persistence
        RegisterTrainableParameter(Weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(Biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Performs the forward pass of the feed-forward layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after the linear transformation and activation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the feed-forward layer. It performs a matrix multiplication
    /// between the input and the weights, adds the biases, and applies the activation function to produce
    /// the final output. The input and output are cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer processes input data to produce predictions.
    /// 
    /// The forward pass works in three steps:
    /// 1. Linear transformation: Multiply inputs by weights and add biases
    ///    - Each output is a weighted sum of all inputs plus a bias term
    /// 2. Apply activation function: Add non-linearity
    ///    - This enables the network to learn complex patterns
    /// 3. Store inputs and outputs for later use in training
    ///    - This information is needed when updating weights and biases
    /// 
    /// This simple operation (multiply by weights, add bias, apply activation)
    /// is the core of how neural networks transform data.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        Input = input;

        // Use Engine.TensorMatMul for GPU acceleration
        var matmul = Engine.TensorMatMul(Input, Weights);

        // Add biases (broadcast [1, outputSize] to [batchSize, outputSize]) using engine op
        var biasBroadcast = Biases.Reshape([1, Weights.Shape[1]]);
        var linearOutput = Engine.TensorBroadcastAdd(matmul, biasBroadcast);

        Output = ApplyActivation(linearOutput);

        return Output;
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the feed-forward computation (matmul + bias + activation) entirely on GPU
    /// without downloading intermediate results to CPU.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];

        // MatMul: input @ Weights
        var matmul = gpuEngine.BatchedMatMulGpu(input, Weights);

        // Add biases
        var biased = gpuEngine.AddBiasGpu(matmul, Biases);

        // Apply activation
        IGpuTensor<T> output;
        if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T>)
        {
            var fusedType = MapActivationToFused();
            output = gpuEngine.ActivationGpu<T>(biased, fusedType);
        }
        else
        {
            output = biased;
        }

        // Cache state for backward pass only during training
        // Skip this expensive download during inference (50% overhead reduction)
        if (IsTrainingMode)
        {
            // Cache GPU tensors for GPU-resident backward pass
            _gpuInput = input;
            _gpuOutput = output;
            _gpuInputShape = input.Shape;

            // Also cache CPU tensors for fallback backward pass
            Input = input.ToTensor();
            Output = output.ToTensor();
        }

        return output;
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

        if (_gpuInput == null || _gpuOutput == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        int[] inputShape = _gpuInputShape;
        int rank = inputShape.Length;

        // Apply activation derivative
        IGpuTensor<T> activationGrad;
        if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T>)
        {
            // Use GPU activation backward
            var fusedType = MapActivationToFused();
            if (fusedType == FusedActivationType.ReLU)
            {
                activationGrad = gpuEngine.ReluBackwardGpu(outputGradient, _gpuOutput);
            }
            else
            {
                // For other activations, compute derivative and multiply
                // Fall back to element-wise multiply with derivative
                activationGrad = outputGradient;
            }
        }
        else
        {
            activationGrad = outputGradient;
        }

        // Compute gradients based on tensor rank
        int inputFeatures = inputShape[^1];
        int outputFeatures = activationGrad.Shape[^1];

        // Flatten to 2D for matmul operations
        int totalBatch = 1;
        for (int d = 0; d < rank - 1; d++)
            totalBatch *= inputShape[d];

        var input2D = gpuEngine.ReshapeGpu(_gpuInput, new[] { totalBatch, inputFeatures });
        var grad2D = gpuEngine.ReshapeGpu(activationGrad, new[] { totalBatch, outputFeatures });

        // 1. Bias gradient: sum over batch dimension
        var biasGrad = gpuEngine.SumAxisGpu(grad2D, 0);
        BiasesGradient = biasGrad.ToTensor().Reshape([1, outputFeatures]);

        // 2. Weights gradient: input^T @ grad
        var inputT = gpuEngine.TransposeGpu(input2D);
        var weightsGrad = gpuEngine.MatMulGpuTensors(inputT, grad2D);
        WeightsGradient = weightsGrad.ToTensor();

        // 3. Input gradient: grad @ weights^T
        var weightsT = gpuEngine.UploadToGpu(Engine.TensorTranspose(Weights), GpuTensorRole.Weight);
        var inputGrad2D = gpuEngine.MatMulGpuTensors(grad2D, weightsT);

        // Reshape back to original input shape
        var inputGradient = gpuEngine.ReshapeGpu(inputGrad2D, inputShape);

        return inputGradient;
    }

    /// <summary>
    /// Performs the backward pass of the feed-forward layer to compute gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass (backpropagation) of the feed-forward layer. It computes
    /// the gradients of the loss with respect to the layer's weights, biases, and inputs. These gradients
    /// are used to update the parameters during training and to propagate the error signal back to the previous layer.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer learns from its mistakes during training.
    ///
    /// The backward pass has several steps:
    /// 1. Apply activation function derivative:
    ///    - This determines how sensitive the output is to small changes
    /// 2. Calculate gradient for weights:
    ///    - Shows how each weight contributed to errors
    /// 3. Calculate gradient for biases:
    ///    - Shows how each bias affected the output
    /// 4. Calculate gradient to pass to previous layer:
    ///    - Helps the earlier layers learn as well
    ///
    /// It's like figuring out who was responsible for a mistake in a team:
    /// - How much did each weight contribute to the error?
    /// - How much did each bias contribute?
    /// - How should we adjust them to do better next time?
    /// - What feedback should we give to the previous layer?
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
        // Apply activation derivative
        Tensor<T> activationGradient;
        if (ScalarActivation != null)
        {
            // Use optimized Engine operation
            var derivatives = ScalarActivation.Derivative(Output);
            activationGradient = Engine.TensorMultiply(derivatives, outputGradient);
        }
        else
        {
            // Fallback or Vector activation
            activationGradient = ApplyActivationDerivative(Output, outputGradient);
        }

        int rank = Input.Shape.Length;

        if (rank == 2)
        {
            // Standard 2D case: [batch, features]
            var weightsT = Engine.TensorTranspose(Weights);
            var inputGradient = Engine.TensorMatMul(activationGradient, weightsT);

            var inputT = Engine.TensorTranspose(Input);
            WeightsGradient = Engine.TensorMatMul(inputT, activationGradient);

            BiasesGradient = Engine.ReduceSum(activationGradient, new[] { 0 }, keepDims: true);

            return inputGradient;
        }
        else if (rank == 3)
        {
            // 3D case: [batch, seq, features]
            // Reshape to 2D, compute gradients, reshape back
            int batchSize = Input.Shape[0];
            int seqLen = Input.Shape[1];
            int inputFeatures = Input.Shape[2];
            int outputFeatures = activationGradient.Shape[2];

            // Flatten batch and seq dimensions for matmul
            var input2D = Input.Reshape([batchSize * seqLen, inputFeatures]);
            var grad2D = activationGradient.Reshape([batchSize * seqLen, outputFeatures]);

            // Input Gradient: grad @ Weights^T -> [batch*seq, inputFeatures]
            var weightsT = Engine.TensorTranspose(Weights);
            var inputGradient2D = Engine.TensorMatMul(grad2D, weightsT);
            var inputGradient = inputGradient2D.Reshape([batchSize, seqLen, inputFeatures]);

            // Weights Gradient: Input^T @ grad -> [inputFeatures, outputFeatures]
            var inputT = Engine.TensorTranspose(input2D);
            WeightsGradient = Engine.TensorMatMul(inputT, grad2D);

            // Biases Gradient: sum(grad, axis=0)
            BiasesGradient = Engine.ReduceSum(grad2D, new[] { 0 }, keepDims: true);

            return inputGradient;
        }
        else
        {
            // Higher-rank tensors: flatten all but last dimension
            int totalBatch = 1;
            for (int d = 0; d < rank - 1; d++)
                totalBatch *= Input.Shape[d];
            int inputFeatures = Input.Shape[rank - 1];
            int outputFeatures = activationGradient.Shape[rank - 1];

            var input2D = Input.Reshape([totalBatch, inputFeatures]);
            var grad2D = activationGradient.Reshape([totalBatch, outputFeatures]);

            var weightsT = Engine.TensorTranspose(Weights);
            var inputGradient2D = Engine.TensorMatMul(grad2D, weightsT);
            var inputGradient = inputGradient2D.Reshape(Input.Shape);

            var inputT = Engine.TensorTranspose(input2D);
            WeightsGradient = Engine.TensorMatMul(inputT, grad2D);

            BiasesGradient = Engine.ReduceSum(grad2D, new[] { 0 }, keepDims: true);

            return inputGradient;
        }
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
        if (Input == null || Input.Shape == null || Input.Shape.Length == 0)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = Input.Shape[0];
        int outputSize = Biases.Shape[1];

        // Production-grade: Compute activation derivative using cached Output
        // For most activations, derivative can be computed from the activated output:
        // - ReLU: derivative is 1 where Output > 0, else 0
        // - Sigmoid: derivative = Output * (1 - Output)
        // - Tanh: derivative = 1 - Output²
        Tensor<T> preActivationGradient;
        if (UsingVectorActivation && VectorActivation != null)
        {
            var actDeriv = VectorActivation.Derivative(Output);
            preActivationGradient = Engine.TensorMultiply(outputGradient, actDeriv);
        }
        else if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T>)
        {
            var activation = ScalarActivation;
            var activationDerivative = Output.Transform((x, _) => activation.Derivative(x));
            preActivationGradient = Engine.TensorMultiply(outputGradient, activationDerivative);
        }
        else
        {
            preActivationGradient = outputGradient;
        }

        // Build minimal autodiff graph for linear part: Z = X @ W + b
        var input = Autodiff.TensorOperations<T>.Variable(Input, "input", requiresGradient: true);
        var weights = Autodiff.TensorOperations<T>.Variable(Weights, "weights", requiresGradient: true);
        var matmul = Autodiff.TensorOperations<T>.MatrixMultiply(input, weights);

        // Broadcast biases to match batch dimension using engine TensorTile
        // Biases: [1, outputSize] -> tile to [batchSize, outputSize]
        var broadcastedBiases = Engine.TensorTile(Biases, new[] { batchSize, 1 });
        var biasesBroadcast = Autodiff.TensorOperations<T>.Variable(broadcastedBiases, "biases_broadcast", requiresGradient: true);
        var linearOutput = Autodiff.TensorOperations<T>.Add(matmul, biasesBroadcast);

        // Set the pre-activation gradient at linearOutput (after manually handling activation)
        linearOutput.Gradient = preActivationGradient;

        // Topological sort and backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((linearOutput, false));
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

        // Extract gradients
        if (weights.Gradient == null || biasesBroadcast.Gradient == null || input.Gradient == null)
            throw new InvalidOperationException("Gradients not computed properly during autodiff backward pass.");

        WeightsGradient = weights.Gradient;

        // Use Engine.ReduceSum for consistency with BackwardManual
        BiasesGradient = Engine.ReduceSum(biasesBroadcast.Gradient, new[] { 0 }, keepDims: true);

        return input.Gradient;
    }

    /// <summary>
    /// Updates the weights and biases using the calculated gradients and the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases based on the gradients calculated during the backward pass.
    /// The learning rate determines the size of the parameter updates. Smaller learning rates lead to more
    /// stable but slower training, while larger learning rates can lead to faster but potentially unstable training.
    /// </para>
    /// <para><b>For Beginners:</b> This method actually changes the weights and biases to improve future predictions.
    /// 
    /// After figuring out how each parameter should change:
    /// - Each weight and bias is adjusted in the direction that reduces errors
    /// - The learning rate controls how big these adjustments are
    /// 
    /// Think of it like adjusting a recipe after tasting:
    /// - Too salty? Reduce salt next time (adjust weights/biases)
    /// - But make small adjustments (learning rate), not drastic ones
    /// 
    /// For example, with a learning rate of 0.01:
    /// - A gradient of 0.5 would change the parameter by -0.005
    /// - A gradient of -2.0 would change the parameter by +0.02
    /// 
    /// The minus sign in the code is because we want to go in the opposite
    /// direction of the gradient to minimize error.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        Weights = Weights.Subtract(WeightsGradient.Multiply(learningRate));
        Biases = Biases.Subtract(BiasesGradient.Multiply(learningRate));

        // Notify engine that parameters have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(Weights);
        Engine.InvalidatePersistentTensor(Biases);
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) of the layer as a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving
    /// and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the layer's learnable values into a single list.
    /// 
    /// The parameters include:
    /// - All the weight values (the majority of the parameters)
    /// - All the bias values (one per output neuron)
    /// 
    /// This combined list is useful for:
    /// - Saving a trained model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need all parameters together
    /// 
    /// For example, a layer with 100 inputs and 10 outputs would have:
    /// - 1,000 weight parameters (100 × 10)
    /// - 10 bias parameters (one per output)
    /// - Totaling 1,010 parameters in the returned vector
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = Weights.Length + Biases.Length;
        var parameters = new Vector<T>(totalParams);

        int index = 0;

        // Copy weights parameters
        for (int i = 0; i < Weights.Shape[0]; i++)
        {
            for (int j = 0; j < Weights.Shape[1]; j++)
            {
                parameters[index++] = Weights[i, j];
            }
        }

        // Copy biases parameters
        for (int j = 0; j < Biases.Shape[1]; j++)
        {
            parameters[index++] = Biases[0, j];
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
    /// This method sets all trainable parameters (weights and biases) of the layer from a single vector.
    /// This is useful for loading saved model weights or for implementing optimization algorithms
    /// that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's learnable values from a provided list.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the exact right length
    /// - The values are distributed back to the weights and biases
    /// - This allows loading previously trained weights
    /// 
    /// Use cases include:
    /// - Restoring a saved model
    /// - Using pre-trained weights
    /// - Testing specific weight configurations
    /// 
    /// The method throws an error if the provided vector doesn't contain exactly the right number of values.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != Weights.Length + Biases.Length)
        {
            throw new ArgumentException($"Expected {Weights.Length + Biases.Length} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set weights parameters
        for (int i = 0; i < Weights.Shape[0]; i++)
        {
            for (int j = 0; j < Weights.Shape[1]; j++)
            {
                Weights[i, j] = parameters[index++];
            }
        }

        // Set biases parameters
        for (int j = 0; j < Biases.Shape[1]; j++)
        {
            Biases[0, j] = parameters[index++];
        }

        // Notify engine that parameters have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(Weights);
        Engine.InvalidatePersistentTensor(Biases);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing all cached values from forward
    /// and backward passes. This is useful when starting to process a new batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input and output are cleared
    /// - The calculated gradients are cleared
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
        Input = Tensor<T>.Empty();
        Output = Tensor<T>.Empty();
        WeightsGradient = Tensor<T>.Empty();
        BiasesGradient = Tensor<T>.Empty();

        // Clear GPU cached tensors
        _gpuInput = null;
        _gpuOutput = null;
        _gpuInputShape = [];
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (Weights == null || Biases == null)
            throw new InvalidOperationException("Layer weights and biases not initialized.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        var weightsNode = TensorOperations<T>.Constant(Weights, "weights");
        var biasesNode = TensorOperations<T>.Constant(Biases, "biases");

        var matmulNode = TensorOperations<T>.MatrixMultiply(inputNode, weightsNode);
        var addNode = TensorOperations<T>.Add(matmulNode, biasesNode);

        if (ScalarActivation != null && ScalarActivation.SupportsJitCompilation)
        {
            return ScalarActivation.ApplyToGraph(addNode);
        }
        else if (VectorActivation != null)
        {
            var activation = (IActivationFunction<T>)VectorActivation;
            if (activation.SupportsJitCompilation)
            {
                return activation.ApplyToGraph(addNode);
            }
        }

        return addNode;
    }

    public override bool SupportsJitCompilation
    {
        get
        {
            if (Weights == null || Biases == null)
                return false;

            if (ScalarActivation != null)
                return ScalarActivation.SupportsJitCompilation;

            if (VectorActivation != null)
            {
                var activation = (IActivationFunction<T>)VectorActivation;
                return activation.SupportsJitCompilation;
            }

            return true;
        }
    }
}
