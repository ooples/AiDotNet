namespace AiDotNet.NeuralNetworks.Layers;

using AiDotNet.Gpu;
using AiDotNet.Autodiff;

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
/// - Output before activation: [0.2�0.1 + 0.5�0.4 + 0.1�0.7 + 0.1, 0.2�0.8 + 0.5�0.3 + 0.1�0.2 - 0.2]
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
    /// // Create a layer with 784 inputs (e.g., from a 28�28 image), 
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
    /// <para>
    /// <b>GPU Acceleration:</b> When GPU acceleration is available (IsGpuAccelerationAvailable is true),
    /// large matrix operations automatically use GPU for 10-100x speedup. Small operations stay on CPU
    /// to avoid transfer overhead.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer processes input data to produce predictions.
    ///
    /// The forward pass works in three steps:
    /// 1. Linear transformation: Multiply inputs by weights and add biases
    ///    - Each output is a weighted sum of all inputs plus a bias term
    ///    - GPU-accelerated for large matrices (10-100x faster!)
    /// 2. Apply activation function: Add non-linearity
    ///    - This enables the network to learn complex patterns
    ///    - GPU-accelerated for large tensors
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

        // Use GPU acceleration if available and beneficial
        if (IsGpuAccelerationAvailable && typeof(T) == typeof(float))
        {
            Output = ForwardGpu(input);
        }
        else
        {
            // CPU fallback
            var linearOutput = Input.MatrixMultiply(Weights).Add(Biases);
            Output = ApplyActivation(linearOutput);
        }

        return Output;
    }

    /// <summary>
    /// GPU-accelerated forward pass implementation.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method uses GPU operations for matrix multiplication and activation functions.
    /// Operations are automatically placed on GPU or CPU based on tensor size.
    /// </para>
    /// </remarks>
    private Tensor<T> ForwardGpu(Tensor<T> input)
    {
        var backend = GpuContext!.GpuBackend as IlgpuBackend<float>;
        if (backend == null)
            return ForwardCpu(input); // Fallback

        // Convert tensors to float for GPU operations
        var inputFloat = input as Tensor<float> ?? throw new InvalidOperationException("GPU forward requires float tensors");
        var weightsFloat = Weights as Tensor<float> ?? throw new InvalidOperationException("GPU forward requires float weights");
        var biasesFloat = Biases as Tensor<float> ?? throw new InvalidOperationException("GPU forward requires float biases");

        Tensor<float> result;

        // Check if tensors are large enough to benefit from GPU
        bool useGpu = GpuContext.ShouldUseGpu(inputFloat) || GpuContext.ShouldUseGpu(weightsFloat);

        if (useGpu)
        {
            // GPU path: MatMul + Add + Activation
            using var gpuInput = backend.ToGpu(inputFloat);
            using var gpuWeights = backend.ToGpu(weightsFloat);
            using var gpuBiases = backend.ToGpu(biasesFloat);

            // MatMul: input @ weights
            using var gpuMatMul = backend.MatMul(gpuInput, gpuWeights);

            // Add bias
            using var gpuLinear = backend.Add(gpuMatMul, gpuBiases);

            // Apply activation (currently only ReLU is GPU-accelerated)
            GpuTensor<float> gpuActivated;
            if (ScalarActivation is Activations.ReLUActivation<float>)
            {
                gpuActivated = backend.ReLU(gpuLinear);
            }
            else
            {
                // For other activations, transfer back to CPU
                var linear = backend.ToCpu(gpuLinear);
                return ApplyActivation(linear as Tensor<T> ?? throw new InvalidOperationException()) as Tensor<float>
                    ?? throw new InvalidOperationException();
            }

            result = backend.ToCpu(gpuActivated);
            gpuActivated.Dispose();
        }
        else
        {
            // CPU path for small tensors
            result = ForwardCpu(inputFloat);
        }

        return result as Tensor<T> ?? throw new InvalidOperationException();
    }

    /// <summary>
    /// CPU fallback forward pass implementation.
    /// </summary>
    private Tensor<T> ForwardCpu(Tensor<T> input)
    {
        var linearOutput = input.MatrixMultiply(Weights).Add(Biases);
        return ApplyActivation(linearOutput);
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
    /// <remarks>
    /// <para>
    /// <b>GPU Acceleration:</b> When GPU acceleration is available, gradient computations for large tensors
    /// automatically use GPU for significant speedup. Matrix multiplications and transposes benefit most.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        // Use GPU acceleration if available and beneficial
        if (IsGpuAccelerationAvailable && typeof(T) == typeof(float))
        {
            return BackwardGpu(outputGradient);
        }
        else
        {
            return BackwardCpu(outputGradient);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass implementation.
    /// </summary>
    private Tensor<T> BackwardGpu(Tensor<T> outputGradient)
    {
        var backend = GpuContext!.GpuBackend as IlgpuBackend<float>;
        if (backend == null)
            return BackwardCpu(outputGradient);

        // Convert to float tensors
        var gradFloat = outputGradient as Tensor<float> ?? throw new InvalidOperationException("GPU backward requires float tensors");
        var inputFloat = Input as Tensor<float> ?? throw new InvalidOperationException("GPU backward requires float input");
        var outputFloat = Output as Tensor<float> ?? throw new InvalidOperationException("GPU backward requires float output");
        var weightsFloat = Weights as Tensor<float> ?? throw new InvalidOperationException("GPU backward requires float weights");

        // Check if large enough for GPU
        bool useGpu = GpuContext.ShouldUseGpu(gradFloat) || GpuContext.ShouldUseGpu(weightsFloat);

        if (useGpu)
        {
            // Apply activation derivative
            var activationGradient = ApplyActivationDerivative(gradFloat as Tensor<T> ?? throw new InvalidOperationException(),
                                                              outputFloat as Tensor<T> ?? throw new InvalidOperationException()) as Tensor<float>
                                                              ?? throw new InvalidOperationException();

            Tensor<float> inputGradient, weightsGradient, biasesGradient;

            using (var gpuActivationGrad = backend.ToGpu(activationGradient))
            using (var gpuInput = backend.ToGpu(inputFloat))
            using (var gpuWeights = backend.ToGpu(weightsFloat))
            {
                // Input gradient = activationGradient @ weights^T
                using var gpuWeightsT = backend.Transpose(gpuWeights);
                using var gpuInputGrad = backend.MatMul(gpuActivationGrad, gpuWeightsT);
                inputGradient = backend.ToCpu(gpuInputGrad);

                // Weights gradient = input^T @ activationGradient
                using var gpuInputT = backend.Transpose(gpuInput);
                using var gpuWeightsGrad = backend.MatMul(gpuInputT, gpuActivationGrad);
                weightsGradient = backend.ToCpu(gpuWeightsGrad);

                // Biases gradient = sum(activationGradient, axis=0)
                using var gpuBiasesGrad = backend.Sum(gpuActivationGrad);
                biasesGradient = backend.ToCpu(gpuBiasesGrad);
            }

            WeightsGradient = weightsGradient as Tensor<T> ?? throw new InvalidOperationException();
            BiasesGradient = biasesGradient as Tensor<T> ?? throw new InvalidOperationException();

            return inputGradient as Tensor<T> ?? throw new InvalidOperationException();
        }
        else
        {
            return BackwardCpu(outputGradient);
        }
    }

    /// <summary>
    /// CPU fallback backward pass implementation.
    /// </summary>
    private Tensor<T> BackwardCpu(Tensor<T> outputGradient)
    {
        var activationGradient = ApplyActivationDerivative(outputGradient, Output);

        var inputGradient = activationGradient.MatrixMultiply(Weights.Transpose(new[] { 1, 0 }));
        var weightsGradient = Input.Transpose(new[] { 1, 0 }).MatrixMultiply(activationGradient);
        var biasesGradient = activationGradient.Sum(new[] { 0 });

        WeightsGradient = weightsGradient;
        BiasesGradient = biasesGradient;

        return inputGradient;
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

        // Convert to computation nodes
        var input = Autodiff.TensorOperations<T>.Variable(Input, "input", requiresGradient: true);
        var weights = Autodiff.TensorOperations<T>.Variable(Weights, "weights", requiresGradient: true);
        var biases = Autodiff.TensorOperations<T>.Variable(Biases, "biases", requiresGradient: true);

        // Forward computation using autodiff ops
        // output = input @ weights + biases
        var matmul = Autodiff.TensorOperations<T>.MatrixMultiply(input, weights);
        var linearOutput = Autodiff.TensorOperations<T>.Add(matmul, biases);

        // Apply activation using autodiff
        var activated = ApplyActivationAutodiff(linearOutput);

        // Set the gradient at the output and propagate backward
        activated.Gradient = outputGradient;

        // Perform topological sort and backward pass
        var topoOrder = GetTopologicalOrder(activated);

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
        if (weights.Gradient == null || biases.Gradient == null || input.Gradient == null)
            throw new InvalidOperationException("Gradients not computed properly during autodiff backward pass.");

        WeightsGradient = weights.Gradient;
        BiasesGradient = biases.Gradient;

        return input.Gradient;
    }

    /// <summary>
    /// Gets the topological order of nodes in the computation graph.
    /// </summary>
    private List<Autodiff.ComputationNode<T>> GetTopologicalOrder(Autodiff.ComputationNode<T> root)
    {
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var result = new List<Autodiff.ComputationNode<T>>();

        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((root, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
            {
                continue;
            }

            if (processed)
            {
                visited.Add(node);
                result.Add(node);
            }
            else
            {
                stack.Push((node, true));

                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                        {
                            stack.Push((parent, false));
                        }
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Applies the activation function using automatic differentiation operations.
    /// </summary>
    private Autodiff.ComputationNode<T> ApplyActivationAutodiff(Autodiff.ComputationNode<T> input)
    {
        // Check if using scalar activation
        if (!UsingVectorActivation && ScalarActivation != null)
        {
            // Map scalar activation to autodiff operation
            var activationName = ScalarActivation.GetType().Name;

            if (activationName.Contains("ReLU"))
                return Autodiff.TensorOperations<T>.ReLU(input);
            else if (activationName.Contains("Sigmoid"))
                return Autodiff.TensorOperations<T>.Sigmoid(input);
            else if (activationName.Contains("Tanh"))
                return Autodiff.TensorOperations<T>.Tanh(input);
            else
                throw new NotSupportedException($"Scalar activation {activationName} not supported with autodiff. Use manual backward pass or implement autodiff support for this activation.");
        }

        // Check if using vector activation
        if (UsingVectorActivation && VectorActivation != null)
        {
            var activationName = VectorActivation.GetType().Name;

            if (activationName.Contains("Softmax"))
                return Autodiff.TensorOperations<T>.Softmax(input);
            else if (activationName.Contains("ReLU"))
                return Autodiff.TensorOperations<T>.ReLU(input);
            else if (activationName.Contains("Sigmoid"))
                return Autodiff.TensorOperations<T>.Sigmoid(input);
            else if (activationName.Contains("Tanh"))
                return Autodiff.TensorOperations<T>.Tanh(input);
            else
                throw new NotSupportedException($"Vector activation {activationName} not supported with autodiff. Use manual backward pass or implement autodiff support for this activation.");
        }

        // No activation function, return input as-is
        return input;
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
    /// - 1,000 weight parameters (100 � 10)
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
    }
}