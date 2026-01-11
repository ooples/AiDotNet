using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents an expert module in a Mixture-of-Experts architecture, containing a sequence of layers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// An Expert is a container for a sequence of neural network layers that are executed sequentially.
/// In a Mixture-of-Experts (MoE) architecture, multiple experts process the same input, and their outputs
/// are combined based on learned routing weights. Each expert can specialize in processing different
/// types of inputs or patterns.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of an Expert as a mini neural network that specializes in a particular task.
///
/// In a Mixture-of-Experts system:
/// - You have multiple "experts" (mini-networks), each with their own layers
/// - Each expert learns to be good at handling certain types of inputs
/// - A routing mechanism decides which experts should process each input
/// - The final output combines the predictions from the selected experts
///
/// For example, in a language model:
/// - One expert might specialize in technical vocabulary
/// - Another might handle conversational language
/// - Another might focus on formal writing
/// - The router learns to send each input to the most appropriate expert(s)
///
/// This allows the model to scale to very large sizes while keeping computation efficient,
/// since only a subset of experts are activated for each input.
/// </para>
/// </remarks>
public class ExpertLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The sequence of layers that make up this expert.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These layers are executed in order during the forward pass and in reverse order during backpropagation.
    /// Each layer processes the output from the previous layer, creating a pipeline of transformations.
    /// </para>
    /// <para><b>For Beginners:</b> This is the list of layers that define what this expert does.
    ///
    /// Think of it as a recipe:
    /// - Each layer is a step in the recipe
    /// - Data flows through each step in order
    /// - The output of one step becomes the input to the next
    ///
    /// For example, an expert might have:
    /// 1. A dense layer to extract features
    /// 2. An activation layer to add non-linearity
    /// 3. Another dense layer to produce the final output
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _layers;

    /// <summary>
    /// Stores the pre-activation output for use in backpropagation.
    /// </summary>
    private Tensor<T>? _lastPreActivationOutput;

    /// <summary>
    /// Gets a value indicating whether this expert supports training through backpropagation.
    /// </summary>
    /// <value>
    /// <c>true</c> if any of the contained layers support training; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// An expert supports training if at least one of its layers has trainable parameters.
    /// This determines whether gradients will be computed during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you whether this expert can learn from data.
    ///
    /// An expert can learn if:
    /// - At least one of its layers has adjustable parameters
    /// - Those parameters can be updated during training
    ///
    /// If all layers in an expert are fixed (like certain activation layers), the expert
    /// won't be trainable, but it can still process data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _layers.Any(l => l.SupportsTraining);

    /// <summary>
    /// Gets a value indicating whether this expert supports GPU execution.
    /// Returns true if all contained layers support GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution =>
        _layers.All(l => l is LayerBase<T> lb && lb.CanExecuteOnGpu);

    /// <summary>
    /// Gets the total number of trainable parameters across all layers in this expert.
    /// </summary>
    /// <value>
    /// The sum of parameter counts from all contained layers.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property calculates the total number of trainable parameters by summing the
    /// parameter counts of all layers in the expert's sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This counts all the numbers that can be adjusted during training.
    ///
    /// The total includes:
    /// - Weights from all dense layers
    /// - Biases from all layers that use them
    /// - Any other learnable parameters in the layers
    ///
    /// A higher parameter count means the expert can represent more complex patterns,
    /// but also requires more memory and computation.
    /// </para>
    /// </remarks>
    public override int ParameterCount => _layers.Sum(l => l.ParameterCount);

    /// <summary>
    /// Initializes a new instance of the <see cref="ExpertLayer{T}"/> class with the specified layers.
    /// </summary>
    /// <param name="layers">The sequence of layers that make up this expert.</param>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <param name="activationFunction">Optional activation function to apply after all layers (defaults to identity).</param>
    /// <exception cref="ArgumentException">Thrown when the layers list is empty.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates an expert module from a sequence of layers. The layers are executed
    /// in the order provided during forward pass, and in reverse order during backpropagation.
    /// The input and output shapes should match the first layer's input shape and last layer's output shape.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new expert by chaining together multiple layers.
    ///
    /// When creating an expert:
    /// - Provide a list of layers in the order they should execute
    /// - The first layer should accept your input shape
    /// - The last layer should produce your desired output shape
    /// - Each intermediate layer's output should match the next layer's expected input
    ///
    /// For example, to create an expert that reduces dimensions:
    /// <code>
    /// var layers = new List&lt;ILayer&lt;float&gt;&gt;
    /// {
    ///     new DenseLayer&lt;float&gt;(100, 50, new ReLUActivation&lt;float&gt;()),
    ///     new DenseLayer&lt;float&gt;(50, 25, new ReLUActivation&lt;float&gt;())
    /// };
    /// var expert = new ExpertLayer&lt;float&gt;(layers, new[] { 100 }, new[] { 25 });
    /// </code>
    /// </para>
    /// </remarks>
    public ExpertLayer(List<ILayer<T>> layers, int[] inputShape, int[] outputShape, IActivationFunction<T>? activationFunction = null)
        : base(inputShape, outputShape, activationFunction ?? new IdentityActivation<T>())
    {
        if (layers == null || layers.Count == 0)
        {
            throw new ArgumentException("Expert must contain at least one layer.", nameof(layers));
        }

        _layers = layers;
    }

    /// <summary>
    /// Processes the input data through all layers in sequence.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the forward pass by sequentially passing the input through each layer.
    /// The output of each layer becomes the input to the next layer. After all layers have processed
    /// the data, the expert's activation function (if any) is applied to the final output.
    /// </para>
    /// <para><b>For Beginners:</b> This method runs the data through all the expert's layers in order.
    ///
    /// The forward pass works like an assembly line:
    /// 1. Start with the input data
    /// 2. Pass it through the first layer
    /// 3. Take that output and pass it to the second layer
    /// 4. Continue until all layers have processed the data
    /// 5. Apply the expert's activation function (if specified)
    /// 6. Return the final result
    ///
    /// Each layer transforms the data in some way, building up more complex representations
    /// as the data flows through the expert.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var output = input;

        // Pass through each layer sequentially
        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }

        // Store pre-activation output for backpropagation
        _lastPreActivationOutput = output;

        // Apply the expert's activation function if specified
        return ApplyActivation(output);
    }

    /// <summary>
    /// Performs the forward pass on GPU tensors by chaining through all layers.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after processing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// This method executes the GPU forward pass by sequentially passing the input through each layer's
    /// ForwardGpu method. If any layer doesn't support GPU execution, falls back to CPU.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var output = inputs[0];

        // Chain through each layer's ForwardGpu
        foreach (var layer in _layers)
        {
            if (layer is LayerBase<T> layerBase && layerBase.CanExecuteOnGpu)
            {
                output = layerBase.ForwardGpu(output);
            }
            else
            {
                // Fall back to CPU for this layer
                var cpuInput = output.ToTensor();
                var cpuOutput = layer.Forward(cpuInput);
                output = gpuEngine.UploadToGpu(cpuOutput, GpuTensorRole.Activation);
            }
        }

        // Store pre-activation output for backpropagation
        _lastPreActivationOutput = output.ToTensor();

        // Apply the expert's activation function if specified
        if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T>)
        {
            // Apply activation on GPU if possible
            var activationType = GetActivationType();
            if (activationType != FusedActivationType.None)
            {
                output = gpuEngine.ActivationGpu(output, activationType);
            }
            else
            {
                // CPU fallback for unsupported activations
                var cpuOutput = output.ToTensor();
                var activated = ApplyActivation(cpuOutput);
                output = gpuEngine.UploadToGpu(activated, GpuTensorRole.Activation);
            }
        }

        return output;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the input on the GPU.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        var grad = outputGradient;

        // Apply activation backward if needed
        if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T> && _lastPreActivationOutput != null)
        {
            var activationType = GetActivationType();
            if (activationType != FusedActivationType.None)
            {
                var preActGpu = gpuEngine.UploadToGpu<T>(_lastPreActivationOutput, GpuTensorRole.Activation);
                grad = activationType switch
                {
                    FusedActivationType.ReLU => gpuEngine.ReluBackwardGpu<T>(outputGradient, preActGpu),
                    FusedActivationType.Sigmoid => gpuEngine.SigmoidBackwardGpu<T>(outputGradient, preActGpu),
                    FusedActivationType.Tanh => gpuEngine.TanhBackwardGpu<T>(outputGradient, preActGpu),
                    _ => outputGradient
                };
                preActGpu.Dispose();
            }
        }

        // Backpropagate through each layer in reverse order
        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            var layer = _layers[i];
            if (layer is LayerBase<T> layerBase)
            {
                var layerType = layerBase.GetType();
                var backwardGpuMethod = layerType.GetMethod("BackwardGpu", new[] { typeof(IGpuTensor<T>) });
                if (backwardGpuMethod != null)
                {
                    grad = (IGpuTensor<T>)backwardGpuMethod.Invoke(layerBase, new object[] { grad })!;
                }
                else
                {
                    // Fallback to CPU backward
                    var cpuGrad = grad.ToTensor();
                    var cpuInputGrad = layer.Backward(cpuGrad);
                    grad = gpuEngine.UploadToGpu<T>(cpuInputGrad, GpuTensorRole.Gradient);
                }
            }
        }

        return grad;
    }

    /// <summary>
    /// Gets the FusedActivationType for the expert's activation function.
    /// </summary>
    private FusedActivationType GetActivationType()
    {
        if (ScalarActivation is ReLUActivation<T>)
            return FusedActivationType.ReLU;
        if (ScalarActivation is TanhActivation<T>)
            return FusedActivationType.Tanh;
        if (ScalarActivation is SigmoidActivation<T>)
            return FusedActivationType.Sigmoid;
        if (ScalarActivation is GELUActivation<T>)
            return FusedActivationType.GELU;
        if (ScalarActivation is IdentityActivation<T>)
            return FusedActivationType.None;
        return FusedActivationType.None;
    }

    /// <summary>
    /// Calculates gradients by backpropagating through all layers in reverse order.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this expert's output.</param>
    /// <returns>The gradient of the loss with respect to this expert's input.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the backward pass by propagating gradients through layers in reverse order.
    /// Each layer computes gradients for its parameters and passes the input gradient to the previous layer.
    /// The gradients are stored in each layer for the subsequent parameter update step.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps all layers learn from their mistakes by passing error information backward.
    ///
    /// The backward pass works in reverse:
    /// 1. Start with information about how wrong the output was
    /// 2. Apply the derivative of the expert's activation function
    /// 3. Pass this error information to the last layer
    /// 4. That layer calculates how to improve and passes error info to the previous layer
    /// 5. Continue in reverse until reaching the first layer
    /// 6. Return the gradient for the input (so earlier layers can learn too)
    ///
    /// This is the core of how neural networks learn - each layer figures out how to
    /// adjust its parameters to reduce the error.
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
        // Apply the derivative of the expert's activation function
        // Use the stored pre-activation output from the forward pass
        if (_lastPreActivationOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward pass.");
        }

        var gradient = ApplyActivationDerivative(_lastPreActivationOutput, outputGradient);

        // Backpropagate through layers in reverse order
        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            gradient = _layers[i].Backward(gradient);
        }

        return gradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation by delegating to the autodiff implementations
    /// of the constituent layers in this expert. Each sublayer will use its own autodiff if available.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // Apply the derivative of the expert's activation function
        if (_lastPreActivationOutput == null)
            throw new InvalidOperationException("Forward pass must be called before Backward pass.");

        var gradient = ApplyActivationDerivative(_lastPreActivationOutput, outputGradient);

        // Composite layer: backpropagate through layers in reverse order
        // The sublayers will handle their own autodiff if they support it
        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            gradient = _layers[i].Backward(gradient);
        }

        return gradient;
    }


    /// <summary>
    /// Updates all trainable parameters in all layers using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of all layers that support training. The learning rate
    /// controls the step size of the updates - larger values make bigger changes but may cause instability,
    /// while smaller values make more gradual, stable updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies all the learned improvements to every layer.
    ///
    /// After the backward pass has calculated how each layer should change:
    /// - This method actually makes those changes
    /// - It goes through each layer in order
    /// - Each layer updates its weights and biases
    /// - The learning rate controls how big the changes are
    ///
    /// Think of it like this:
    /// - Small learning rate = careful, small adjustments (slower but safer)
    /// - Large learning rate = bold, big adjustments (faster but riskier)
    ///
    /// After calling this method, the expert should perform slightly better than before.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in _layers.Where(l => l.SupportsTraining))
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Gets all trainable parameters from all layers as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters from all layers, concatenated in layer order.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts all trainable parameters from all layers and concatenates them into
    /// a single vector. The parameters are ordered by layer (first layer's parameters, then second layer's, etc.).
    /// This is useful for optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learned values from every layer into one list.
    ///
    /// The returned vector contains:
    /// - All parameters from the first layer
    /// - Then all parameters from the second layer
    /// - And so on for all layers
    ///
    /// This is useful for:
    /// - Saving the expert's knowledge to disk
    /// - Transferring learned parameters to another expert
    /// - Advanced optimization techniques
    /// - Analyzing what the expert has learned
    ///
    /// You can think of it as packaging up everything the expert knows into one container.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector<T>.Concatenate for production-grade parameter collection
        var layerParams = _layers
            .Where(l => l.ParameterCount > 0)
            .Select(l => l.GetParameters())
            .ToArray();

        return layerParams.Length > 0 ? Vector<T>.Concatenate(layerParams) : new Vector<T>(0);
    }

    /// <summary>
    /// Sets all trainable parameters in all layers from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for all layers, concatenated in layer order.</param>
    /// <exception cref="ArgumentException">Thrown when the parameter vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method distributes parameters from a single vector to all layers. The parameters should be
    /// in the same order as returned by GetParameters() - first layer's parameters, then second layer's, etc.
    /// This is useful for loading pre-trained models or implementing advanced optimization algorithms.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads previously saved knowledge back into all the layers.
    ///
    /// When setting parameters:
    /// - The vector must contain exactly the right number of parameters
    /// - Parameters are distributed to layers in order (first layer first, etc.)
    /// - Each layer receives its parameters and updates its weights and biases
    ///
    /// This is the opposite of GetParameters() - instead of collecting knowledge, it distributes it.
    ///
    /// Use cases:
    /// - Loading a saved model from disk
    /// - Transferring knowledge from one expert to another
    /// - Initializing an expert with pre-trained parameters
    /// - Implementing custom optimization algorithms
    ///
    /// If the parameter count doesn't match, an error will be thrown to prevent corruption.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, but got {parameters.Length}.",
                nameof(parameters));
        }

        // === Vectorized Parameter Distribution (Phase B: US-GPU-015) ===
        int offset = 0;
        foreach (var layer in _layers.Where(l => l.ParameterCount > 0))
        {
            var layerParamCount = layer.ParameterCount;
            var layerParamsVec = parameters.Slice(offset, layerParamCount);
            layer.SetParameters(layerParamsVec);
            offset += layerParamCount;
        }
    }

    /// <summary>
    /// Resets the internal state of all layers, clearing any cached values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method calls ResetState() on all contained layers, clearing any cached values from
    /// forward/backward passes. This should be called between different training batches or when
    /// switching between training and inference modes.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the expert's "short-term memory".
    ///
    /// During processing, layers remember:
    /// - Recent inputs they processed
    /// - Intermediate calculations
    /// - Gradients from backpropagation
    ///
    /// ResetState() clears all of this temporary information:
    /// - Frees up memory
    /// - Prevents information from one batch affecting another
    /// - Prepares the expert for processing new data
    ///
    /// Think of it like cleaning a whiteboard before starting a new problem - you want a
    /// fresh start without old information interfering.
    ///
    /// When to call this:
    /// - Between different training batches
    /// - When switching from training to testing
    /// - Before processing a completely new input
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        foreach (var layer in _layers)
        {
            layer.ResetState();
        }
    }

    /// <summary>
    /// Creates a deep copy of this expert, including all contained layers.
    /// </summary>
    /// <returns>A new Expert instance with the same configuration and parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete copy of the expert, including all layers and their parameters.
    /// The clone is independent of the original - changes to one won't affect the other.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an identical copy of the expert.
    ///
    /// Cloning is useful when you want to:
    /// - Experiment with different training approaches on the same starting point
    /// - Create an ensemble of similar but independent experts
    /// - Save a checkpoint while continuing to train
    /// - Implement certain training algorithms that need multiple copies
    ///
    /// The clone has:
    /// - The same layer structure
    /// - The same parameter values
    /// - But is completely independent (changes to one don't affect the other)
    ///
    /// It's like photocopying a document - you get an identical copy that you can
    /// modify without changing the original.
    /// </para>
    /// </remarks>
    public override LayerBase<T> Clone()
    {
        // Clone all layers
        var clonedLayers = _layers.Select(l =>
        {
            if (l is LayerBase<T> layerBase)
            {
                return (ILayer<T>)layerBase.Clone();
            }
            return l; // If not cloneable, use the same reference (not ideal but safe for most cases)
        }).ToList();

        return new ExpertLayer<T>(clonedLayers, InputShape, OutputShape, ScalarActivation);
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Check if all inner layers support JIT
        foreach (var layer in _layers)
        {
            if (layer is LayerBase<T> layerBase && !layerBase.SupportsJitCompilation)
                throw new InvalidOperationException($"Inner layer does not support JIT compilation.");
        }

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Chain layers sequentially
        var currentNode = inputNode;
        foreach (var layer in _layers)
        {
            if (layer is LayerBase<T> layerBase)
            {
                var layerInputNodes = new List<ComputationNode<T>>();
                currentNode = layerBase.ExportComputationGraph(layerInputNodes);
            }
        }

        // Apply expert's activation function if specified
        if (ScalarActivation != null && ScalarActivation.SupportsJitCompilation)
        {
            currentNode = ScalarActivation.ApplyToGraph(currentNode);
        }

        return currentNode;
    }

    public override bool SupportsJitCompilation =>
        _layers.All(l => l is LayerBase<T> layerBase && layerBase.SupportsJitCompilation);

}
