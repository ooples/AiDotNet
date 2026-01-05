using System;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a deconvolutional layer (also known as transposed convolution) in a neural network.
/// </summary>
/// <remarks>
/// <para>
/// A deconvolutional layer performs the opposite operation of a convolutional layer. While convolution
/// reduces spatial dimensions by applying filters, deconvolution expands spatial dimensions by applying
/// learnable filters to upsample the input. This is particularly useful in generative models and
/// image segmentation networks where upsampling is required.
/// </para>
/// <para><b>For Beginners:</b> A deconvolutional layer is like zooming in on an image in a smart way.
/// 
/// Think of it like the reverse of a convolutional layer:
/// - A convolutional layer summarizes information (making images smaller)
/// - A deconvolutional layer expands information (making images larger)
/// 
/// For example, if you have a small feature map representing "cat features," a deconvolutional layer
/// could expand it back to a cat-shaped image.
/// 
/// This is particularly useful for:
/// - Generating images from small encoded representations
/// - Increasing the resolution of feature maps
/// - Creating detailed outputs from simplified inputs
/// 
/// Applications include image generation, super-resolution, and segmentation tasks where
/// you need to expand the spatial dimensions of your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DeconvolutionalLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The collection of filter kernels used for the deconvolution operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the weight values for all kernels used in the layer. It has dimensions
    /// [InputDepth, OutputDepth, KernelSize, KernelSize], where each kernel is a set of weights
    /// that define a specific pattern to generate.
    /// </para>
    /// <para><b>For Beginners:</b> These are the "pattern generators" that the layer uses.
    /// 
    /// Each kernel:
    /// - Is a grid of numbers (weights)
    /// - Creates a specific pattern in the output
    /// - Is learned during training
    /// 
    /// The layer has multiple kernels to generate different patterns, and these kernels
    /// are what actually get updated when the network learns.
    /// </para>
    /// </remarks>
    private Tensor<T> _kernels;

    /// <summary>
    /// The bias values added to the deconvolution results for each output channel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the bias values for each output channel. Biases are constants that are
    /// added to the deconvolution results before applying the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> Biases are like "base values" for each generated pattern.
    /// 
    /// Think of biases as:
    /// - A starting point or baseline value
    /// - Added to the result after applying the pattern generator
    /// - Helping the network be more flexible in what it can create
    /// 
    /// For example, biases help the network generate patterns with different intensities
    /// or brightness levels.
    /// </para>
    /// </remarks>
    private Tensor<T> _biases;

    /// <summary>
    /// Stored input data from the most recent forward pass, used for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During the backward pass (training), the layer needs access to the input data from the forward
    /// pass to calculate the gradients for the kernels. This tensor stores that input data.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the network's "short-term memory" of what it just saw.
    /// 
    /// The layer remembers:
    /// - The last data it processed
    /// - So it can figure out how to improve when learning
    /// 
    /// This is similar to remembering what ingredients you used in a recipe,
    /// so you can adjust them if the dish didn't turn out perfectly.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored output data from the most recent forward pass, used for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During the backward pass (training), the layer needs access to the output data from the forward
    /// pass to calculate the gradients for the activation function. This tensor stores that output data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the network's memory of what result it produced.
    /// 
    /// The layer remembers:
    /// - What output it generated for the last input
    /// - So it can calculate how to improve
    /// 
    /// This allows the network to compare what it created with the expected result
    /// and adjust its internal values to make better outputs next time.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Calculated gradients for the kernels during the backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the calculated gradients for the kernels during backpropagation.
    /// These gradients indicate how the kernels should be adjusted to reduce the loss.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the "improvement directions" for each pattern generator.
    /// 
    /// During training:
    /// - The layer calculates how each kernel weight should change
    /// - These changes are stored here temporarily
    /// - They're later applied to the actual kernels
    /// 
    /// Think of it like a set of instructions for how to adjust each knob
    /// to make the output better next time.
    /// </para>
    /// </remarks>
    private Tensor<T>? _kernelsGradient;

    /// <summary>
    /// Calculated gradients for the biases during the backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the calculated gradients for the biases during backpropagation.
    /// These gradients indicate how the biases should be adjusted to reduce the loss.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the "improvement directions" for each bias value.
    /// 
    /// During training:
    /// - The layer calculates how each bias should change
    /// - These changes are stored here temporarily
    /// - They're later applied to the actual biases
    /// 
    /// Think of it like instructions for how to adjust each baseline value
    /// to make the output better next time.
    /// </para>
    /// </remarks>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Gets the depth (number of channels) of the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The input depth represents the number of feature channels in the input data. In a neural
    /// network, this typically corresponds to the number of features or patterns detected by
    /// previous layers.
    /// </para>
    /// <para><b>For Beginners:</b> Input depth is the number of different features in your input data.
    /// 
    /// Think of it like:
    /// - The number of different patterns the previous layer detected
    /// - The number of "aspects" of the data you're working with
    /// 
    /// For example, in a deep network, the input depth might be 64 or 128,
    /// representing many different detected features.
    /// </para>
    /// </remarks>
    public int InputDepth { get; }

    /// <summary>
    /// Gets the depth (number of channels) of the output data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The output depth represents the number of feature channels that will be generated in the output.
    /// Each output channel is produced by a different set of kernels and captures different aspects
    /// of the upsampled data.
    /// </para>
    /// <para><b>For Beginners:</b> Output depth is how many different types of patterns this layer will create.
    /// 
    /// For example:
    /// - If output depth is 3, the layer might generate RGB color channels
    /// - If output depth is 32, the layer creates 32 different feature maps
    /// 
    /// A higher number usually means more detailed or varied outputs, but
    /// also requires more processing power.
    /// </para>
    /// </remarks>
    public int OutputDepth { get; }

    /// <summary>
    /// Gets the size of each filter (kernel) used in the deconvolution operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The kernel size determines the area of the output that is influenced by each input value.
    /// A larger kernel size means each input value affects a larger area of the output, potentially
    /// creating more detailed or smooth upsampling.
    /// </para>
    /// <para><b>For Beginners:</b> Kernel size is how big each "pattern generator" is.
    /// 
    /// For example:
    /// - A kernel size of 3 means a 3×3 grid (9 weights)
    /// - A kernel size of 5 means a 5×5 grid (25 weights)
    /// 
    /// Larger kernels:
    /// - Can create more complex patterns
    /// - Affect larger areas of the output
    /// - But require more computation
    /// </para>
    /// </remarks>
    public int KernelSize { get; }

    /// <summary>
    /// Gets the step size for positioning the kernel across the output data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In deconvolution, the stride determines how much the output size increases relative to the input.
    /// A stride of 2 typically doubles the spatial dimensions, while a stride of 1 increases them by a smaller amount.
    /// </para>
    /// <para><b>For Beginners:</b> Stride controls how much upsampling (enlargement) happens.
    /// 
    /// Think of it like:
    /// - Stride of 1: Minimal enlargement
    /// - Stride of 2: Roughly doubles the size
    /// - Stride of 4: Roughly quadruples the size
    /// 
    /// For example, if your input is 16×16 pixels and you use a stride of 2,
    /// the output might be around 32×32 pixels (the exact size depends on other factors too).
    /// </para>
    /// </remarks>
    public int Stride { get; }

    /// <summary>
    /// Gets the amount of padding applied during the deconvolution operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In deconvolution, padding actually reduces the output size. This might seem counterintuitive,
    /// but it allows for more control over the exact output dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> Padding in deconvolution works differently than in convolution.
    /// 
    /// In deconvolution:
    /// - More padding makes the output smaller
    /// - Zero padding means maximum enlargement
    /// - It helps control the exact output size
    /// 
    /// This is the opposite of regular convolution, where padding makes outputs larger.
    /// </para>
    /// </remarks>
    public int Padding { get; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Always returns <c>true</c> for deconvolutional layers, as they contain trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation. Deconvolutional
    /// layers have trainable parameters (kernel weights and biases), so they support training.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// For deconvolutional layers:
    /// - The value is always true
    /// - This means the layer can adjust its pattern generators (filters) during training
    /// - It will improve its upsampling abilities as it processes more data
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="DeconvolutionalLayer{T}"/> class with the specified 
    /// parameters and a scalar activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input data.</param>
    /// <param name="outputDepth">The number of output channels to create.</param>
    /// <param name="kernelSize">The size of each filter kernel (width and height).</param>
    /// <param name="stride">The step size for positioning the kernel. Defaults to 1.</param>
    /// <param name="padding">The amount of padding to apply. Defaults to 0.</param>
    /// <param name="activationFunction">The activation function to apply. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a deconvolutional layer with the specified configuration. The output shape is
    /// calculated based on the input shape, kernel size, stride, and padding. The kernels and biases are
    /// initialized with scaled random values.
    /// </para>
    /// <para><b>For Beginners:</b> This setup method creates a new deconvolutional layer with specific settings.
    /// 
    /// When creating the layer, you specify:
    /// - Input details: The shape of your data
    /// - How many output channels to create (outputDepth)
    /// - How big each pattern generator is (kernelSize)
    /// - How much enlargement to apply (stride)
    /// - How to adjust the exact output size (padding)
    /// - What mathematical function to apply to the results (activation)
    /// 
    /// The layer then creates all the necessary pattern generators with random starting values
    /// that will be improved during training.
    /// </para>
    /// </remarks>
    public DeconvolutionalLayer(int[] inputShape, int outputDepth, int kernelSize, int stride = 1, int padding = 0,
                                IActivationFunction<T>? activationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape, outputDepth, kernelSize, stride, padding),
               activationFunction ?? new ReLUActivation<T>())
    {
        InputDepth = inputShape[1];
        OutputDepth = outputDepth;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;

        _kernels = new Tensor<T>([InputDepth, OutputDepth, KernelSize, KernelSize]);
        _biases = new Tensor<T>([OutputDepth]);

        InitializeParameters();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DeconvolutionalLayer{T}"/> class with the specified
    /// parameters and a vector activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input data.</param>
    /// <param name="outputDepth">The number of output channels to create.</param>
    /// <param name="kernelSize">The size of each filter kernel (width and height).</param>
    /// <param name="stride">The step size for positioning the kernel. Defaults to 1.</param>
    /// <param name="padding">The amount of padding to apply. Defaults to 0.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a deconvolutional layer with the specified configuration and a vector activation function,
    /// which operates on entire vectors rather than individual elements. This can be useful when applying more complex
    /// activation functions or when performance is a concern.
    /// </para>
    /// <para><b>For Beginners:</b> This setup method is similar to the previous one, but uses a different type of
    /// activation function.
    /// 
    /// A vector activation function:
    /// - Works on entire groups of numbers at once
    /// - Can be more efficient for certain types of calculations
    /// - Otherwise works the same as the regular activation function
    /// 
    /// You would choose this option if you have a specific mathematical operation that
    /// needs to be applied to groups of outputs rather than individual values.
    /// </para>
    /// </remarks>
    public DeconvolutionalLayer(int[] inputShape, int outputDepth, int kernelSize, int stride = 1, int padding = 0,
                                IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape, outputDepth, kernelSize, stride, padding),
               vectorActivationFunction ?? new ReLUActivation<T>())
    {
        InputDepth = inputShape[1];
        OutputDepth = outputDepth;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;

        _kernels = new Tensor<T>([InputDepth, OutputDepth, KernelSize, KernelSize]);
        _biases = new Tensor<T>([OutputDepth]);

        InitializeParameters();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Calculates the output shape after applying the deconvolution operation.
    /// </summary>
    /// <param name="inputShape">The shape of the input data.</param>
    /// <param name="outputDepth">The number of output channels to create.</param>
    /// <param name="kernelSize">The size of each filter kernel.</param>
    /// <param name="stride">The step size for positioning the kernel.</param>
    /// <param name="padding">The amount of padding to apply.</param>
    /// <returns>The calculated output shape after deconvolution.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape of the data after applying the deconvolution operation.
    /// The formula used is (inputDim - 1) * stride - 2 * padding + kernelSize, which determines
    /// how much the spatial dimensions will increase.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how big your output will be after enlarging the input.
    /// 
    /// The formula takes into account:
    /// - How big your input is
    /// - How much enlargement to apply (stride)
    /// - How big your pattern generators are (kernel size)
    /// - Any adjustments needed (padding)
    /// 
    /// For example, if you have a 16×16 input and use stride 2, kernel size 3, and no padding:
    /// - The output height will be (16-1)*2 - 0 + 3 = 33
    /// - The output width will be (16-1)*2 - 0 + 3 = 33
    /// 
    /// So your 16×16 input becomes approximately 33×33 output (about 4 times larger in area).
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int outputDepth, int kernelSize, int stride, int padding)
    {
        int outputHeight = (inputShape[2] - 1) * stride - 2 * padding + kernelSize;
        int outputWidth = (inputShape[3] - 1) * stride - 2 * padding + kernelSize;

        return [inputShape[0], outputDepth, outputHeight, outputWidth];
    }

    /// <summary>
    /// Initializes the kernel weights and biases with appropriate random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the kernel weights using the Xavier/Glorot initialization method,
    /// which scales the random values based on the number of input and output connections.
    /// This helps improve training convergence. The biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for the pattern generators.
    /// 
    /// When initializing weights:
    /// - Random values are created for each pattern generator
    /// - The values are carefully scaled to work well for training
    /// - Biases start at zero
    /// 
    /// Good initialization is important because:
    /// - It helps the network learn faster
    /// - It prevents certain mathematical problems during training
    /// - It gives each pattern generator a different starting point
    /// 
    /// This uses a technique called "Xavier/Glorot initialization" which works well
    /// with many neural networks.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // Xavier/Glorot initialization
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(NumOps.FromDouble(2.0), NumOps.FromDouble(InputDepth + OutputDepth)));
        for (int i = 0; i < _kernels.Length; i++)
        {
            _kernels[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }

        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Processes the input data through the deconvolutional layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after deconvolution and activation.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the forward pass of the deconvolutional layer. For each position in the output,
    /// it computes the contribution from all relevant input positions, multiplied by the appropriate kernel weights.
    /// The results are summed, the bias is added, and the activation function is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This method enlarges the input data using learned patterns.
    /// 
    /// During the forward pass:
    /// - Each value in the input helps create a region in the output
    /// - The pattern generators (kernels) determine what that region looks like
    /// - The layer combines all these regions to form a larger, detailed output
    /// - The activation function then adjusts these values
    /// 
    /// Think of it like painting a mural by stamping many small patterns next to each other,
    /// where each stamp design comes from your pattern generators (kernels).
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Get the fused activation type for optimal GPU/CPU performance
        var fusedActivation = GetFusedActivationType();

        Tensor<T> result;

        if (fusedActivation != FusedActivationType.None)
        {
            // Use FusedConvTranspose2D for optimal GPU kernel fusion (conv transpose + bias + activation)
            result = Engine.FusedConvTranspose2D(
                input, _kernels, _biases,
                Stride, Stride,
                Padding, Padding,
                0, 0,  // output padding
                fusedActivation);
        }
        else
        {
            // Fallback for unsupported activations: use separate operations
            var stride = new int[] { Stride, Stride };
            var padding = new int[] { Padding, Padding };
            var outputPadding = new int[] { 0, 0 };

            var output = Engine.ConvTranspose2D(input, _kernels, stride, padding, outputPadding);

            // Add bias using broadcast: reshape [OutputDepth] to [1, OutputDepth, 1, 1] for NCHW format
            var biasReshaped = _biases.Reshape([1, OutputDepth, 1, 1]);
            var biasedOutput = Engine.TensorBroadcastAdd(output, biasReshaped);

            result = ApplyActivation(biasedOutput);
        }

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        return result;
    }

    /// <summary>
    /// Calculates gradients for the input, kernels, and biases during backpropagation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method performs the backward pass of the deconvolutional layer during training. It calculates
    /// the gradient of the loss with respect to the input, kernel weights, and biases. The calculated input
    /// gradient is returned for propagation to earlier layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps the layer learn from its mistakes.
    ///
    /// During the backward pass:
    /// - The layer receives information about how wrong its output was
    /// - It calculates how to adjust its pattern generators to be more accurate
    /// - It prepares the gradients for updating kernels and biases
    /// - It passes information back to previous layers so they can learn too
    ///
    /// This is where the actual "learning" happens. The layer figures out how to
    /// adjust all its internal values to make better outputs next time.
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
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        // Calculate bias gradient: sum over batch, height, width
        _biasesGradient = Engine.ReduceSum(activationGradient, new[] { 0, 2, 3 }, keepDims: false);

        var stride = new int[] { Stride, Stride };
        var padding = new int[] { Padding, Padding };

        // Calculate input gradient
        var inputGradient = Engine.ConvTranspose2DBackwardInput(activationGradient, _kernels, _lastInput.Shape, stride, padding);

        // Calculate kernel gradient
        _kernelsGradient = Engine.ConvTranspose2DBackwardKernel(activationGradient, _lastInput, _kernels.Shape, stride, padding);

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. Currently, deconvolution operations
    /// are not yet available in TensorOperations, so this method falls back to the manual implementation.
    /// </para>
    /// <para>
    /// Once deconvolution operations are added to TensorOperations, this method will provide:
    /// - Automatic gradient computation through the computation graph
    /// - Verification of manual gradient implementations
    /// - Support for rapid prototyping with custom modifications
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Convert parameters to computation nodes
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);
        var kernelNode = Autodiff.TensorOperations<T>.Variable(_kernels, "kernel", requiresGradient: true);

        // Convert biases from Vector to Tensor for TensorOperations
        var biasesTensor = new Tensor<T>(new int[] { OutputDepth });
        for (int i = 0; i < OutputDepth; i++)
        {
            biasesTensor[i] = _biases[i];
        }
        var biasNode = Autodiff.TensorOperations<T>.Variable(biasesTensor, "bias", requiresGradient: true);

        // Forward pass using autodiff ConvTranspose2D operation
        var stride = new int[] { Stride, Stride };
        var padding = new int[] { Padding, Padding };

        var deconvOutput = Autodiff.TensorOperations<T>.ConvTranspose2D(
            inputNode,
            kernelNode,
            biasNode,
            stride,
            padding,
            null  // outputPadding - not used in this layer
        );

        // Apply activation if present
        // If vector activation is configured, fall back to manual path
        Autodiff.ComputationNode<T> activated;
        if (VectorActivation != null)
        {
            // Vector activations not supported in autodiff - use manual path
            return BackwardManual(outputGradient);
        }
        activated = ApplyScalarActivationAutodiff(deconvOutput);

        // Perform backward pass with inline topological sort
        activated.Gradient = outputGradient;

        // Production-grade: Inline topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((activated, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
                continue;

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
        _kernelsGradient = kernelNode.Gradient;

        if (biasNode.Gradient != null)
        {
            _biasesGradient = biasNode.Gradient;
        }

        return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
    }

    private Autodiff.ComputationNode<T> ApplyScalarActivationAutodiff(Autodiff.ComputationNode<T> input)
    {
        if (ScalarActivation == null)
            return input;

        // Use generic activation support - works for ALL 39 built-in activations
        return Autodiff.TensorOperations<T>.ApplyActivation(input, ScalarActivation);
    }

    /// <summary>
    /// Updates the layer's parameters (kernel weights and biases) using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the layer's parameters (kernel weights and biases) based on the gradients
    /// calculated during the backward pass. The learning rate controls the step size of the update.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the lessons learned during training.
    /// 
    /// When updating parameters:
    /// - The learning rate controls how big each adjustment is
    /// - Small learning rate = small, careful changes
    /// - Large learning rate = big, faster changes (but might overshoot)
    /// 
    /// The layer takes the gradients calculated during backward pass and uses them to 
    /// update all its kernels and biases, making them slightly better for next time.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_kernelsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _kernels = Engine.TensorSubtract(_kernels, Engine.TensorMultiplyScalar(_kernelsGradient, learningRate));
        _biases = Engine.TensorSubtract(_biases, Engine.TensorMultiplyScalar(_biasesGradient, learningRate));

        // Invalidate GPU cache after parameter update
        Engine.InvalidatePersistentTensor(_kernels);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all kernel weights and biases.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts all trainable parameters (kernel weights and biases) from the layer
    /// and returns them as a single vector. This is useful for optimization algorithms that operate
    /// on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method gathers all the learned values from the layer.
    /// 
    /// The parameters include:
    /// - All values from all pattern generators (kernels)
    /// - All bias values
    /// 
    /// These are combined into a single long list (vector), which can be used for:
    /// - Saving the model
    /// - Sharing parameters between layers
    /// - Advanced optimization techniques
    /// 
    /// This provides access to all the "knowledge" the layer has learned.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(new Vector<T>(_kernels.ToArray()), new Vector<T>(_biases.ToArray()));
    }

    /// <summary>
    /// Sets all trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters (kernel weights and biases) of the layer from a single
    /// vector. The vector must have the exact length required for all parameters of the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's learned values at once.
    /// 
    /// When setting parameters:
    /// - The vector must have exactly the right number of values
    /// - The values are assigned to the kernels and biases in a specific order
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Copying parameters from another model
    /// - Setting parameters that were optimized externally
    /// 
    /// It's like replacing all the "knowledge" in the layer with new information.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedLength = _kernels.Length + _biases.Length;
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, but got {parameters.Length}");
        }

        var kernelVec = parameters.Slice(0, _kernels.Length);
        var biasVec = parameters.Slice(_kernels.Length, _biases.Length);

        _kernels = new Tensor<T>([InputDepth, OutputDepth, KernelSize, KernelSize], kernelVec);
        _biases = new Tensor<T>([OutputDepth], biasVec);

        // Invalidate GPU cache after parameter update
        Engine.InvalidatePersistentTensor(_kernels);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears the cached input and output values from the most recent forward pass,
    /// as well as the gradients calculated during the backward pass. This is useful when starting
    /// to process a new batch or when implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The layer forgets the last input it processed
    /// - It forgets the last output it produced
    /// - It clears any calculated gradients
    /// 
    /// This is useful for:
    /// - Processing a new, unrelated set of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Think of it like wiping a whiteboard clean before starting a new calculation.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _kernelsGradient = null;
        _biasesGradient = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_kernels == null || _biases == null)
            throw new InvalidOperationException("Layer weights not initialized.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        var kernelNode = TensorOperations<T>.Constant(_kernels, "kernel");
        var biasNode = TensorOperations<T>.Constant(_biases, "bias");

        var deconvNode = TensorOperations<T>.ConvTranspose2D(inputNode, kernelNode, biasNode, stride: new[] { Stride, Stride }, padding: new[] { Padding, Padding });

        if (ScalarActivation != null && ScalarActivation.SupportsJitCompilation)
        {
            return ScalarActivation.ApplyToGraph(deconvNode);
        }

        return deconvNode;
    }

    public override bool SupportsJitCompilation
    {
        get
        {
            if (_kernels == null || _biases == null)
                return false;

            if (ScalarActivation != null)
                return ScalarActivation.SupportsJitCompilation;

            return true;
        }
    }
}
