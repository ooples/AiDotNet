using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Radial Basis Function (RBF) layer for neural networks.
/// </summary>
/// <remarks>
/// <para>
/// The RBF layer implements a type of artificial neural network that uses radial basis functions as 
/// activation functions. Each neuron in this layer has a center point in the input space and responds
/// most strongly to inputs near that center. The response decreases as the distance from the center
/// increases, controlled by the width parameter of each neuron.
/// </para>
/// <para><b>For Beginners:</b> This layer works like a collection of specialized detectors.
/// 
/// Think of each neuron in this layer as a spotlight:
/// - Each spotlight has a specific location (center) in the input space
/// - Each spotlight has a certain brightness range (width)
/// - When input comes in, spotlights that are close to that input light up brightly
/// - Spotlights far from the input barely light up at all
/// 
/// For example, if you're recognizing handwritten digits:
/// - One spotlight might be positioned to detect curved lines (like in "8")
/// - Another might detect vertical lines (like in "1")
/// - When a "3" comes in, the spotlights for curves light up strongly, while others stay dim
/// 
/// This layer is particularly good at classification problems and function approximation
/// where the relationship between inputs and outputs is complex or non-linear.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RBFLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Tensor storing the center positions of each RBF neuron in the input space.
    /// </summary>
    /// <remarks>
    /// This tensor has shape [outputSize, inputSize], where each row represents the coordinates
    /// of a center point for one RBF neuron. These centers are the primary trainable parameters of
    /// the layer and determine where in the input space each neuron responds most strongly.
    /// </remarks>
    private Tensor<T> _centers;

    /// <summary>
    /// Tensor storing the width parameters for each RBF neuron.
    /// </summary>
    /// <remarks>
    /// This tensor has shape [outputSize], where each element controls how quickly the response of
    /// the corresponding RBF neuron decreases as the distance from its center increases. Larger
    /// width values mean the neuron responds more broadly, while smaller values make the response
    /// more focused around the center.
    /// </remarks>
    private Tensor<T> _widths;

    /// <summary>
    /// The radial basis function implementation used to compute neuron activations.
    /// </summary>
    /// <remarks>
    /// This interface provides methods to compute the activation of an RBF neuron based on the
    /// distance from the center, as well as derivatives for these computations needed during
    /// backpropagation. Common implementations include Gaussian, Multiquadric, and Inverse Quadratic
    /// functions, each providing different response patterns.
    /// </remarks>
    private IRadialBasisFunction<T> _rbf;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached input is needed during the backward pass to compute gradients. It holds the
    /// batch of input vectors that were processed in the most recent forward pass. The tensor
    /// is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the output tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached output is needed during the backward pass to compute certain derivatives.
    /// It holds the batch of output vectors that were produced in the most recent forward pass.
    /// The tensor is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Stores the gradients of the loss with respect to the center parameters.
    /// </summary>
    /// <remarks>
    /// This tensor holds the accumulated gradients for all center parameters during the backward pass.
    /// It has the same shape as the _centers tensor and is used to update the centers during
    /// the parameter update step. The tensor is null before the first backward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _centersGradient;

    /// <summary>
    /// Stores the gradients of the loss with respect to the width parameters.
    /// </summary>
    /// <remarks>
    /// This tensor holds the accumulated gradients for all width parameters during the backward pass.
    /// It has the same shape as the _widths tensor and is used to update the widths during
    /// the parameter update step. The tensor is null before the first backward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _widthsGradient;

    /// <summary>
    /// Number of RBF neurons (output size).
    /// </summary>
    private readonly int _numCenters;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private readonly int _inputSize;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for RBF layers, indicating that the layer can be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the RBF layer has trainable parameters (centers and widths) that
    /// can be optimized during the training process using backpropagation. The gradients of these parameters
    /// are calculated during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has values (centers and widths) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process of the neural network
    /// 
    /// When you train a neural network containing this layer, the centers and widths will 
    /// automatically adjust to better match the patterns in your specific data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="RBFLayer{T}"/> class with specified dimensions and radial basis function.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer.</param>
    /// <param name="outputSize">The size of the output from the layer (number of RBF neurons).</param>
    /// <param name="rbf">The radial basis function to use for computing neuron activations.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RBF layer with the specified dimensions and radial basis function.
    /// The centers are initialized randomly around the origin, and the widths are initialized with random
    /// values between 0 and 1. The scale of the random initialization for centers depends on the layer dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new RBF layer for your neural network.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many numbers come into the layer (features of your data)
    /// - outputSize: How many RBF neurons to create (more neurons can capture more complex patterns)
    /// - rbf: The specific mathematical function that determines how neurons respond to input
    /// 
    /// For example, if you're analyzing images with 784 pixels and want 100 different pattern detectors,
    /// you might use inputSize=784 and outputSize=100.
    /// 
    /// Common radial basis functions include Gaussian (bell-shaped), Multiquadric, and Inverse Quadratic.
    /// Each creates a different pattern of responsiveness around the neuron centers.
    /// </para>
    /// </remarks>
    public RBFLayer(int inputSize, int outputSize, IRadialBasisFunction<T> rbf)
        : base([inputSize], [outputSize])
    {
        _inputSize = inputSize;
        _numCenters = outputSize;
        _centers = new Tensor<T>([outputSize, inputSize]);
        _widths = new Tensor<T>([outputSize]);
        _rbf = rbf;

        InitializeParameters();
    }

    /// <summary>
    /// Performs the forward pass of the RBF layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after RBF processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the RBF layer. For each neuron (center), it calculates
    /// the Euclidean distance between the input vector and the center, then applies the radial basis function
    /// to this distance to produce the neuron's activation. The input and output are cached for use during
    /// the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the RBF neurons.
    /// 
    /// During the forward pass:
    /// 1. For each input example, the layer measures how far it is from each neuron's center
    /// 2. The distance is plugged into the radial basis function (like a mathematical formula)
    /// 3. The result determines how strongly each neuron activates
    /// 4. Neurons with centers close to the input activate strongly; distant ones activate weakly
    /// 
    /// This is like asking "How similar is this input to each of my known patterns?" The output
    /// tells you the similarity scores for each pattern.
    /// 
    /// The layer saves the input and output for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Handle unbatched input (1D) by adding batch dimension
        bool wasUnbatched = input.Rank == 1;
        var processedInput = wasUnbatched
            ? input.Reshape([1, input.Shape[0]])
            : input;

        // Use Engine.RBFKernel for GPU/CPU acceleration
        // This computes exp(-epsilon * ||x - center||²) for Gaussian RBF
        // Convert widths to epsilons: epsilon = 1 / (2 * width²)
        var epsilons = ComputeEpsilonsFromWidths();
        var output = Engine.RBFKernel(processedInput, _centers, epsilons);

        // Remove batch dimension if input was unbatched
        _lastOutput = wasUnbatched ? output.Reshape([output.Shape[1]]) : output;

        return _lastOutput;
    }

    /// <summary>
    /// Converts width parameters to epsilon values for RBF kernel.
    /// epsilon = 1 / (2 * width²) for Gaussian RBF.
    /// </summary>
    private Tensor<T> ComputeEpsilonsFromWidths()
    {
        var epsilons = new Tensor<T>(_widths.Shape);
        var two = NumOps.FromDouble(2.0);
        for (int i = 0; i < _numCenters; i++)
        {
            var widthSquared = NumOps.Multiply(_widths[i], _widths[i]);
            var twoWidthSquared = NumOps.Multiply(two, widthSquared);
            epsilons[i] = NumOps.Divide(NumOps.One, twoWidthSquared);
        }
        return epsilons;
    }

    /// <summary>
    /// Performs the backward pass of the RBF layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the RBF layer, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradients of the loss
    /// with respect to the centers and widths (to update the layer's parameters) and with respect to
    /// the input (to propagate back to previous layers).
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The error gradient from the next layer is received
    /// 2. The layer calculates how each center should move to reduce the error
    /// 3. The layer calculates how each width should change to reduce the error
    /// 4. The layer calculates how the previous layer's output should change
    /// 
    /// This is like saying "Based on the mistakes we made, how should we adjust our pattern detectors
    /// to be more accurate next time?" The gradients tell us both how to update this layer and
    /// how to guide the previous layers.
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
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Use Engine.RBFKernelBackward for GPU/CPU acceleration
        var epsilons = ComputeEpsilonsFromWidths();
        var (gradInput, gradCenters, gradEpsilons) = Engine.RBFKernelBackward(
            outputGradient, _lastInput, _centers, epsilons, _lastOutput);

        _centersGradient = gradCenters;

        // Convert epsilon gradients back to width gradients
        // epsilon = 1/(2*w²), depsilon/dw = -2/(2*w³) = -1/w³
        // dL/dw = dL/depsilon * depsilon/dw = dL/depsilon * (-1/w³)
        _widthsGradient = ConvertEpsilonGradientsToWidthGradients(gradEpsilons);

        return gradInput;
    }

    /// <summary>
    /// Converts epsilon gradients to width gradients using chain rule.
    /// </summary>
    private Tensor<T> ConvertEpsilonGradientsToWidthGradients(Tensor<T> gradEpsilons)
    {
        var gradWidths = new Tensor<T>(_widths.Shape);
        for (int i = 0; i < _numCenters; i++)
        {
            // depsilon/dwidth = -1/width³
            var widthCubed = NumOps.Multiply(_widths[i], NumOps.Multiply(_widths[i], _widths[i]));
            var dEpsilonDWidth = NumOps.Negate(NumOps.Divide(NumOps.One, widthCubed));
            gradWidths[i] = NumOps.Multiply(gradEpsilons[i], dEpsilonDWidth);
        }
        return gradWidths;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation via the RBFKernel operation to compute gradients.
    /// The operation handles Gaussian RBF computations with proper gradient flow.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Create computation nodes - _centers and _widths are already Tensors
        var inputNode = Autodiff.TensorOperations<T>.Variable(
            _lastInput,
            "input",
            requiresGradient: true);

        var centersNode = Autodiff.TensorOperations<T>.Variable(
            _centers,
            "centers",
            requiresGradient: true);

        var widthsNode = Autodiff.TensorOperations<T>.Variable(
            _widths,
            "widths",
            requiresGradient: true);

        // Apply RBFKernel operation
        var outputNode = Autodiff.TensorOperations<T>.RBFKernel(
            inputNode,
            centersNode,
            widthsNode);

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

        // Update parameter gradients (already Tensor types)
        if (centersNode.Gradient != null)
            _centersGradient = centersNode.Gradient;

        if (widthsNode.Gradient != null)
            _widthsGradient = widthsNode.Gradient;

        // Return input gradient
        return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
    }


    /// <summary>
    /// Updates the parameters of the RBF layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when UpdateParameters is called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the centers and widths of the RBF layer based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter
    /// updates. This method should be called after the backward pass to apply the calculated updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// 1. The center positions are adjusted based on their gradients
    /// 2. The widths are adjusted based on their gradients
    /// 3. The learning rate controls how big each update step is
    /// 
    /// Imagine each RBF neuron as a spotlight:
    /// - Updating the centers moves where the spotlight is pointing
    /// - Updating the widths changes how broad or narrow the spotlight beam is
    /// 
    /// Smaller learning rates mean slower but more stable learning, while larger learning rates
    /// mean faster but potentially unstable learning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_centersGradient == null || _widthsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Use Engine.TensorSubtract and TensorMultiplyScalar for GPU/CPU acceleration
        var scaledCentersGradient = Engine.TensorMultiplyScalar(_centersGradient, learningRate);
        _centers = Engine.TensorSubtract(_centers, scaledCentersGradient);

        var scaledWidthsGradient = Engine.TensorMultiplyScalar(_widthsGradient, learningRate);
        _widths = Engine.TensorSubtract(_widths, scaledWidthsGradient);
    }

    /// <summary>
    /// Gets all trainable parameters of the RBF layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (centers and widths).</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (centers and widths) of the RBF layer as a
    /// single vector. The centers are stored first, followed by the widths. This is useful for optimization
    /// algorithms that operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the RBF layer.
    /// 
    /// The parameters:
    /// - Are the centers and widths that the RBF layer learns during training
    /// - Control where and how widely each neuron responds to inputs
    /// - Are returned as a single list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The centers are stored first in the vector, followed by all the width values.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Convert tensors to vectors and concatenate
        var centersData = _centers.ToArray();
        var widthsData = _widths.ToArray();

        var centersVector = new Vector<T>(centersData);
        var widthsVector = new Vector<T>(widthsData);

        return Vector<T>.Concatenate(centersVector, widthsVector);
    }

    /// <summary>
    /// Sets the trainable parameters of the RBF layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters (centers and widths) to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters (centers and widths) of the RBF layer from a single vector.
    /// The vector should contain the center values first, followed by the width values. This is useful for loading
    /// saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the centers and widths in the RBF layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct total length
    /// - The first part of the vector is used for the centers (positions of the neurons)
    /// - The second part of the vector is used for the widths (how broadly each neuron responds)
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
        int centersSize = _numCenters * _inputSize;
        int totalParams = centersSize + _numCenters;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        // Extract and reshape centers from the first portion of parameters
        var centersVector = parameters.Slice(0, centersSize);
        _centers = Tensor<T>.FromVector(centersVector, [_numCenters, _inputSize]);

        // Extract widths from the remaining portion
        var widthsVector = parameters.Slice(centersSize, _numCenters);
        _widths = Tensor<T>.FromVector(widthsVector, [_numCenters]);
    }

    /// <summary>
    /// Resets the internal state of the RBF layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the RBF layer, including the cached inputs and outputs
    /// from the forward pass, and the gradients from the backward pass. This is useful when starting to
    /// process a new sequence or batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs from previous calculations are cleared
    /// - Calculated gradients are cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// The centers and widths (the learned parameters) are not reset,
    /// only the temporary state information.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _centersGradient = null;
        _widthsGradient = null;
    }

    /// <summary>
    /// Initializes the centers and widths of the RBF layer with random values.
    /// </summary>
    /// <remarks>
    /// This private method initializes the centers with random values scaled by the input and output dimensions,
    /// and initializes the widths with random values between 0 and 1. This provides a good starting point for
    /// training the RBF layer.
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_numCenters + _inputSize)));
        for (int i = 0; i < _numCenters; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _centers[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }

            _widths[i] = NumOps.FromDouble(Random.NextDouble());
        }
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input [batch, inputSize]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // _centers is already a Tensor [numCenters, inputSize]
        var centersNode = TensorOperations<T>.Constant(_centers, "centers");

        // Convert widths to epsilons: epsilon = 1 / (2 * width²) for Gaussian RBF
        var epsilonsTensor = ComputeEpsilonsFromWidths();
        var epsilonsNode = TensorOperations<T>.Constant(epsilonsTensor, "epsilons");

        // Use RBFKernel operation: computes exp(-epsilon * distance²)
        return TensorOperations<T>.RBFKernel(inputNode, centersNode, epsilonsNode);
    }

    public override bool SupportsJitCompilation => true;

}
