using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that computes the mean (average) of input values along a specified axis.
/// </summary>
/// <remarks>
/// <para>
/// The MeanLayer reduces the dimensionality of data by taking the average of values along a specified axis.
/// This operation is useful for aggregating feature information or reducing sequence data to a fixed-size
/// representation. The output shape has one fewer dimension than the input shape, with the specified axis
/// being removed.
/// </para>
/// <para><b>For Beginners:</b> This layer calculates the average of values in your data along one direction.
/// 
/// Think of it like calculating the average test score for each student across multiple subjects:
/// - Input: A table of scores where rows are students and columns are subjects
/// - MeanLayer with axis=1 (columns): Gives each student's average score across all subjects
/// 
/// Some practical examples:
/// - In image processing: Taking the average across color channels
/// - In text analysis: Taking the average of word embeddings to get a sentence representation
/// - In time series: Taking the average across time steps to get a summary
/// 
/// For instance, if you have data with shape [10, 5, 20] (e.g., 10 batches, 5 time steps, 20 features),
/// a MeanLayer with axis=1 would output shape [10, 20], giving you the average across all time steps.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MeanLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the axis along which the mean is calculated.
    /// </summary>
    /// <value>
    /// The index of the axis for mean calculation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates which dimension of the input tensor will be averaged and removed in the output.
    /// For example, with a 3D input tensor, axis=0 would average across batches, axis=1 would average across
    /// the second dimension (often time steps or rows), and axis=2 would average across the third dimension
    /// (often features or columns).
    /// </para>
    /// <para><b>For Beginners:</b> The axis tells the layer which direction to calculate averages in.
    /// 
    /// Think of your data as a multi-dimensional array:
    /// - axis=0: First dimension (often batch samples)
    /// - axis=1: Second dimension (often rows or time steps)
    /// - axis=2: Third dimension (often columns or features)
    /// 
    /// For example, with image data shaped as [batch, height, width, channels]:
    /// - axis=1 would average across the height dimension
    /// - axis=3 would average across the channels dimension
    /// 
    /// The axis you choose determines what kind of summary you get from your data.
    /// </para>
    /// </remarks>
    public int Axis { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>false</c> because the MeanLayer has no trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that MeanLayer cannot be trained through backpropagation. Since the mean
    /// operation is a fixed mathematical procedure with no learnable parameters, this layer always returns
    /// false for SupportsTraining.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer doesn't learn from data.
    /// 
    /// A value of false means:
    /// - The layer has no internal values that change during training
    /// - It always performs the same mathematical operation (averaging)
    /// - It's a fixed transformation rather than a learned one
    /// 
    /// Many layers in neural networks learn patterns from data (like Convolutional or Dense layers),
    /// but some layers, like MeanLayer, simply apply a fixed mathematical operation.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// The input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the input tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the output tensor from the most recent forward pass, which may be
    /// useful for certain operations or debugging.
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Initializes a new instance of the <see cref="MeanLayer{T}"/> class with the specified input shape and axis.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="axis">The axis along which to compute the mean.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a MeanLayer that computes the mean along the specified axis. The output shape
    /// is calculated by removing the specified axis from the input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary information.
    /// 
    /// When creating a MeanLayer, you need to specify:
    /// - inputShape: The shape of your data (e.g., [32, 10, 128] for 32 samples, 10 time steps, 128 features)
    /// - axis: Which dimension to average over (e.g., 1 to average over the 10 time steps)
    /// 
    /// The constructor automatically calculates what shape your data will have after averaging.
    /// For example, with inputShape=[32, 10, 128] and axis=1, the output shape would be [32, 128].
    /// </para>
    /// </remarks>
    public MeanLayer(int[] inputShape, int axis)
        : base(inputShape, CalculateOutputShape(inputShape, axis))
    {
        Axis = axis;
    }

    /// <summary>
    /// Calculates the output shape of the mean layer based on the input shape and axis.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="axis">The axis along which to compute the mean.</param>
    /// <returns>The calculated output shape for the mean layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape by removing the dimension specified by the axis parameter.
    /// For example, if the input shape is [10, 20, 30] and axis is 1, the output shape will be [10, 30].
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of your data after taking the average.
    /// 
    /// The output shape is calculated by:
    /// - Taking all dimensions from the input shape
    /// - Removing the dimension you're averaging over (the axis)
    /// 
    /// For example:
    /// - Input shape: [5, 10, 15]
    /// - Axis: 1
    /// - Output shape: [5, 15]
    /// 
    /// The dimension you average over disappears because you're collapsing that information
    /// into a single average value.
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int axis)
    {
        var outputShape = new int[inputShape.Length - 1];
        int outputIndex = 0;
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (i != axis)
            {
                outputShape[outputIndex++] = inputShape[i];
            }
        }
        return outputShape;
    }

    /// <summary>
    /// Performs the forward pass of the mean layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after mean calculation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the mean layer. It computes the mean of the input tensor
    /// along the specified axis and returns a tensor with one fewer dimension. The input and output tensors
    /// are cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method performs the actual averaging operation on your data.
    /// 
    /// During the forward pass:
    /// - The layer receives input data
    /// - It calculates the average along the specified axis
    /// - It returns the averaged result with one fewer dimension
    /// - It also saves both the input and output for later use during training
    /// 
    /// The averaging works by:
    /// 1. Creating an output tensor with the correct shape
    /// 2. For each position in the output, averaging all corresponding values in the input
    /// 3. Storing this average in the output tensor
    /// 
    /// For example, with a 2D array like [[1,2,3], [4,5,6]] and axis=0, 
    /// the result would be [2.5, 3.5, 4.5] (average of each column).
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        var output = new Tensor<T>(OutputShape);
        int axisSize = input.Shape[Axis];
        T axisScale = NumOps.FromDouble(1.0 / axisSize);
        
        // Iterate over all dimensions except the mean axis
        var indices = new int[input.Shape.Length];
        IterateOverDimensions(input, output, indices, 0, Axis, (input, output, indices) =>
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                indices[Axis] = i;
                sum = NumOps.Add(sum, input[indices]);
            }
            output[indices] = NumOps.Multiply(sum, axisScale);
        });
        
        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Performs the backward pass of the mean layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the mean layer, which is used during training to propagate
    /// error gradients back through the network. Since the mean operation averages multiple input values to
    /// produce each output value, during backpropagation, the gradient for each output value is distributed
    /// equally among all corresponding input values.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// should change to reduce errors.
    ///
    /// During the backward pass:
    /// - The layer receives the error gradient from the next layer
    /// - It needs to distribute this gradient back to its inputs
    /// - For a mean operation, each input that contributed to an average receives an equal portion of the gradient
    ///
    /// For example:
    /// If 5 values were averaged to produce one output, and that output's gradient is 10,
    /// each of the 5 input values would receive a gradient of 10/5 = 2.
    ///
    /// This process is part of the "backpropagation" algorithm that helps neural networks learn.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Performs the backward pass using manual gradient calculation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        int axisSize = _lastInput.Shape[Axis];
        T axisScale = NumOps.FromDouble(1.0 / axisSize);

        // Iterate over all dimensions except the mean axis
        var indices = new int[_lastInput.Shape.Length];
        IterateOverDimensions(_lastInput, outputGradient, indices, 0, Axis, (_, outputGradient, indices) =>
        {
            for (int i = 0; i < axisSize; i++)
            {
                indices[Axis] = i;
                inputGradient[indices] = NumOps.Multiply(outputGradient[indices], axisScale);
            }
        });

        return inputGradient;
    }

    /// <summary>
    /// Performs the backward pass using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Create computation node for the input
        var inputNode = TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Replay forward pass using autodiff
        // ReduceMean takes axes parameter as array
        var outputNode = TensorOperations<T>.ReduceMean(inputNode, axes: new int[] { Axis }, keepDims: false);

        // Set gradient at output and perform backward pass
        outputNode.Gradient = outputGradient;

        // Perform topological sort and backward pass
        var topoOrder = GetTopologicalOrder(outputNode);

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract and return input gradient
        if (inputNode.Gradient == null)
            throw new InvalidOperationException("Gradient computation failed in automatic differentiation.");

        return inputNode.Gradient;
    }

    /// <summary>
    /// Gets the topological order of nodes in the computation graph.
    /// </summary>
    private List<ComputationNode<T>> GetTopologicalOrder(ComputationNode<T> root)
    {
        var visited = new HashSet<ComputationNode<T>>();
        var result = new List<ComputationNode<T>>();

        var stack = new Stack<(ComputationNode<T> node, bool processed)>();
        stack.Push((root, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
                continue;

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
    /// Updates the parameters of the mean layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is part of the training process, but since MeanLayer has no trainable parameters,
    /// this method does nothing.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally update a layer's internal values during training.
    /// 
    /// However, since MeanLayer just performs a fixed mathematical operation (averaging) and doesn't
    /// have any internal values that can be learned or adjusted, this method is empty.
    /// 
    /// This is unlike layers such as Dense or Convolutional layers, which have weights and biases
    /// that get updated during training.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // MeanLayer has no learnable parameters, so this method is empty
    }

    /// <summary>
    /// Recursively iterates over all dimensions of the input tensor, skipping the specified dimension.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="output">The output tensor.</param>
    /// <param name="indices">The current indices being processed.</param>
    /// <param name="currentDim">The current dimension being processed.</param>
    /// <param name="skipDim">The dimension to skip (the axis for mean calculation).</param>
    /// <param name="action">The action to perform when all indices except the skip dimension have been determined.</param>
    /// <remarks>
    /// <para>
    /// This private helper method is used to iterate over all positions in the input and output tensors.
    /// It recursively explores all dimensions, skipping the dimension specified by skipDim. When all other
    /// dimensions have been fixed, it calls the provided action with the current indices.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper method that efficiently processes multi-dimensional data.
    /// 
    /// Imagine you need to visit every cell in a 3D cube, but you want to handle entire slices at once:
    /// - This method navigates through the dimensions one by one
    /// - It skips the dimension you're averaging over
    /// - When it has fixed positions in all other dimensions, it calls your provided function
    /// 
    /// This is a common pattern for processing multi-dimensional arrays efficiently.
    /// It's like having nested for-loops, but implemented recursively for any number of dimensions.
    /// </para>
    /// </remarks>
    private void IterateOverDimensions(Tensor<T> input, Tensor<T> output, int[] indices, int currentDim, int skipDim, Action<Tensor<T>, Tensor<T>, int[]> action)
    {
        if (currentDim == input.Shape.Length)
        {
            action(input, output, indices);
            return;
        }
        
        if (currentDim == skipDim)
        {
            IterateOverDimensions(input, output, indices, currentDim + 1, skipDim, action);
        }
        else
        {
            for (int i = 0; i < input.Shape[currentDim]; i++)
            {
                indices[currentDim] = i;
                IterateOverDimensions(input, output, indices, currentDim + 1, skipDim, action);
            }
        }
    }

    /// <summary>
    /// Gets all trainable parameters from the mean layer as a single vector.
    /// </summary>
    /// <returns>An empty vector since MeanLayer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer as a single vector. Since MeanLayer
    /// has no trainable parameters, it returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns all the learnable values in the layer.
    /// 
    /// Since MeanLayer:
    /// - Only performs fixed mathematical operations (averaging)
    /// - Has no weights, biases, or other learnable parameters
    /// - The method returns an empty list
    /// 
    /// This is different from layers like Dense layers, which would return their weights and biases.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // MeanLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the mean layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the mean layer, including the cached inputs and outputs.
    /// This is useful when starting to process a new sequence or batch.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs from previous processing are cleared
    /// - The layer forgets any information from previous data batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Ensuring clean state before a new training epoch
    /// - Preventing information from one batch affecting another
    /// 
    /// While the MeanLayer doesn't maintain long-term state across samples (unlike recurrent layers),
    /// clearing these cached values helps with memory management and ensuring a clean processing pipeline.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _lastOutput = null;
    }
}