using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that computes the logarithm of variance along a specified axis in the input tensor.
/// </summary>
/// <remarks>
/// <para>
/// The LogVarianceLayer calculates the statistical variance of values along a specified axis of the input tensor,
/// and then computes the natural logarithm of that variance. This is often used in neural networks for calculating
/// statistical measures, normalizing data, or as part of variational autoencoders (VAEs).
/// </para>
/// <para><b>For Beginners:</b> This layer measures how much the values in your data spread out from their average (variance),
/// and then takes the logarithm of that spread.
/// 
/// Think of it like measuring how consistent or varied your data is:
/// - Low values mean the data points are very similar to each other
/// - High values mean the data points vary widely
/// 
/// For example, if you have a set of images:
/// - Images that are very similar would produce low log-variance
/// - Images that are very different would produce high log-variance
/// 
/// This is often used in AI models that need to understand the variation in the data,
/// such as in models that generate new data similar to what they've been trained on.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LogVarianceLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the axis along which the variance is calculated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates the dimension of the input tensor along which the variance will be calculated.
    /// For example, if Axis is 1 and the input tensor has shape [batch, features], the variance will be calculated
    /// across the features dimension, resulting in one variance value per batch item.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the layer which direction to look when calculating variance.
    /// 
    /// For example, with a 2D data array (like a table):
    /// - Axis 0 means calculate variance down each column
    /// - Axis 1 means calculate variance across each row
    /// 
    /// The value depends on how your data is organized and what kind of variance you want to measure.
    /// </para>
    /// </remarks>
    public int Axis { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property returns false because the LogVarianceLayer does not have any trainable parameters,
    /// though it does support backward pass for gradient propagation through the network.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the layer can learn from training data.
    /// 
    /// A value of false means:
    /// - This layer doesn't have any values that get updated during training
    /// - It performs a fixed mathematical calculation (log of variance)
    /// - However, during training, it still helps gradients flow backward through the network
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the input tensor from the most recent forward pass. It is needed during the backward pass
    /// to compute gradients correctly. This field is reset when ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the most recent data that was fed into the layer.
    /// 
    /// The layer needs to remember the input:
    /// - To calculate how each input value affected the output
    /// - To determine how to propagate gradients during training
    /// - To ensure the backward pass works correctly
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The output tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor from the most recent forward pass. It is needed during the backward pass
    /// because the derivative of the logarithm function depends on the output value. This field is reset when ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the most recent result that came out of the layer.
    /// 
    /// The layer needs to remember its output:
    /// - Because the derivative of log(x) is 1/x, so we need x (our output) during backpropagation
    /// - To avoid recalculating values during the backward pass
    /// - To make the training process more efficient
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// The mean values calculated during the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the mean values calculated during the most recent forward pass. These values are needed
    /// during both the variance calculation and the backward pass. This field is reset when ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the average values calculated during the first step.
    /// 
    /// The layer needs to remember these mean values:
    /// - They're used when calculating the variance (which measures deviation from the mean)
    /// - They're needed again during the backward pass
    /// - Storing them avoids having to recalculate them multiple times
    /// 
    /// Think of it as saving an intermediate result that will be reused later.
    /// </para>
    /// </remarks>
    private Tensor<T>? _meanValues;

    /// <summary>
    /// Initializes a new instance of the <see cref="LogVarianceLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="axis">The axis along which to calculate variance.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a LogVarianceLayer that will calculate the variance along the specified axis
    /// of the input tensor. The output shape is determined by removing the specified axis from the input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new log-variance layer with your desired settings.
    /// 
    /// When setting up this layer:
    /// - inputShape defines the expected size and dimensions of your data
    /// - axis specifies which dimension to calculate variance along
    /// 
    /// The layer will reduce the data along the specified axis, meaning the output
    /// will have one fewer dimension than the input.
    /// </para>
    /// </remarks>
    public LogVarianceLayer(int[] inputShape, int axis)
        : base(inputShape, CalculateOutputShape(inputShape, axis))
    {
        Axis = axis;
    }

    /// <summary>
    /// Calculates the output shape of the log-variance layer based on the input shape and the axis along which variance is calculated.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="axis">The axis along which to calculate variance.</param>
    /// <returns>The calculated output shape for the log-variance layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape by removing the dimension specified by the axis parameter
    /// from the input shape. This is because the variance calculation reduces the data along that axis.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of the data that will come out of this layer.
    /// 
    /// When calculating variance along an axis:
    /// - That dimension gets "collapsed" into a single value
    /// - The output shape has one fewer dimension than the input
    /// 
    /// For example, if your input has shape [10, 20, 30] (a 3D array) and you calculate 
    /// variance along axis 1, the output shape would be [10, 30].
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
    /// Performs the forward pass of the log-variance layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor containing the log-variance values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the log-variance calculation. It first computes the mean along
    /// the specified axis, then calculates the variance by summing squared differences from the mean, and finally
    /// takes the natural logarithm of the variance (with a small epsilon added for numerical stability).
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the layer, calculating the log-variance.
    /// 
    /// The calculation happens in these steps:
    /// 1. Calculate the average (mean) of values along the specified axis
    /// 2. For each value, find how far it is from the average
    /// 3. Square these differences and add them up
    /// 4. Divide by the number of values to get the variance
    /// 5. Take the natural logarithm of the variance
    /// 
    /// A small value (epsilon) is added to prevent errors when taking the logarithm of zero or very small numbers.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Use Engine operations for GPU/CPU acceleration
        _meanValues = Engine.ReduceMean(input, [Axis], keepDims: true);
        _lastOutput = Engine.ReduceLogVariance(input, [Axis], keepDims: false, epsilon: 1e-8);

        return _lastOutput;
    }

    /// <summary>
    /// Performs GPU-accelerated forward pass for log-variance reduction.
    /// </summary>
    /// <param name="inputs">Input GPU tensors (uses first input).</param>
    /// <returns>GPU-resident output tensor with log-variance values.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        int[] shape = input.Shape;
        int inputRank = shape.Length;

        // Calculate output shape by removing the axis dimension
        int[] outputShape = CalculateOutputShape(shape, Axis);
        int axisSize = shape[Axis];
        float scale = 1.0f / axisSize;
        const float epsilon = 1e-8f;

        // GPU-resident variance calculation using computational formula:
        // variance = E[X^2] - E[X]^2 = mean(x*x) - mean(x)^2
        // This avoids the need for broadcast subtraction

        // If axis is not last, permute to move it to the last position
        IGpuTensor<T> processedInput = input;
        bool needsPermute = Axis != inputRank - 1;

        if (needsPermute)
        {
            var perm = new int[inputRank];
            int j = 0;
            for (int i = 0; i < inputRank; i++)
                if (i != Axis) perm[j++] = i;
            perm[inputRank - 1] = Axis;
            processedInput = gpuEngine.PermuteGpu(input, perm);
        }

        int outerSize = processedInput.ElementCount / axisSize;

        // Step 1: Compute mean = sum(x) / n
        var sumBuffer = backend.AllocateBuffer(outerSize);
        backend.SumAxis(processedInput.Buffer, sumBuffer, outerSize, axisSize);

        var meanBuffer = backend.AllocateBuffer(outerSize);
        backend.Scale(sumBuffer, meanBuffer, scale, outerSize);
        sumBuffer.Dispose();

        // Step 2: Compute x^2 element-wise
        int totalSize = processedInput.ElementCount;
        var xSquaredBuffer = backend.AllocateBuffer(totalSize);
        backend.Multiply(processedInput.Buffer, processedInput.Buffer, xSquaredBuffer, totalSize);

        // Step 3: Compute mean(x^2) = sum(x^2) / n
        var sumXSquaredBuffer = backend.AllocateBuffer(outerSize);
        backend.SumAxis(xSquaredBuffer, sumXSquaredBuffer, outerSize, axisSize);
        xSquaredBuffer.Dispose();

        var meanXSquaredBuffer = backend.AllocateBuffer(outerSize);
        backend.Scale(sumXSquaredBuffer, meanXSquaredBuffer, scale, outerSize);
        sumXSquaredBuffer.Dispose();

        // Step 4: Compute mean^2
        var meanSquaredBuffer = backend.AllocateBuffer(outerSize);
        backend.Multiply(meanBuffer, meanBuffer, meanSquaredBuffer, outerSize);

        // Step 5: Compute variance = mean(x^2) - mean^2
        var varianceBuffer = backend.AllocateBuffer(outerSize);
        backend.Subtract(meanXSquaredBuffer, meanSquaredBuffer, varianceBuffer, outerSize);
        meanXSquaredBuffer.Dispose();
        meanSquaredBuffer.Dispose();

        // Step 6: Add epsilon to variance for numerical stability
        // Create buffer with epsilon values
        var epsilonData = new float[outerSize];
        Array.Fill(epsilonData, epsilon);
        using var epsilonBuffer = backend.AllocateBuffer(epsilonData);

        var variancePlusEpsilonBuffer = backend.AllocateBuffer(outerSize);
        backend.Add(varianceBuffer, epsilonBuffer, variancePlusEpsilonBuffer, outerSize);
        varianceBuffer.Dispose();

        // Step 7: Compute log(variance + epsilon)
        var outputBuffer = backend.AllocateBuffer(outerSize);
        backend.Log(variancePlusEpsilonBuffer, outputBuffer, outerSize);
        variancePlusEpsilonBuffer.Dispose();

        // Dispose permuted tensor if we created one
        if (needsPermute)
            processedInput.Dispose();

        // Cache for backward pass (only download if training)
        if (IsTrainingMode)
        {
            var inputData = backend.DownloadBuffer(input.Buffer);
            _lastInput = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(inputData), shape);

            int[] meanShape = (int[])shape.Clone();
            meanShape[Axis] = 1;
            var meanData = backend.DownloadBuffer(meanBuffer);
            _meanValues = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanData), meanShape);

            var outputData = backend.DownloadBuffer(outputBuffer);
            _lastOutput = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputData), outputShape);
        }

        meanBuffer.Dispose();

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Performs the backward pass of the log-variance layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the log-variance layer, which is used during training to propagate
    /// error gradients backward through the network. It calculates how changes in the output affect the input, 
    /// taking into account the derivatives of the logarithm and variance calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how changes in the output
    /// would affect the input.
    /// 
    /// During the backward pass:
    /// - The layer receives information about how its output affected the overall error
    /// - It calculates how each input value contributed to that error
    /// - This information is passed backward to earlier layers
    /// 
    /// The mathematics here are complex but involve the chain rule from calculus:
    /// - For the log function: the derivative is 1/x
    /// - For variance: it involves how each value's difference from the mean contributed
    /// 
    /// This process is part of how neural networks learn from their mistakes.
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
        if (_lastInput == null || _lastOutput == null || _meanValues == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Compute variance from log variance: variance = exp(log_variance)
        var varianceData = _lastOutput.ToArray();
        for (int i = 0; i < varianceData.Length; i++)
        {
            varianceData[i] = NumOps.Exp(varianceData[i]);
        }
        var variance = new Tensor<T>(_lastOutput.Shape, new Vector<T>(varianceData));

        // Use Engine operation for backward pass
        return Engine.ReduceLogVarianceBackward(outputGradient, _lastInput, _meanValues, variance, [Axis]);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation via the ReduceLogVariance operation to compute gradients.
    /// The operation handles the full forward and backward pass for log-variance computation.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Create computation node for input
        var inputNode = Autodiff.TensorOperations<T>.Variable(
            _lastInput,
            "input",
            requiresGradient: true);

        // Apply ReduceLogVariance operation
        var outputNode = Autodiff.TensorOperations<T>.ReduceLogVariance(
            inputNode,
            axis: Axis,
            epsilon: 1e-8);

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
                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                        stack.Push((parent, false));
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

        // Return input gradient
        return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
    }

    /// <summary>
    /// Updates the parameters of the layer based on the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is empty because the LogVarianceLayer has no trainable parameters to update.
    /// However, it must be implemented to satisfy the base class contract.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally update the layer's internal values during training.
    /// 
    /// However, since this layer doesn't have any trainable parameters:
    /// - There's nothing to update
    /// - The method exists but doesn't do anything
    /// - This is normal for layers that perform fixed mathematical operations
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // LogVarianceLayer has no learnable parameters, so this method is empty
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>An empty vector since this layer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector because the LogVarianceLayer has no trainable parameters.
    /// However, it must be implemented to satisfy the base class contract.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally return all the values that can be learned during training.
    /// 
    /// Since this layer has no learnable values:
    /// - It returns an empty list (vector with length 0)
    /// - This is expected for mathematical operation layers
    /// - Other layers, like those with weights, would return those weights
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // LogVarianceLayer has no trainable parameters
        return new Vector<T>(0);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears any cached data from previous forward passes, essentially resetting the layer
    /// to its initial state. This is useful when starting to process a new batch of data or when
    /// implementing recurrent neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and calculated values are cleared
    /// - The layer forgets any information from previous data
    /// - This is important when processing a new, unrelated batch of data
    /// 
    /// Think of it like wiping a calculator's memory before starting a new calculation.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _lastOutput = null;
        _meanValues = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return TensorOperations<T>.ReduceLogVariance(inputNode, axis: Axis);
    }

    public override bool SupportsJitCompilation => true;
}
