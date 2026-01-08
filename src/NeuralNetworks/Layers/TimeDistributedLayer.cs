using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a wrapper layer that applies an inner layer to each time step of a sequence independently.
/// </summary>
/// <remarks>
/// <para>
/// A time distributed layer applies the same inner layer (and its operations) to each time step of a sequence 
/// independently. This is particularly useful for processing sequential data where the same transformation needs 
/// to be applied to each element in the sequence. The layer maintains the temporal structure of the data while 
/// allowing each time step to be processed by the inner layer.
/// </para>
/// <para><b>For Beginners:</b> This layer helps process sequences of data by applying the same operation to each step.
/// 
/// Think of it like an assembly line worker who performs the same task on each item that passes by:
/// - You have a sequence of items (like frames in a video or words in a sentence)
/// - You want to apply the same operation to each item independently
/// - This layer automates that process while preserving the original sequence order
/// 
/// For example, if you have a video with 30 frames per second, and you want to detect objects in each frame:
/// - A normal layer would need to process all frames together
/// - This time distributed layer would apply your object detection layer to each frame separately
/// - The result would be object detections for each frame, still organized as a sequence
/// 
/// This makes it much easier to work with sequential data like videos, sentences, or time series.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class TimeDistributedLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The inner layer that is applied to each time step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the inner layer that will be applied to each time step in the sequence. The same layer
    /// instance is reused for processing each time step, which means its weights are shared across all time steps.
    /// </para>
    /// <para><b>For Beginners:</b> This is the operation that gets applied to each item in the sequence.
    /// 
    /// The inner layer:
    /// - Could be any type of neural network layer (convolutional, dense, etc.)
    /// - Processes each time step with the exact same weights and parameters
    /// - Learns patterns that are consistent across different time steps
    /// 
    /// For example, if processing text, the inner layer might be a dense layer that converts
    /// each word into a semantic representation, with the time distributed wrapper ensuring
    /// that each word is processed independently but in sequence.
    /// </para>
    /// </remarks>
    private readonly LayerBase<T> _innerLayer;

    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the input tensor from the most recent forward pass, which is needed during the backward
    /// pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This is the layer's memory of what sequence it last processed.
    /// 
    /// Storing the input is necessary because:
    /// - During training, the layer needs to remember what sequence it processed
    /// - This information helps calculate how to improve the inner layer's performance
    /// - It enables the backward pass to compute gradients correctly
    /// 
    /// Think of it as keeping a copy of the work so the layer can analyze what it did right
    /// or wrong during the learning process.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// The output tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor from the most recent forward pass, which is needed during the backward
    /// pass to compute gradients with respect to the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This is the layer's memory of what result it produced last time.
    /// 
    /// Storing the output is necessary because:
    /// - During training, the layer needs to know what results it produced
    /// - Some activation functions need their original output to calculate how to improve
    /// - It helps compute the correct gradients during the backward pass
    /// 
    /// This cached output helps the layer understand how its processing affected the final result,
    /// which is crucial for learning.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> if the inner layer supports training; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the time distributed layer can be trained. It simply forwards the value of
    /// the inner layer's SupportsTraining property, as the time distributed layer's trainability depends entirely
    /// on whether its inner layer can be trained.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// Rather than having its own answer, this layer checks if the inner layer can learn:
    /// - If the inner layer supports training, this layer also supports training
    /// - If the inner layer doesn't support training, this layer also doesn't support training
    /// 
    /// This makes sense because:
    /// - The time distributed layer doesn't have its own trainable parameters
    /// - It just organizes how the inner layer is applied to sequences
    /// - The actual learning happens in the inner layer
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _innerLayer.SupportsTraining;

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0) throw new ArgumentException("TimeDistributedLayer requires an input tensor.");
        var input = inputs[0];

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        int batch = input.Shape[0];
        int time = input.Shape[1];
        int[] inputShape = input.Shape;

        int[] flattenedShape = new int[inputShape.Length - 1];
        flattenedShape[0] = batch * time;
        Array.Copy(inputShape, 2, flattenedShape, 1, inputShape.Length - 2);

        // Process all timesteps in parallel by flattening the time dimension into the batch dimension
        var reshapedInput = gpuEngine.ReshapeGpu(input, flattenedShape);
        var innerOutput = _innerLayer.ForwardGpu(reshapedInput);

        int[] innerOutputShape = innerOutput.Shape;
        int[] outputShape = new int[innerOutputShape.Length + 1];
        outputShape[0] = batch;
        outputShape[1] = time;
        Array.Copy(innerOutputShape, 1, outputShape, 2, innerOutputShape.Length - 1);

        var output = gpuEngine.ReshapeGpu(innerOutput, outputShape);

        var fusedOp = MapActivationToFused();
        if (fusedOp != FusedActivationType.None)
        {
            output = gpuEngine.ActivationGpu(output, fusedOp);
        }

        if (IsTrainingMode)
        {
            _lastInput = input.ToTensor();
            _lastOutput = output.ToTensor();
            _originalInputShape = input.Shape;
        }

        return output;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TimeDistributedLayer{T}"/> class with scalar activation function.
    /// </summary>
    /// <param name="innerLayer">The layer to apply to each time step.</param>
    /// <param name="activationFunction">The activation function to apply after processing. Defaults to ReLU if not specified.</param>
    /// <param name="inputShape">Optional explicit input shape. If not provided, derived from the inner layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a time distributed layer that applies the specified inner layer to each time step of a
    /// sequence. It also applies the specified scalar activation function to the output. The input shape can be
    /// explicitly provided or derived from the inner layer's input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new time distributed layer.
    /// 
    /// The parameters you provide determine:
    /// - innerLayer: What operation to apply to each time step in the sequence
    /// - activationFunction: What mathematical function to apply to the results (ReLU is default)
    /// - inputShape: The expected shape of incoming data (optional, can be figured out automatically)
    /// 
    /// For example, if processing a sequence of images, you might wrap a convolutional layer
    /// with this time distributed layer to apply the same convolutional operations to each
    /// image frame independently.
    /// </para>
    /// </remarks>
    public TimeDistributedLayer(LayerBase<T> innerLayer, IActivationFunction<T>? activationFunction = null, int[]? inputShape = null)
        : base(CalculateInputShape(innerLayer, inputShape), CalculateOutputShape(innerLayer, inputShape), activationFunction ?? new ReLUActivation<T>())
    {
        _innerLayer = innerLayer;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TimeDistributedLayer{T}"/> class with vector activation function.
    /// </summary>
    /// <param name="innerLayer">The layer to apply to each time step.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after processing. Defaults to ReLU if not specified.</param>
    /// <param name="inputShape">Optional explicit input shape. If not provided, derived from the inner layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a time distributed layer that applies the specified inner layer to each time step of a
    /// sequence. It also applies the specified vector activation function to the output. The input shape can be
    /// explicitly provided or derived from the inner layer's input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is similar to the previous one, but uses vector activations.
    /// 
    /// Vector activations:
    /// - Process entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements
    /// - Allow for more complex transformations
    /// 
    /// This version is useful when you need more sophisticated processing that considers
    /// how different features relate to each other, rather than treating each feature independently.
    /// </para>
    /// </remarks>
    public TimeDistributedLayer(LayerBase<T> innerLayer, IVectorActivationFunction<T>? vectorActivationFunction = null, int[]? inputShape = null)
        : base(CalculateInputShape(innerLayer, inputShape), CalculateOutputShape(innerLayer, inputShape), vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _innerLayer = innerLayer;
    }

    /// <summary>
    /// Calculates the input shape of the time distributed layer.
    /// </summary>
    /// <param name="innerLayer">The inner layer to be applied to each time step.</param>
    /// <param name="inputShape">Optional explicit input shape. If not provided, derived from the inner layer.</param>
    /// <returns>The calculated input shape as an array of integers.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the input shape of the time distributed layer based on either the explicitly provided
    /// input shape or the inner layer's input shape. It adds a time dimension at the beginning of the shape.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of data that should go into this layer.
    /// 
    /// It works by:
    /// - Taking the shape of data that the inner layer expects
    /// - Adding an extra dimension at the beginning for the sequence/time steps
    /// - Using -1 for the time dimension to indicate it can be any length
    /// 
    /// For example, if the inner layer expects images of shape [224, 224, 3], 
    /// this would create an input shape of [-1, 224, 224, 3], where -1 means 
    /// "any number of time steps" (like any number of video frames).
    /// </para>
    /// </remarks>
    private static int[] CalculateInputShape(LayerBase<T> innerLayer, int[]? inputShape)
    {
        int[] result;
        if (inputShape != null && inputShape.Length >= 2)
        {
            result = new int[inputShape.Length];
            result[0] = -1;
            Array.Copy(inputShape, 1, result, 1, inputShape.Length - 1);

            return result;
        }

        int[] innerShape = innerLayer.GetInputShape();
        result = new int[innerShape.Length + 1];
        result[0] = -1;
        Array.Copy(innerShape, 0, result, 1, innerShape.Length);

        return result;
    }

    /// <summary>
    /// Calculates the output shape of the time distributed layer.
    /// </summary>
    /// <param name="innerLayer">The inner layer to be applied to each time step.</param>
    /// <param name="inputShape">Optional explicit input shape. If not provided, derived from the inner layer.</param>
    /// <returns>The calculated output shape as an array of integers.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape of the time distributed layer based on the inner layer's output shape
    /// and optionally the provided input shape. It adds a time dimension at the beginning of the shape.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of data that will come out of this layer.
    /// 
    /// It works by:
    /// - Taking the shape of data that the inner layer produces
    /// - Adding an extra dimension at the beginning for the sequence/time steps
    /// - Using -1 for the time dimension to indicate it can be any length
    /// 
    /// For example, if the inner layer outputs feature vectors of shape [128],
    /// this would create an output shape of [-1, 128], meaning "a sequence of 
    /// 128-dimensional feature vectors of any length."
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(LayerBase<T> innerLayer, int[]? inputShape)
    {
        int[] result;
        if (inputShape != null && inputShape.Length >= 2)
        {
            int[] innerOutputShape = innerLayer.GetOutputShape();
            result = new int[innerOutputShape.Length + 1];
            result[0] = -1;
            result[1] = inputShape[1];
            Array.Copy(innerOutputShape, 1, result, 2, innerOutputShape.Length - 1);

            return result;
        }

        int[] innerShape = innerLayer.GetOutputShape();
        result = new int[innerShape.Length + 1];
        result[0] = -1;
        Array.Copy(innerShape, 0, result, 1, innerShape.Length);

        return result;
    }

    /// <summary>
    /// Performs the forward pass of the time distributed layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing each time step.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the time distributed layer. It iterates over each time step in the
    /// input sequence, applies the inner layer to that time step, and collects the results into an output sequence.
    /// Finally, it applies the activation function to the entire output.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input sequence through the layer.
    /// 
    /// During the forward pass:
    /// 1. The layer receives a sequence of inputs
    /// 2. For each step in the sequence:
    ///    - It extracts just that step's data
    ///    - It passes that data through the inner layer
    ///    - It collects the result
    /// 3. All the individual results are combined back into a sequence
    /// 4. The activation function is applied to the entire output
    /// 
    /// For example, with a video input, this would:
    /// - Process each frame individually through the inner layer
    /// - Maintain the original frame order in the output
    /// - Apply the activation function to all processed frames
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        if (rank < 2)
            throw new ArgumentException($"TimeDistributedLayer requires at least 2D input, got {rank}D");

        int batchSize = input.Shape[0];
        int timeSteps = rank == 2 ? 1 : input.Shape[1];

        var processInput = rank == 2
            ? input.Reshape([batchSize, 1, input.Shape[1]])
            : input;

        _lastInput = processInput;

        var innerOutputShape = _innerLayer.GetOutputShape();
        var outputShape = new[] { batchSize, timeSteps }.Concat(innerOutputShape).ToArray();
        var output = new Tensor<T>(outputShape);

        for (int t = 0; t < timeSteps; t++)
        {
            var stepInput = processInput.Slice(1, t, t + 1);
            stepInput = SqueezeAxis(stepInput, 1);
            var stepOutput = _innerLayer.Forward(stepInput);
            output.SetSlice(1, t, stepOutput);
        }

        var activated = ApplyActivation(output);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            activated = activated.Reshape(new[] { batchSize }.Concat(innerOutputShape).ToArray());
        }

        _lastOutput = activated;
        return _lastOutput;
    }

    /// <summary>
    /// Performs the backward pass of the time distributed layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when trying to perform a backward pass before a forward pass.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the time distributed layer, which is used during training to
    /// propagate error gradients back through the network. It first computes the gradient with respect to the activation
    /// function, then iterates over each time step, applies the inner layer's backward pass to that time step's gradient,
    /// and collects the results into an input gradient sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// should change to reduce errors.
    ///
    /// During the backward pass:
    /// 1. The layer receives information about how its output should change (outputGradient)
    /// 2. It first adjusts this gradient based on the activation function
    /// 3. For each step in the sequence:
    ///    - It extracts just that step's gradient
    ///    - It passes that gradient backward through the inner layer
    ///    - It collects the resulting input gradient
    /// 4. All the individual input gradients are combined back into a sequence
    ///
    /// This process tells the layer how its inputs should change to reduce errors,
    /// while maintaining the same time-step-by-time-step processing as the forward pass.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Performs the backward pass using manual gradient computation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when trying to perform a backward pass before a forward pass.</exception>
    /// <remarks>
    /// <para>
    /// This is the original optimized manual implementation that directly computes gradients by iterating
    /// through each time step and delegating to the inner layer's backward pass.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _lastInput.Shape[0];
        int timeSteps = _lastInput.Shape[1];

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        Tensor<T> outputGrad = outputGradient;
        Tensor<T> lastOutput = _lastOutput;
        if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            var innerOutputShape = _innerLayer.GetOutputShape();
            var expandedShape = new[] { batchSize, 1 }.Concat(innerOutputShape).ToArray();
            outputGrad = outputGradient.Reshape(expandedShape);
            lastOutput = _lastOutput.Reshape(expandedShape);
        }

        if (ScalarActivation != null)
        {
            outputGrad = outputGrad.ElementwiseMultiply(lastOutput.Transform((x, _) => ScalarActivation.Derivative(x)));
        }
        else if (VectorActivation != null)
        {
            outputGrad = outputGrad.ElementwiseMultiply(VectorActivation.Derivative(lastOutput));
        }

        for (int t = 0; t < timeSteps; t++)
        {
            var stepOutputGradient = outputGrad.Slice(1, t, t + 1);
            stepOutputGradient = SqueezeAxis(stepOutputGradient, 1);
            var stepInputGradient = _innerLayer.Backward(stepOutputGradient);
            inputGradient.SetSlice(1, t, stepInputGradient);
        }

        if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }


    /// <summary>
    /// Performs the backward pass using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when trying to perform a backward pass before a forward pass.</exception>
    /// <remarks>
    /// <para>
    /// This implementation uses the TensorOperations autodiff framework to compute gradients automatically.
    /// Since TimeDistributedLayer is a wrapper that applies the inner layer independently to each timestep,
    /// the autodiff implementation simply delegates gradient computation to the inner layer for each timestep,
    /// similar to the manual implementation. The inner layer is responsible for building its own computation
    /// graph when autodiff is enabled.
    /// </para>
    /// <para>
    /// Note: TimeDistributedLayer does not build a unified computation graph across all timesteps because:
    /// 1. Each timestep is processed independently with no temporal dependencies
    /// 2. The inner layer builds its own computation graph when it uses autodiff
    /// 3. Slicing operations are not differentiable in the current TensorOperations framework
    /// Therefore, this autodiff path effectively delegates to the inner layer's autodiff implementation
    /// per timestep, while handling the activation function derivatives at the wrapper level.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _lastInput.Shape[0];
        int timeSteps = _lastInput.Shape[1];

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        Tensor<T> outputGrad = outputGradient;
        Tensor<T> lastOutput = _lastOutput;
        if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            var innerOutputShape = _innerLayer.GetOutputShape();
            var expandedShape = new[] { batchSize, 1 }.Concat(innerOutputShape).ToArray();
            outputGrad = outputGradient.Reshape(expandedShape);
            lastOutput = _lastOutput.Reshape(expandedShape);
        }

        if (ScalarActivation != null)
        {
            outputGrad = outputGrad.ElementwiseMultiply(lastOutput.Transform((x, _) => ScalarActivation.Derivative(x)));
        }
        else if (VectorActivation != null)
        {
            outputGrad = outputGrad.ElementwiseMultiply(VectorActivation.Derivative(lastOutput));
        }

        for (int t = 0; t < timeSteps; t++)
        {
            var stepOutputGradient = outputGrad.Slice(1, t, t + 1);
            stepOutputGradient = SqueezeAxis(stepOutputGradient, 1);
            var stepInputGradient = _innerLayer.Backward(stepOutputGradient);
            inputGradient.SetSlice(1, t, stepInputGradient);
        }

        if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    private static Tensor<T> SqueezeAxis(Tensor<T> tensor, int axis)
    {
        if (axis < 0 || axis >= tensor.Shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis));

        if (tensor.Shape[axis] != 1)
            return tensor;

        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int d = 0, nd = 0; d < tensor.Shape.Length; d++)
        {
            if (d == axis)
                continue;
            newShape[nd++] = tensor.Shape[d];
        }

        return newShape.Length == 0 ? tensor.Reshape([1]) : tensor.Reshape(newShape);
    }



    /// <summary>
    /// Updates the parameters of the inner layer.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of the inner layer based on the gradients calculated during the backward pass.
    /// The time distributed layer itself doesn't have trainable parameters; it simply delegates the update to the inner layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the inner layer's parameters during training.
    /// 
    /// The time distributed layer:
    /// - Doesn't have its own parameters to update
    /// - Simply passes the learning rate to the inner layer
    /// - Lets the inner layer adjust its own parameters
    /// 
    /// This works because the time distributed layer is just a wrapper that changes how
    /// the inner layer is applied to sequences, but doesn't change the inner layer's
    /// learning process.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        _innerLayer.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters of the inner layer.
    /// </summary>
    /// <returns>A vector containing all trainable parameters from the inner layer.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the inner layer. The time distributed layer itself doesn't
    /// have trainable parameters; it simply delegates to the inner layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the inner layer.
    /// 
    /// Since the time distributed layer:
    /// - Doesn't have its own parameters to learn
    /// - Simply applies the inner layer multiple times
    /// 
    /// It returns the inner layer's parameters, which are:
    /// - The numbers that the neural network learns during training
    /// - The same across all time steps (parameter sharing)
    /// 
    /// This parameter sharing is a key feature - it means the layer learns patterns
    /// that can be applied to any time step, rather than learning different patterns
    /// for different positions in the sequence.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Return the parameters of the inner layer
        return _innerLayer.GetParameters();
    }

    /// <summary>
    /// Resets the internal state of the layer and its inner layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the time distributed layer and its inner layer. It clears the cached
    /// input and output tensors and delegates to the inner layer to reset its state as well.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The layer forgets what inputs and outputs it recently processed
    /// - It also tells its inner layer to reset its own state
    /// - This prepares the layer to process new, unrelated sequences
    /// 
    /// This is important when:
    /// - Starting to process a new, unrelated sequence
    /// - Testing the layer with fresh inputs
    /// - Beginning a new training episode
    /// 
    /// Think of it like clearing your mind before starting a completely new task.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Reset the inner layer's state
        _innerLayer.ResetState();

        // Clear cached values
        _lastInput = null;
        _lastOutput = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        // Check if inner layer supports JIT
        if (!_innerLayer.SupportsJitCompilation)
            throw new NotSupportedException("TimeDistributed inner layer does not support JIT compilation.");

        // TimeDistributedLayer JIT delegates to the inner layer:
        // For a fixed sequence length, we apply the inner layer to the entire sequence
        // treating the time dimension as part of the batch dimension.

        var input = inputNodes[0];

        // Apply inner layer's computation graph
        // The inner layer will process the input with time steps treated as batch samples
        var output = _innerLayer.ExportComputationGraph(inputNodes);

        // Apply layer activation
        output = ApplyActivationToGraph(output);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> if the inner layer supports JIT compilation; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation for TimeDistributed delegates to the inner layer. The time
    /// distributed behavior is achieved by reshaping the input so that time steps
    /// are treated as batch samples, allowing the inner layer to process all
    /// time steps in parallel.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => _innerLayer.SupportsJitCompilation;

}
