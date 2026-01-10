using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a bidirectional layer that processes input sequences in both forward and backward directions.
/// </summary>
/// <remarks>
/// <para>
/// A bidirectional layer processes input sequences in two directions: forward (from first to last) and backward 
/// (from last to first). This approach allows the layer to capture patterns that depend on both past and future 
/// context. The outputs from both directions can either be merged (typically added together) or kept separate, 
/// depending on the configuration.
/// </para>
/// <para><b>For Beginners:</b> This layer looks at input data in two ways at the same time - both forward and backward.
/// 
/// Think of it like reading a sentence:
/// - Forward reading: "The cat sat on the mat" (left to right)
/// - Backward reading: "mat the on sat cat The" (right to left)
/// 
/// By processing data in both directions:
/// - The layer can understand context from both past and future elements
/// - It can discover patterns that might be missed if only looking in one direction
/// - It often improves performance on sequence tasks like text processing
/// 
/// For example, in the sentence "The bank is by the river", the meaning of "bank" 
/// depends on both previous words ("The") and future words ("by the river").
/// A bidirectional layer helps capture these relationships.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BidirectionalLayer<T> : LayerBase<T>
{
    private readonly LayerBase<T> _forwardLayer;
    private readonly LayerBase<T> _backwardLayer;
    private readonly bool _mergeMode;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastForwardOutput;
    private Tensor<T>? _lastBackwardOutput;

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> if either the forward or backward layer supports training; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the bidirectional layer can be trained through backpropagation. 
    /// The layer supports training if either of its internal layers (forward or backward) supports training.
    /// This is typically the case for layers that have trainable parameters, such as weights or biases.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its internal values during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// The bidirectional layer supports training if either of its two internal layers
    /// (forward or backward) supports training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _forwardLayer.SupportsTraining || _backwardLayer.SupportsTraining;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-accelerated forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The bidirectional layer supports GPU execution when both the forward and backward inner layers
    /// support GPU execution. This ensures that the entire bidirectional processing can be done on the GPU.
    /// </para>
    /// <para><b>For Beginners:</b> This property indicates whether this layer can use the GPU for faster processing.
    /// Since the bidirectional layer wraps two inner layers, it can only use the GPU if both of those layers
    /// support GPU execution.
    /// </para>
    /// </remarks>
    protected override bool SupportsGpuExecution =>
        _forwardLayer.CanExecuteOnGpu && _backwardLayer.CanExecuteOnGpu;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-resident training.
    /// </summary>
    /// <remarks>
    /// The bidirectional layer supports GPU training when both inner layers support GPU training.
    /// </remarks>
    public override bool SupportsGpuTraining =>
        _forwardLayer.SupportsGpuTraining && _backwardLayer.SupportsGpuTraining;

    #region GPU Training Fields
    private IGpuTensor<T>? _gpuLastInput;
    private IGpuTensor<T>? _gpuLastForwardOutput;
    private IGpuTensor<T>? _gpuLastBackwardOutput;
    #endregion

    /// <summary>
    /// Initializes a new instance of the <see cref="BidirectionalLayer{T}"/> class with the specified inner layer
    /// and a ReLU activation function.
    /// </summary>
    /// <param name="innerLayer">The layer to be used for both forward and backward processing.</param>
    /// <param name="mergeMode">If <c>true</c>, outputs from both directions are added; otherwise, they are kept separate.</param>
    /// <param name="activationFunction">The activation function to apply after processing. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a bidirectional layer using the specified inner layer for both forward and backward
    /// processing. A copy of the inner layer is created for backward processing to ensure independent parameters.
    /// The mergeMode parameter determines how outputs from both directions are combined.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new bidirectional layer with a standard activation function.
    /// 
    /// When you create a bidirectional layer this way:
    /// - The same type of layer is used for both forward and backward processing
    /// - The mergeMode parameter decides how to combine the results from both directions
    /// - The ReLU activation function is used by default, which helps the network learn non-linear patterns
    /// 
    /// For example, if innerLayer is an LSTM layer, this creates a bidirectional LSTM that
    /// processes sequences in both directions.
    /// </para>
    /// </remarks>
    public BidirectionalLayer(
        LayerBase<T> innerLayer,
        bool mergeMode = true,
        IActivationFunction<T>? activationFunction = null,
        IEngine? engine = null)
        : base(innerLayer.GetInputShape(), CalculateOutputShape(innerLayer.GetOutputShape(), mergeMode), activationFunction ?? new ReLUActivation<T>())
    {
        _forwardLayer = innerLayer;
        _backwardLayer = innerLayer.Clone();
        _mergeMode = mergeMode;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="BidirectionalLayer{T}"/> class with the specified inner layer
    /// and a vector activation function.
    /// </summary>
    /// <param name="innerLayer">The layer to be used for both forward and backward processing.</param>
    /// <param name="mergeMode">If <c>true</c>, outputs from both directions are added; otherwise, they are kept separate.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a bidirectional layer using the specified inner layer for both forward and backward
    /// processing. A copy of the inner layer is created for backward processing to ensure independent parameters.
    /// This overload accepts a vector activation function, which operates on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new bidirectional layer with a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the output
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor is useful when you need more complex activation patterns
    /// that consider the relationships between different outputs.
    /// </para>
    /// </remarks>
    public BidirectionalLayer(
        LayerBase<T> innerLayer,
        bool mergeMode = true,
        IVectorActivationFunction<T>? vectorActivationFunction = null,
        IEngine? engine = null)
        : base(innerLayer.GetInputShape(), CalculateOutputShape(innerLayer.GetOutputShape(), mergeMode), vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _forwardLayer = innerLayer;
        _backwardLayer = innerLayer.Clone();
        _mergeMode = mergeMode;
    }

    /// <summary>
    /// Calculates the output shape of the bidirectional layer based on the inner layer's output shape and merge mode.
    /// </summary>
    /// <param name="innerOutputShape">The output shape of the inner layer.</param>
    /// <param name="mergeMode">If <c>true</c>, outputs are merged; otherwise, they are kept separate.</param>
    /// <returns>The calculated output shape for the bidirectional layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape of the bidirectional layer. If mergeMode is true, the output shape
    /// is the same as the inner layer's output shape because the forward and backward outputs are added together.
    /// If mergeMode is false, an additional dimension of size 2 is added at the beginning to accommodate the separate
    /// forward and backward outputs.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of the data that will come out of this layer.
    /// 
    /// The output shape depends on whether the forward and backward results are combined:
    /// - If mergeMode is true: The shape is the same as the inner layer's output
    /// - If mergeMode is false: The shape adds an extra dimension to hold both directions separately
    /// 
    /// For example, if the inner layer outputs data with shape [32, 128] (32 time steps, 128 features):
    /// - With mergeMode=true: The output shape remains [32, 128]
    /// - With mergeMode=false: The output shape becomes [2, 32, 128], where the first dimension
    ///   represents the two directions (forward and backward)
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] innerOutputShape, bool mergeMode)
    {
        if (mergeMode)
        {
            return innerOutputShape;
        }
        else
        {
            var newShape = new int[innerOutputShape.Length + 1];
            newShape[0] = 2;
            Array.Copy(innerOutputShape, 0, newShape, 1, innerOutputShape.Length);

            return newShape;
        }
    }

    /// <summary>
    /// Performs the forward pass of the bidirectional layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after bidirectional processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the bidirectional layer. It processes the input in both forward
    /// and backward directions using the respective inner layers, and then combines the outputs according to the
    /// merge mode. The input and outputs are cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input data through both forward and backward layers.
    /// 
    /// During the forward pass:
    /// - The original input is sent through the forward layer
    /// - A reversed version of the input is sent through the backward layer
    /// - The results from both directions are either combined or kept separate
    /// 
    /// This method also saves the inputs and outputs for later use during training.
    /// 
    /// For example, with a text sequence, the forward layer sees "Hello world" while
    /// the backward layer sees "world Hello", and then the results are combined.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Forward pass
        var forwardInput = input;
        _lastForwardOutput = _forwardLayer.Forward(forwardInput);

        // Backward pass (reverse the input sequence)
        var backwardInput = ReverseSequence(input);
        _lastBackwardOutput = _backwardLayer.Forward(backwardInput);

        // Merge outputs
        return MergeOutputs(_lastForwardOutput, _lastBackwardOutput);
    }

    /// <summary>
    /// Performs a GPU-resident forward pass of the bidirectional layer.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensor(s).</param>
    /// <returns>GPU-resident output tensor after bidirectional processing.</returns>
    /// <exception cref="ArgumentException">Thrown when no input tensor is provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when GPU backend is unavailable or inner layers don't support GPU.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the GPU-optimized version of the Forward method.
    /// All data stays on the GPU throughout the computation, avoiding expensive CPU-GPU transfers.
    /// The input sequence is processed in both forward and backward directions using GPU operations,
    /// and the results are merged on the GPU.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        if (!_forwardLayer.CanExecuteOnGpu || !_backwardLayer.CanExecuteOnGpu)
        {
            throw new InvalidOperationException(
                "BidirectionalLayer requires both inner layers to support GPU execution.");
        }

        var input = inputs[0];
        var shape = input.Shape;

        // Expected input shape: [batch, timeSteps, features] or [timeSteps, features]
        int batchSize, timeSteps, features;
        if (shape.Length == 2)
        {
            batchSize = 1;
            timeSteps = shape[0];
            features = shape[1];
        }
        else if (shape.Length == 3)
        {
            batchSize = shape[0];
            timeSteps = shape[1];
            features = shape[2];
        }
        else
        {
            throw new ArgumentException(
                $"BidirectionalLayer expects 2D or 3D input, got {shape.Length}D tensor.");
        }

        // Forward pass through forward layer
        var forwardOutput = _forwardLayer.ForwardGpu(input);

        // Reverse the input sequence for backward layer
        var reversedInput = ReverseSequenceGpu(backend, input, batchSize, timeSteps, features);

        // Forward pass through backward layer (with reversed input)
        var backwardOutput = _backwardLayer.ForwardGpu(reversedInput);

        // Reverse the backward output to match the original sequence order
        var reversedBackwardOutput = ReverseSequenceGpu(backend, backwardOutput, batchSize, timeSteps,
            backwardOutput.Shape[backwardOutput.Shape.Length - 1]);

        // Dispose intermediate tensors
        reversedInput.Dispose();
        backwardOutput.Dispose();

        // Merge outputs based on merge mode
        var output = MergeOutputsGpu(backend, forwardOutput, reversedBackwardOutput, batchSize, timeSteps);

        // Cache for GPU-resident training
        if (IsTrainingMode)
        {
            _gpuLastInput = input;
            _gpuLastForwardOutput = forwardOutput;
            _gpuLastBackwardOutput = reversedBackwardOutput;
        }

        return output;
    }

    /// <summary>
    /// Reverses a sequence along the time dimension on the GPU.
    /// </summary>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="input">Input tensor to reverse.</param>
    /// <param name="batchSize">Batch size.</param>
    /// <param name="timeSteps">Number of time steps.</param>
    /// <param name="features">Feature dimension size.</param>
    /// <returns>GPU tensor with the sequence reversed along the time dimension.</returns>
    private static IGpuTensor<T> ReverseSequenceGpu(
        IDirectGpuBackend backend,
        IGpuTensor<T> input,
        int batchSize,
        int timeSteps,
        int features)
    {
        int sliceSize = batchSize * features;
        int totalSize = batchSize * timeSteps * features;

        var outputBuffer = backend.AllocateBuffer(totalSize);

        // Reverse by copying each time step to its mirror position
        // For input[b, t, f], output[b, t, f] = input[b, timeSteps-1-t, f]
        for (int t = 0; t < timeSteps; t++)
        {
            int srcOffset = (timeSteps - 1 - t) * sliceSize;
            int dstOffset = t * sliceSize;

            // Use Copy2DStrided to copy the slice
            var srcView = input.CreateView(srcOffset, [sliceSize]);
            backend.Copy2DStrided(srcView.Buffer, outputBuffer, 1, sliceSize, totalSize, dstOffset);
        }

        return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-accelerated backward pass for the bidirectional layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the GPU-optimized backward pass that propagates gradients
    /// through both forward and backward inner layers while keeping all data on the GPU.
    /// </para>
    /// </remarks>
    public IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var shape = outputGradient.Shape;
        int batchSize, timeSteps, features;

        // Determine input dimensions
        if (_mergeMode)
        {
            // In merge mode, output shape matches inner layer output
            if (shape.Length == 2)
            {
                batchSize = 1;
                timeSteps = shape[0];
                features = shape[1];
            }
            else if (shape.Length == 3)
            {
                batchSize = shape[0];
                timeSteps = shape[1];
                features = shape[2];
            }
            else
            {
                throw new ArgumentException($"BidirectionalLayer backward expects 2D or 3D gradient, got {shape.Length}D.");
            }
        }
        else
        {
            // Non-merge mode has an extra dimension [2, batch, timeSteps, features]
            if (shape.Length == 3)
            {
                batchSize = 1;
                timeSteps = shape[1];
                features = shape[2];
            }
            else if (shape.Length == 4)
            {
                batchSize = shape[1];
                timeSteps = shape[2];
                features = shape[3];
            }
            else
            {
                throw new ArgumentException($"BidirectionalLayer non-merge backward expects 3D or 4D gradient, got {shape.Length}D.");
            }
        }

        IGpuTensor<T> forwardGradient, backwardGradient;

        if (_mergeMode)
        {
            // In merge mode, both directions receive the same gradient
            forwardGradient = outputGradient;
            backwardGradient = outputGradient;
        }
        else
        {
            // In non-merge mode, split the gradient for each direction
            int sliceSize = outputGradient.ElementCount / 2;
            int[] sliceShape = shape.Length == 3 ? [timeSteps, features] : [batchSize, timeSteps, features];

            var forwardBuffer = backend.AllocateBuffer(sliceSize);
            var backwardBuffer = backend.AllocateBuffer(sliceSize);

            backend.Copy(outputGradient.Buffer, 0, forwardBuffer, 0, sliceSize);
            backend.Copy(outputGradient.Buffer, sliceSize, backwardBuffer, 0, sliceSize);

            forwardGradient = new GpuTensor<T>(backend, forwardBuffer, sliceShape, GpuTensorRole.Gradient, ownsBuffer: true);
            backwardGradient = new GpuTensor<T>(backend, backwardBuffer, sliceShape, GpuTensorRole.Gradient, ownsBuffer: true);
        }

        // Propagate gradient through forward layer
        IGpuTensor<T> forwardInputGradient;
        var forwardLayerType = _forwardLayer.GetType();
        var forwardBackwardGpuMethod = forwardLayerType.GetMethod("BackwardGpu", new[] { typeof(IGpuTensor<T>) });

        if (forwardBackwardGpuMethod is not null)
        {
            forwardInputGradient = (IGpuTensor<T>)forwardBackwardGpuMethod.Invoke(_forwardLayer, new object[] { forwardGradient })!;
        }
        else
        {
            // CPU fallback: download, compute, upload
            var gradData = backend.DownloadBuffer(forwardGradient.Buffer);
            var cpuGrad = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradData), forwardGradient.Shape);
            var cpuResult = _forwardLayer.Backward(cpuGrad);
            forwardInputGradient = gpuEngine.UploadToGpu(cpuResult, GpuTensorRole.Gradient);
        }

        // Reverse the backward gradient before propagating
        var reversedBackwardGradient = ReverseSequenceGpu(backend, backwardGradient, batchSize, timeSteps, features);

        // Propagate gradient through backward layer
        IGpuTensor<T> backwardInputGradient;
        var backwardLayerType = _backwardLayer.GetType();
        var backwardBackwardGpuMethod = backwardLayerType.GetMethod("BackwardGpu", new[] { typeof(IGpuTensor<T>) });

        if (backwardBackwardGpuMethod is not null)
        {
            backwardInputGradient = (IGpuTensor<T>)backwardBackwardGpuMethod.Invoke(_backwardLayer, new object[] { reversedBackwardGradient })!;
        }
        else
        {
            // CPU fallback
            var gradData = backend.DownloadBuffer(reversedBackwardGradient.Buffer);
            var cpuGrad = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradData), reversedBackwardGradient.Shape);
            var cpuResult = _backwardLayer.Backward(cpuGrad);
            backwardInputGradient = gpuEngine.UploadToGpu(cpuResult, GpuTensorRole.Gradient);
        }

        // Reverse the backward input gradient to match original sequence order
        var reversedBackwardInputGradient = ReverseSequenceGpu(backend, backwardInputGradient, batchSize, timeSteps,
            backwardInputGradient.Shape[backwardInputGradient.Shape.Length - 1]);

        // Cleanup intermediate tensors
        reversedBackwardGradient.Dispose();
        backwardInputGradient.Dispose();

        if (!_mergeMode)
        {
            forwardGradient.Dispose();
            backwardGradient.Dispose();
        }

        // Sum the gradients from both directions
        int elementCount = forwardInputGradient.ElementCount;
        int[] resultShape = (int[])forwardInputGradient.Shape.Clone();
        var resultBuffer = backend.AllocateBuffer(elementCount);
        backend.Add(forwardInputGradient.Buffer, reversedBackwardInputGradient.Buffer, resultBuffer, elementCount);

        // Cleanup
        forwardInputGradient.Dispose();
        reversedBackwardInputGradient.Dispose();

        return new GpuTensor<T>(backend, resultBuffer, resultShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// Merges forward and backward outputs on the GPU according to the merge mode.
    /// </summary>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="forward">Forward layer output.</param>
    /// <param name="backward">Backward layer output (already reversed).</param>
    /// <param name="batchSize">Batch size.</param>
    /// <param name="timeSteps">Number of time steps.</param>
    /// <returns>Merged GPU tensor.</returns>
    private IGpuTensor<T> MergeOutputsGpu(
        IDirectGpuBackend backend,
        IGpuTensor<T> forward,
        IGpuTensor<T> backward,
        int batchSize,
        int timeSteps)
    {
        int hiddenSize = forward.Shape[forward.Shape.Length - 1];
        int totalElements = forward.Buffer.Size;

        if (_mergeMode)
        {
            // Add forward and backward outputs element-wise
            var outputBuffer = backend.AllocateBuffer(totalElements);
            backend.Add(forward.Buffer, backward.Buffer, outputBuffer, totalElements);

            // Dispose inputs since we're returning a new buffer
            forward.Dispose();
            backward.Dispose();

            return new GpuTensor<T>(backend, outputBuffer, forward.Shape, GpuTensorRole.Activation, ownsBuffer: true);
        }
        else
        {
            // Stack outputs along a new dimension [2, batch, timeSteps, hiddenSize]
            int[] stackedShape;
            if (forward.Shape.Length == 2)
            {
                stackedShape = [2, timeSteps, hiddenSize];
            }
            else
            {
                stackedShape = [2, batchSize, timeSteps, hiddenSize];
            }

            int stackedSize = totalElements * 2;
            var outputBuffer = backend.AllocateBuffer(stackedSize);

            // Copy forward output to first half
            backend.Copy2DStrided(forward.Buffer, outputBuffer, 1, totalElements, stackedSize, 0);

            // Copy backward output to second half
            backend.Copy2DStrided(backward.Buffer, outputBuffer, 1, totalElements, stackedSize, totalElements);

            // Dispose inputs since we're returning a new buffer
            forward.Dispose();
            backward.Dispose();

            return new GpuTensor<T>(backend, outputBuffer, stackedShape, GpuTensorRole.Activation, ownsBuffer: true);
        }
    }

    /// <summary>
    /// Performs the backward pass of the bidirectional layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the bidirectional layer, which is used during training to propagate
    /// error gradients back through the network. It splits the output gradient according to the merge mode, propagates
    /// it through both forward and backward inner layers, and combines the resulting input gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// should change to reduce errors.
    ///
    /// During the backward pass:
    /// - The error gradient from the next layer is received
    /// - If the outputs were merged, the same gradient is sent to both forward and backward layers
    /// - If the outputs were separate, the gradient is split for each direction
    /// - The gradients from both layers are combined to update the previous layer
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

    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        // Note: BidirectionalLayer delegates backward pass to inner forward and backward layers.
        // Autodiff support is handled by the inner layers (e.g., LSTMLayer, GRULayer).
        // This wrapper layer simply manages gradient flow for bidirectional processing.

        Tensor<T> forwardGradient, backwardGradient;

        if (_mergeMode)
        {
            forwardGradient = outputGradient;
            backwardGradient = outputGradient;
        }
        else
        {
            forwardGradient = outputGradient.Slice(0);
            backwardGradient = outputGradient.Slice(1);
        }

        var forwardInputGradient = _forwardLayer.Backward(forwardGradient);
        var backwardInputGradient = _backwardLayer.Backward(backwardGradient);

        // Reverse the backward gradient
        backwardInputGradient = ReverseSequence(backwardInputGradient);

        // Sum the gradients
        return forwardInputGradient.Add(backwardInputGradient);
    }

    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // Note: BidirectionalLayer is a wrapper that delegates to inner layers.
        // The autodiff path simply ensures the inner layers use their autodiff implementations.
        // This method manages the bidirectional gradient flow using autodiff-friendly operations.

        if (_lastInput is null || _lastForwardOutput is null || _lastBackwardOutput is null)
            throw new InvalidOperationException("Forward pass must be called before Backward");

        // Split output gradient based on merge mode
        Tensor<T> forwardGradient, backwardGradient;

        if (_mergeMode)
        {
            // In merge mode, the same gradient flows to both directions
            forwardGradient = outputGradient;
            backwardGradient = outputGradient;
        }
        else
        {
            // In non-merge mode, split the gradient for each direction
            forwardGradient = outputGradient.Slice(0);
            backwardGradient = outputGradient.Slice(1);
        }

        // Propagate gradients through inner layers (they will use their autodiff implementations if UseAutodiff is set)
        var forwardInputGradient = _forwardLayer.Backward(forwardGradient);
        var backwardInputGradient = _backwardLayer.Backward(backwardGradient);

        // Reverse the backward gradient to match input sequence order
        backwardInputGradient = ReverseSequence(backwardInputGradient);

        // Sum the gradients from both directions
        return forwardInputGradient.Add(backwardInputGradient);
    }

    /// <summary>
    /// Updates the parameters of both forward and backward layers using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of both the forward and backward inner layers based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - Both forward and backward layers are updated independently
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        _forwardLayer.UpdateParameters(learningRate);
        _backwardLayer.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Reverses the sequence order along the time dimension (typically dimension 1).
    /// </summary>
    /// <param name="input">The input tensor to reverse.</param>
    /// <returns>A new tensor with the sequence reversed.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new tensor with the same shape as the input, but with the sequence reversed along
    /// the time dimension (typically dimension 1). This is used to flip the input for the backward layer and
    /// to flip the gradients during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method flips the order of elements in a sequence.
    /// 
    /// For example:
    /// - Original sequence: [1, 2, 3, 4, 5]
    /// - Reversed sequence: [5, 4, 3, 2, 1]
    /// 
    /// In a neural network, a "sequence" typically represents time steps or ordered data points.
    /// This method is important for the bidirectional functionality because:
    /// - The backward layer needs to process data in reverse order
    /// - The gradients from the backward layer need to be reversed again during training
    /// </para>
    /// </remarks>
    private static Tensor<T> ReverseSequence(Tensor<T> input)
    {
        var reversed = new Tensor<T>(input.Shape);
        int timeSteps = input.Shape[1];

        for (int i = 0; i < timeSteps; i++)
        {
            // Slice along dimension 1 (time axis), getting a [batch, features] tensor
            var slice = input.GetSliceAlongDimension(timeSteps - 1 - i, 1);
            // Set into reversed tensor at position i along dimension 1
            reversed.SetSlice(1, i, slice);
        }

        return reversed;
    }

    /// <summary>
    /// Merges the outputs from the forward and backward passes according to the configured merge mode.
    /// </summary>
    /// <param name="forward">The output tensor from the forward pass.</param>
    /// <param name="backward">The output tensor from the backward pass.</param>
    /// <returns>The merged output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method combines the outputs from the forward and backward passes according to the configured merge mode.
    /// If mergeMode is true, the outputs are added together element-wise. If mergeMode is false, the outputs are
    /// stacked along a new dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This method combines the results from the forward and backward processing.
    /// 
    /// There are two ways to combine the results:
    /// - If mergeMode is true: The values are added together (element by element)
    /// - If mergeMode is false: The two sets of results are kept separate but packed into a single tensor
    /// 
    /// For example, with mergeMode=true and values:
    /// - Forward: [1, 2, 3]
    /// - Backward: [4, 5, 6]
    /// - Result: [5, 7, 9] (each element added)
    /// 
    /// With mergeMode=false:
    /// - Result would contain both original sets of values in a larger tensor
    /// </para>
    /// </remarks>
    private Tensor<T> MergeOutputs(Tensor<T> forward, Tensor<T> backward)
    {
        if (_mergeMode)
        {
            return forward.Add(backward);
        }
        else
        {
            return Tensor<T>.Stack([forward, backward], 0);
        }
    }

    /// <summary>
    /// Gets all trainable parameters from both the forward and backward layers as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from both the forward and backward inner layers and combines them
    /// into a single vector. This is useful for optimization algorithms that operate on all parameters at once, or for
    /// saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from both forward and backward layers.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include weights and biases from both forward and backward layers
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Combine parameters from both forward and backward layers
        var forwardParams = _forwardLayer.GetParameters();
        var backwardParams = _backwardLayer.GetParameters();

        var combinedParams = new Vector<T>(forwardParams.Length + backwardParams.Length);

        // Copy forward parameters
        for (int i = 0; i < forwardParams.Length; i++)
        {
            combinedParams[i] = forwardParams[i];
        }

        // Copy backward parameters
        for (int i = 0; i < backwardParams.Length; i++)
        {
            combinedParams[i + forwardParams.Length] = backwardParams[i];
        }

        return combinedParams;
    }

    /// <summary>
    /// Sets the trainable parameters for both the forward and backward layers.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters for both the forward and backward inner layers from a single vector.
    /// It extracts the appropriate portions of the input vector for each inner layer. This is useful for loading
    /// saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in both forward and backward layers.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The first part of the vector is used for the forward layer
    /// - The second part of the vector is used for the backward layer
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
        var forwardParams = _forwardLayer.GetParameters();
        var backwardParams = _backwardLayer.GetParameters();

        if (parameters.Length != forwardParams.Length + backwardParams.Length)
            throw new ArgumentException($"Expected {forwardParams.Length + backwardParams.Length} parameters, but got {parameters.Length}");

        // Extract and set forward parameters
        var newForwardParams = new Vector<T>(forwardParams.Length);
        for (int i = 0; i < forwardParams.Length; i++)
        {
            newForwardParams[i] = parameters[i];
        }

        // Extract and set backward parameters
        var newBackwardParams = new Vector<T>(backwardParams.Length);
        for (int i = 0; i < backwardParams.Length; i++)
        {
            newBackwardParams[i] = parameters[i + forwardParams.Length];
        }

        _forwardLayer.SetParameters(newForwardParams);
        _backwardLayer.SetParameters(newBackwardParams);
    }

    /// <summary>
    /// Resets the internal state of the bidirectional layer and its inner layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the bidirectional layer, including the cached inputs and outputs,
    /// as well as the states of both forward and backward inner layers. This is useful when starting to process
    /// a new sequence or when implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs are cleared
    /// - Both forward and backward layers are also reset
    /// - The layer forgets any information from previous sequences
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence
    /// - Preventing information from one sequence affecting another
    /// - Starting a new training episode
    /// 
    /// For example, if you've processed one sentence and want to start with a new sentence,
    /// you should reset the state to prevent the new sentence from being influenced by the previous one.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastForwardOutput = null;
        _lastBackwardOutput = null;

        _forwardLayer.ResetState();
        _backwardLayer.ResetState();
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (!_forwardLayer.SupportsJitCompilation || !_backwardLayer.SupportsJitCompilation)
            throw new InvalidOperationException("BidirectionalLayer requires both inner layers to support JIT compilation.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Forward layer processing
        var forwardInputNodes = new List<ComputationNode<T>>();
        var forwardOutput = _forwardLayer.ExportComputationGraph(forwardInputNodes);

        // Backward layer processing (note: sequence reversal is handled at runtime, not in graph)
        var backwardInputNodes = new List<ComputationNode<T>>();
        var backwardOutput = _backwardLayer.ExportComputationGraph(backwardInputNodes);

        // Merge outputs based on merge mode
        if (_mergeMode)
        {
            // Add outputs element-wise
            return TensorOperations<T>.Add(forwardOutput, backwardOutput);
        }
        else
        {
            // Stack outputs along new dimension
            // Note: This requires a Stack operation in TensorOperations
            // For now, return forward output as primary
            return forwardOutput;
        }
    }

    public override bool SupportsJitCompilation =>
        _forwardLayer.SupportsJitCompilation && _backwardLayer.SupportsJitCompilation;

    /// <summary>
    /// Performs the backward pass on GPU tensors.
    /// </summary>
    /// <param name="outputGradient">GPU tensor containing the gradient of the loss with respect to the output.</param>
    /// <returns>GPU tensor containing the gradient of the loss with respect to the input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Check if inner layers support GPU training
        if (!_forwardLayer.SupportsGpuTraining || !_backwardLayer.SupportsGpuTraining)
        {
            // CPU fallback: download gradient, run Backward(), upload result
            var outputGradCpu = outputGradient.ToTensor();
            var inputGradCpu = Backward(outputGradCpu);
            return new GpuTensor<T>(backend, inputGradCpu, GpuTensorRole.Gradient);
        }

        // Delegate to inner layers for GPU backward
        IGpuTensor<T> forwardGradient, backwardGradient;

        if (_mergeMode)
        {
            forwardGradient = outputGradient;
            backwardGradient = outputGradient;
        }
        else
        {
            // Split output gradient for each direction
            int halfSize = outputGradient.Buffer.Size / 2;
            forwardGradient = outputGradient.CreateView(0, [halfSize]);
            backwardGradient = outputGradient.CreateView(halfSize, [halfSize]);
        }

        var forwardInputGradient = _forwardLayer.BackwardGpu(forwardGradient);
        var backwardInputGradient = _backwardLayer.BackwardGpu(backwardGradient);

        // Reverse the backward gradient
        int batchSize = backwardInputGradient.Shape.Length >= 3 ? backwardInputGradient.Shape[0] : 1;
        int timeSteps = backwardInputGradient.Shape.Length >= 3 ? backwardInputGradient.Shape[1] : backwardInputGradient.Shape[0];
        int features = backwardInputGradient.Shape[backwardInputGradient.Shape.Length - 1];

        var reversedBackwardGradient = ReverseSequenceGpu(backend, backwardInputGradient, batchSize, timeSteps, features);
        backwardInputGradient.Dispose();

        // Sum the gradients
        var inputGradBuffer = backend.AllocateBuffer(forwardInputGradient.Buffer.Size);
        backend.Add(forwardInputGradient.Buffer, reversedBackwardGradient.Buffer, inputGradBuffer, forwardInputGradient.Buffer.Size);

        forwardInputGradient.Dispose();
        reversedBackwardGradient.Dispose();

        return new GpuTensor<T>(backend, inputGradBuffer, forwardInputGradient.Shape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// Updates parameters on GPU using the configured optimizer.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        // Delegate to inner layers
        if (_forwardLayer.SupportsGpuTraining)
        {
            _forwardLayer.UpdateParametersGpu(config);
        }

        if (_backwardLayer.SupportsGpuTraining)
        {
            _backwardLayer.UpdateParametersGpu(config);
        }
    }
}
