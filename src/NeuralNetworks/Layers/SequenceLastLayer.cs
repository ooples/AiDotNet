using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A layer that extracts the last timestep from a sequence.
/// </summary>
/// <remarks>
/// <para>
/// This layer is used after recurrent layers (RNN, LSTM, GRU) when the task requires
/// a single output from the entire sequence, such as sequence classification.
/// </para>
/// <para>
/// <b>For Beginners:</b> When processing sequences (like sentences or time series),
/// recurrent layers output a value for each timestep. For tasks like classification,
/// we often only need the final output (after seeing the whole sequence). This layer
/// extracts just that last output.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SequenceLastLayer<T> : LayerBase<T>
{
    private readonly int _featureSize;
    private int _lastSequenceLength;
    private int[]? _originalShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>false</c> because this layer has no trainable parameters.
    /// </value>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Indicates whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new SequenceLastLayer.
    /// </summary>
    /// <param name="featureSize">The size of the feature dimension (last dimension of input).</param>
    public SequenceLastLayer(int featureSize)
        : base([featureSize], [featureSize], new IdentityActivation<T>() as IActivationFunction<T>)
    {
        _featureSize = featureSize;
    }

    /// <inheritdoc/>
    public override int ParameterCount => 0;

    /// <summary>
    /// Extracts the last timestep from the input sequence.
    /// </summary>
    /// <param name="input">Input tensor of shape [seqLen, features] or [seqLen, batch, features].</param>
    /// <returns>Output tensor of shape [features] or [batch, features].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalShape = input.Shape;
        int rank = input.Shape.Length;

        if (rank == 1)
        {
            // Already a 1D vector, just pass through
            _lastSequenceLength = 1;
            return input;
        }
        else if (rank == 2)
        {
            // Shape: [seqLen, features] -> [features]
            int seqLen = input.Shape[0];
            int features = input.Shape[1];
            _lastSequenceLength = seqLen;

            // Extract last row
            var result = new Tensor<T>([features]);
            int offset = (seqLen - 1) * features;
            for (int i = 0; i < features; i++)
            {
                result.Data[i] = input.Data[offset + i];
            }
            return result;
        }
        else if (rank == 3)
        {
            // Shape: [seqLen, batch, features] -> [batch, features]
            int seqLen = input.Shape[0];
            int batch = input.Shape[1];
            int features = input.Shape[2];
            _lastSequenceLength = seqLen;

            var result = new Tensor<T>([batch, features]);
            int stride = batch * features;
            int lastOffset = (seqLen - 1) * stride;
            for (int i = 0; i < batch * features; i++)
            {
                result.Data[i] = input.Data[lastOffset + i];
            }
            return result;
        }
        else
        {
            throw new ArgumentException($"SequenceLastLayer expects 1D, 2D, or 3D input, got {rank}D.");
        }
    }

    /// <summary>
    /// GPU-accelerated forward pass that extracts the last timestep from a sequence.
    /// Uses zero-copy CreateView to extract the last slice directly on GPU.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensors.</param>
    /// <returns>GPU-resident output tensor containing the last timestep.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var input = inputs[0];
        var shape = input.Shape;
        int rank = shape.Length;

        _originalShape = shape;

        if (rank == 1)
        {
            // Already a 1D vector, just pass through
            _lastSequenceLength = 1;
            return input;
        }
        else if (rank == 2)
        {
            // Shape: [seqLen, features] -> [features]
            int seqLen = shape[0];
            int features = shape[1];
            _lastSequenceLength = seqLen;

            // Zero-copy view of the last row
            int offset = (seqLen - 1) * features;
            return input.CreateView(offset, [features]);
        }
        else if (rank == 3)
        {
            // Shape: [seqLen, batch, features] -> [batch, features]
            int seqLen = shape[0];
            int batch = shape[1];
            int features = shape[2];
            _lastSequenceLength = seqLen;

            // Zero-copy view of the last slice
            int offset = (seqLen - 1) * batch * features;
            return input.CreateView(offset, [batch, features]);
        }
        else
        {
            throw new ArgumentException($"SequenceLastLayer expects 1D, 2D, or 3D input, got {rank}D.");
        }
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the input on the GPU.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input (zeros except at last timestep).</returns>
    public IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        if (_originalShape == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var backend = gpuEngine.GetBackend() ?? throw new InvalidOperationException("GPU backend unavailable.");

        int inputSize = 1;
        foreach (var dim in _originalShape) inputSize *= dim;

        // Create zero-initialized buffer for input gradient
        var gradInputBuffer = backend.AllocateBuffer(inputSize);
        backend.Fill(gradInputBuffer, 0.0f, inputSize);

        int rank = _originalShape.Length;

        if (rank == 1)
        {
            // Pass through
            backend.Copy(outputGradient.Buffer, gradInputBuffer, outputGradient.ElementCount);
        }
        else if (rank == 2)
        {
            // Shape: [seqLen, features] - copy gradient to last row
            int features = _originalShape[1];
            int offset = (_lastSequenceLength - 1) * features;
            backend.Copy(outputGradient.Buffer, 0, gradInputBuffer, offset, features);
        }
        else if (rank == 3)
        {
            // Shape: [seqLen, batch, features] - copy gradient to last slice
            int batch = _originalShape[1];
            int features = _originalShape[2];
            int offset = (_lastSequenceLength - 1) * batch * features;
            int copySize = batch * features;
            backend.Copy(outputGradient.Buffer, 0, gradInputBuffer, offset, copySize);
        }

        return new GpuTensor<T>(backend, gradInputBuffer, _originalShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// Backward pass: distributes gradient to the last timestep only.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer.</param>
    /// <returns>Gradient with shape matching the original input (zeros except at last timestep).</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_originalShape == null)
            throw new InvalidOperationException("Forward must be called before Backward.");

        // Create zero gradient tensor with original input shape
        var inputGradient = new Tensor<T>(_originalShape);
        int rank = _originalShape.Length;

        if (rank == 1)
        {
            // Pass through
            for (int i = 0; i < outputGradient.Length; i++)
            {
                inputGradient.Data[i] = outputGradient.Data[i];
            }
        }
        else if (rank == 2)
        {
            // Shape: [seqLen, features]
            int features = _originalShape[1];
            int offset = (_lastSequenceLength - 1) * features;
            for (int i = 0; i < features; i++)
            {
                inputGradient.Data[offset + i] = outputGradient.Data[i];
            }
        }
        else if (rank == 3)
        {
            // Shape: [seqLen, batch, features]
            int batch = _originalShape[1];
            int features = _originalShape[2];
            int stride = batch * features;
            int lastOffset = (_lastSequenceLength - 1) * stride;
            for (int i = 0; i < batch * features; i++)
            {
                inputGradient.Data[lastOffset + i] = outputGradient.Data[i];
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Returns an empty vector since this layer has no trainable parameters.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Update parameters is a no-op since this layer has no trainable parameters.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update
    }

    /// <summary>
    /// Reset state is a no-op since this layer maintains no state between forward passes.
    /// </summary>
    public override void ResetState()
    {
        // No state to reset (except cached shape for backward pass)
        _originalShape = null;
        _lastSequenceLength = 0;
    }

    /// <summary>
    /// Exports the computation graph for this layer.
    /// </summary>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // For sequence last, we're essentially doing a slice operation
        // For simplicity, return the input node as this is a pass-through in terms of graph structure
        return inputNode;
    }
}
