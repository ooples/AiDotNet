using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
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
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, ChangesShape = true, TestInputShape = "4, 4", TestConstructorArgs = "4")]
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
    /// Indicates whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override bool SupportsGpuTraining => true;

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
        _originalShape = input._shape;
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
                result.Data.Span[i] = input.Data.Span[offset + i];
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
                result.Data.Span[i] = input.Data.Span[lastOffset + i];
            }
            return result;
        }
        else
        {
            // Higher rank (>= 4): treat dim[0] as seqLen, extract last slice with remaining dims
            int seqLen = input.Shape[0];
            _lastSequenceLength = seqLen;

            // Output shape is input shape minus the first dimension
            var outputShape = new int[rank - 1];
            int sliceSize = 1;
            for (int d = 1; d < rank; d++)
            {
                outputShape[d - 1] = input.Shape[d];
                sliceSize *= input.Shape[d];
            }

            var result = new Tensor<T>(outputShape);
            int lastOffset = (seqLen - 1) * sliceSize;
            for (int i = 0; i < sliceSize; i++)
            {
                result.Data.Span[i] = input.Data.Span[lastOffset + i];
            }
            return result;
        }
    }

    /// <summary>
    /// GPU-accelerated forward pass that extracts the last timestep from a sequence.
    /// Uses zero-copy CreateView to extract the last slice directly on GPU.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensors.</param>
    /// <returns>GPU-resident output tensor containing the last timestep.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var input = inputs[0];
        var shape = input._shape;
        int rank = shape.Length;

        _originalShape = shape;

        if (rank == 1)
        {
            _lastSequenceLength = 1;
            return input;
        }
        else if (rank >= 2)
        {
            // Shape: [seqLen, ...rest] -> [...rest]
            int seqLen = shape[0];
            _lastSequenceLength = seqLen;

            // Output shape is input shape minus the first dimension
            var outputShape = new int[rank - 1];
            int sliceSize = 1;
            for (int d = 1; d < rank; d++)
            {
                outputShape[d - 1] = shape[d];
                sliceSize *= shape[d];
            }

            // Zero-copy view of the last slice
            int offset = (seqLen - 1) * sliceSize;
            return input.CreateView(offset, outputShape);
        }
        else
        {
            throw new ArgumentException($"SequenceLastLayer expects at least 1D input, got {rank}D.");
        }
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
}
