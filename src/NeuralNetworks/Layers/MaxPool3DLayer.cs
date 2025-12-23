using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a 3D max pooling layer for volumetric data, reducing spatial dimensions
/// by taking the maximum value in each pooling window.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A 3D max pooling layer reduces the spatial dimensions (depth, height, width) of volumetric
/// data while preserving the most prominent features. This helps reduce computational cost
/// and provides translation invariance.
/// </para>
/// <para><b>For Beginners:</b> Max pooling works like summarizing a 3D region by keeping only
/// the largest value.
/// 
/// Think of it like this:
/// - You have a 3D grid of numbers
/// - You divide it into small cubes (e.g., 2x2x2)
/// - For each cube, you keep only the largest number
/// - This makes your data smaller while keeping the important features
/// 
/// This is useful because:
/// - It reduces the amount of computation needed
/// - It helps the network focus on the most important features
/// - It makes the network more robust to small position changes
/// </para>
/// </remarks>
public class MaxPool3DLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the size of the pooling window.
    /// </summary>
    public int PoolSize { get; private set; }

    /// <summary>
    /// Gets the stride (step size) for moving the pooling window.
    /// </summary>
    public int Stride { get; private set; }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => InputShape != null && InputShape.Length > 0;

    private int[,,,,,]? _maxIndices;
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Creates a new 3D max pooling layer with the specified parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input [channels, depth, height, width].</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="stride">The stride for moving the pooling window.</param>
    public MaxPool3DLayer(int[] inputShape, int poolSize, int stride)
        : base(inputShape, CalculateOutputShape(inputShape, poolSize, stride))
    {
        PoolSize = poolSize;
        Stride = stride;
    }

    private static int[] CalculateOutputShape(int[] inputShape, int poolSize, int stride)
    {
        int channels = inputShape[0];
        int depth = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int outputDepth = (depth - poolSize) / stride + 1;
        int outputHeight = (height - poolSize) / stride + 1;
        int outputWidth = (width - poolSize) / stride + 1;

        return [channels, outputDepth, outputHeight, outputWidth];
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        bool hasBatch = input.Rank == 5;
        int batch = hasBatch ? input.Shape[0] : 1;
        int channels = hasBatch ? input.Shape[1] : input.Shape[0];
        int inD = hasBatch ? input.Shape[2] : input.Shape[1];
        int inH = hasBatch ? input.Shape[3] : input.Shape[2];
        int inW = hasBatch ? input.Shape[4] : input.Shape[3];

        int outD = (inD - PoolSize) / Stride + 1;
        int outH = (inH - PoolSize) / Stride + 1;
        int outW = (inW - PoolSize) / Stride + 1;

        var outputData = new T[batch * channels * outD * outH * outW];
        var output = new Tensor<T>(outputData, hasBatch
            ? [batch, channels, outD, outH, outW]
            : [channels, outD, outH, outW]);

        // Store indices for backward pass [batch, channels, outD, outH, outW, 3]
        _maxIndices = new int[batch, channels, outD, outH, outW, 3];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int od = 0; od < outD; od++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            T maxVal = NumOps.MinValue;
                            int maxD = 0, maxH = 0, maxW = 0;

                            for (int pd = 0; pd < PoolSize; pd++)
                            {
                                for (int ph = 0; ph < PoolSize; ph++)
                                {
                                    for (int pw = 0; pw < PoolSize; pw++)
                                    {
                                        int id = od * Stride + pd;
                                        int ih = oh * Stride + ph;
                                        int iw = ow * Stride + pw;

                                        T val = hasBatch ? input[b, c, id, ih, iw] : input[c, id, ih, iw];
                                        if (NumOps.GreaterThan(val, maxVal))
                                        {
                                            maxVal = val;
                                            maxD = id;
                                            maxH = ih;
                                            maxW = iw;
                                        }
                                    }
                                }
                            }

                            if (hasBatch)
                                output[b, c, od, oh, ow] = maxVal;
                            else
                                output[c, od, oh, ow] = maxVal;

                            _maxIndices[b, c, od, oh, ow, 0] = maxD;
                            _maxIndices[b, c, od, oh, ow, 1] = maxH;
                            _maxIndices[b, c, od, oh, ow, 2] = maxW;
                        }
                    }
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _maxIndices == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        bool hasBatch = _lastInput.Rank == 5;
        int batch = hasBatch ? _lastInput.Shape[0] : 1;
        int channels = hasBatch ? _lastInput.Shape[1] : _lastInput.Shape[0];
        int inD = hasBatch ? _lastInput.Shape[2] : _lastInput.Shape[1];
        int inH = hasBatch ? _lastInput.Shape[3] : _lastInput.Shape[2];
        int inW = hasBatch ? _lastInput.Shape[4] : _lastInput.Shape[3];

        int outD = hasBatch ? outputGradient.Shape[2] : outputGradient.Shape[1];
        int outH = hasBatch ? outputGradient.Shape[3] : outputGradient.Shape[2];
        int outW = hasBatch ? outputGradient.Shape[4] : outputGradient.Shape[3];

        var inputGradData = new T[batch * channels * inD * inH * inW];
        var inputGrad = new Tensor<T>(inputGradData, _lastInput.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int od = 0; od < outD; od++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            T gradVal = hasBatch ? outputGradient[b, c, od, oh, ow] : outputGradient[c, od, oh, ow];
                            int maxD = _maxIndices[b, c, od, oh, ow, 0];
                            int maxH = _maxIndices[b, c, od, oh, ow, 1];
                            int maxW = _maxIndices[b, c, od, oh, ow, 2];

                            if (hasBatch)
                            {
                                T current = inputGrad[b, c, maxD, maxH, maxW];
                                inputGrad[b, c, maxD, maxH, maxW] = NumOps.Add(current, gradVal);
                            }
                            else
                            {
                                T current = inputGrad[c, maxD, maxH, maxW];
                                inputGrad[c, maxD, maxH, maxW] = NumOps.Add(current, gradVal);
                            }
                        }
                    }
                }
            }
        }

        return inputGrad;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        // Max pooling has no trainable parameters
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Empty();
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _maxIndices = null;
    }

    /// <inheritdoc />
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(PoolSize);
        writer.Write(Stride);
    }

    /// <inheritdoc />
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        PoolSize = reader.ReadInt32();
        Stride = reader.ReadInt32();
    }

    /// <inheritdoc />
    public override IEnumerable<ActivationFunction> GetActivationTypes()
    {
        return [];
    }

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input: [1, channels, depth, height, width]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Note: MaxPool3D operation not yet in TensorOperations - return input for now
        return inputNode;
    }
}
