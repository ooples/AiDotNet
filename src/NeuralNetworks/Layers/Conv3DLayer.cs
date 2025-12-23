using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a 3D convolutional layer for processing volumetric data like voxel grids.
/// </summary>
/// <remarks>
/// <para>
/// A 3D convolutional layer applies learnable filters to volumetric input data to extract
/// spatial features across all three dimensions. This is essential for processing 3D data
/// such as voxelized point clouds, medical imaging (CT/MRI), or video sequences.
/// </para>
/// <para><b>For Beginners:</b> A 3D convolutional layer is like a 2D convolution but extended
/// to work with volumetric data.
/// 
/// Think of it like examining a 3D cube of data:
/// - A 2D convolution slides a filter across height and width
/// - A 3D convolution slides a filter across depth, height, and width
/// 
/// This is useful for:
/// - Recognizing 3D shapes from voxel grids (like ModelNet40)
/// - Analyzing medical scans (CT, MRI)
/// - Processing video frames as a 3D volume
/// 
/// The layer learns to detect 3D patterns like edges, surfaces, and volumes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class Conv3DLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    public int InputChannels { get; private set; }

    /// <summary>
    /// Gets the number of output channels (filters).
    /// </summary>
    public int OutputChannels { get; private set; }

    /// <summary>
    /// Gets the size of the 3D convolution kernel.
    /// </summary>
    public int KernelSize { get; private set; }

    /// <summary>
    /// Gets the stride of the convolution.
    /// </summary>
    public int Stride { get; private set; }

    /// <summary>
    /// Gets the padding applied to all sides.
    /// </summary>
    public int Padding { get; private set; }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => _kernels != null && _biases != null && CanActivationBeJitted();

    private Tensor<T> _kernels;
    private Tensor<T> _biases;
    private Tensor<T>? _kernelsGradient;
    private Tensor<T>? _biasesGradient;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Initializes a new instance of the Conv3DLayer class.
    /// </summary>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="outputChannels">Number of output channels (filters).</param>
    /// <param name="kernelSize">Size of the 3D convolution kernel.</param>
    /// <param name="inputDepth">Depth of the input volume.</param>
    /// <param name="inputHeight">Height of the input volume.</param>
    /// <param name="inputWidth">Width of the input volume.</param>
    /// <param name="stride">Stride of the convolution. Defaults to 1.</param>
    /// <param name="padding">Zero-padding added to all sides. Defaults to 0.</param>
    /// <param name="activation">The activation function to apply. Defaults to ReLU.</param>
    public Conv3DLayer(
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int inputDepth,
        int inputHeight,
        int inputWidth,
        int stride = 1,
        int padding = 0,
        IActivationFunction<T>? activation = null)
        : base(
            CalculateInputShape(inputChannels, inputDepth, inputHeight, inputWidth),
            CalculateOutputShape(outputChannels, inputDepth, inputHeight, inputWidth, kernelSize, stride, padding),
            activation ?? new ReLUActivation<T>())
    {
        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;

        // Initialize kernels: [outputChannels, inputChannels, kernelSize, kernelSize, kernelSize]
        _kernels = new Tensor<T>([outputChannels, inputChannels, kernelSize, kernelSize, kernelSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();
    }

    /// <summary>
    /// Initializes a new instance of the Conv3DLayer class with a vector activation function.
    /// </summary>
    public Conv3DLayer(
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int inputDepth,
        int inputHeight,
        int inputWidth,
        int stride = 1,
        int padding = 0,
        IVectorActivationFunction<T>? vectorActivation = null)
        : base(
            CalculateInputShape(inputChannels, inputDepth, inputHeight, inputWidth),
            CalculateOutputShape(outputChannels, inputDepth, inputHeight, inputWidth, kernelSize, stride, padding),
            vectorActivation ?? new ReLUActivation<T>())
    {
        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;

        _kernels = new Tensor<T>([outputChannels, inputChannels, kernelSize, kernelSize, kernelSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();
    }

    private static int[] CalculateInputShape(int channels, int depth, int height, int width)
    {
        return [channels, depth, height, width];
    }

    private static int[] CalculateOutputShape(int channels, int inputDepth, int inputHeight, int inputWidth, int kernelSize, int stride, int padding)
    {
        int outputDepth = (inputDepth + 2 * padding - kernelSize) / stride + 1;
        int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;
        return [channels, outputDepth, outputHeight, outputWidth];
    }

    private void InitializeWeights()
    {
        // He initialization for ReLU activations
        int fanIn = InputChannels * KernelSize * KernelSize * KernelSize;
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(
            NumOps.FromDouble(2.0),
            NumOps.FromDouble(fanIn)));

        for (int oc = 0; oc < OutputChannels; oc++)
        {
            for (int ic = 0; ic < InputChannels; ic++)
            {
                for (int kd = 0; kd < KernelSize; kd++)
                {
                    for (int kh = 0; kh < KernelSize; kh++)
                    {
                        for (int kw = 0; kw < KernelSize; kw++)
                        {
                            _kernels[oc, ic, kd, kh, kw] = NumOps.Multiply(
                                scale,
                                NumOps.FromDouble(Random.NextDouble() * 2 - 1));
                        }
                    }
                }
            }
            _biases[oc] = NumOps.Zero;
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Input shape: [batch, channels, depth, height, width] or [channels, depth, height, width]
        bool hasBatch = input.Rank == 5;
        int batch = hasBatch ? input.Shape[0] : 1;
        int inC = hasBatch ? input.Shape[1] : input.Shape[0];
        int inD = hasBatch ? input.Shape[2] : input.Shape[1];
        int inH = hasBatch ? input.Shape[3] : input.Shape[2];
        int inW = hasBatch ? input.Shape[4] : input.Shape[3];

        int outD = (inD + 2 * Padding - KernelSize) / Stride + 1;
        int outH = (inH + 2 * Padding - KernelSize) / Stride + 1;
        int outW = (inW + 2 * Padding - KernelSize) / Stride + 1;

        var outputData = new T[batch * OutputChannels * outD * outH * outW];
        var output = new Tensor<T>(outputData, hasBatch 
            ? [batch, OutputChannels, outD, outH, outW] 
            : [OutputChannels, outD, outH, outW]);

        // Perform 3D convolution
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < OutputChannels; oc++)
            {
                for (int od = 0; od < outD; od++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            T sum = _biases[oc];

                            for (int ic = 0; ic < inC; ic++)
                            {
                                for (int kd = 0; kd < KernelSize; kd++)
                                {
                                    for (int kh = 0; kh < KernelSize; kh++)
                                    {
                                        for (int kw = 0; kw < KernelSize; kw++)
                                        {
                                            int id = od * Stride + kd - Padding;
                                            int ih = oh * Stride + kh - Padding;
                                            int iw = ow * Stride + kw - Padding;

                                            if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                            {
                                                T inputVal = hasBatch ? input[b, ic, id, ih, iw] : input[ic, id, ih, iw];
                                                T kernelVal = _kernels[oc, ic, kd, kh, kw];
                                                sum = NumOps.Add(sum, NumOps.Multiply(inputVal, kernelVal));
                                            }
                                        }
                                    }
                                }
                            }

                            if (hasBatch)
                                output[b, oc, od, oh, ow] = sum;
                            else
                                output[oc, od, oh, ow] = sum;
                        }
                    }
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Apply activation derivative
        var delta = ApplyActivationDerivative(_lastOutput, outputGradient);

        bool hasBatch = _lastInput.Rank == 5;
        int batch = hasBatch ? _lastInput.Shape[0] : 1;
        int inC = hasBatch ? _lastInput.Shape[1] : _lastInput.Shape[0];
        int inD = hasBatch ? _lastInput.Shape[2] : _lastInput.Shape[1];
        int inH = hasBatch ? _lastInput.Shape[3] : _lastInput.Shape[2];
        int inW = hasBatch ? _lastInput.Shape[4] : _lastInput.Shape[3];

        int outD = hasBatch ? delta.Shape[2] : delta.Shape[1];
        int outH = hasBatch ? delta.Shape[3] : delta.Shape[2];
        int outW = hasBatch ? delta.Shape[4] : delta.Shape[3];

        // Initialize gradients
        var inputGradData = new T[batch * inC * inD * inH * inW];
        var inputGrad = new Tensor<T>(inputGradData, _lastInput.Shape);

        _kernelsGradient = new Tensor<T>(_kernels.Shape);
        _biasesGradient = new Tensor<T>(_biases.Shape);

        // Compute gradients
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < OutputChannels; oc++)
            {
                for (int od = 0; od < outD; od++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            T gradVal = hasBatch ? delta[b, oc, od, oh, ow] : delta[oc, od, oh, ow];

                            // Bias gradient
                            _biasesGradient[oc] = NumOps.Add(_biasesGradient[oc], gradVal);

                            for (int ic = 0; ic < inC; ic++)
                            {
                                for (int kd = 0; kd < KernelSize; kd++)
                                {
                                    for (int kh = 0; kh < KernelSize; kh++)
                                    {
                                        for (int kw = 0; kw < KernelSize; kw++)
                                        {
                                            int id = od * Stride + kd - Padding;
                                            int ih = oh * Stride + kh - Padding;
                                            int iw = ow * Stride + kw - Padding;

                                            if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                            {
                                                // Kernel gradient
                                                T inputVal = hasBatch ? _lastInput[b, ic, id, ih, iw] : _lastInput[ic, id, ih, iw];
                                                T currentKernelGrad = _kernelsGradient[oc, ic, kd, kh, kw];
                                                _kernelsGradient[oc, ic, kd, kh, kw] = NumOps.Add(
                                                    currentKernelGrad,
                                                    NumOps.Multiply(gradVal, inputVal));

                                                // Input gradient
                                                T kernelVal = _kernels[oc, ic, kd, kh, kw];
                                                if (hasBatch)
                                                {
                                                    T currentInputGrad = inputGrad[b, ic, id, ih, iw];
                                                    inputGrad[b, ic, id, ih, iw] = NumOps.Add(
                                                        currentInputGrad,
                                                        NumOps.Multiply(gradVal, kernelVal));
                                                }
                                                else
                                                {
                                                    T currentInputGrad = inputGrad[ic, id, ih, iw];
                                                    inputGrad[ic, id, ih, iw] = NumOps.Add(
                                                        currentInputGrad,
                                                        NumOps.Multiply(gradVal, kernelVal));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGrad;
    }

    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // Fallback to manual for now - autodiff Conv3D not yet implemented
        return BackwardManual(outputGradient);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_kernelsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _kernels = _kernels.Subtract(_kernelsGradient.Multiply(learningRate));
        _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_kernels.ToArray()),
            new Vector<T>(_biases.ToArray()));
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int expected = _kernels.Length + _biases.Length;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, but got {parameters.Length}");

        int index = 0;
        _kernels = new Tensor<T>(_kernels.Shape, parameters.Slice(index, _kernels.Length));
        index += _kernels.Length;
        _biases = new Tensor<T>(_biases.Shape, parameters.Slice(index, _biases.Length));
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _kernelsGradient = null;
        _biasesGradient = null;
    }

    /// <inheritdoc />
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(InputChannels);
        writer.Write(OutputChannels);
        writer.Write(KernelSize);
        writer.Write(Stride);
        writer.Write(Padding);

        // Serialize kernels
        for (int i = 0; i < _kernels.Length; i++)
        {
            writer.Write(NumOps.ToDouble(_kernels[i]));
        }

        // Serialize biases
        for (int i = 0; i < _biases.Length; i++)
        {
            writer.Write(NumOps.ToDouble(_biases[i]));
        }
    }

    /// <inheritdoc />
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        InputChannels = reader.ReadInt32();
        OutputChannels = reader.ReadInt32();
        KernelSize = reader.ReadInt32();
        Stride = reader.ReadInt32();
        Padding = reader.ReadInt32();

        // Deserialize kernels
        _kernels = new Tensor<T>([OutputChannels, InputChannels, KernelSize, KernelSize, KernelSize]);
        for (int i = 0; i < _kernels.Length; i++)
        {
            _kernels[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Deserialize biases
        _biases = new Tensor<T>([OutputChannels]);
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_kernels == null || _biases == null)
            throw new InvalidOperationException("Layer weights not initialized.");

        // Create symbolic input: [1, channels, depth, height, width]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create constant nodes for kernels and biases
        var kernelNode = TensorOperations<T>.Constant(_kernels, "kernel");
        var biasNode = TensorOperations<T>.Constant(_biases, "bias");

        // Note: Conv3D operation not yet in TensorOperations - return input for now
        // This is a placeholder for future JIT support
        var activatedOutput = ApplyActivationToGraph(inputNode);
        return activatedOutput;
    }

    /// <inheritdoc />
    public override Tensor<T> GetBiases() => _biases;

    /// <summary>
    /// Gets the filter kernels of the convolutional layer.
    /// </summary>
    public Tensor<T> GetFilters() => _kernels;
}
