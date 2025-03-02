namespace AiDotNet.NeuralNetworks.Layers;

public class DeconvolutionalLayer<T> : LayerBase<T>
{
    private Tensor<T> _kernels;
    private Vector<T> _biases;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _kernelsGradient;
    private Vector<T>? _biasesGradient;

    public int InputDepth { get; }
    public int OutputDepth { get; }
    public int KernelSize { get; }
    public int Stride { get; }
    public int Padding { get; }

    public override bool SupportsTraining => true;

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
        _biases = new Vector<T>(OutputDepth);

        InitializeParameters();
    }

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
        _biases = new Vector<T>(OutputDepth);

        InitializeParameters();
    }

    private static int[] CalculateOutputShape(int[] inputShape, int outputDepth, int kernelSize, int stride, int padding)
    {
        int outputHeight = (inputShape[2] - 1) * stride - 2 * padding + kernelSize;
        int outputWidth = (inputShape[3] - 1) * stride - 2 * padding + kernelSize;

        return [inputShape[0], outputDepth, outputHeight, outputWidth];
    }

    private void InitializeParameters()
    {
        // Xavier/Glorot initialization
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (InputDepth + OutputDepth)));
        for (int i = 0; i < _kernels.Length; i++)
        {
            _kernels[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }

        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Zero;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int inputHeight = input.Shape[2];
        int inputWidth = input.Shape[3];
        int outputHeight = OutputShape[2];
        int outputWidth = OutputShape[3];

        var output = new Tensor<T>([batchSize, OutputDepth, outputHeight, outputWidth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int od = 0; od < OutputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = _biases[od];
                        for (int id = 0; id < InputDepth; id++)
                        {
                            for (int kh = 0; kh < KernelSize; kh++)
                            {
                                for (int kw = 0; kw < KernelSize; kw++)
                                {
                                    int ih = (oh + Padding - kh) / Stride;
                                    int iw = (ow + Padding - kw) / Stride;
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                    {
                                        sum = NumOps.Add(sum, NumOps.Multiply(input[b, id, ih, iw], _kernels[id, od, kh, kw]));
                                    }
                                }
                            }
                        }

                        output[b, od, oh, ow] = sum;
                    }
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int inputHeight = _lastInput.Shape[2];
        int inputWidth = _lastInput.Shape[3];
        int outputHeight = OutputShape[2];
        int outputWidth = OutputShape[3];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _kernelsGradient = new Tensor<T>(_kernels.Shape);
        _biasesGradient = new Vector<T>(OutputDepth);

        for (int b = 0; b < batchSize; b++)
        {
            for (int od = 0; od < OutputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T outGrad = activationGradient[b, od, oh, ow];
                        _biasesGradient[od] = NumOps.Add(_biasesGradient[od], outGrad);

                        for (int id = 0; id < InputDepth; id++)
                        {
                            for (int kh = 0; kh < KernelSize; kh++)
                            {
                                for (int kw = 0; kw < KernelSize; kw++)
                                {
                                    int ih = (oh + Padding - kh) / Stride;
                                    int iw = (ow + Padding - kw) / Stride;
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                    {
                                        T inputVal = _lastInput[b, id, ih, iw];
                                        _kernelsGradient[id, od, kh, kw] = NumOps.Add(_kernelsGradient[id, od, kh, kw], NumOps.Multiply(outGrad, inputVal));
                                        inputGradient[b, id, ih, iw] = NumOps.Add(inputGradient[b, id, ih, iw], NumOps.Multiply(outGrad, _kernels[id, od, kh, kw]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_kernelsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        for (int i = 0; i < _kernels.Length; i++)
        {
            _kernels[i] = NumOps.Subtract(_kernels[i], NumOps.Multiply(learningRate, _kernelsGradient[i]));
        }

        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Subtract(_biases[i], NumOps.Multiply(learningRate, _biasesGradient[i]));
        }
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _kernels.Length + _biases.Length;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy kernel parameters
        for (int id = 0; id < InputDepth; id++)
        {
            for (int od = 0; od < OutputDepth; od++)
            {
                for (int kh = 0; kh < KernelSize; kh++)
                {
                    for (int kw = 0; kw < KernelSize; kw++)
                    {
                        parameters[index++] = _kernels[id, od, kh, kw];
                    }
                }
            }
        }
    
        // Copy bias parameters
        for (int od = 0; od < OutputDepth; od++)
        {
            parameters[index++] = _biases[od];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _kernels.Length + _biases.Length)
        {
            throw new ArgumentException($"Expected {_kernels.Length + _biases.Length} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set kernel parameters
        for (int id = 0; id < InputDepth; id++)
        {
            for (int od = 0; od < OutputDepth; od++)
            {
                for (int kh = 0; kh < KernelSize; kh++)
                {
                    for (int kw = 0; kw < KernelSize; kw++)
                    {
                        _kernels[id, od, kh, kw] = parameters[index++];
                    }
    }
}
        }
    
        // Set bias parameters
        for (int od = 0; od < OutputDepth; od++)
        {
            _biases[od] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _kernelsGradient = null;
        _biasesGradient = null;
    }
}