namespace AiDotNet.NeuralNetworks.Layers;

public class DilatedConvolutionalLayer<T> : LayerBase<T>
{
    private readonly int _inputDepth;
    private readonly int _outputDepth;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;
    private readonly int _dilation;

    private Tensor<T> _kernels;
    private Vector<T> _biases;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _kernelGradients;
    private Vector<T>? _biasGradients;

    public DilatedConvolutionalLayer(int inputDepth, int outputDepth, int kernelSize, int inputHeight, int inputWidth, 
                                     int dilation, int stride = 1, int padding = 0, 
                                     IActivationFunction<T>? activation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth), 
               CalculateOutputShape(outputDepth, CalculateOutputDimension(inputHeight, kernelSize, stride, padding, dilation), 
                   CalculateOutputDimension(inputWidth, kernelSize, stride, padding, dilation)), 
               activation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _dilation = dilation;

        _kernels = new Tensor<T>([_outputDepth, _inputDepth, _kernelSize, _kernelSize]);
        _biases = new Vector<T>(_outputDepth);

        InitializeWeights();
    }

    public DilatedConvolutionalLayer(int inputDepth, int outputDepth, int kernelSize, int inputHeight, int inputWidth, 
                                     int dilation, int stride = 1, int padding = 0, 
                                     IVectorActivationFunction<T>? vectorActivation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth), 
               CalculateOutputShape(outputDepth, CalculateOutputDimension(inputHeight, kernelSize, stride, padding, dilation), 
                   CalculateOutputDimension(inputWidth, kernelSize, stride, padding, dilation)), 
               vectorActivation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _dilation = dilation;

        _kernels = new Tensor<T>([_outputDepth, _inputDepth, _kernelSize, _kernelSize]);
        _biases = new Vector<T>(_outputDepth);

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        // Xavier initialization
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputDepth * _kernelSize * _kernelSize + _outputDepth)));

        for (int i = 0; i < _kernels.Shape[0]; i++)
        {
            for (int j = 0; j < _kernels.Shape[1]; j++)
            {
                for (int k = 0; k < _kernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _kernels.Shape[3]; l++)
                    {
                        _kernels[i, j, k, l] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                    }
                }
            }
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
        int inputHeight = input.Shape[1];
        int inputWidth = input.Shape[2];
        int outputHeight = CalculateOutputDimension(inputHeight, _kernelSize, _stride, _padding, _dilation);
        int outputWidth = CalculateOutputDimension(inputWidth, _kernelSize, _stride, _padding, _dilation);

        var output = new Tensor<T>([batchSize, outputHeight, outputWidth, _outputDepth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    for (int od = 0; od < _outputDepth; od++)
                    {
                        T sum = _biases[od];

                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                int ih = oh * _stride + kh * _dilation - _padding;
                                int iw = ow * _stride + kw * _dilation - _padding;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                {
                                    for (int id = 0; id < _inputDepth; id++)
                                    {
                                        sum = NumOps.Add(sum, NumOps.Multiply(input[b, ih, iw, id], _kernels[od, id, kh, kw]));
                                    }
                                }
                            }
                        }

                        output[b, oh, ow, od] = sum;
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
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _lastInput.Shape[0];
        int inputHeight = _lastInput.Shape[1];
        int inputWidth = _lastInput.Shape[2];
        int outputHeight = outputGradient.Shape[1];
        int outputWidth = outputGradient.Shape[2];

        // Initialize gradients
        var kernelGradients = new Tensor<T>(_kernels.Shape);
        var biasGradients = new Vector<T>(_biases.Length);
        var inputGradients = new Tensor<T>(_lastInput.Shape);

        // Compute gradients
        for (int b = 0; b < batchSize; b++)
        {
            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    for (int od = 0; od < _outputDepth; od++)
                    {
                        T outputGrad = outputGradient[b, oh, ow, od];
                        biasGradients[od] = NumOps.Add(biasGradients[od], outputGrad);

                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                int ih = oh * _stride + kh * _dilation - _padding;
                                int iw = ow * _stride + kw * _dilation - _padding;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                {
                                    for (int id = 0; id < _inputDepth; id++)
                                    {
                                        T inputVal = _lastInput[b, ih, iw, id];
                                        kernelGradients[od, id, kh, kw] = NumOps.Add(kernelGradients[od, id, kh, kw], NumOps.Multiply(outputGrad, inputVal));
                                        inputGradients[b, ih, iw, id] = NumOps.Add(inputGradients[b, ih, iw, id], NumOps.Multiply(outputGrad, _kernels[od, id, kh, kw]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Store gradients for parameter update
        _kernelGradients = kernelGradients;
        _biasGradients = biasGradients;

        return inputGradients;
    }

    private new static int[][] CalculateInputShape(int inputDepth, int inputHeight, int inputWidth)
    {
        return [[inputHeight, inputWidth, inputDepth]];
    }

    private new static int[] CalculateOutputShape(int outputDepth, int outputHeight, int outputWidth)
    {
        return [outputHeight, outputWidth, outputDepth];
    }

    private static int CalculateOutputDimension(int inputDim, int kernelSize, int stride, int padding, int dilation)
    {
        return (inputDim + 2 * padding - dilation * (kernelSize - 1) - 1) / stride + 1;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_kernelGradients == null || _biasGradients == null)
        {
            throw new InvalidOperationException("UpdateParameters called before Backward.");
        }

        for (int i = 0; i < _kernels.Shape[0]; i++)
        {
            for (int j = 0; j < _kernels.Shape[1]; j++)
            {
                for (int k = 0; k < _kernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _kernels.Shape[3]; l++)
                    {
                        _kernels[i, j, k, l] = NumOps.Subtract(_kernels[i, j, k, l], NumOps.Multiply(learningRate, _kernelGradients[i, j, k, l]));
                    }
                }
            }
        }

        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Subtract(_biases[i], NumOps.Multiply(learningRate, _biasGradients[i]));
        }

        // Reset gradients
        _kernelGradients = null;
        _biasGradients = null;
    }
}