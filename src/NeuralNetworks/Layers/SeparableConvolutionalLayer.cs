namespace AiDotNet.NeuralNetworks.Layers;

public class SeparableConvolutionalLayer<T> : LayerBase<T>
{
    private Tensor<T> _depthwiseKernels;
    private Tensor<T> _pointwiseKernels;
    private Vector<T> _biases;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    private Tensor<T>? _depthwiseKernelsGradient;
    private Tensor<T>? _pointwiseKernelsGradient;
    private Vector<T>? _biasesGradient;

    private readonly int _inputDepth;
    private readonly int _outputDepth;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;

    private Tensor<T>? _depthwiseKernelsVelocity;
    private Tensor<T>? _pointwiseKernelsVelocity;
    private Vector<T>? _biasesVelocity;

    public SeparableConvolutionalLayer(int[] inputShape, int outputDepth, int kernelSize, int stride = 1, int padding = 0, IActivationFunction<T>? scalarActivation = null)
        : base(inputShape, CalculateOutputShape(inputShape, outputDepth, kernelSize, stride, padding), scalarActivation ?? new IdentityActivation<T>())
    {
        _inputDepth = inputShape[3];
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        // Initialize depthwise kernels
        _depthwiseKernels = new Tensor<T>([_inputDepth, _kernelSize, _kernelSize, 1]);
        
        // Initialize pointwise kernels
        _pointwiseKernels = new Tensor<T>([_inputDepth, 1, 1, _outputDepth]);
        
        // Initialize biases
        _biases = new Vector<T>(_outputDepth);


        InitializeParameters();
    }

    public SeparableConvolutionalLayer(int[] inputShape, int outputDepth, int kernelSize, int stride = 1, int padding = 0, IVectorActivationFunction<T>? vectorActivation = null)
        : base(inputShape, CalculateOutputShape(inputShape, outputDepth, kernelSize, stride, padding), vectorActivation ?? new IdentityActivation<T>())
    {
        _inputDepth = inputShape[3];
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        // Initialize depthwise kernels
        _depthwiseKernels = new Tensor<T>([_inputDepth, _kernelSize, _kernelSize, 1]);
        
        // Initialize pointwise kernels
        _pointwiseKernels = new Tensor<T>([_inputDepth, 1, 1, _outputDepth]);
        
        // Initialize biases
        _biases = new Vector<T>(_outputDepth);


        InitializeParameters();
    }

    private static int[] CalculateOutputShape(int[] inputShape, int outputDepth, int kernelSize, int stride, int padding)
    {
        int outputHeight = (inputShape[1] - kernelSize + 2 * padding) / stride + 1;
        int outputWidth = (inputShape[2] - kernelSize + 2 * padding) / stride + 1;

        return [inputShape[0], outputHeight, outputWidth, outputDepth];
    }

    private void InitializeParameters()
    {
        // Use He initialization for kernels
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_kernelSize * _kernelSize * _inputDepth)));
        InitializeTensor(_depthwiseKernels, scale);
        InitializeTensor(_pointwiseKernels, scale);

        // Initialize biases to zero
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Zero;
        }
    }

    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int inputHeight = input.Shape[1];
        int inputWidth = input.Shape[2];
        int outputHeight = OutputShape[1];
        int outputWidth = OutputShape[2];

        var depthwiseOutput = new Tensor<T>([batchSize, outputHeight, outputWidth, _inputDepth]);
        var output = new Tensor<T>(OutputShape);

        // Depthwise convolution
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int d = 0; d < _inputDepth; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int ki = 0; ki < _kernelSize; ki++)
                        {
                            for (int kj = 0; kj < _kernelSize; kj++)
                            {
                                int ii = i * _stride + ki - _padding;
                                int jj = j * _stride + kj - _padding;
                                if (ii >= 0 && ii < inputHeight && jj >= 0 && jj < inputWidth)
                                {
                                    sum = NumOps.Add(sum, NumOps.Multiply(input[b, ii, jj, d], _depthwiseKernels[d, ki, kj, 0]));
                                }
                            }
                        }

                        depthwiseOutput[b, i, j, d] = sum;
                    }
                }
            }
        }

        // Pointwise convolution
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int od = 0; od < _outputDepth; od++)
                    {
                        T sum = _biases[od];
                        for (int id = 0; id < _inputDepth; id++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(depthwiseOutput[b, i, j, id], _pointwiseKernels[id, 0, 0, od]));
                        }

                        output[b, i, j, od] = sum;
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

        Tensor<T> activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);
        outputGradient = Tensor<T>.ElementwiseMultiply(outputGradient, activationGradient);

        int batchSize = _lastInput.Shape[0];
        int inputHeight = _lastInput.Shape[1];
        int inputWidth = _lastInput.Shape[2];
        int outputHeight = OutputShape[1];
        int outputWidth = OutputShape[2];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _depthwiseKernelsGradient = new Tensor<T>(_depthwiseKernels.Shape);
        _pointwiseKernelsGradient = new Tensor<T>(_pointwiseKernels.Shape);
        _biasesGradient = new Vector<T>(_outputDepth);

        var depthwiseOutputGradient = new Tensor<T>([batchSize, outputHeight, outputWidth, _inputDepth]);

        // Compute gradients for pointwise convolution
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int od = 0; od < _outputDepth; od++)
                    {
                        T outputGrad = outputGradient[b, i, j, od];
                        _biasesGradient[od] = NumOps.Add(_biasesGradient[od], outputGrad);

                        for (int id = 0; id < _inputDepth; id++)
                        {
                            T depthwiseOutput = _lastOutput[b, i, j, id];
                            _pointwiseKernelsGradient[id, 0, 0, od] = NumOps.Add(_pointwiseKernelsGradient[id, 0, 0, od], NumOps.Multiply(outputGrad, depthwiseOutput));
                            depthwiseOutputGradient[b, i, j, id] = NumOps.Add(depthwiseOutputGradient[b, i, j, id], NumOps.Multiply(outputGrad, _pointwiseKernels[id, 0, 0, od]));
                        }
                    }
                }
            }
        }

        // Compute gradients for depthwise convolution
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int d = 0; d < _inputDepth; d++)
                    {
                        T depthwiseGrad = depthwiseOutputGradient[b, i, j, d];
                        for (int ki = 0; ki < _kernelSize; ki++)
                        {
                            for (int kj = 0; kj < _kernelSize; kj++)
                            {
                                int ii = i * _stride + ki - _padding;
                                int jj = j * _stride + kj - _padding;
                                if (ii >= 0 && ii < inputHeight && jj >= 0 && jj < inputWidth)
                                {
                                    T inputValue = _lastInput[b, ii, jj, d];
                                    _depthwiseKernelsGradient[d, ki, kj, 0] = NumOps.Add(_depthwiseKernelsGradient[d, ki, kj, 0], NumOps.Multiply(depthwiseGrad, inputValue));
                                    inputGradient[b, ii, jj, d] = NumOps.Add(inputGradient[b, ii, jj, d], NumOps.Multiply(depthwiseGrad, _depthwiseKernels[d, ki, kj, 0]));
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
        if (_depthwiseKernelsGradient == null || _pointwiseKernelsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T momentum = NumOps.FromDouble(0.9); // Momentum factor
        T l2RegularizationFactor = NumOps.FromDouble(0.0001); // L2 regularization factor

        // Initialize velocity tensors if they don't exist
        if (_depthwiseKernelsVelocity == null)
            _depthwiseKernelsVelocity = new Tensor<T>(_depthwiseKernels.Shape);
        if (_pointwiseKernelsVelocity == null)
            _pointwiseKernelsVelocity = new Tensor<T>(_pointwiseKernels.Shape);
        if (_biasesVelocity == null)
            _biasesVelocity = new Vector<T>(_biases.Length);

        // Update depthwise kernels
        for (int i = 0; i < _depthwiseKernels.Length; i++)
        {
            T gradient = _depthwiseKernelsGradient[i];
            T l2Regularization = NumOps.Multiply(l2RegularizationFactor, _depthwiseKernels[i]);
            _depthwiseKernelsVelocity[i] = NumOps.Add(
                NumOps.Multiply(momentum, _depthwiseKernelsVelocity[i]),
                NumOps.Multiply(learningRate, NumOps.Add(gradient, l2Regularization))
            );
            _depthwiseKernels[i] = NumOps.Subtract(_depthwiseKernels[i], _depthwiseKernelsVelocity[i]);
        }

        // Update pointwise kernels
        for (int i = 0; i < _pointwiseKernels.Length; i++)
        {
            T gradient = _pointwiseKernelsGradient[i];
            T l2Regularization = NumOps.Multiply(l2RegularizationFactor, _pointwiseKernels[i]);
            _pointwiseKernelsVelocity[i] = NumOps.Add(
                NumOps.Multiply(momentum, _pointwiseKernelsVelocity[i]),
                NumOps.Multiply(learningRate, NumOps.Add(gradient, l2Regularization))
            );
            _pointwiseKernels[i] = NumOps.Subtract(_pointwiseKernels[i], _pointwiseKernelsVelocity[i]);
        }

        // Update biases
        for (int i = 0; i < _biases.Length; i++)
        {
            T gradient = _biasesGradient[i];
            _biasesVelocity[i] = NumOps.Add(
                NumOps.Multiply(momentum, _biasesVelocity[i]),
                NumOps.Multiply(learningRate, gradient)
            );
            _biases[i] = NumOps.Subtract(_biases[i], _biasesVelocity[i]);
        }

        // Clear gradients
        _depthwiseKernelsGradient = null;
        _pointwiseKernelsGradient = null;
        _biasesGradient = null;
    }
}