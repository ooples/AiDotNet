namespace AiDotNet.NeuralNetworks.Layers;

public class DepthwiseSeparableConvolutionalLayer<T> : LayerBase<T>
{
    private Tensor<T> _depthwiseKernels;
    private Tensor<T> _pointwiseKernels;
    private Vector<T> _biases;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastDepthwiseOutput;
    private Tensor<T>? _lastOutput;

    private Tensor<T>? _depthwiseKernelsGradient;
    private Tensor<T>? _pointwiseKernelsGradient;
    private Vector<T>? _biasesGradient;

    private readonly int _inputDepth;
    private readonly int _outputDepth;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;

    public override bool SupportsTraining => true;

    public DepthwiseSeparableConvolutionalLayer(int inputDepth, int outputDepth, int kernelSize, int inputHeight, int inputWidth,
                                                int stride = 1, int padding = 0, IActivationFunction<T>? activation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth),
               CalculateOutputShape(outputDepth, CalculateOutputDimension(inputHeight, kernelSize, stride, padding),
                   CalculateOutputDimension(inputWidth, kernelSize, stride, padding)),
               activation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        _depthwiseKernels = new Tensor<T>([inputDepth, 1, kernelSize, kernelSize]);
        _pointwiseKernels = new Tensor<T>([outputDepth, inputDepth, 1, 1]);
        _biases = new Vector<T>(outputDepth);

        InitializeParameters();
    }

    public DepthwiseSeparableConvolutionalLayer(int inputDepth, int outputDepth, int kernelSize, int inputHeight, int inputWidth,
                                                int stride = 1, int padding = 0, IVectorActivationFunction<T>? vectorActivation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth),
               CalculateOutputShape(outputDepth, CalculateOutputDimension(inputHeight, kernelSize, stride, padding),
                   CalculateOutputDimension(inputWidth, kernelSize, stride, padding)),
               vectorActivation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        _depthwiseKernels = new Tensor<T>([inputDepth, 1, kernelSize, kernelSize]);
        _pointwiseKernels = new Tensor<T>([outputDepth, inputDepth, 1, 1]);
        _biases = new Vector<T>(outputDepth);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T depthwiseScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_kernelSize * _kernelSize)));
        T pointwiseScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / _inputDepth));

        InitializeTensor(_depthwiseKernels, depthwiseScale);
        InitializeTensor(_pointwiseKernels, pointwiseScale);

        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Zero;
        }
    }

    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                for (int k = 0; k < tensor.Shape[2]; k++)
                {
                    for (int l = 0; l < tensor.Shape[3]; l++)
                    {
                        tensor[i, j, k, l] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                    }
                }
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int outputHeight = CalculateOutputDimension(input.Shape[1], _kernelSize, _stride, _padding);
        int outputWidth = CalculateOutputDimension(input.Shape[2], _kernelSize, _stride, _padding);

        // Depthwise convolution
        var depthwiseOutput = DepthwiseConvolution(input);
        _lastDepthwiseOutput = depthwiseOutput;

        // Pointwise convolution
        var pointwiseOutput = PointwiseConvolution(depthwiseOutput);

        // Add biases and apply activation
        var output = new Tensor<T>([batchSize, outputHeight, outputWidth, _outputDepth]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < outputHeight; h++)
            {
                for (int w = 0; w < outputWidth; w++)
                {
                    for (int c = 0; c < _outputDepth; c++)
                    {
                        T value = NumOps.Add(pointwiseOutput[b, h, w, c], _biases[c]);
                        output[b, h, w, c] = ApplyActivation(value);
                    }
                }
            }
        }

        _lastOutput = output;
        return output;
    }

    private Tensor<T> DepthwiseConvolution(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int inputHeight = input.Shape[1];
        int inputWidth = input.Shape[2];
        int outputHeight = CalculateOutputDimension(inputHeight, _kernelSize, _stride, _padding);
        int outputWidth = CalculateOutputDimension(inputWidth, _kernelSize, _stride, _padding);

        var output = new Tensor<T>([batchSize, outputHeight, outputWidth, _inputDepth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _inputDepth; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = NumOps.Zero;
                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                int ih = oh * _stride + kh - _padding;
                                int iw = ow * _stride + kw - _padding;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                {
                                    sum = NumOps.Add(sum, NumOps.Multiply(input[b, ih, iw, c], _depthwiseKernels[c, 0, kh, kw]));
                                }
                            }
                        }

                        output[b, oh, ow, c] = sum;
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> PointwiseConvolution(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int height = input.Shape[1];
        int width = input.Shape[2];

        var output = new Tensor<T>([batchSize, height, width, _outputDepth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int oc = 0; oc < _outputDepth; oc++)
                    {
                        T sum = NumOps.Zero;
                        for (int ic = 0; ic < _inputDepth; ic++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(input[b, h, w, ic], _pointwiseKernels[oc, ic, 0, 0]));
                        }
                        output[b, h, w, oc] = sum;
                    }
                }
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastDepthwiseOutput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = outputGradient.Shape[0];
        int outputHeight = outputGradient.Shape[1];
        int outputWidth = outputGradient.Shape[2];
        int inputHeight = _lastInput.Shape[1];
        int inputWidth = _lastInput.Shape[2];

        // Initialize gradients
        _depthwiseKernelsGradient = new Tensor<T>(_depthwiseKernels.Shape);
        _pointwiseKernelsGradient = new Tensor<T>(_pointwiseKernels.Shape);
        _biasesGradient = new Vector<T>(_biases.Length);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute gradients for biases and pointwise kernels
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < outputHeight; h++)
            {
                for (int w = 0; w < outputWidth; w++)
                {
                    for (int oc = 0; oc < _outputDepth; oc++)
                    {
                        T gradValue = ApplyActivationDerivative(outputGradient[b, h, w, oc], _lastOutput[b, h, w, oc]);
                        _biasesGradient[oc] = NumOps.Add(_biasesGradient[oc], gradValue);

                        for (int ic = 0; ic < _inputDepth; ic++)
                        {
                            _pointwiseKernelsGradient[oc, ic, 0, 0] = NumOps.Add(_pointwiseKernelsGradient[oc, ic, 0, 0],
                                NumOps.Multiply(gradValue, _lastDepthwiseOutput[b, h, w, ic]));
                        }
                    }
                }
            }
        }

        // Compute gradients for depthwise kernels and input
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _inputDepth; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T gradValue = NumOps.Zero;
                        for (int oc = 0; oc < _outputDepth; oc++)
                        {
                            gradValue = NumOps.Add(gradValue, NumOps.Multiply(
                                ApplyActivationDerivative(outputGradient[b, oh, ow, oc], _lastOutput[b, oh, ow, oc]),
                                _pointwiseKernels[oc, c, 0, 0]));
                        }

                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                int ih = oh * _stride + kh - _padding;
                                int iw = ow * _stride + kw - _padding;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                {
                                    _depthwiseKernelsGradient[c, 0, kh, kw] = NumOps.Add(_depthwiseKernelsGradient[c, 0, kh, kw],
                                        NumOps.Multiply(gradValue, _lastInput[b, ih, iw, c]));
                                    inputGradient[b, ih, iw, c] = NumOps.Add(inputGradient[b, ih, iw, c],
                                        NumOps.Multiply(gradValue, _depthwiseKernels[c, 0, kh, kw]));
                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    protected new T ApplyActivationDerivative(T gradient, T output)
    {
        if (UsingVectorActivation)
        {
            if (VectorActivation == null)
                throw new InvalidOperationException("Vector activation function is not set.");

            // Create a vector with a single element
            var outputVector = new Vector<T>([output]);
        
            // Get the derivative matrix (1x1 in this case)
            var derivativeMatrix = VectorActivation.Derivative(outputVector);
        
            // Multiply the gradient with the single element of the derivative matrix
            return NumOps.Multiply(gradient, derivativeMatrix[0, 0]);
        }
        else
        {
            if (ScalarActivation == null)
                throw new InvalidOperationException("Scalar activation function is not set.");

            // For scalar activation, we directly multiply the gradient with the derivative
            return NumOps.Multiply(gradient, ScalarActivation.Derivative(output));
        }
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_depthwiseKernelsGradient == null || _pointwiseKernelsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Update depthwise kernels
        for (int i = 0; i < _depthwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _depthwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _depthwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _depthwiseKernels.Shape[3]; l++)
                    {
                        _depthwiseKernels[i, j, k, l] = NumOps.Subtract(_depthwiseKernels[i, j, k, l],
                            NumOps.Multiply(learningRate, _depthwiseKernelsGradient[i, j, k, l]));
                    }
                }
            }
        }

        // Update pointwise kernels
        for (int i = 0; i < _pointwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _pointwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _pointwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _pointwiseKernels.Shape[3]; l++)
                    {
                        _pointwiseKernels[i, j, k, l] = NumOps.Subtract(_pointwiseKernels[i, j, k, l],
                            NumOps.Multiply(learningRate, _pointwiseKernelsGradient[i, j, k, l]));
                    }
                }
            }
        }

        // Update biases
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Subtract(_biases[i], NumOps.Multiply(learningRate, _biasesGradient[i]));
        }
    }

    private new static int[] CalculateInputShape(int inputDepth, int inputHeight, int inputWidth)
    {
        return [inputHeight, inputWidth, inputDepth];
    }

    private new static int[] CalculateOutputShape(int outputDepth, int outputHeight, int outputWidth)
    {
        return [outputHeight, outputWidth, outputDepth];
    }

    private static int CalculateOutputDimension(int inputDimension, int kernelSize, int stride, int padding)
    {
        return (inputDimension - kernelSize + 2 * padding) / stride + 1;
    }

    private T ApplyActivation(T value)
    {
        if (UsingVectorActivation)
        {
            return VectorActivation!.Activate(new Vector<T>([value]))[0];
        }
        else
        {
            return ScalarActivation!.Activate(value);
        }
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _depthwiseKernels.Length + _pointwiseKernels.Length + _biases.Length;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy depthwise kernel parameters
        for (int i = 0; i < _depthwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _depthwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _depthwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _depthwiseKernels.Shape[3]; l++)
                    {
                        parameters[index++] = _depthwiseKernels[i, j, k, l];
                    }
                }
            }
        }
    
        // Copy pointwise kernel parameters
        for (int i = 0; i < _pointwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _pointwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _pointwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _pointwiseKernels.Shape[3]; l++)
                    {
                        parameters[index++] = _pointwiseKernels[i, j, k, l];
                    }
                }
            }
        }
    
        // Copy bias parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            parameters[index++] = _biases[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _depthwiseKernels.Length + _pointwiseKernels.Length + _biases.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set depthwise kernel parameters
        for (int i = 0; i < _depthwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _depthwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _depthwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _depthwiseKernels.Shape[3]; l++)
                    {
                        _depthwiseKernels[i, j, k, l] = parameters[index++];
                    }
                }
            }
        }
    
        // Set pointwise kernel parameters
        for (int i = 0; i < _pointwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _pointwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _pointwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _pointwiseKernels.Shape[3]; l++)
                    {
                        _pointwiseKernels[i, j, k, l] = parameters[index++];
                    }
    }
}
        }
    
        // Set bias parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastDepthwiseOutput = null;
        _lastOutput = null;
        _depthwiseKernelsGradient = null;
        _pointwiseKernelsGradient = null;
        _biasesGradient = null;
    }
}