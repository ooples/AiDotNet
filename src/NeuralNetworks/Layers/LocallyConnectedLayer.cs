namespace AiDotNet.NeuralNetworks.Layers;

public class LocallyConnectedLayer<T> : LayerBase<T>
{
    private Tensor<T> _weights;
    private Vector<T> _biases;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _weightGradients;
    private Vector<T>? _biasGradients;

    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _inputChannels;
    private readonly int _outputHeight;
    private readonly int _outputWidth;
    private readonly int _outputChannels;
    private readonly int _kernelSize;
    private readonly int _stride;

    public LocallyConnectedLayer(
        int inputHeight, 
        int inputWidth, 
        int inputChannels, 
        int outputChannels, 
        int kernelSize, 
        int stride, 
        IActivationFunction<T>? activationFunction = null)
        : base(
            [inputHeight, inputWidth, inputChannels], 
            [
                (inputHeight - kernelSize) / stride + 1, 
                (inputWidth - kernelSize) / stride + 1, 
                outputChannels
            ], 
            activationFunction ?? new ReLUActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _outputHeight = (inputHeight - kernelSize) / stride + 1;
        _outputWidth = (inputWidth - kernelSize) / stride + 1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;

        // Initialize weights and biases
        _weights = new Tensor<T>([_outputHeight, _outputWidth, _outputChannels, _kernelSize, _kernelSize, _inputChannels]);
        _biases = new Vector<T>(_outputChannels);

        InitializeParameters();
    }

    public LocallyConnectedLayer(
        int inputHeight, 
        int inputWidth, 
        int inputChannels, 
        int outputChannels, 
        int kernelSize, 
        int stride, 
        IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(
            [inputHeight, inputWidth, inputChannels], 
            [
                (inputHeight - kernelSize) / stride + 1, 
                (inputWidth - kernelSize) / stride + 1, 
                outputChannels
            ], 
            vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _outputHeight = (inputHeight - kernelSize) / stride + 1;
        _outputWidth = (inputWidth - kernelSize) / stride + 1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;

        // Initialize weights and biases
        _weights = new Tensor<T>([_outputHeight, _outputWidth, _outputChannels, _kernelSize, _kernelSize, _inputChannels]);
        _biases = new Vector<T>(_outputChannels);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Xavier initialization for weights
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_kernelSize * _kernelSize * _inputChannels + _outputChannels)));
        
        for (int h = 0; h < _outputHeight; h++)
        {
            for (int w = 0; w < _outputWidth; w++)
            {
                for (int oc = 0; oc < _outputChannels; oc++)
                {
                    for (int kh = 0; kh < _kernelSize; kh++)
                    {
                        for (int kw = 0; kw < _kernelSize; kw++)
                        {
                            for (int ic = 0; ic < _inputChannels; ic++)
                            {
                                _weights[h, w, oc, kh, kw, ic] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                            }
                        }
                    }
                }
            }
        }

        // Initialize biases to zero
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Zero;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        var output = new Tensor<T>([batchSize, _outputHeight, _outputWidth, _outputChannels]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < _outputHeight; h++)
            {
                for (int w = 0; w < _outputWidth; w++)
                {
                    for (int oc = 0; oc < _outputChannels; oc++)
                    {
                        T sum = NumOps.Zero;
                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                for (int ic = 0; ic < _inputChannels; ic++)
                                {
                                    int ih = h * _stride + kh;
                                    int iw = w * _stride + kw;
                                    sum = NumOps.Add(sum, NumOps.Multiply(input[b, ih, iw, ic], _weights[h, w, oc, kh, kw, ic]));
                                }
                            }
                        }
                        sum = NumOps.Add(sum, _biases[oc]);
                        output[b, h, w, oc] = sum;
                    }
                }
            }
        }

        // Apply activation function
        return ApplyActivation(output);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _weightGradients = new Tensor<T>(_weights.Shape);
        _biasGradients = new Vector<T>(_biases.Length);

        // Apply activation derivative
        outputGradient = ApplyActivationDerivative(_lastInput, outputGradient);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < _outputHeight; h++)
            {
                for (int w = 0; w < _outputWidth; w++)
                {
                    for (int oc = 0; oc < _outputChannels; oc++)
                    {
                        T gradOutput = outputGradient[b, h, w, oc];
                        _biasGradients[oc] = NumOps.Add(_biasGradients[oc], gradOutput);

                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                for (int ic = 0; ic < _inputChannels; ic++)
                                {
                                    int ih = h * _stride + kh;
                                    int iw = w * _stride + kw;
                                    T inputValue = _lastInput[b, ih, iw, ic];
                                    _weightGradients[h, w, oc, kh, kw, ic] = NumOps.Add(_weightGradients[h, w, oc, kh, kw, ic], NumOps.Multiply(gradOutput, inputValue));
                                    inputGradient[b, ih, iw, ic] = NumOps.Add(inputGradient[b, ih, iw, ic], NumOps.Multiply(gradOutput, _weights[h, w, oc, kh, kw, ic]));
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
        if (_weightGradients == null || _biasGradients == null)
        {
            throw new InvalidOperationException("UpdateParameters called before Backward. Gradients are null.");
        }

        // Update weights
        for (int h = 0; h < _outputHeight; h++)
        {
            for (int w = 0; w < _outputWidth; w++)
            {
                for (int oc = 0; oc < _outputChannels; oc++)
                {
                    for (int kh = 0; kh < _kernelSize; kh++)
                    {
                        for (int kw = 0; kw < _kernelSize; kw++)
                        {
                            for (int ic = 0; ic < _inputChannels; ic++)
                            {
                                T update = NumOps.Multiply(learningRate, _weightGradients[h, w, oc, kh, kw, ic]);
                                _weights[h, w, oc, kh, kw, ic] = NumOps.Subtract(_weights[h, w, oc, kh, kw, ic], update);
                            }
                        }
                    }
                }
            }
        }

        // Update biases
        for (int oc = 0; oc < _outputChannels; oc++)
        {
            T update = NumOps.Multiply(learningRate, _biasGradients[oc]);
            _biases[oc] = NumOps.Subtract(_biases[oc], update);
        }
    }
}