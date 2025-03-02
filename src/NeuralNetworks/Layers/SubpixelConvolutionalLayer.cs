namespace AiDotNet.NeuralNetworks.Layers;

public class SubpixelConvolutionalLayer<T> : LayerBase<T>
{
    private readonly int _inputDepth;
    private readonly int _outputDepth;
    private readonly int _upscaleFactor;
    private readonly int _kernelSize;

    private Tensor<T> _kernels;
    private Vector<T> _biases;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _kernelGradients;
    private Vector<T>? _biasGradients;

    private Tensor<T>? _kernelMomentum;
    private Vector<T>? _biasMomentum;
    private readonly T _momentumFactor;
    private readonly T _weightDecay;

    public override bool SupportsTraining => true;

    public SubpixelConvolutionalLayer(int inputDepth, int outputDepth, int upscaleFactor, int kernelSize, int inputHeight, int inputWidth,
                                      IActivationFunction<T>? activation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth),
               CalculateOutputShape(outputDepth, inputHeight * upscaleFactor, inputWidth * upscaleFactor),
               activation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _upscaleFactor = upscaleFactor;
        _kernelSize = kernelSize;
        _momentumFactor = NumOps.FromDouble(0.9);
        _weightDecay = NumOps.FromDouble(0.0001);

        _kernels = new Tensor<T>([_outputDepth * _upscaleFactor * _upscaleFactor, _inputDepth, _kernelSize, _kernelSize]);
        _biases = new Vector<T>(_outputDepth * _upscaleFactor * _upscaleFactor);

        InitializeWeights();
    }

    public SubpixelConvolutionalLayer(int inputDepth, int outputDepth, int upscaleFactor, int kernelSize, int inputHeight, int inputWidth,
                                      IVectorActivationFunction<T>? vectorActivation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth),
               CalculateOutputShape(outputDepth, inputHeight * upscaleFactor, inputWidth * upscaleFactor),
               vectorActivation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _upscaleFactor = upscaleFactor;
        _kernelSize = kernelSize;
        _momentumFactor = NumOps.FromDouble(0.9);
        _weightDecay = NumOps.FromDouble(0.0001);

        _kernels = new Tensor<T>([_outputDepth * _upscaleFactor * _upscaleFactor, _inputDepth, _kernelSize, _kernelSize]);
        _biases = new Vector<T>(_outputDepth * _upscaleFactor * _upscaleFactor);

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        // Xavier initialization
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputDepth * _kernelSize * _kernelSize + _outputDepth * _upscaleFactor * _upscaleFactor)));

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
        int outputHeight = inputHeight * _upscaleFactor;
        int outputWidth = inputWidth * _upscaleFactor;

        var convOutput = new Tensor<T>([batchSize, inputHeight, inputWidth, _outputDepth * _upscaleFactor * _upscaleFactor]);

        // Perform convolution
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < inputHeight; h++)
            {
                for (int w = 0; w < inputWidth; w++)
                {
                    for (int od = 0; od < _outputDepth * _upscaleFactor * _upscaleFactor; od++)
                    {
                        T sum = _biases[od];

                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                int ih = h + kh - _kernelSize / 2;
                                int iw = w + kw - _kernelSize / 2;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                {
                                    for (int id = 0; id < _inputDepth; id++)
                                    {
                                        sum = NumOps.Add(sum, NumOps.Multiply(input[b, ih, iw, id], _kernels[od, id, kh, kw]));
                                    }
                                }
                            }
                        }

                        convOutput[b, h, w, od] = sum;
                    }
                }
            }
        }

        // Perform pixel shuffle
        var output = new Tensor<T>([batchSize, outputHeight, outputWidth, _outputDepth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < inputHeight; h++)
            {
                for (int w = 0; w < inputWidth; w++)
                {
                    for (int c = 0; c < _outputDepth * _upscaleFactor * _upscaleFactor; c++)
                    {
                        int outputChannel = c % _outputDepth;
                        int offsetY = (c / _outputDepth) / _upscaleFactor;
                        int offsetX = (c / _outputDepth) % _upscaleFactor;
                        int outputY = h * _upscaleFactor + offsetY;
                        int outputX = w * _upscaleFactor + offsetX;

                        output[b, outputY, outputX, outputChannel] = convOutput[b, h, w, c];
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

        int batchSize = _lastInput.Shape[0];
        int inputHeight = _lastInput.Shape[1];
        int inputWidth = _lastInput.Shape[2];
        int outputHeight = inputHeight * _upscaleFactor;
        int outputWidth = inputWidth * _upscaleFactor;

        // Step 1: Compute gradient with respect to the activation function
        Tensor<T> activationGradient = ComputeActivationGradient(outputGradient, _lastOutput);

        // Step 2: Reverse pixel shuffle
        var convOutputGradient = new Tensor<T>([batchSize, inputHeight, inputWidth, _outputDepth * _upscaleFactor * _upscaleFactor]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < inputHeight; h++)
            {
                for (int w = 0; w < inputWidth; w++)
                {
                    for (int c = 0; c < _outputDepth * _upscaleFactor * _upscaleFactor; c++)
                    {
                        int outputChannel = c % _outputDepth;
                        int offsetY = (c / _outputDepth) / _upscaleFactor;
                        int offsetX = (c / _outputDepth) % _upscaleFactor;
                        int outputY = h * _upscaleFactor + offsetY;
                        int outputX = w * _upscaleFactor + offsetX;

                        convOutputGradient[b, h, w, c] = activationGradient[b, outputY, outputX, outputChannel];
                    }
                }
            }
        }

        // Step 3: Initialize gradients
        _kernelGradients = new Tensor<T>(_kernels.Shape);
        _biasGradients = new Vector<T>(_biases.Length);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Step 4: Compute gradients
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < inputHeight; h++)
            {
                for (int w = 0; w < inputWidth; w++)
                {
                    for (int od = 0; od < _outputDepth * _upscaleFactor * _upscaleFactor; od++)
                    {
                        T gradOutput = convOutputGradient[b, h, w, od];
                        _biasGradients[od] = NumOps.Add(_biasGradients[od], gradOutput);

                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                int ih = h + kh - _kernelSize / 2;
                                int iw = w + kw - _kernelSize / 2;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                {
                                    for (int id = 0; id < _inputDepth; id++)
                                    {
                                        T inputValue = _lastInput[b, ih, iw, id];
                                        _kernelGradients[od, id, kh, kw] = NumOps.Add(_kernelGradients[od, id, kh, kw], NumOps.Multiply(gradOutput, inputValue));
                                        inputGradient[b, ih, iw, id] = NumOps.Add(inputGradient[b, ih, iw, id], NumOps.Multiply(gradOutput, _kernels[od, id, kh, kw]));
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

    private Tensor<T> ComputeActivationGradient(Tensor<T> outputGradient, Tensor<T> lastOutput)
    {
        if (UsingVectorActivation)
        {
            return VectorActivation!.Derivative(lastOutput).PointwiseMultiply(outputGradient);
        }
        else
        {
            var result = new Tensor<T>(outputGradient.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
            {
                result[i] = NumOps.Multiply(ScalarActivation!.Derivative(lastOutput[i]), outputGradient[i]);
            }

            return result;
        }
    }

    private new static int[][] CalculateInputShape(int inputDepth, int inputHeight, int inputWidth)
    {
        return [[inputHeight, inputWidth, inputDepth]];
    }

    private new static int[] CalculateOutputShape(int outputDepth, int outputHeight, int outputWidth)
    {
        return [outputHeight, outputWidth, outputDepth];
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_kernelGradients == null || _biasGradients == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Initialize momentum if not already done
        _kernelMomentum ??= new Tensor<T>(_kernels.Shape);
        _biasMomentum ??= new Vector<T>(_biases.Length);

        // Update kernels
        for (int i = 0; i < _kernels.Length; i++)
        {
            // Compute momentum
            _kernelMomentum[i] = NumOps.Add(
                NumOps.Multiply(_momentumFactor, _kernelMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, _momentumFactor), _kernelGradients[i])
            );

            // Update kernel with momentum and weight decay
            _kernels[i] = NumOps.Subtract(
                NumOps.Subtract(_kernels[i], NumOps.Multiply(learningRate, _kernelMomentum[i])),
                NumOps.Multiply(NumOps.Multiply(learningRate, _weightDecay), _kernels[i])
            );
        }

        // Update biases
        for (int i = 0; i < _biases.Length; i++)
        {
            // Compute momentum
            _biasMomentum[i] = NumOps.Add(
                NumOps.Multiply(_momentumFactor, _biasMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, _momentumFactor), _biasGradients[i])
            );

            // Update bias with momentum (no weight decay for biases)
            _biases[i] = NumOps.Subtract(_biases[i], NumOps.Multiply(learningRate, _biasMomentum[i]));
        }

        // Clear gradients after update
        _kernelGradients = null;
        _biasGradients = null;
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int kernelParamCount = _kernels.Length;
        int biasParamCount = _biases.Length;
        int totalParamCount = kernelParamCount + biasParamCount;
    
        // Create a vector to hold all parameters
        var parameters = new Vector<T>(totalParamCount);
    
        // Copy kernel parameters
        int index = 0;
        for (int i = 0; i < _kernels.Length; i++)
        {
            parameters[index++] = _kernels[i];
        }
    
        // Copy bias parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            parameters[index++] = _biases[i];
        }
    
        return parameters;
    }

    public override void ResetState()
    {
        // Clear cached values
        _lastInput = null;
        _lastOutput = null;
        _kernelGradients = null;
        _biasGradients = null;
    
        // Reset momentum if using momentum
        _kernelMomentum = null;
        _biasMomentum = null;
    
        // Reinitialize weights
        InitializeWeights();
    }
}