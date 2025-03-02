namespace AiDotNet.NeuralNetworks.Layers;

public class PrimaryCapsuleLayer<T> : LayerBase<T>
{
    private Matrix<T> _convWeights;
    private Vector<T> _convBias;
    private Matrix<T>? _convWeightsGradient;
    private Vector<T>? _convBiasGradient;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    private readonly int _inputChannels;
    private readonly int _capsuleChannels;
    private readonly int _capsuleDimension;
    private readonly int _kernelSize;
    private readonly int _stride;

    public override bool SupportsTraining => true;

    public PrimaryCapsuleLayer(int inputChannels, int capsuleChannels, int capsuleDimension, int kernelSize, int stride, IActivationFunction<T>? scalarActivation = null)
        : base([inputChannels], [capsuleChannels * capsuleDimension], scalarActivation ?? new SquashActivation<T>())
    {
        _inputChannels = inputChannels;
        _capsuleChannels = capsuleChannels;
        _capsuleDimension = capsuleDimension;
        _kernelSize = kernelSize;
        _stride = stride;

        _convWeights = new Matrix<T>(capsuleChannels * capsuleDimension, inputChannels * kernelSize * kernelSize);
        _convBias = new Vector<T>(capsuleChannels * capsuleDimension);

        InitializeParameters();
    }

    public PrimaryCapsuleLayer(int inputChannels, int capsuleChannels, int capsuleDimension, int kernelSize, int stride, IVectorActivationFunction<T>? vectorActivation = null)
        : base([inputChannels], [capsuleChannels * capsuleDimension], vectorActivation ?? new SquashActivation<T>())
    {
        _inputChannels = inputChannels;
        _capsuleChannels = capsuleChannels;
        _capsuleDimension = capsuleDimension;
        _kernelSize = kernelSize;
        _stride = stride;

        _convWeights = new Matrix<T>(capsuleChannels * capsuleDimension, inputChannels * kernelSize * kernelSize);
        _convBias = new Vector<T>(capsuleChannels * capsuleDimension);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_convWeights.Rows + _convWeights.Columns)));
        for (int i = 0; i < _convWeights.Rows; i++)
        {
            for (int j = 0; j < _convWeights.Columns; j++)
            {
                _convWeights[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }

        for (int i = 0; i < _convBias.Length; i++)
        {
            _convBias[i] = NumOps.Zero;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int inputHeight = input.Shape[1];
        int inputWidth = input.Shape[2];

        int outputHeight = (inputHeight - _kernelSize) / _stride + 1;
        int outputWidth = (inputWidth - _kernelSize) / _stride + 1;

        var output = new Tensor<T>([batchSize, outputHeight, outputWidth, _capsuleChannels, _capsuleDimension]);

        // Perform convolution and reshape
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    var patch = ExtractPatch(input, b, i * _stride, j * _stride);
                    var capsule = _convWeights.Multiply(patch).Add(_convBias);
                    for (int c = 0; c < _capsuleChannels; c++)
                    {
                        for (int d = 0; d < _capsuleDimension; d++)
                        {
                            output[b, i, j, c, d] = capsule[c * _capsuleDimension + d];
                        }
                    }
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    private Vector<T> ExtractPatch(Tensor<T> input, int batch, int startY, int startX)
    {
        var patch = new Vector<T>(_inputChannels * _kernelSize * _kernelSize);
        int index = 0;
        for (int c = 0; c < _inputChannels; c++)
        {
            for (int i = 0; i < _kernelSize; i++)
            {
                for (int j = 0; j < _kernelSize; j++)
                {
                    patch[index++] = input[batch, startY + i, startX + j, c];
                }
            }
        }

        return patch;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int inputHeight = _lastInput.Shape[1];
        int inputWidth = _lastInput.Shape[2];
        int outputHeight = activationGradient.Shape[1];
        int outputWidth = activationGradient.Shape[2];

        _convWeightsGradient = new Matrix<T>(_convWeights.Rows, _convWeights.Columns);
        _convBiasGradient = new Vector<T>(_convBias.Length);

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    var patch = ExtractPatch(_lastInput, b, i * _stride, j * _stride);
                    var capsuleGradient = new Vector<T>(_capsuleChannels * _capsuleDimension);

                    for (int c = 0; c < _capsuleChannels; c++)
                    {
                        for (int d = 0; d < _capsuleDimension; d++)
                        {
                            capsuleGradient[c * _capsuleDimension + d] = activationGradient[b, i, j, c, d];
                        }
                    }

                    _convWeightsGradient = _convWeightsGradient.Add(capsuleGradient.OuterProduct(patch));
                    _convBiasGradient = _convBiasGradient.Add(capsuleGradient);

                    var patchGradient = _convWeights.Transpose().Multiply(capsuleGradient);
                    int index = 0;
                    for (int c = 0; c < _inputChannels; c++)
                    {
                        for (int ki = 0; ki < _kernelSize; ki++)
                        {
                            for (int kj = 0; kj < _kernelSize; kj++)
                            {
                                inputGradient[b, i * _stride + ki, j * _stride + kj, c] = NumOps.Add(
                                    inputGradient[b, i * _stride + ki, j * _stride + kj, c],
                                    patchGradient[index++]
                                );
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
        if (_convWeightsGradient == null || _convBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _convWeights = _convWeights.Subtract(_convWeightsGradient.Multiply(learningRate));
        _convBias = _convBias.Subtract(_convBiasGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _convWeights.Rows * _convWeights.Columns + _convBias.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy convolution weights
        for (int i = 0; i < _convWeights.Rows; i++)
        {
            for (int j = 0; j < _convWeights.Columns; j++)
            {
                parameters[index++] = _convWeights[i, j];
            }
        }
    
        // Copy convolution bias
        for (int i = 0; i < _convBias.Length; i++)
        {
            parameters[index++] = _convBias[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _convWeights.Rows * _convWeights.Columns + _convBias.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set convolution weights
        for (int i = 0; i < _convWeights.Rows; i++)
        {
            for (int j = 0; j < _convWeights.Columns; j++)
            {
                _convWeights[i, j] = parameters[index++];
            }
        }
    
        // Set convolution bias
        for (int i = 0; i < _convBias.Length; i++)
        {
            _convBias[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _convWeightsGradient = null;
        _convBiasGradient = null;
    }
}