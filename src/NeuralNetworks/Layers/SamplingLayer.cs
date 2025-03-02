namespace AiDotNet.NeuralNetworks.Layers;

public class SamplingLayer<T> : LayerBase<T>
{
    public int PoolSize { get; private set; }
    public int Strides { get; private set; }
    public SamplingType SamplingType { get; private set; }

    public override bool SupportsTraining => true;

    private Func<Tensor<T>, Tensor<T>>? _forwardStrategy;
    private Func<Tensor<T>, Tensor<T>, Tensor<T>>? _backwardStrategy;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _maxIndices;

    public SamplingLayer(int[] inputShape, int poolSize, int strides, SamplingType samplingType) 
        : base(inputShape, CalculateOutputShape(inputShape, poolSize, strides))
    {
        PoolSize = poolSize;
        Strides = strides;
        SamplingType = samplingType;

        SetStrategies();
    }

    private static int[] CalculateOutputShape(int[] inputShape, int poolSize, int strides)
    {
        int outputHeight = (inputShape[1] - poolSize) / strides + 1;
        int outputWidth = (inputShape[2] - poolSize) / strides + 1;

        return [inputShape[0], outputHeight, outputWidth];
    }

    private void SetStrategies()
    {
        switch (SamplingType)
        {
            case SamplingType.Max:
                _forwardStrategy = MaxPoolForward;
                _backwardStrategy = MaxPoolBackward;
                break;
            case SamplingType.Average:
                _forwardStrategy = AveragePoolForward;
                _backwardStrategy = AveragePoolBackward;
                break;
            case SamplingType.L2Norm:
                _forwardStrategy = L2NormPoolForward;
                _backwardStrategy = L2NormPoolBackward;
                break;
            default:
                throw new NotImplementedException($"Sampling type {SamplingType} not implemented.");
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (_forwardStrategy == null)
        {
            throw new InvalidOperationException("Forward strategy is not set.");
        }

        _lastInput = input;
        return _forwardStrategy(input);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_backwardStrategy == null)
        {
            throw new InvalidOperationException("Backward strategy is not set.");
        }

        if (_lastInput == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }

        return _backwardStrategy(_lastInput, outputGradient);
    }

    private Tensor<T> MaxPoolForward(Tensor<T> input)
    {
        var output = new Tensor<T>(OutputShape);
        var maxIndices = new Tensor<T>(OutputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T maxVal = NumOps.MinValue;
                    int maxIndex = -1;
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                T val = input[b, ih, iw];
                                if (maxIndex == -1 || NumOps.GreaterThan(val, maxVal))
                                {
                                    maxVal = val;
                                    maxIndex = ph * PoolSize + pw;
                                }
                            }
                        }
                    }

                    output[b, h, w] = maxVal;
                    maxIndices[b, h, w] = NumOps.FromDouble(maxIndex);
                }
            }
        }

        _maxIndices = maxIndices;
        return output;
    }

    private Tensor<T> MaxPoolBackward(Tensor<T> input, Tensor<T> outputGradient)
    {
        if (_maxIndices == null)
        {
            throw new InvalidOperationException("MaxPoolBackward called before MaxPoolForward.");
        }

        var inputGradient = new Tensor<T>(InputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    var maxIndex = _maxIndices[b, h, w];
                    var ph = NumOps.Divide(maxIndex, NumOps.FromDouble(PoolSize));
                    var pw = MathHelper.Modulo(maxIndex, NumOps.FromDouble(PoolSize));
                    var ih = NumOps.Add(NumOps.Multiply(NumOps.FromDouble(h), NumOps.FromDouble(Strides)), ph);
                    var iw = NumOps.Add(NumOps.Multiply(NumOps.FromDouble(w), NumOps.FromDouble(Strides)), pw);
            
                    var ihInt = (int)Convert.ToDouble(ih);
                    var iwInt = (int)Convert.ToDouble(iw);
            
                    if (ihInt < InputShape[1] && iwInt < InputShape[2])
                    {
                        inputGradient[b, ihInt, iwInt] = NumOps.Add(inputGradient[b, ihInt, iwInt], outputGradient[b, h, w]);
                    }
                }
            }
        }

        return inputGradient;
    }

    private Tensor<T> AveragePoolForward(Tensor<T> input)
    {
        var output = new Tensor<T>(OutputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T sum = NumOps.Zero;
                    int count = 0;
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                sum = NumOps.Add(sum, input[b, ih, iw]);
                                count++;
                            }
                        }
                    }

                    output[b, h, w] = NumOps.Divide(sum, NumOps.FromDouble(count));
                }
            }
        }

        return output;
    }

    private Tensor<T> AveragePoolBackward(Tensor<T> input, Tensor<T> outputGradient)
    {
        var inputGradient = new Tensor<T>(InputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T gradientValue = NumOps.Divide(outputGradient[b, h, w], NumOps.FromDouble(PoolSize * PoolSize));
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                inputGradient[b, ih, iw] = NumOps.Add(inputGradient[b, ih, iw], gradientValue);
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    private Tensor<T> L2NormPoolForward(Tensor<T> input)
    {
        var output = new Tensor<T>(OutputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T sumSquares = NumOps.Zero;
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                T val = input[b, ih, iw];
                                sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(val, val));
                            }
                        }
                    }

                    output[b, h, w] = NumOps.Sqrt(sumSquares);
                }
            }
        }

        return output;
    }

    private Tensor<T> L2NormPoolBackward(Tensor<T> input, Tensor<T> outputGradient)
    {
        if (_lastInput == null)
        {
            throw new InvalidOperationException("L2NormPoolBackward called before Forward pass. _lastInput is null.");
        }

        var inputGradient = new Tensor<T>(InputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T l2Norm = _lastInput[b, h, w];
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                T val = input[b, ih, iw];
                                T gradientValue = NumOps.Multiply(outputGradient[b, h, w], 
                                    NumOps.Divide(val, l2Norm));
                                inputGradient[b, ih, iw] = NumOps.Add(inputGradient[b, ih, iw], gradientValue);
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
        // Sampling layers typically don't have trainable parameters
    }

    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);

        writer.Write(PoolSize);
        writer.Write(Strides);
        writer.Write((int)SamplingType);
    }

    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);

        PoolSize = reader.ReadInt32();
        Strides = reader.ReadInt32();
        SamplingType = (SamplingType)reader.ReadInt32();

        SetStrategies();
    }

    public override IEnumerable<ActivationType> GetActivationTypes()
    {
        // Sampling layers don't have activation functions
        return [];
    }

    public override Vector<T> GetParameters()
    {
        // Sampling layers have no trainable parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _maxIndices = null;
    }
}