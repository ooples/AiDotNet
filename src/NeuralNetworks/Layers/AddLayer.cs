namespace AiDotNet.NeuralNetworks.Layers;

public class AddLayer<T> : LayerBase<T>
{
    private Tensor<T>[]? _lastInputs;
    private Tensor<T>? _lastOutput;

    public override bool SupportsTraining => false;

    public AddLayer(int[][] inputShapes, IActivationFunction<T>? activationFunction = null)
        : base(inputShapes, inputShapes[0], activationFunction ?? new IdentityActivation<T>())
    {
        ValidateInputShapes(inputShapes);
    }

    public AddLayer(int[][] inputShapes, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShapes, inputShapes[0], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        ValidateInputShapes(inputShapes);
    }

    private static void ValidateInputShapes(int[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            throw new ArgumentException("AddLayer requires at least two input tensors.", nameof(inputShapes));
        }

        var firstShape = inputShapes[0];
        for (int i = 1; i < inputShapes.Length; i++)
        {
            if (!firstShape.SequenceEqual(inputShapes[i]))
            {
                throw new ArgumentException("All input shapes must be identical for AddLayer.", nameof(inputShapes));
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new NotSupportedException("AddLayer requires multiple inputs. Use Forward(params Tensor<T>[] inputs) instead.");
    }

    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length < 2)
        {
            throw new ArgumentException("AddLayer requires at least two input tensors.", nameof(inputs));
        }

        _lastInputs = inputs;

        var result = inputs[0].Copy();
        for (int i = 1; i < inputs.Length; i++)
        {
            result = result.Add(inputs[i]);
        }

        _lastOutput = ApplyActivation(result);
        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInputs == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        Tensor<T> gradientWithActivation;
        if (UsingVectorActivation && VectorActivation != null)
        {
            gradientWithActivation = VectorActivation.Derivative(_lastOutput).Multiply(outputGradient);
        }
        else if (ScalarActivation != null)
        {
            gradientWithActivation = _lastOutput.Transform((x, i) => NumOps.Multiply(ScalarActivation.Derivative(x), outputGradient[i]));
        }
        else
        {
            gradientWithActivation = outputGradient;
        }

        var inputGradients = new Tensor<T>[_lastInputs.Length];
        for (int i = 0; i < _lastInputs.Length; i++)
        {
            inputGradients[i] = gradientWithActivation.Copy();
        }

        return inputGradients[0];
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    public override Vector<T> GetParameters()
    {
        // Add layers don't have parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        _lastInputs = null;
        _lastOutput = null;
    }
}