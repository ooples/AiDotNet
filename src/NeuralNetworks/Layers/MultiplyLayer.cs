namespace AiDotNet.NeuralNetworks.Layers;

public class MultiplyLayer<T> : LayerBase<T>
{
    private Tensor<T>[]? _lastInputs;
    private Tensor<T>? _lastOutput;

    public override bool SupportsTraining => true;

    public MultiplyLayer(int[][] inputShapes, IActivationFunction<T>? activationFunction = null)
        : base(inputShapes, inputShapes[0], activationFunction ?? new LinearActivation<T>())
    {
        ValidateInputShapes(inputShapes);
    }

    public MultiplyLayer(int[][] inputShapes, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShapes, inputShapes[0], vectorActivationFunction ?? new LinearActivation<T>())
    {
        ValidateInputShapes(inputShapes);
    }

    private static void ValidateInputShapes(int[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            throw new ArgumentException("MultiplyLayer requires at least two inputs.");
        }

        for (int i = 1; i < inputShapes.Length; i++)
        {
            if (!inputShapes[i].SequenceEqual(inputShapes[0]))
            {
                throw new ArgumentException("All input shapes must be identical for MultiplyLayer.");
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new NotSupportedException("MultiplyLayer requires multiple inputs. Use Forward(params Tensor<T>[] inputs) instead.");
    }

    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length < 2)
        {
            throw new ArgumentException("MultiplyLayer requires at least two inputs.");
        }

        _lastInputs = inputs;

        var result = inputs[0].Copy();
        for (int i = 1; i < inputs.Length; i++)
        {
            result = result.ElementwiseMultiply(inputs[i]);
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

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        var inputGradients = new Tensor<T>[_lastInputs.Length];
        for (int i = 0; i < _lastInputs.Length; i++)
        {
            inputGradients[i] = activationGradient.Copy();
            for (int j = 0; j < _lastInputs.Length; j++)
            {
                if (i != j)
                {
                    inputGradients[i] = inputGradients[i].ElementwiseMultiply(_lastInputs[j]);
                }
            }
        }

        return Tensor<T>.Stack(inputGradients);
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    public override Vector<T> GetParameters()
    {
        // MultiplyLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInputs = null;
        _lastOutput = null;
    }
}