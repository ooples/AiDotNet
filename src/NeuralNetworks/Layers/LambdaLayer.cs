namespace AiDotNet.NeuralNetworks.Layers;

public class LambdaLayer<T> : LayerBase<T>
{
    private readonly Func<Tensor<T>, Tensor<T>> _forwardFunction;
    private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>>? _backwardFunction;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    public LambdaLayer(int[] inputShape, int[] outputShape, 
                       Func<Tensor<T>, Tensor<T>> forwardFunction, 
                       Func<Tensor<T>, Tensor<T>, Tensor<T>>? backwardFunction = null,
                       IActivationFunction<T>? activationFunction = null)
        : base(inputShape, outputShape, activationFunction ?? new ReLUActivation<T>())
    {
        _forwardFunction = forwardFunction;
        _backwardFunction = backwardFunction;
    }

    public LambdaLayer(int[] inputShape, int[] outputShape, 
                       Func<Tensor<T>, Tensor<T>> forwardFunction, 
                       Func<Tensor<T>, Tensor<T>, Tensor<T>>? backwardFunction = null,
                       IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShape, outputShape, vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _forwardFunction = forwardFunction;
        _backwardFunction = backwardFunction;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        var output = _forwardFunction(input);
        
        if (ScalarActivation != null)
        {
            output = ApplyActivation(output);
        }
        else if (VectorActivation != null)
        {
            output = ApplyVectorActivation(output);
        }

        _lastOutput = output;
        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        Tensor<T> gradient = outputGradient;

        if (ScalarActivation != null)
        {
            gradient = ApplyActivationDerivative(gradient, _lastOutput);
        }
        else if (VectorActivation != null)
        {
            gradient = ApplyVectorActivationDerivative(gradient, _lastOutput);
        }

        if (_backwardFunction != null)
        {
            gradient = _backwardFunction(_lastInput, gradient);
        }
        else
        {
            throw new InvalidOperationException("Backward function not provided for this Lambda layer.");
        }

        return gradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // Lambda layers typically don't have trainable parameters
    }

    private new Tensor<T> ApplyActivation(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = ScalarActivation!.Activate(input[i]);
        }

        return output;
    }

    private Tensor<T> ApplyVectorActivation(Tensor<T> input)
    {
        return VectorActivation!.Activate(input);
    }

    private new Tensor<T> ApplyActivationDerivative(Tensor<T> gradient, Tensor<T> output)
    {
        var result = new Tensor<T>(gradient.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            result[i] = NumOps.Multiply(gradient[i], ScalarActivation!.Derivative(output[i]));
        }

        return result;
    }

    private Tensor<T> ApplyVectorActivationDerivative(Tensor<T> gradient, Tensor<T> output)
    {
        return gradient.ElementwiseMultiply(VectorActivation!.Derivative(output));
    }
}