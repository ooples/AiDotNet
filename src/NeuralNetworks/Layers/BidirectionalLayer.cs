namespace AiDotNet.NeuralNetworks.Layers;

public class BidirectionalLayer<T> : LayerBase<T>
{
    private readonly LayerBase<T> _forwardLayer;
    private readonly LayerBase<T> _backwardLayer;
    private readonly bool _mergeMode;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastForwardOutput;
    private Tensor<T>? _lastBackwardOutput;

    public override bool SupportsTraining => _forwardLayer.SupportsTraining || _backwardLayer.SupportsTraining;

    public BidirectionalLayer(
        LayerBase<T> innerLayer, 
        bool mergeMode = true, 
        IActivationFunction<T>? activationFunction = null)
        : base(innerLayer.GetInputShape(), CalculateOutputShape(innerLayer.GetOutputShape(), mergeMode), activationFunction ?? new ReLUActivation<T>())
    {
        _forwardLayer = innerLayer;
        _backwardLayer = innerLayer.Copy();
        _mergeMode = mergeMode;
    }

    public BidirectionalLayer(
        LayerBase<T> innerLayer, 
        bool mergeMode = true, 
        IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(innerLayer.GetInputShape(), CalculateOutputShape(innerLayer.GetOutputShape(), mergeMode), vectorActivationFunction ?? new LinearActivation<T>())
    {
        _forwardLayer = innerLayer;
        _backwardLayer = innerLayer.Copy();
        _mergeMode = mergeMode;
    }

    private static int[] CalculateOutputShape(int[] innerOutputShape, bool mergeMode)
    {
        if (mergeMode)
        {
            return innerOutputShape;
        }
        else
        {
            var newShape = new int[innerOutputShape.Length + 1];
            newShape[0] = 2;
            Array.Copy(innerOutputShape, 0, newShape, 1, innerOutputShape.Length);

            return newShape;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Forward pass
        var forwardInput = input;
        _lastForwardOutput = _forwardLayer.Forward(forwardInput);

        // Backward pass (reverse the input sequence)
        var backwardInput = ReverseSequence(input);
        _lastBackwardOutput = _backwardLayer.Forward(backwardInput);

        // Merge outputs
        return MergeOutputs(_lastForwardOutput, _lastBackwardOutput);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        Tensor<T> forwardGradient, backwardGradient;

        if (_mergeMode)
        {
            forwardGradient = outputGradient;
            backwardGradient = outputGradient;
        }
        else
        {
            forwardGradient = outputGradient.Slice(0);
            backwardGradient = outputGradient.Slice(1);
        }

        var forwardInputGradient = _forwardLayer.Backward(forwardGradient);
        var backwardInputGradient = _backwardLayer.Backward(backwardGradient);

        // Reverse the backward gradient
        backwardInputGradient = ReverseSequence(backwardInputGradient);

        // Sum the gradients
        return forwardInputGradient.Add(backwardInputGradient);
    }

    public override void UpdateParameters(T learningRate)
    {
        _forwardLayer.UpdateParameters(learningRate);
        _backwardLayer.UpdateParameters(learningRate);
    }

    private static Tensor<T> ReverseSequence(Tensor<T> input)
    {
        var reversed = new Tensor<T>(input.Shape);
        int timeSteps = input.Shape[1];

        for (int i = 0; i < timeSteps; i++)
        {
            reversed.SetSlice(i, input.Slice(timeSteps - 1 - i));
        }

        return reversed;
    }

    private Tensor<T> MergeOutputs(Tensor<T> forward, Tensor<T> backward)
    {
        if (_mergeMode)
        {
            return forward.Add(backward);
        }
        else
        {
            return Tensor<T>.Stack([forward, backward], 0);
        }
    }

    public override Vector<T> GetParameters()
    {
        // Combine parameters from both forward and backward layers
        var forwardParams = _forwardLayer.GetParameters();
        var backwardParams = _backwardLayer.GetParameters();
        
        var combinedParams = new Vector<T>(forwardParams.Length + backwardParams.Length);
        
        // Copy forward parameters
        for (int i = 0; i < forwardParams.Length; i++)
        {
            combinedParams[i] = forwardParams[i];
        }
        
        // Copy backward parameters
        for (int i = 0; i < backwardParams.Length; i++)
        {
            combinedParams[i + forwardParams.Length] = backwardParams[i];
        }
        
        return combinedParams;
    }
    
    public override void SetParameters(Vector<T> parameters)
    {
        var forwardParams = _forwardLayer.GetParameters();
        var backwardParams = _backwardLayer.GetParameters();
        
        if (parameters.Length != forwardParams.Length + backwardParams.Length)
            throw new ArgumentException($"Expected {forwardParams.Length + backwardParams.Length} parameters, but got {parameters.Length}");
        
        // Extract and set forward parameters
        var newForwardParams = new Vector<T>(forwardParams.Length);
        for (int i = 0; i < forwardParams.Length; i++)
        {
            newForwardParams[i] = parameters[i];
        }
        
        // Extract and set backward parameters
        var newBackwardParams = new Vector<T>(backwardParams.Length);
        for (int i = 0; i < backwardParams.Length; i++)
        {
            newBackwardParams[i] = parameters[i + forwardParams.Length];
        }
        
        _forwardLayer.SetParameters(newForwardParams);
        _backwardLayer.SetParameters(newBackwardParams);
    }
    
    public override void ResetState()
    {
        _lastInput = null;
        _lastForwardOutput = null;
        _lastBackwardOutput = null;
        
        _forwardLayer.ResetState();
        _backwardLayer.ResetState();
    }
}