namespace AiDotNet.NeuralNetworks.Layers;

public class InputLayer<T> : LayerBase<T>
{
    public override bool SupportsTraining => false;

    public InputLayer(int inputSize)
        : base([inputSize], [inputSize], new IdentityActivation<T>() as IActivationFunction<T>)
    {
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        return input;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return outputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // Input layer has no parameters to update
    }

    public override Vector<T> GetParameters()
    {
        // InputLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // InputLayer has no state to reset
    }
}