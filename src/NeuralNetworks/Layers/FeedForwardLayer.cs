namespace AiDotNet.NeuralNetworks.Layers;

public class FeedForwardLayer<T> : LayerBase<T>
{
    private Tensor<T> Weights { get; set; }
    private Tensor<T> Biases { get; set; }
    private Tensor<T> Input { get; set; }
    private Tensor<T> Output { get; set; }
    private Tensor<T> WeightsGradient { get; set; }
    private Tensor<T> BiasesGradient { get; set; }

    public override bool SupportsTraining => true;

    public FeedForwardLayer(int inputSize, int outputSize, IActivationFunction<T> activationFunction)
        : base(new[] { inputSize }, new[] { outputSize })
    {
        Weights = Tensor<T>.CreateRandom(new[] { inputSize, outputSize });
        Biases = Tensor<T>.CreateDefault(new[] { 1, outputSize }, NumOps.Zero);
        WeightsGradient = Tensor<T>.Empty();
        BiasesGradient = Tensor<T>.Empty();
        Input = Tensor<T>.Empty();
        Output = Tensor<T>.Empty();
    }

    public FeedForwardLayer(int inputSize, int outputSize, IVectorActivationFunction<T> activationFunction)
        : base(new[] { inputSize }, new[] { outputSize })
    {
        Weights = Tensor<T>.CreateRandom(new[] { inputSize, outputSize });
        Biases = Tensor<T>.CreateDefault(new[] { 1, outputSize }, NumOps.Zero);
        WeightsGradient = Tensor<T>.Empty();
        BiasesGradient = Tensor<T>.Empty();
        Input = Tensor<T>.Empty();
        Output = Tensor<T>.Empty();
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        Input = input;
        var linearOutput = Input.MatrixMultiply(Weights).Add(Biases);
        Output = ApplyActivation(linearOutput);

        return Output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var activationGradient = ApplyActivationDerivative(outputGradient, Output);

        var inputGradient = activationGradient.MatrixMultiply(Weights.Transpose(new[] { 1, 0 }));
        var weightsGradient = Input.Transpose(new[] { 1, 0 }).MatrixMultiply(activationGradient);
        var biasesGradient = activationGradient.Sum(new[] { 0 });

        WeightsGradient = weightsGradient;
        BiasesGradient = biasesGradient;

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        Weights = Weights.Subtract(WeightsGradient.Multiply(learningRate));
        Biases = Biases.Subtract(BiasesGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = Weights.Length + Biases.Length;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy weights parameters
        for (int i = 0; i < Weights.Shape[0]; i++)
        {
            for (int j = 0; j < Weights.Shape[1]; j++)
            {
                parameters[index++] = Weights[i, j];
            }
        }
    
        // Copy biases parameters
        for (int j = 0; j < Biases.Shape[1]; j++)
        {
            parameters[index++] = Biases[0, j];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != Weights.Length + Biases.Length)
        {
            throw new ArgumentException($"Expected {Weights.Length + Biases.Length} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set weights parameters
        for (int i = 0; i < Weights.Shape[0]; i++)
        {
            for (int j = 0; j < Weights.Shape[1]; j++)
            {
                Weights[i, j] = parameters[index++];
            }
        }
    
        // Set biases parameters
        for (int j = 0; j < Biases.Shape[1]; j++)
        {
            Biases[0, j] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        Input = Tensor<T>.Empty();
        Output = Tensor<T>.Empty();
        WeightsGradient = Tensor<T>.Empty();
        BiasesGradient = Tensor<T>.Empty();
    }
}