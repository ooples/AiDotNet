namespace AiDotNet.Models.Options;

public class MultilayerPerceptronOptions<T> : NonLinearRegressionOptions
{
    public List<int> LayerSizes { get; set; } = new List<int> { 1, 10, 1 };  // Default: 1 input, 1 hidden layer with 10 neurons, 1 output
    public int MaxEpochs { get; set; } = 1000;
    public int BatchSize { get; set; } = 32;
    public double LearningRate { get; set; } = 0.001;
    public bool Verbose { get; set; } = false;

    public Func<T, T> HiddenActivationFunction { get; set; } = NeuralNetworkHelper<T>.ReLU;
    public Func<T, T> HiddenActivationFunctionDerivative { get; set; } = NeuralNetworkHelper<T>.ReLUDerivative;
    public Func<T, T> OutputActivationFunction { get; set; } = NeuralNetworkHelper<T>.Linear;
    public Func<T, T> OutputActivationFunctionDerivative { get; set; } = NeuralNetworkHelper<T>.LinearDerivative;
    public Func<Vector<T>, Vector<T>, T> LossFunction { get; set; } = NeuralNetworkHelper<T>.MeanSquaredError;
    public Func<Vector<T>, Vector<T>, Vector<T>> LossFunctionDerivative { get; set; } = NeuralNetworkHelper<T>.MeanSquaredErrorDerivative;

    public IOptimizer<T> Optimizer { get; set; } = new AdamOptimizer<T>(new AdamOptimizerOptions
    {
        LearningRate = 0.001,
        Beta1 = 0.9,
        Beta2 = 0.999,
        Epsilon = 1e-8
    });
}