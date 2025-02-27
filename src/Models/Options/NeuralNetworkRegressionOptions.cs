

namespace AiDotNet.Models.Options;

public class NeuralNetworkRegressionOptions<T> : NonLinearRegressionOptions
{
    public List<int> LayerSizes { get; set; } = [1, 10, 1];  // Default: 1 input, 1 hidden layer with 10 neurons, 1 output
    public int Epochs { get; set; } = 1000;
    public int BatchSize { get; set; } = 32;
    public double LearningRate { get; set; } = 0.01;
    public Func<T, T> HiddenActivationFunction { get; set; } = NeuralNetworkHelper<T>.ReLU;
    public Func<T, T> HiddenActivationFunctionDerivative { get; set; } = NeuralNetworkHelper<T>.ReLUDerivative;
    public Func<T, T> OutputActivationFunction { get; set; } = (x) => x; // Linear for regression
    public Func<T, T> OutputActivationFunctionDerivative { get; set; } = (x) => MathHelper.GetNumericOperations<T>().One;
    public Func<Vector<T>, Vector<T>, T> LossFunction { get; set; } = NeuralNetworkHelper<T>.MeanSquaredError;
    public Func<Vector<T>, Vector<T>, Vector<T>> LossFunctionDerivative { get; set; } = NeuralNetworkHelper<T>.MeanSquaredErrorDerivative;
    public IOptimizer<T>? Optimizer { get; set; }
}