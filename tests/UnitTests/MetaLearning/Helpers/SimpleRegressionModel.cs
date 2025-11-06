using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.MetaLearning.Helpers;

/// <summary>
/// Simple polynomial regression model for sine wave approximation.
/// Uses a polynomial basis to approximate sine functions.
/// </summary>
public class SimpleRegressionModel : IFullModel<double, Tensor<double>, Tensor<double>>
{
    private Vector<double> _weights;
    private readonly int _polynomialDegree;
    private readonly double _learningRate;

    public SimpleRegressionModel(int polynomialDegree = 5, double learningRate = 0.01)
    {
        _polynomialDegree = polynomialDegree;
        _learningRate = learningRate;
        // Initialize weights with small random values
        _weights = new Vector<double>(polynomialDegree + 1);
        var random = new Random(42);
        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = (random.NextDouble() - 0.5) * 0.1;
        }
    }

    public Vector<double> GetParameters() => _weights.Copy();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters.Length != _weights.Length)
        {
            throw new ArgumentException($"Parameter count mismatch: expected {_weights.Length}, got {parameters.Length}");
        }
        _weights = parameters.Copy();
    }

    public int ParameterCount => _weights.Length;

    public IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters)
    {
        var newModel = new SimpleRegressionModel(_polynomialDegree, _learningRate);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Computes polynomial features [1, x, x^2, ..., x^degree] for a single input value.
    /// </summary>
    private Vector<double> ComputePolynomialFeatures(double x)
    {
        var features = new Vector<double>(_polynomialDegree + 1);
        features[0] = 1.0; // Bias term
        for (int i = 1; i <= _polynomialDegree; i++)
        {
            features[i] = Math.Pow(x, i);
        }
        return features;
    }

    public void Train(Tensor<double> input, Tensor<double> expectedOutput)
    {
        // Perform one gradient descent step
        // Input shape: [num_samples, 1]
        // Output shape: [num_samples, 1]

        int numSamples = input.Shape[0];
        var gradient = new Vector<double>(_weights.Length);

        // Compute gradient: ∇L = (1/n) Σ (y_pred - y_true) * features
        for (int i = 0; i < numSamples; i++)
        {
            double x = input[new[] { i, 0 }];
            double y_true = expectedOutput[new[] { i, 0 }];

            var features = ComputePolynomialFeatures(x);
            double y_pred = features.DotProduct(_weights);
            double error = y_pred - y_true;

            // Accumulate gradient
            for (int j = 0; j < _weights.Length; j++)
            {
                gradient[j] += error * features[j];
            }
        }

        // Average gradient
        for (int j = 0; j < _weights.Length; j++)
        {
            gradient[j] /= numSamples;
        }

        // Update weights: w = w - α * ∇L
        for (int j = 0; j < _weights.Length; j++)
        {
            _weights[j] -= _learningRate * gradient[j];
        }
    }

    public Tensor<double> Predict(Tensor<double> input)
    {
        // Input shape: [num_samples, 1]
        // Output shape: [num_samples, 1]
        int numSamples = input.Shape[0];
        var output = new Tensor<double>(new[] { numSamples, 1 });

        for (int i = 0; i < numSamples; i++)
        {
            double x = input[new[] { i, 0 }];
            var features = ComputePolynomialFeatures(x);
            double prediction = features.DotProduct(_weights);
            output[new[] { i, 0 }] = prediction;
        }

        return output;
    }

    public ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double>();
    public void Save(string filePath) { }
    public void Load(string filePath) { }

    public IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
    {
        var copy = new SimpleRegressionModel(_polynomialDegree, _learningRate);
        copy.SetParameters(_weights);
        return copy;
    }

    public IFullModel<double, Tensor<double>, Tensor<double>> Clone() => DeepCopy();

    public int InputFeatureCount => 1;
    public int OutputFeatureCount => 1;
    public string[] FeatureNames { get; set; } = new[] { "x" };
    public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();
}
