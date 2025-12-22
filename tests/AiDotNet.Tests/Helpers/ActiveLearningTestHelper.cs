using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Test helper utilities for Active Learning strategy tests.
/// </summary>
public static class ActiveLearningTestHelper
{
    /// <summary>
    /// Creates a mock model for testing active learning strategies.
    /// </summary>
    public static MockFullModel<double> CreateMockModel(int numClasses = 3)
    {
        return new MockFullModel<double>(numClasses);
    }

    /// <summary>
    /// Creates test data representing an unlabeled pool.
    /// </summary>
    /// <param name="numSamples">Number of samples in the pool.</param>
    /// <param name="numFeatures">Number of features per sample.</param>
    /// <returns>A tensor representing the unlabeled pool.</returns>
    public static Tensor<double> CreateUnlabeledPool(int numSamples, int numFeatures)
    {
        var random = new Random(42);
        var data = new Vector<double>(numSamples * numFeatures);

        for (int i = 0; i < numSamples * numFeatures; i++)
        {
            data[i] = random.NextDouble();
        }

        return new Tensor<double>([numSamples, numFeatures], data);
    }

    /// <summary>
    /// Creates test data with known patterns for deterministic testing.
    /// </summary>
    public static Tensor<double> CreateDeterministicPool(int numSamples, int numFeatures)
    {
        var data = new Vector<double>(numSamples * numFeatures);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                data[i * numFeatures + j] = (i + 1.0) / numSamples * (j + 1.0) / numFeatures;
            }
        }

        return new Tensor<double>([numSamples, numFeatures], data);
    }
}

/// <summary>
/// Mock implementation of IFullModel for testing active learning strategies.
/// </summary>
public class MockFullModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    private readonly int _numClasses;
    private readonly int _numParameters;
    private readonly Random _random;
    private readonly Vector<T> _parameters;
    private readonly INumericOperations<T> _ops;
    private List<int> _activeFeatureIndices;

    public MockFullModel(int numClasses = 3, int numParameters = 100, int seed = 42)
    {
        _numClasses = numClasses;
        _numParameters = numParameters;
        _random = new Random(seed);
        _ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _parameters = new Vector<T>(numParameters);
        _activeFeatureIndices = Enumerable.Range(0, 10).ToList();

        for (int i = 0; i < numParameters; i++)
        {
            _parameters[i] = _ops.FromDouble(_random.NextDouble() * 2 - 1);
        }
    }

    // IFullModel<T> - DefaultLossFunction
    public ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    // IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>>
    /// <summary>
    /// Predicts class probabilities for each sample.
    /// Returns logits that when softmaxed give probabilities.
    /// </summary>
    public Tensor<T> Predict(Tensor<T> input)
    {
        var numSamples = input.Shape[0];
        var outputSize = numSamples * _numClasses;
        var data = new Vector<T>(outputSize);

        for (int i = 0; i < numSamples; i++)
        {
            // Generate logits for each class
            for (int c = 0; c < _numClasses; c++)
            {
                // Create varying uncertainty based on sample index
                double baseLogit = _random.NextDouble() * 2 - 1;
                // Make earlier samples more uncertain (closer logits)
                double uncertainty = 1.0 - (double)i / numSamples;
                baseLogit *= (1.0 - uncertainty * 0.8);
                data[i * _numClasses + c] = _ops.FromDouble(baseLogit);
            }
        }

        return new Tensor<T>([numSamples, _numClasses], data);
    }

    public void Train(Tensor<T> inputs, Tensor<T> targets)
    {
        // Mock training - no-op
    }

    public ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>();
    }

    // IModelSerializer
    public byte[] Serialize()
    {
        return [];
    }

    public void Deserialize(byte[] data)
    {
        // No-op for mock
    }

    public void SaveModel(string filePath)
    {
        // No-op for mock
    }

    public void LoadModel(string filePath)
    {
        // No-op for mock
    }

    // ICheckpointableModel
    public void SaveState(Stream stream)
    {
        // No-op for mock
    }

    public void LoadState(Stream stream)
    {
        // No-op for mock
    }

    // IParameterizable<T, Tensor<T>, Tensor<T>>
    public Vector<T> GetParameters()
    {
        return _parameters;
    }

    public void SetParameters(Vector<T> parameters)
    {
        for (int i = 0; i < Math.Min(parameters.Length, _parameters.Length); i++)
        {
            _parameters[i] = parameters[i];
        }
    }

    public int ParameterCount => _numParameters;

    public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new MockFullModel<T>(_numClasses, _numParameters);
        newModel.SetParameters(parameters);
        return newModel;
    }

    // IFeatureAware
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        return _activeFeatureIndices;
    }

    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        _activeFeatureIndices = featureIndices.ToList();
    }

    public bool IsFeatureUsed(int featureIndex)
    {
        return _activeFeatureIndices.Contains(featureIndex);
    }

    // IFeatureImportance<T>
    public Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        for (int i = 0; i < 10; i++)
        {
            importance[$"feature_{i}"] = _ops.FromDouble(0.1);
        }
        return importance;
    }

    // ICloneable<IFullModel<T, Tensor<T>, Tensor<T>>>
    public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = new MockFullModel<T>(_numClasses, _numParameters);
        copy.SetParameters(_parameters);
        return copy;
    }

    public IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        return DeepCopy();
    }

    // IGradientComputable<T, Tensor<T>, Tensor<T>>
    public Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Simple mock - return random gradients
        var grads = new Vector<T>(_numParameters);
        for (int i = 0; i < _numParameters; i++)
        {
            grads[i] = _ops.FromDouble(_random.NextDouble() * 0.1 - 0.05);
        }
        return grads;
    }

    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        for (int i = 0; i < Math.Min(gradients.Length, _parameters.Length); i++)
        {
            var update = _ops.Multiply(gradients[i], learningRate);
            _parameters[i] = _ops.Subtract(_parameters[i], update);
        }
    }

    // IJitCompilable<T>
    public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("Mock model does not support JIT compilation");
    }

    public bool SupportsJitCompilation => false;
}
