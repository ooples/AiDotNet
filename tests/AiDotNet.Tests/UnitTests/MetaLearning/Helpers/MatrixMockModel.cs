using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

namespace AiDotNet.Tests.UnitTests.MetaLearning.Helpers;

/// <summary>
/// Mock model for testing meta-learning algorithms with Matrix/Vector inputs/outputs.
/// Implements ICloneable to support meta-learning model cloning.
/// </summary>
public class MatrixMockModel : IFullModel<double, Matrix<double>, Vector<double>>, ICloneable
{
    private Vector<double> _parameters;
    private readonly int _inputFeatureCount;
    private readonly int _outputFeatureCount;
    public int TrainCallCount { get; private set; }
    public int PredictCallCount { get; private set; }

    public MatrixMockModel(int inputFeatureCount, int outputFeatureCount)
    {
        _inputFeatureCount = inputFeatureCount;
        _outputFeatureCount = outputFeatureCount;
        int paramCount = inputFeatureCount * outputFeatureCount + outputFeatureCount;
        _parameters = new Vector<double>(paramCount);
        // Initialize with small values
        for (int i = 0; i < paramCount; i++)
        {
            _parameters[i] = 0.01 * i;
        }
        TrainCallCount = 0;
        PredictCallCount = 0;
    }

    public Vector<double> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
    {
        var newModel = new MatrixMockModel(_inputFeatureCount, _outputFeatureCount);
        newModel.SetParameters(parameters);
        return newModel;
    }

    public void Train(Matrix<double> input, Vector<double> expectedOutput)
    {
        TrainCallCount++;
        // Simple update to simulate training
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] += 0.001;
        }
    }

    public Vector<double> Predict(Matrix<double> input)
    {
        PredictCallCount++;
        // Return a vector with length matching output feature count
        int batchSize = input.Rows;
        return new Vector<double>(batchSize);
    }

    public ModelMetadata<double> GetModelMetadata()
    {
        return new ModelMetadata<double>
        {
            Name = "MatrixMockModel",
            FeatureCount = _inputFeatureCount,
            Complexity = _parameters.Length
        };
    }

    // IModelSerializer implementation
    public byte[] Serialize() => Array.Empty<byte>();
    public void Deserialize(byte[] data) { }
    public void SaveModel(string filePath) { }
    public void LoadModel(string filePath) { }

    // ICheckpointableModel implementation
    public void SaveState(Stream stream) { }
    public void LoadState(Stream stream) { }

    // IFeatureAware implementation
    public int InputFeatureCount => _inputFeatureCount;
    public int OutputFeatureCount => _outputFeatureCount;
    public string[] FeatureNames { get; set; } = Array.Empty<string>();
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputFeatureCount);
    public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _inputFeatureCount;

    // IFeatureImportance implementation
    public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();

    // ICloneable implementation
    public IFullModel<double, Matrix<double>, Vector<double>> DeepCopy()
    {
        var copy = new MatrixMockModel(_inputFeatureCount, _outputFeatureCount);
        copy.SetParameters(_parameters);
        return copy;
    }

    IFullModel<double, Matrix<double>, Vector<double>> ICloneable<IFullModel<double, Matrix<double>, Vector<double>>>.Clone()
    {
        return DeepCopy();
    }

    // System.ICloneable explicit implementation for meta-learning support
    object ICloneable.Clone()
    {
        return DeepCopy();
    }

    // IGradientComputable implementation
    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    public Vector<double> ComputeGradients(Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null)
    {
        // Return non-zero gradients for testing
        var gradients = new Vector<double>(ParameterCount);
        for (int i = 0; i < ParameterCount; i++)
        {
            gradients[i] = 0.1 * (i + 1);
        }
        return gradients;
    }

    public void ApplyGradients(Vector<double> gradients, double learningRate)
    {
        for (int i = 0; i < Math.Min(gradients.Length, _parameters.Length); i++)
        {
            _parameters[i] -= learningRate * gradients[i];
        }
    }

    // IJitCompilable implementation
    public bool SupportsJitCompilation => false;

    public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation not supported in mock model");
    }
}
