using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.MetaLearning.Helpers;

/// <summary>
/// Simple mock model for testing that tracks parameter updates.
/// Implements ISecondOrderGradientComputable for full MAML testing.
/// Implements ICloneable to support meta-learning model cloning.
/// </summary>
public class SimpleMockModel : IFullModel<double, Tensor<double>, Tensor<double>>,
    ISecondOrderGradientComputable<double, Tensor<double>, Tensor<double>>,
    ICloneable
{
    private Vector<double> _parameters;
    public int TrainCallCount { get; private set; }
    public int PredictCallCount { get; private set; }

    public SimpleMockModel(int parameterCount)
    {
        _parameters = new Vector<double>(parameterCount);
        // Initialize with small random values
        for (int i = 0; i < parameterCount; i++)
        {
            _parameters[i] = 0.1 * i;
        }
        TrainCallCount = 0;
        PredictCallCount = 0;
    }

    public Vector<double> GetParameters() => _parameters.Clone();

    public void SetParameters(Vector<double> parameters)
    {
        if (parameters.Length != _parameters.Length)
        {
            throw new ArgumentException($"Parameter count mismatch: expected {_parameters.Length}, got {parameters.Length}");
        }
        _parameters = parameters.Clone();
    }

    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters)
    {
        var newModel = new SimpleMockModel(_parameters.Length);
        newModel.SetParameters(parameters);
        return newModel;
    }

    public void Train(Tensor<double> input, Tensor<double> expectedOutput)
    {
        TrainCallCount++;
        // Simple update: add a small value to each parameter to simulate training
        for (int i = 0; i < _parameters.Length; i++)
        {
            _parameters[i] += 0.01;
        }
    }

    public Tensor<double> Predict(Tensor<double> input)
    {
        PredictCallCount++;
        // Return output matching the batch size (first dimension of input)
        // Input shape is typically [batch_size, features] for meta-learning tasks
        // Target shape is [batch_size] (one label per sample)
        int batchSize = input.Shape.Length > 0 ? input.Shape[0] : 1;
        return new Tensor<double>(new int[] { batchSize });
    }

    public ModelMetadata<double> GetModelMetadata()
    {
        return new ModelMetadata<double>();
    }

    // IModelSerializer implementation
    public void SaveModel(string filePath) { }
    public void LoadModel(string filePath) { }
    public byte[] Serialize() => Array.Empty<byte>();
    public void Deserialize(byte[] data) { }

    // ICheckpointableModel implementation
    public void SaveState(Stream stream) { }
    public void LoadState(Stream stream) { }

    public IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
    {
        var copy = new SimpleMockModel(_parameters.Length);
        copy.SetParameters(_parameters);
        return copy;
    }

    public IFullModel<double, Tensor<double>, Tensor<double>> Clone()
    {
        return DeepCopy();
    }

    // ICloneable explicit implementation
    object ICloneable.Clone()
    {
        return DeepCopy();
    }

    // IFeatureAware implementation
    public int InputFeatureCount => 10;
    public int OutputFeatureCount => 1;
    public string[] FeatureNames { get; set; } = Array.Empty<string>();
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, InputFeatureCount);
    public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < InputFeatureCount;

    // IFeatureImportance implementation
    public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();

    // IGradientComputable implementation
    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

    public Vector<double> ComputeGradients(Tensor<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null)
    {
        // Return non-zero gradients so that meta-learning parameter updates work
        var gradients = new Vector<double>(ParameterCount);
        for (int i = 0; i < ParameterCount; i++)
        {
            gradients[i] = 0.1 * (i + 1);  // Non-zero values for testing
        }
        return gradients;
    }

    public void ApplyGradients(Vector<double> gradients, double learningRate)
    {
        // Mock implementation - simple parameter update
        for (int i = 0; i < Math.Min(gradients.Length, _parameters.Length); i++)
        {
            _parameters[i] -= learningRate * gradients[i];
        }
    }

    // IJitCompilable implementation
    public bool SupportsJitCompilation => true;

    public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
    {
        // Create a simple linear computation graph: output = sum(input * parameters)
        var inputShape = new int[] { 1, _parameters.Length };
        var inputTensor = new Tensor<double>(inputShape);
        var inputNode = TensorOperations<double>.Variable(inputTensor, "input");
        inputNodes.Add(inputNode);

        // Create parameter node
        var paramTensor = new Tensor<double>(new int[] { _parameters.Length }, _parameters);
        var paramNode = TensorOperations<double>.Variable(paramTensor, "parameters");
        inputNodes.Add(paramNode);

        // Compute element-wise multiply and sum
        var mulNode = TensorOperations<double>.ElementwiseMultiply(inputNode, paramNode);
        var outputNode = TensorOperations<double>.Sum(mulNode);
        return outputNode;
    }

    // ISecondOrderGradientComputable implementation
    public Vector<double> ComputeSecondOrderGradients(
        List<(Tensor<double> input, Tensor<double> target)> adaptationSteps,
        Tensor<double> queryInput,
        Tensor<double> queryTarget,
        ILossFunction<double> lossFunction,
        double innerLearningRate)
    {
        // Return non-zero gradients to allow meta-learning parameter updates
        // In a real implementation this would compute gradients through the adaptation steps
        var gradients = new Vector<double>(ParameterCount);
        for (int i = 0; i < ParameterCount; i++)
        {
            gradients[i] = 0.05 * (i + 1);  // Non-zero values for testing
        }
        return gradients;
    }
}
