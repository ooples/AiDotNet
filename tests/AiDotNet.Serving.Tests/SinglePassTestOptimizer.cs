using AiDotNet.Interfaces;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Test-only optimizer that returns the provided model without running a full optimization loop.
/// </summary>
internal sealed class SinglePassTestOptimizer : IOptimizer<double, Matrix<double>, Vector<double>>
{
    private readonly IFullModel<double, Matrix<double>, Vector<double>> _model;
    private readonly OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>> _options;

    public SinglePassTestOptimizer(IFullModel<double, Matrix<double>, Vector<double>> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _options = new OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 1,
            UseEarlyStopping = false
        };
    }

    public OptimizationResult<double, Matrix<double>, Vector<double>> Optimize(OptimizationInputData<double, Matrix<double>, Vector<double>> inputData)
    {
        if (inputData == null)
        {
            throw new ArgumentNullException(nameof(inputData));
        }

        return new OptimizationResult<double, Matrix<double>, Vector<double>>
        {
            BestSolution = _model.DeepCopy(),
            Iterations = 1
        };
    }

    public bool ShouldEarlyStop() => false;

    public OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>> GetOptions() => _options;

    public void Reset()
    {
    }

    public byte[] Serialize() => Array.Empty<byte>();

    public void Deserialize(byte[] data)
    {
    }

    public void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Array.Empty<byte>());
    }

    public void LoadModel(string filePath)
    {
    }

    public void SetModel(IFullModel<double, Matrix<double>, Vector<double>> model)
    {
        // No-op for test optimizer
    }
}

