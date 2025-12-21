using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.Tests.FederatedLearning;

internal sealed class FederatedNoOpOptimizer : IOptimizer<double, Matrix<double>, Vector<double>>
{
    private readonly OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>> _options;
    private readonly IFullModel<double, Matrix<double>, Vector<double>> _model;

    public FederatedNoOpOptimizer(
        IFullModel<double, Matrix<double>, Vector<double>> model,
        OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>? options = null)
    {
        _model = model;
        _options = options ?? new OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>();
    }

    public OptimizationResult<double, Matrix<double>, Vector<double>> Optimize(OptimizationInputData<double, Matrix<double>, Vector<double>> inputData)
    {
        var best = inputData.InitialSolution ?? _model;
        return new OptimizationResult<double, Matrix<double>, Vector<double>>
        {
            BestSolution = best.WithParameters(best.GetParameters()),
            Iterations = _options.MaxIterations
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
    }

    public void LoadModel(string filePath)
    {
    }
}

