using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;

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

        // For models that require training data (like KNN), we need to actually train them
        // even though this is a "no-op" optimizer - otherwise Predict will fail
        if (best is NonLinearRegressionBase<double> nonLinearModel)
        {
            // Call Train to populate training data for distance-based models like KNN
            nonLinearModel.Train(inputData.XTrain, inputData.YTrain);
        }

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

