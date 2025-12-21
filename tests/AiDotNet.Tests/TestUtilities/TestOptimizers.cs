using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Tests.TestUtilities;

internal sealed class PassthroughOptimizer<T, TInput, TOutput> : IOptimizer<T, TInput, TOutput>
{
    private readonly IFullModel<T, TInput, TOutput> _model;

    public PassthroughOptimizer(IFullModel<T, TInput, TOutput> model)
    {
        _model = model;
    }

    public OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        return new OptimizationResult<T, TInput, TOutput>
        {
            BestSolution = _model
        };
    }

    public bool ShouldEarlyStop()
    {
        return false;
    }

    public OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return new OptimizationAlgorithmOptions<T, TInput, TOutput>();
    }

    public void Reset()
    {
    }

    public byte[] Serialize()
    {
        return [];
    }

    public void Deserialize(byte[] data)
    {
    }

    public void SaveModel(string filePath)
    {
        throw new NotSupportedException();
    }

    public void LoadModel(string filePath)
    {
        throw new NotSupportedException();
    }
}

internal sealed class SingleStepTrainOptimizer<T, TInput, TOutput> : IOptimizer<T, TInput, TOutput>
{
    private readonly IFullModel<T, TInput, TOutput> _model;

    public SingleStepTrainOptimizer(IFullModel<T, TInput, TOutput> model)
    {
        _model = model;
    }

    public OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        _model.Train(inputData.XTrain, inputData.YTrain);
        return new OptimizationResult<T, TInput, TOutput> { BestSolution = _model };
    }

    public bool ShouldEarlyStop()
    {
        return false;
    }

    public OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return new OptimizationAlgorithmOptions<T, TInput, TOutput>();
    }

    public void Reset()
    {
    }

    public byte[] Serialize()
    {
        return [];
    }

    public void Deserialize(byte[] data)
    {
    }

    public void SaveModel(string filePath)
    {
        throw new NotSupportedException();
    }

    public void LoadModel(string filePath)
    {
        throw new NotSupportedException();
    }
}

internal sealed class DeterministicNeuralNetworkParameterOptimizer<TInput, TOutput> : IOptimizer<double, TInput, TOutput>
{
    private readonly IFullModel<double, TInput, TOutput> _model;

    public DeterministicNeuralNetworkParameterOptimizer(IFullModel<double, TInput, TOutput> model)
    {
        _model = model;
    }

    public OptimizationResult<double, TInput, TOutput> Optimize(OptimizationInputData<double, TInput, TOutput> inputData)
    {
        var nn = _model as NeuralNetworkModel<double>;
        if (nn != null)
        {
            var parameters = nn.Network.GetParameters();
            for (int i = 0; i < parameters.Length; i++)
            {
                parameters[i] = (i + 1) * 0.01;
            }
            nn.Network.UpdateParameters(parameters);
        }

        return new OptimizationResult<double, TInput, TOutput> { BestSolution = _model };
    }

    public bool ShouldEarlyStop()
    {
        return false;
    }

    public OptimizationAlgorithmOptions<double, TInput, TOutput> GetOptions()
    {
        return new OptimizationAlgorithmOptions<double, TInput, TOutput>();
    }

    public void Reset()
    {
    }

    public byte[] Serialize()
    {
        return [];
    }

    public void Deserialize(byte[] data)
    {
    }

    public void SaveModel(string filePath)
    {
        throw new NotSupportedException();
    }

    public void LoadModel(string filePath)
    {
        throw new NotSupportedException();
    }
}
