namespace AiDotNet.Interfaces;

public interface IOptimizer<T> : IModelSerializer
{
    OptimizationResult<T> Optimize(OptimizationInputData<T> inputData);
    bool ShouldEarlyStop();
    OptimizationAlgorithmOptions GetOptions();
}