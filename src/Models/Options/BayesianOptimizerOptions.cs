global using AiDotNet.Kernels;

namespace AiDotNet.Models.Options;

public class BayesianOptimizerOptions<T> : OptimizationAlgorithmOptions
{
    public int InitialSamples { get; set; } = 5;
    public int AcquisitionOptimizationSamples { get; set; } = 1000;
    public double LowerBound { get; set; } = -10;
    public double UpperBound { get; set; } = 10;
    public double ExplorationFactor { get; set; } = 2.0;
    public AcquisitionFunctionType AcquisitionFunction { get; set; } = AcquisitionFunctionType.UpperConfidenceBound;
    public IKernelFunction<T> KernelFunction { get; set; } = new RBFKernel<T>();
    public int Seed { get; set; } = 42;
}