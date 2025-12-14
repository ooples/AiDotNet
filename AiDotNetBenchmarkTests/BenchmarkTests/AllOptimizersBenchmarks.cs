using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Optimizer Options creation in AiDotNet
/// Tests the creation performance of optimizer configuration objects
/// Note: Actual optimizer instances require a model, so we benchmark configuration creation
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class AllOptimizersBenchmarks
{
    #region Gradient-Based Optimizer Options

    [Benchmark(Baseline = true)]
    public GradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt01_GradientDescentOptions()
    {
        return new GradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt02_SGDOptions()
    {
        return new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public MiniBatchGradientDescentOptions<double, Tensor<double>, Tensor<double>> Opt03_MiniBatchOptions()
    {
        return new MiniBatchGradientDescentOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public MomentumOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt04_MomentumOptions()
    {
        return new MomentumOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public NesterovAcceleratedGradientOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt05_NAGOptions()
    {
        return new NesterovAcceleratedGradientOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    #endregion

    #region Adaptive Learning Rate Optimizer Options

    [Benchmark]
    public AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt06_AdamOptions()
    {
        return new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public NadamOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt07_NadamOptions()
    {
        return new NadamOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public AMSGradOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt08_AMSGradOptions()
    {
        return new AMSGradOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public AdaMaxOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt09_AdaMaxOptions()
    {
        return new AdaMaxOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public AdagradOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt10_AdaGradOptions()
    {
        return new AdagradOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public AdaDeltaOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt11_AdaDeltaOptions()
    {
        return new AdaDeltaOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt12_RMSpropOptions()
    {
        return new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    #endregion

    #region Specialized Optimizer Options

    [Benchmark]
    public LionOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt13_LionOptions()
    {
        return new LionOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public FTRLOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt14_FTRLOptions()
    {
        return new FTRLOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public ProximalGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt15_ProximalGDOptions()
    {
        return new ProximalGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    #endregion

    #region Metaheuristic Optimizer Options

    [Benchmark]
    public GeneticAlgorithmOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt16_GAOptions()
    {
        return new GeneticAlgorithmOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public ParticleSwarmOptimizationOptions<double, Tensor<double>, Tensor<double>> Opt17_PSOOptions()
    {
        return new ParticleSwarmOptimizationOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public DifferentialEvolutionOptions<double, Tensor<double>, Tensor<double>> Opt18_DEOptions()
    {
        return new DifferentialEvolutionOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public SimulatedAnnealingOptions<double, Tensor<double>, Tensor<double>> Opt19_SAOptions()
    {
        return new SimulatedAnnealingOptions<double, Tensor<double>, Tensor<double>>();
    }

    #endregion

    #region Second-Order Optimizer Options

    [Benchmark]
    public LBFGSOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt20_LBFGSOptions()
    {
        return new LBFGSOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    [Benchmark]
    public BFGSOptimizerOptions<double, Tensor<double>, Tensor<double>> Opt21_BFGSOptions()
    {
        return new BFGSOptimizerOptions<double, Tensor<double>, Tensor<double>>();
    }

    #endregion
}
