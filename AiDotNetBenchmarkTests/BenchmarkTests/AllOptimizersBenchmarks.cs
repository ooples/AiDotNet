using AiDotNet.Optimizers;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for ALL 35 Optimizers in AiDotNet
/// Each optimizer tested individually for parameter update performance
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class AllOptimizersBenchmarks
{
    [Params(1000, 5000)]
    public int ParameterSize { get; set; }

    private Vector<double> _parameters = null!;
    private Vector<double> _gradients = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _parameters = new Vector<double>(ParameterSize);
        _gradients = new Vector<double>(ParameterSize);

        for (int i = 0; i < ParameterSize; i++)
        {
            _parameters[i] = random.NextDouble() * 2 - 1;
            _gradients[i] = random.NextDouble() * 0.1 - 0.05;
        }
    }

    [Benchmark(Baseline = true)]
    public Vector<double> Opt01_GradientDescent()
    {
        var opt = new GradientDescentOptimizer<double>(learningRate: 0.01);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt02_StochasticGradientDescent()
    {
        var opt = new StochasticGradientDescentOptimizer<double>(learningRate: 0.01);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt03_MiniBatchGradientDescent()
    {
        var opt = new MiniBatchGradientDescentOptimizer<double>(learningRate: 0.01, batchSize: 32);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt04_Momentum()
    {
        var opt = new MomentumOptimizer<double>(learningRate: 0.01, momentum: 0.9);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt05_NesterovAcceleratedGradient()
    {
        var opt = new NesterovAcceleratedGradientOptimizer<double>(learningRate: 0.01, momentum: 0.9);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt06_Adam()
    {
        var opt = new AdamOptimizer<double>(learningRate: 0.001);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt07_Nadam()
    {
        var opt = new NadamOptimizer<double>(learningRate: 0.001);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt08_AMSGrad()
    {
        var opt = new AMSGradOptimizer<double>(learningRate: 0.001);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt09_AdaMax()
    {
        var opt = new AdaMaxOptimizer<double>(learningRate: 0.001);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt10_AdaGrad()
    {
        var opt = new AdagradOptimizer<double>(learningRate: 0.01);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt11_AdaDelta()
    {
        var opt = new AdaDeltaOptimizer<double>(rho: 0.95);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt12_RMSprop()
    {
        var opt = new RootMeanSquarePropagationOptimizer<double>(learningRate: 0.001);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt13_Lion()
    {
        var opt = new LionOptimizer<double>(learningRate: 0.0001);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt14_FTRL()
    {
        var opt = new FTRLOptimizer<double>(learningRate: 0.01);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt15_ProximalGradientDescent()
    {
        var opt = new ProximalGradientDescentOptimizer<double>(learningRate: 0.01);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt16_CoordinateDescent()
    {
        var opt = new CoordinateDescentOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt17_ConjugateGradient()
    {
        var opt = new ConjugateGradientOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt18_NewtonMethod()
    {
        var opt = new NewtonMethodOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt19_BFGS()
    {
        var opt = new BFGSOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt20_LBFGS()
    {
        var opt = new LBFGSOptimizer<double>(memorySize: 10);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt21_DFP()
    {
        var opt = new DFPOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt22_LevenbergMarquardt()
    {
        var opt = new LevenbergMarquardtOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt23_TrustRegion()
    {
        var opt = new TrustRegionOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt24_NelderMead()
    {
        var opt = new NelderMeadOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt25_Powell()
    {
        var opt = new PowellOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt26_GeneticAlgorithm()
    {
        var opt = new GeneticAlgorithmOptimizer<double>(populationSize: 50);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt27_ParticleSwarm()
    {
        var opt = new ParticleSwarmOptimizer<double>(swarmSize: 30);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt28_DifferentialEvolution()
    {
        var opt = new DifferentialEvolutionOptimizer<double>(populationSize: 50);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt29_SimulatedAnnealing()
    {
        var opt = new SimulatedAnnealingOptimizer<double>(initialTemperature: 100);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt30_AntColony()
    {
        var opt = new AntColonyOptimizer<double>(numAnts: 20);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt31_TabuSearch()
    {
        var opt = new TabuSearchOptimizer<double>(tabuListSize: 20);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt32_CMAES()
    {
        var opt = new CMAESOptimizer<double>(populationSize: 50);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt33_Bayesian()
    {
        var opt = new BayesianOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt34_ADMM()
    {
        var opt = new ADMMOptimizer<double>(rho: 1.0);
        return opt.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Opt35_Normal()
    {
        var opt = new NormalOptimizer<double>();
        return opt.UpdateParameters(_parameters, _gradients);
    }
}
