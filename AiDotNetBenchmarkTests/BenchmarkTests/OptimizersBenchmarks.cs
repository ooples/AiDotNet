using AiDotNet.Optimizers;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Optimizer operations comparing different optimization algorithms
/// Tests single step performance and convergence characteristics
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class OptimizersBenchmarks
{
    [Params(100, 1000, 10000)]
    public int ParameterSize { get; set; }

    private Vector<double> _parameters = null!;
    private Vector<double> _gradients = null!;

    // Gradient-based optimizers
    private GradientDescentOptimizer<double> _gd = null!;
    private StochasticGradientDescentOptimizer<double> _sgd = null!;
    private MomentumOptimizer<double> _momentum = null!;
    private NesterovAcceleratedGradientOptimizer<double> _nag = null!;

    // Adaptive optimizers
    private AdamOptimizer<double> _adam = null!;
    private AdagradOptimizer<double> _adagrad = null!;
    private RootMeanSquarePropagationOptimizer<double> _rmsprop = null!;
    private AdaDeltaOptimizer<double> _adadelta = null!;
    private NadamOptimizer<double> _nadam = null!;
    private AMSGradOptimizer<double> _amsgrad = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize parameters and gradients
        _parameters = new Vector<double>(ParameterSize);
        _gradients = new Vector<double>(ParameterSize);

        for (int i = 0; i < ParameterSize; i++)
        {
            _parameters[i] = random.NextDouble() * 2 - 1; // Range [-1, 1]
            _gradients[i] = random.NextDouble() * 0.2 - 0.1; // Range [-0.1, 0.1]
        }

        // Initialize optimizers with standard learning rates
        double learningRate = 0.001;

        _gd = new GradientDescentOptimizer<double>(learningRate);
        _sgd = new StochasticGradientDescentOptimizer<double>(learningRate);
        _momentum = new MomentumOptimizer<double>(learningRate, momentum: 0.9);
        _nag = new NesterovAcceleratedGradientOptimizer<double>(learningRate, momentum: 0.9);

        _adam = new AdamOptimizer<double>(learningRate, beta1: 0.9, beta2: 0.999);
        _adagrad = new AdagradOptimizer<double>(learningRate);
        _rmsprop = new RootMeanSquarePropagationOptimizer<double>(learningRate);
        _adadelta = new AdaDeltaOptimizer<double>(rho: 0.95);
        _nadam = new NadamOptimizer<double>(learningRate);
        _amsgrad = new AMSGradOptimizer<double>(learningRate);
    }

    #region Gradient Descent Variants

    [Benchmark(Baseline = true)]
    public Vector<double> GradientDescent_Step()
    {
        return _gd.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> SGD_Step()
    {
        return _sgd.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Momentum_Step()
    {
        return _momentum.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> NAG_Step()
    {
        return _nag.UpdateParameters(_parameters, _gradients);
    }

    #endregion

    #region Adaptive Optimizers

    [Benchmark]
    public Vector<double> Adam_Step()
    {
        return _adam.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Adagrad_Step()
    {
        return _adagrad.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> RMSprop_Step()
    {
        return _rmsprop.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> AdaDelta_Step()
    {
        return _adadelta.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> Nadam_Step()
    {
        return _nadam.UpdateParameters(_parameters, _gradients);
    }

    [Benchmark]
    public Vector<double> AMSGrad_Step()
    {
        return _amsgrad.UpdateParameters(_parameters, _gradients);
    }

    #endregion

    #region Multi-Step Optimization (Convergence Test)

    [Benchmark]
    public Vector<double> Adam_100Steps()
    {
        var params_ = _parameters.Clone();
        for (int i = 0; i < 100; i++)
        {
            params_ = _adam.UpdateParameters(params_, _gradients);
        }
        return params_;
    }

    [Benchmark]
    public Vector<double> Momentum_100Steps()
    {
        var params_ = _parameters.Clone();
        for (int i = 0; i < 100; i++)
        {
            params_ = _momentum.UpdateParameters(params_, _gradients);
        }
        return params_;
    }

    [Benchmark]
    public Vector<double> RMSprop_100Steps()
    {
        var params_ = _parameters.Clone();
        for (int i = 0; i < 100; i++)
        {
            params_ = _rmsprop.UpdateParameters(params_, _gradients);
        }
        return params_;
    }

    #endregion
}
