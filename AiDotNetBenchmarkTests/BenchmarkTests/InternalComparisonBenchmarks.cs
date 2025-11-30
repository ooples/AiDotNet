using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using AiDotNet.Optimizers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Kernels;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Internal comparison benchmarks for AiDotNet
/// Compares different implementations and algorithms within the library
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class InternalComparisonBenchmarks
{
    [Params(500, 2000)]
    public int SampleCount { get; set; }

    [Params(10, 30)]
    public int FeatureCount { get; set; }

    private Matrix<double> _trainX = null!;
    private Vector<double> _trainY = null!;
    private Vector<double> _parameters = null!;
    private Vector<double> _gradients = null!;
    private Vector<double> _vectorA = null!;
    private Vector<double> _vectorB = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize regression data
        _trainX = new Matrix<double>(SampleCount, FeatureCount);
        _trainY = new Vector<double>(SampleCount);

        for (int i = 0; i < SampleCount; i++)
        {
            double target = 0;
            for (int j = 0; j < FeatureCount; j++)
            {
                double value = random.NextDouble() * 10 - 5;
                _trainX[i, j] = value;
                target += value * (j + 1);
            }
            _trainY[i] = target + random.NextDouble() * 2;
        }

        // Initialize optimizer parameters
        _parameters = new Vector<double>(FeatureCount);
        _gradients = new Vector<double>(FeatureCount);
        for (int i = 0; i < FeatureCount; i++)
        {
            _parameters[i] = random.NextDouble() * 2 - 1;
            _gradients[i] = random.NextDouble() * 0.1 - 0.05;
        }

        // Initialize vectors for kernel comparisons
        _vectorA = new Vector<double>(FeatureCount);
        _vectorB = new Vector<double>(FeatureCount);
        for (int i = 0; i < FeatureCount; i++)
        {
            _vectorA[i] = random.NextDouble() * 2 - 1;
            _vectorB[i] = random.NextDouble() * 2 - 1;
        }
    }

    #region Regression Methods Comparison

    [Benchmark(Baseline = true)]
    public MultipleRegression<double> Regression_Standard()
    {
        var model = new MultipleRegression<double>();
        model.Fit(_trainX, _trainY);
        return model;
    }

    [Benchmark]
    public RidgeRegression<double> Regression_Ridge()
    {
        var model = new RidgeRegression<double>(alpha: 1.0);
        model.Fit(_trainX, _trainY);
        return model;
    }

    [Benchmark]
    public LassoRegression<double> Regression_Lasso()
    {
        var model = new LassoRegression<double>(alpha: 1.0);
        model.Fit(_trainX, _trainY);
        return model;
    }

    [Benchmark]
    public ElasticNetRegression<double> Regression_ElasticNet()
    {
        var model = new ElasticNetRegression<double>(alpha: 1.0, l1Ratio: 0.5);
        model.Fit(_trainX, _trainY);
        return model;
    }

    #endregion

    #region Optimizer Comparison (Same Architecture)

    [Benchmark]
    public Vector<double> Optimizer_GradientDescent()
    {
        var optimizer = new GradientDescentOptimizer<double>(learningRate: 0.01);
        var params_ = _parameters.Clone();
        for (int i = 0; i < 100; i++)
        {
            params_ = optimizer.UpdateParameters(params_, _gradients);
        }
        return params_;
    }

    [Benchmark]
    public Vector<double> Optimizer_Momentum()
    {
        var optimizer = new MomentumOptimizer<double>(learningRate: 0.01, momentum: 0.9);
        var params_ = _parameters.Clone();
        for (int i = 0; i < 100; i++)
        {
            params_ = optimizer.UpdateParameters(params_, _gradients);
        }
        return params_;
    }

    [Benchmark]
    public Vector<double> Optimizer_Adam()
    {
        var optimizer = new AdamOptimizer<double>(learningRate: 0.01);
        var params_ = _parameters.Clone();
        for (int i = 0; i < 100; i++)
        {
            params_ = optimizer.UpdateParameters(params_, _gradients);
        }
        return params_;
    }

    [Benchmark]
    public Vector<double> Optimizer_RMSprop()
    {
        var optimizer = new RootMeanSquarePropagationOptimizer<double>(learningRate: 0.01);
        var params_ = _parameters.Clone();
        for (int i = 0; i < 100; i++)
        {
            params_ = optimizer.UpdateParameters(params_, _gradients);
        }
        return params_;
    }

    [Benchmark]
    public Vector<double> Optimizer_AdaGrad()
    {
        var optimizer = new AdagradOptimizer<double>(learningRate: 0.01);
        var params_ = _parameters.Clone();
        for (int i = 0; i < 100; i++)
        {
            params_ = optimizer.UpdateParameters(params_, _gradients);
        }
        return params_;
    }

    [Benchmark]
    public Vector<double> Optimizer_AdaDelta()
    {
        var optimizer = new AdaDeltaOptimizer<double>(rho: 0.95);
        var params_ = _parameters.Clone();
        for (int i = 0; i < 100; i++)
        {
            params_ = optimizer.UpdateParameters(params_, _gradients);
        }
        return params_;
    }

    #endregion

    #region Activation Function Comparison

    [Benchmark]
    public Vector<double> Activation_ReLU()
    {
        var activation = new ReLUActivation<double>();
        return activation.Activate(_vectorA);
    }

    [Benchmark]
    public Vector<double> Activation_LeakyReLU()
    {
        var activation = new LeakyReLUActivation<double>();
        return activation.Activate(_vectorA);
    }

    [Benchmark]
    public Vector<double> Activation_ELU()
    {
        var activation = new ELUActivation<double>();
        return activation.Activate(_vectorA);
    }

    [Benchmark]
    public Vector<double> Activation_GELU()
    {
        var activation = new GELUActivation<double>();
        return activation.Activate(_vectorA);
    }

    [Benchmark]
    public Vector<double> Activation_Swish()
    {
        var activation = new SwishActivation<double>();
        return activation.Activate(_vectorA);
    }

    [Benchmark]
    public Vector<double> Activation_Mish()
    {
        var activation = new MishActivation<double>();
        return activation.Activate(_vectorA);
    }

    [Benchmark]
    public Vector<double> Activation_Tanh()
    {
        var activation = new TanhActivation<double>();
        return activation.Activate(_vectorA);
    }

    [Benchmark]
    public Vector<double> Activation_Sigmoid()
    {
        var activation = new SigmoidActivation<double>();
        return activation.Activate(_vectorA);
    }

    #endregion

    #region Kernel Function Comparison

    [Benchmark]
    public double Kernel_Linear()
    {
        var kernel = new LinearKernel<double>();
        return kernel.Compute(_vectorA, _vectorB);
    }

    [Benchmark]
    public double Kernel_Polynomial_Deg2()
    {
        var kernel = new PolynomialKernel<double>(degree: 2, constant: 1.0);
        return kernel.Compute(_vectorA, _vectorB);
    }

    [Benchmark]
    public double Kernel_Polynomial_Deg3()
    {
        var kernel = new PolynomialKernel<double>(degree: 3, constant: 1.0);
        return kernel.Compute(_vectorA, _vectorB);
    }

    [Benchmark]
    public double Kernel_Gaussian_Sigma05()
    {
        var kernel = new GaussianKernel<double>(sigma: 0.5);
        return kernel.Compute(_vectorA, _vectorB);
    }

    [Benchmark]
    public double Kernel_Gaussian_Sigma10()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        return kernel.Compute(_vectorA, _vectorB);
    }

    [Benchmark]
    public double Kernel_Gaussian_Sigma20()
    {
        var kernel = new GaussianKernel<double>(sigma: 2.0);
        return kernel.Compute(_vectorA, _vectorB);
    }

    [Benchmark]
    public double Kernel_Laplacian()
    {
        var kernel = new LaplacianKernel<double>(sigma: 1.0);
        return kernel.Compute(_vectorA, _vectorB);
    }

    [Benchmark]
    public double Kernel_Sigmoid()
    {
        var kernel = new SigmoidKernel<double>(alpha: 1.0, constant: 0.0);
        return kernel.Compute(_vectorA, _vectorB);
    }

    #endregion

    #region Tree-Based Regression Comparison

    [Benchmark]
    public DecisionTreeRegression<double> TreeRegression_DecisionTree()
    {
        var model = new DecisionTreeRegression<double>(maxDepth: 10);
        model.Fit(_trainX, _trainY);
        return model;
    }

    [Benchmark]
    public RandomForestRegression<double> TreeRegression_RandomForest()
    {
        var model = new RandomForestRegression<double>(numTrees: 10, maxDepth: 10);
        model.Fit(_trainX, _trainY);
        return model;
    }

    [Benchmark]
    public GradientBoostingRegression<double> TreeRegression_GradientBoosting()
    {
        var model = new GradientBoostingRegression<double>(numTrees: 10, maxDepth: 5, learningRate: 0.1);
        model.Fit(_trainX, _trainY);
        return model;
    }

    [Benchmark]
    public ExtremelyRandomizedTreesRegression<double> TreeRegression_ExtraTrees()
    {
        var model = new ExtremelyRandomizedTreesRegression<double>(numTrees: 10, maxDepth: 10);
        model.Fit(_trainX, _trainY);
        return model;
    }

    #endregion
}
