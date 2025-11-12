using AiDotNet.Regression;
using AiDotNet.LinearAlgebra;
using AiDotNet.Kernels;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for ALL Regression Models in AiDotNet - Part 2 (Advanced and Tree-Based Models)
/// Tests training performance for each regression model
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class AllRegressionModelsBenchmarks_Part2
{
    [Params(500, 2000)]
    public int SampleCount { get; set; }

    [Params(10)]
    public int FeatureCount { get; set; }

    private Matrix<double> _X = null!;
    private Vector<double> _y = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _X = new Matrix<double>(SampleCount, FeatureCount);
        _y = new Vector<double>(SampleCount);

        for (int i = 0; i < SampleCount; i++)
        {
            double target = 0;
            for (int j = 0; j < FeatureCount; j++)
            {
                double value = random.NextDouble() * 10 - 5;
                _X[i, j] = value;
                target += value * (j + 1);
            }
            _y[i] = target + random.NextDouble() * 2;
        }
    }

    [Benchmark(Baseline = true)]
    public DecisionTreeRegression<double> Reg20_DecisionTreeRegression()
    {
        var model = new DecisionTreeRegression<double>(maxDepth: 10);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public ConditionalInferenceTreeRegression<double> Reg21_ConditionalInferenceTreeRegression()
    {
        var model = new ConditionalInferenceTreeRegression<double>(maxDepth: 10);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public M5ModelTreeRegression<double> Reg22_M5ModelTreeRegression()
    {
        var model = new M5ModelTreeRegression<double>();
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public RandomForestRegression<double> Reg23_RandomForestRegression()
    {
        var model = new RandomForestRegression<double>(numTrees: 10, maxDepth: 10);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public ExtremelyRandomizedTreesRegression<double> Reg24_ExtremelyRandomizedTreesRegression()
    {
        var model = new ExtremelyRandomizedTreesRegression<double>(numTrees: 10, maxDepth: 10);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public GradientBoostingRegression<double> Reg25_GradientBoostingRegression()
    {
        var model = new GradientBoostingRegression<double>(numTrees: 10, maxDepth: 5, learningRate: 0.1);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public AdaBoostR2Regression<double> Reg26_AdaBoostR2Regression()
    {
        var model = new AdaBoostR2Regression<double>(numEstimators: 10);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public QuantileRegressionForests<double> Reg27_QuantileRegressionForests()
    {
        var model = new QuantileRegressionForests<double>(numTrees: 10, maxDepth: 10);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public KNearestNeighborsRegression<double> Reg28_KNearestNeighborsRegression()
    {
        var model = new KNearestNeighborsRegression<double>(k: 5);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public SupportVectorRegression<double> Reg29_SupportVectorRegression()
    {
        var model = new SupportVectorRegression<double>(C: 1.0, epsilon: 0.1);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public KernelRidgeRegression<double> Reg30_KernelRidgeRegression()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        var model = new KernelRidgeRegression<double>(kernel, alpha: 1.0);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public RadialBasisFunctionRegression<double> Reg31_RadialBasisFunctionRegression()
    {
        var model = new RadialBasisFunctionRegression<double>(numCenters: 10);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public GaussianProcessRegression<double> Reg32_GaussianProcessRegression()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        var model = new GaussianProcessRegression<double>(kernel, noise: 0.1);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public NeuralNetworkRegression<double> Reg33_NeuralNetworkRegression()
    {
        var model = new NeuralNetworkRegression<double>(hiddenLayers: new[] { 32, 16 });
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public MultilayerPerceptronRegression<double> Reg34_MultilayerPerceptronRegression()
    {
        var model = new MultilayerPerceptronRegression<double>(hiddenLayerSizes: new[] { 32, 16 });
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public GeneralizedAdditiveModelRegression<double> Reg35_GeneralizedAdditiveModelRegression()
    {
        var model = new GeneralizedAdditiveModelRegression<double>();
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public TimeSeriesRegression<double> Reg36_TimeSeriesRegression()
    {
        var model = new TimeSeriesRegression<double>(order: 2);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public GeneticAlgorithmRegression<double> Reg37_GeneticAlgorithmRegression()
    {
        var model = new GeneticAlgorithmRegression<double>(populationSize: 50, generations: 10);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public SymbolicRegression<double> Reg38_SymbolicRegression()
    {
        var model = new SymbolicRegression<double>(populationSize: 50, generations: 10);
        model.Fit(_X, _y);
        return model;
    }
}
