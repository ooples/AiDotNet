using AiDotNet.Regression;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for ALL Regression Models in AiDotNet - Part 1 (Linear and Basic Models)
/// Tests training performance for each regression model
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class AllRegressionModelsBenchmarks_Part1
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
    public SimpleRegression<double> Reg01_SimpleRegression()
    {
        var model = new SimpleRegression<double>();
        var singleFeature = new Matrix<double>(SampleCount, 1);
        for (int i = 0; i < SampleCount; i++) singleFeature[i, 0] = _X[i, 0];
        model.Fit(singleFeature, _y);
        return model;
    }

    [Benchmark]
    public MultipleRegression<double> Reg02_MultipleRegression()
    {
        var model = new MultipleRegression<double>();
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public MultivariateRegression<double> Reg03_MultivariateRegression()
    {
        var model = new MultivariateRegression<double>();
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public PolynomialRegression<double> Reg04_PolynomialRegression()
    {
        var model = new PolynomialRegression<double>(degree: 2);
        var singleFeature = new Matrix<double>(SampleCount, 1);
        for (int i = 0; i < SampleCount; i++) singleFeature[i, 0] = _X[i, 0];
        model.Fit(singleFeature, _y);
        return model;
    }

    [Benchmark]
    public OrthogonalRegression<double> Reg05_OrthogonalRegression()
    {
        var model = new OrthogonalRegression<double>();
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public WeightedRegression<double> Reg06_WeightedRegression()
    {
        var model = new WeightedRegression<double>();
        var weights = new Vector<double>(SampleCount);
        for (int i = 0; i < SampleCount; i++) weights[i] = 1.0;
        model.Fit(_X, _y, weights);
        return model;
    }

    [Benchmark]
    public RobustRegression<double> Reg07_RobustRegression()
    {
        var model = new RobustRegression<double>();
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public QuantileRegression<double> Reg08_QuantileRegression()
    {
        var model = new QuantileRegression<double>(quantile: 0.5);
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public IsotonicRegression<double> Reg09_IsotonicRegression()
    {
        var model = new IsotonicRegression<double>();
        var singleFeature = new Matrix<double>(SampleCount, 1);
        for (int i = 0; i < SampleCount; i++) singleFeature[i, 0] = _X[i, 0];
        model.Fit(singleFeature, _y);
        return model;
    }

    [Benchmark]
    public LogisticRegression<double> Reg10_LogisticRegression()
    {
        var model = new LogisticRegression<double>();
        // Convert to binary classification
        var yBinary = new Vector<double>(SampleCount);
        for (int i = 0; i < SampleCount; i++) yBinary[i] = _y[i] > 0 ? 1 : 0;
        model.Fit(_X, yBinary);
        return model;
    }

    [Benchmark]
    public MultinomialLogisticRegression<double> Reg11_MultinomialLogisticRegression()
    {
        var model = new MultinomialLogisticRegression<double>(numClasses: 3);
        var yMulti = new Vector<double>(SampleCount);
        for (int i = 0; i < SampleCount; i++) yMulti[i] = i % 3;
        model.Fit(_X, yMulti);
        return model;
    }

    [Benchmark]
    public PoissonRegression<double> Reg12_PoissonRegression()
    {
        var model = new PoissonRegression<double>();
        var yPoisson = new Vector<double>(SampleCount);
        for (int i = 0; i < SampleCount; i++) yPoisson[i] = Math.Max(0, _y[i]);
        model.Fit(_X, yPoisson);
        return model;
    }

    [Benchmark]
    public NegativeBinomialRegression<double> Reg13_NegativeBinomialRegression()
    {
        var model = new NegativeBinomialRegression<double>();
        var yNB = new Vector<double>(SampleCount);
        for (int i = 0; i < SampleCount; i++) yNB[i] = Math.Max(0, _y[i]);
        model.Fit(_X, yNB);
        return model;
    }

    [Benchmark]
    public BayesianRegression<double> Reg14_BayesianRegression()
    {
        var model = new BayesianRegression<double>();
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public PrincipalComponentRegression<double> Reg15_PrincipalComponentRegression()
    {
        var model = new PrincipalComponentRegression<double>(numComponents: Math.Min(5, FeatureCount));
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public PartialLeastSquaresRegression<double> Reg16_PartialLeastSquaresRegression()
    {
        var model = new PartialLeastSquaresRegression<double>(numComponents: Math.Min(5, FeatureCount));
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public StepwiseRegression<double> Reg17_StepwiseRegression()
    {
        var model = new StepwiseRegression<double>();
        model.Fit(_X, _y);
        return model;
    }

    [Benchmark]
    public SplineRegression<double> Reg18_SplineRegression()
    {
        var model = new SplineRegression<double>(numKnots: 5);
        var singleFeature = new Matrix<double>(SampleCount, 1);
        for (int i = 0; i < SampleCount; i++) singleFeature[i, 0] = _X[i, 0];
        model.Fit(singleFeature, _y);
        return model;
    }

    [Benchmark]
    public LocallyWeightedRegression<double> Reg19_LocallyWeightedRegression()
    {
        var model = new LocallyWeightedRegression<double>(bandwidth: 0.3);
        model.Fit(_X, _y);
        return model;
    }
}
