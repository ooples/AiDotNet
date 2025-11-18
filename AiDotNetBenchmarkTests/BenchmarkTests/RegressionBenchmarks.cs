using AiDotNet.Regression;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Accord.Statistics.Models.Regression.Linear;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Regression models comparing AiDotNet vs Accord.NET and ML.NET
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class RegressionBenchmarks
{
    [Params(100, 1000, 5000)]
    public int SampleCount { get; set; }

    [Params(5, 20)]
    public int FeatureCount { get; set; }

    private Matrix<double> _aiTrainX = null!;
    private Vector<double> _aiTrainY = null!;
    private Matrix<double> _aiTestX = null!;

    private double[][] _accordTrainX = null!;
    private double[] _accordTrainY = null!;

    private MLContext _mlContext = null!;
    private IDataView _mlTrainData = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize AiDotNet data
        _aiTrainX = new Matrix<double>(SampleCount, FeatureCount);
        _aiTrainY = new Vector<double>(SampleCount);
        _aiTestX = new Matrix<double>(10, FeatureCount);

        // Initialize Accord.NET data
        _accordTrainX = new double[SampleCount][];
        _accordTrainY = new double[SampleCount];

        // Generate synthetic regression data: y = sum(x_i * i) + noise
        for (int i = 0; i < SampleCount; i++)
        {
            _accordTrainX[i] = new double[FeatureCount];
            double y = 0;

            for (int j = 0; j < FeatureCount; j++)
            {
                double value = random.NextDouble() * 10 - 5;
                _aiTrainX[i, j] = value;
                _accordTrainX[i][j] = value;
                y += value * (j + 1);
            }

            y += random.NextDouble() * 2 - 1; // Add noise
            _aiTrainY[i] = y;
            _accordTrainY[i] = y;
        }

        // Initialize test data
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                _aiTestX[i, j] = random.NextDouble() * 10 - 5;
            }
        }

        // Initialize ML.NET data
        _mlContext = new MLContext(seed: 42);
        var dataList = new List<RegressionData>();
        for (int i = 0; i < SampleCount; i++)
        {
            var features = new float[FeatureCount];
            for (int j = 0; j < FeatureCount; j++)
            {
                features[j] = (float)_aiTrainX[i, j];
            }
            dataList.Add(new RegressionData { Features = features, Label = (float)_aiTrainY[i] });
        }
        _mlTrainData = _mlContext.Data.LoadFromEnumerable(dataList);
    }

    #region Simple Regression (Single Feature)

    [Benchmark]
    public SimpleRegression<double> AiDotNet_SimpleRegression_Train()
    {
        var model = new SimpleRegression<double>();
        var singleFeature = new Matrix<double>(SampleCount, 1);
        for (int i = 0; i < SampleCount; i++)
        {
            singleFeature[i, 0] = _aiTrainX[i, 0];
        }
        model.Fit(singleFeature, _aiTrainY);
        return model;
    }

    [Benchmark(Baseline = true)]
    public SimpleLinearRegression AccordNet_SimpleRegression_Train()
    {
        var model = new SimpleLinearRegression();
        var x = new double[SampleCount];
        for (int i = 0; i < SampleCount; i++)
        {
            x[i] = _aiTrainX[i, 0];
        }
        model.Regress(x, _accordTrainY);
        return model;
    }

    #endregion

    #region Multiple Regression

    [Benchmark]
    public MultipleRegression<double> AiDotNet_MultipleRegression_Train()
    {
        var model = new MultipleRegression<double>();
        model.Fit(_aiTrainX, _aiTrainY);
        return model;
    }

    [Benchmark]
    public MultipleLinearRegression AccordNet_MultipleRegression_Train()
    {
        var model = new MultipleLinearRegression();
        model.Regress(_accordTrainX, _accordTrainY);
        return model;
    }

    [Benchmark]
    public ITransformer MLNet_LinearRegression_Train()
    {
        var pipeline = _mlContext.Transforms.Concatenate("Features", nameof(RegressionData.Features))
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", maximumNumberOfIterations: 100));
        return pipeline.Fit(_mlTrainData);
    }

    #endregion

    #region Prediction Performance

    [Benchmark]
    public Vector<double> AiDotNet_MultipleRegression_Predict()
    {
        var model = new MultipleRegression<double>();
        model.Fit(_aiTrainX, _aiTrainY);
        return model.Predict(_aiTestX);
    }

    [Benchmark]
    public double[] AccordNet_MultipleRegression_Predict()
    {
        var model = new MultipleLinearRegression();
        model.Regress(_accordTrainX, _accordTrainY);

        var testData = new double[10][];
        for (int i = 0; i < 10; i++)
        {
            testData[i] = new double[FeatureCount];
            for (int j = 0; j < FeatureCount; j++)
            {
                testData[i][j] = _aiTestX[i, j];
            }
        }

        return model.Transform(testData);
    }

    #endregion

    #region Polynomial Regression

    [Benchmark]
    public PolynomialRegression<double> AiDotNet_PolynomialRegression_Train()
    {
        var model = new PolynomialRegression<double>(degree: 2);
        var singleFeature = new Matrix<double>(SampleCount, 1);
        for (int i = 0; i < SampleCount; i++)
        {
            singleFeature[i, 0] = _aiTrainX[i, 0];
        }
        model.Fit(singleFeature, _aiTrainY);
        return model;
    }

    [Benchmark]
    public PolynomialRegression AccordNet_PolynomialRegression_Train()
    {
        var x = new double[SampleCount];
        for (int i = 0; i < SampleCount; i++)
        {
            x[i] = _aiTrainX[i, 0];
        }
        var model = new PolynomialRegression(degree: 2);
        model.Regress(x, _accordTrainY);
        return model;
    }

    #endregion

    #region Ridge Regression

    [Benchmark]
    public RidgeRegression<double> AiDotNet_RidgeRegression_Train()
    {
        var model = new RidgeRegression<double>(alpha: 1.0);
        model.Fit(_aiTrainX, _aiTrainY);
        return model;
    }

    [Benchmark]
    public MultipleLinearRegression AccordNet_RidgeRegression_Train()
    {
        var model = new MultipleLinearRegression();
        model.Regress(_accordTrainX, _accordTrainY);
        return model;
    }

    #endregion
}

// Helper class for ML.NET
public class RegressionData
{
    [VectorType]
    public float[] Features { get; set; } = Array.Empty<float>();

    public float Label { get; set; }
}
