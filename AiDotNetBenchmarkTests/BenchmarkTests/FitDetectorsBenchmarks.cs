using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Evaluation;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for all FitDetector implementations
/// Tests overfitting/underfitting detection performance across 20+ fit detectors
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class FitDetectorsBenchmarks
{
    [Params(100, 500)]
    public int SampleSize { get; set; }

    [Params(10, 50)]
    public int FeatureCount { get; set; }

    private ModelEvaluationData<double, Matrix<double>, Vector<double>> _evaluationData = null!;
    private Matrix<double> _trainingInputs = null!;
    private Vector<double> _trainingOutputs = null!;
    private Matrix<double> _validationInputs = null!;
    private Vector<double> _validationOutputs = null!;
    private Matrix<double> _testInputs = null!;
    private Vector<double> _testOutputs = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Create synthetic data for training, validation, and test sets
        _trainingInputs = new Matrix<double>(SampleSize, FeatureCount);
        _trainingOutputs = new Vector<double>(SampleSize);
        _validationInputs = new Matrix<double>(SampleSize / 2, FeatureCount);
        _validationOutputs = new Vector<double>(SampleSize / 2);
        _testInputs = new Matrix<double>(SampleSize / 2, FeatureCount);
        _testOutputs = new Vector<double>(SampleSize / 2);

        // Generate synthetic data
        for (int i = 0; i < SampleSize; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                _trainingInputs[i, j] = random.NextDouble() * 2 - 1;
            }
            _trainingOutputs[i] = random.NextDouble() * 10;
        }

        for (int i = 0; i < SampleSize / 2; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                _validationInputs[i, j] = random.NextDouble() * 2 - 1;
                _testInputs[i, j] = random.NextDouble() * 2 - 1;
            }
            _validationOutputs[i] = random.NextDouble() * 10;
            _testOutputs[i] = random.NextDouble() * 10;
        }

        // Create evaluation data
        _evaluationData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>
        {
            TrainingSet = CreateDataSetStats(_trainingInputs, _trainingOutputs, _trainingOutputs),
            ValidationSet = CreateDataSetStats(_validationInputs, _validationOutputs, _validationOutputs),
            TestSet = CreateDataSetStats(_testInputs, _testOutputs, _testOutputs)
        };
    }

    private DataSetStats<double, Matrix<double>, Vector<double>> CreateDataSetStats(
        Matrix<double> inputs, Vector<double> actualOutputs, Vector<double> predictedOutputs)
    {
        return new DataSetStats<double, Matrix<double>, Vector<double>>
        {
            Inputs = inputs,
            ActualOutputs = actualOutputs,
            PredictedOutputs = predictedOutputs
        };
    }

    #region Default and Core FitDetectors

    [Benchmark(Baseline = true)]
    public FitDetectorResult<double> FitDetector01_Default()
    {
        var detector = new DefaultFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector02_CrossValidation()
    {
        var detector = new CrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector03_Adaptive()
    {
        var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector04_Ensemble()
    {
        var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    #endregion

    #region Residual-Based FitDetectors

    [Benchmark]
    public FitDetectorResult<double> FitDetector05_ResidualAnalysis()
    {
        var detector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector06_ResidualBootstrap()
    {
        var detector = new ResidualBootstrapFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector07_Autocorrelation()
    {
        var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    #endregion

    #region Statistical FitDetectors

    [Benchmark]
    public FitDetectorResult<double> FitDetector08_InformationCriteria()
    {
        var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector09_GaussianProcess()
    {
        var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector10_CookDistance()
    {
        var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector11_VIF()
    {
        var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    #endregion

    #region Resampling-Based FitDetectors

    [Benchmark]
    public FitDetectorResult<double> FitDetector12_Bootstrap()
    {
        var detector = new BootstrapFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector13_Jackknife()
    {
        var detector = new JackknifeFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector14_TimeSeriesCrossValidation()
    {
        var detector = new TimeSeriesCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    #endregion

    #region Feature and Model Analysis FitDetectors

    [Benchmark]
    public FitDetectorResult<double> FitDetector15_FeatureImportance()
    {
        var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector16_PartialDependencePlot()
    {
        var detector = new PartialDependencePlotFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector17_ShapleyValue()
    {
        var detector = new ShapleyValueFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector18_LearningCurve()
    {
        var detector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    #endregion

    #region Classification-Specific FitDetectors

    [Benchmark]
    public FitDetectorResult<double> FitDetector19_ROCCurve()
    {
        var detector = new ROCCurveFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    [Benchmark]
    public FitDetectorResult<double> FitDetector20_PrecisionRecallCurve()
    {
        var detector = new PrecisionRecallCurveFitDetector<double, Matrix<double>, Vector<double>>();
        return detector.DetectFit(_evaluationData);
    }

    #endregion
}
