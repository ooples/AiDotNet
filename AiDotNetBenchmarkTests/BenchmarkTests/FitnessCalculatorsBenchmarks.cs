using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Evaluation;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for all FitnessCalculator implementations
/// Tests all 26+ fitness calculators used for model evaluation
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class FitnessCalculatorsBenchmarks
{
    [Params(100, 500)]
    public int SampleSize { get; set; }

    [Params(10, 50)]
    public int FeatureCount { get; set; }

    private ModelEvaluationData<double, Matrix<double>, Vector<double>> _evaluationData = null!;
    private DataSetStats<double, Matrix<double>, Vector<double>> _dataSetStats = null!;
    private Matrix<double> _inputs = null!;
    private Vector<double> _actualOutputs = null!;
    private Vector<double> _predictedOutputs = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Create synthetic data
        _inputs = new Matrix<double>(SampleSize, FeatureCount);
        _actualOutputs = new Vector<double>(SampleSize);
        _predictedOutputs = new Vector<double>(SampleSize);

        for (int i = 0; i < SampleSize; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                _inputs[i, j] = random.NextDouble() * 2 - 1;
            }
            _actualOutputs[i] = random.NextDouble() * 10;
            _predictedOutputs[i] = _actualOutputs[i] + (random.NextDouble() - 0.5) * 2; // Add noise
        }

        // Create DataSetStats
        _dataSetStats = new DataSetStats<double, Matrix<double>, Vector<double>>
        {
            Inputs = _inputs,
            ActualOutputs = _actualOutputs,
            PredictedOutputs = _predictedOutputs
        };

        // Create ModelEvaluationData
        _evaluationData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>
        {
            ValidationSet = _dataSetStats
        };
    }

    #region Error-Based Fitness Calculators

    [Benchmark(Baseline = true)]
    public double Fitness01_MeanSquaredError()
    {
        var calculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness02_RootMeanSquaredError()
    {
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness03_MeanAbsoluteError()
    {
        var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness04_HuberLoss()
    {
        var calculator = new HuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness05_ModifiedHuberLoss()
    {
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness06_LogCoshLoss()
    {
        var calculator = new LogCoshLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness07_QuantileLoss()
    {
        var calculator = new QuantileLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    #endregion

    #region R-Squared and Correlation-Based Calculators

    [Benchmark]
    public double Fitness08_RSquared()
    {
        var calculator = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness09_AdjustedRSquared()
    {
        var calculator = new AdjustedRSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>(
            numFeatures: FeatureCount);
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    #endregion

    #region Classification Loss Functions

    [Benchmark]
    public double Fitness10_CrossEntropyLoss()
    {
        var calculator = new CrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness11_BinaryCrossEntropyLoss()
    {
        var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness12_CategoricalCrossEntropyLoss()
    {
        var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness13_WeightedCrossEntropyLoss()
    {
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>(
            positiveWeight: 1.5);
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness14_HingeLoss()
    {
        var calculator = new HingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness15_SquaredHingeLoss()
    {
        var calculator = new SquaredHingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness16_FocalLoss()
    {
        var calculator = new FocalLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    #endregion

    #region Specialized Loss Functions

    [Benchmark]
    public double Fitness17_KullbackLeiblerDivergence()
    {
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness18_ElasticNetLoss()
    {
        var calculator = new ElasticNetLossFitnessCalculator<double, Matrix<double>, Vector<double>>(
            l1Ratio: 0.5, alpha: 1.0);
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness19_PoissonLoss()
    {
        var calculator = new PoissonLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness20_ExponentialLoss()
    {
        var calculator = new ExponentialLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness21_OrdinalRegressionLoss()
    {
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    #endregion

    #region Segmentation and Similarity Losses

    [Benchmark]
    public double Fitness22_JaccardLoss()
    {
        var calculator = new JaccardLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness23_DiceLoss()
    {
        var calculator = new DiceLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness24_CosineSimilarityLoss()
    {
        var calculator = new CosineSimilarityLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness25_ContrastiveLoss()
    {
        var calculator = new ContrastiveLossFitnessCalculator<double, Matrix<double>, Vector<double>>(
            margin: 1.0);
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    [Benchmark]
    public double Fitness26_TripletLoss()
    {
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>(
            margin: 1.0);
        return calculator.CalculateFitnessScore(_evaluationData);
    }

    #endregion

    #region IsBetterFitness Comparison Tests

    [Benchmark]
    public bool Fitness_CompareScores()
    {
        var calculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        double score1 = 0.5;
        double score2 = 0.7;
        return calculator.IsBetterFitness(score1, score2);
    }

    #endregion
}
