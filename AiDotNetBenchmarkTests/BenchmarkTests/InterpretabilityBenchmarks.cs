using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for Interpretability features
/// Tests model explainability (LIME, SHAP, Anchors), fairness evaluation, and bias detection
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class InterpretabilityBenchmarks
{
    [Params(100, 500)]
    public int SampleSize { get; set; }

    [Params(10, 30)]
    public int FeatureCount { get; set; }

    private Matrix<double> _inputs = null!;
    private Vector<double> _predictions = null!;
    private Vector<double> _actualLabels = null!;
    private SimpleTestModel _model = null!;
    private int _sensitiveFeatureIndex;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _sensitiveFeatureIndex = 0; // First feature is sensitive

        // Create synthetic data
        _inputs = new Matrix<double>(SampleSize, FeatureCount);
        _predictions = new Vector<double>(SampleSize);
        _actualLabels = new Vector<double>(SampleSize);

        for (int i = 0; i < SampleSize; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                _inputs[i, j] = random.NextDouble() * 2 - 1;
            }
            // Binary predictions
            _predictions[i] = random.NextDouble() > 0.5 ? 1.0 : 0.0;
            _actualLabels[i] = random.NextDouble() > 0.5 ? 1.0 : 0.0;
        }

        _model = new SimpleTestModel();
    }

    #region Fairness Evaluators

    [Benchmark(Baseline = true)]
    public FairnessMetrics<double> Interpretability01_BasicFairnessEvaluator()
    {
        var evaluator = new BasicFairnessEvaluator<double>();
        return evaluator.EvaluateFairness(_model, _inputs, _sensitiveFeatureIndex, _actualLabels);
    }

    [Benchmark]
    public FairnessMetrics<double> Interpretability02_GroupFairnessEvaluator()
    {
        var evaluator = new GroupFairnessEvaluator<double>();
        return evaluator.EvaluateFairness(_model, _inputs, _sensitiveFeatureIndex, _actualLabels);
    }

    [Benchmark]
    public FairnessMetrics<double> Interpretability03_ComprehensiveFairnessEvaluator()
    {
        var evaluator = new ComprehensiveFairnessEvaluator<double>();
        return evaluator.EvaluateFairness(_model, _inputs, _sensitiveFeatureIndex, _actualLabels);
    }

    #endregion

    #region Bias Detectors

    [Benchmark]
    public BiasDetectionResult<double> Interpretability04_DemographicParityBiasDetector()
    {
        var detector = new DemographicParityBiasDetector<double>();
        return detector.DetectBias(_model, _inputs, _sensitiveFeatureIndex, _actualLabels);
    }

    [Benchmark]
    public BiasDetectionResult<double> Interpretability05_DisparateImpactBiasDetector()
    {
        var detector = new DisparateImpactBiasDetector<double>();
        return detector.DetectBias(_model, _inputs, _sensitiveFeatureIndex, _actualLabels);
    }

    [Benchmark]
    public BiasDetectionResult<double> Interpretability06_EqualOpportunityBiasDetector()
    {
        var detector = new EqualOpportunityBiasDetector<double>();
        return detector.DetectBias(_model, _inputs, _sensitiveFeatureIndex, _actualLabels);
    }

    #endregion

    #region Model Explanation Structures

    [Benchmark]
    public LimeExplanation<double> Interpretability07_CreateLimeExplanation()
    {
        var explanation = new LimeExplanation<double>
        {
            NumFeatures = FeatureCount,
            PredictedValue = 0.75,
            Intercept = 0.1,
            LocalModelScore = 0.85
        };

        for (int i = 0; i < 5; i++)
        {
            explanation.FeatureImportance[i] = 0.1 * i;
        }

        return explanation;
    }

    [Benchmark]
    public AnchorExplanation<double> Interpretability08_CreateAnchorExplanation()
    {
        var explanation = new AnchorExplanation<double>
        {
            Precision = 0.95,
            Coverage = 0.75,
            PredictedClass = 1
        };

        explanation.AnchorRules.Add("Feature_0 > 0.5");
        explanation.AnchorRules.Add("Feature_1 < 0.3");

        return explanation;
    }

    [Benchmark]
    public CounterfactualExplanation<double> Interpretability09_CreateCounterfactualExplanation()
    {
        var explanation = new CounterfactualExplanation<double>
        {
            OriginalInstance = new Vector<double>(FeatureCount),
            CounterfactualInstance = new Vector<double>(FeatureCount),
            OriginalPrediction = 0.0,
            CounterfactualPrediction = 1.0,
            Distance = 0.25
        };

        return explanation;
    }

    #endregion

    #region Helper Methods Performance

    [Benchmark]
    public List<double> Interpretability10_GetUniqueGroups()
    {
        var sensitiveFeature = _inputs.GetColumn(_sensitiveFeatureIndex);
        return InterpretabilityMetricsHelper<double>.GetUniqueGroups(sensitiveFeature);
    }

    [Benchmark]
    public List<int> Interpretability11_GetGroupIndices()
    {
        var sensitiveFeature = _inputs.GetColumn(_sensitiveFeatureIndex);
        var groups = InterpretabilityMetricsHelper<double>.GetUniqueGroups(sensitiveFeature);
        if (groups.Count > 0)
        {
            return InterpretabilityMetricsHelper<double>.GetGroupIndices(sensitiveFeature, groups[0]);
        }
        return new List<int>();
    }

    [Benchmark]
    public Vector<double> Interpretability12_GetSubset()
    {
        var indices = new List<int> { 0, 2, 4, 6, 8 };
        return InterpretabilityMetricsHelper<double>.GetSubset(_predictions, indices);
    }

    [Benchmark]
    public double Interpretability13_ComputePositiveRate()
    {
        return InterpretabilityMetricsHelper<double>.ComputePositiveRate(_predictions);
    }

    [Benchmark]
    public double Interpretability14_ComputeTruePositiveRate()
    {
        return InterpretabilityMetricsHelper<double>.ComputeTruePositiveRate(_predictions, _actualLabels);
    }

    [Benchmark]
    public double Interpretability15_ComputeFalsePositiveRate()
    {
        return InterpretabilityMetricsHelper<double>.ComputeFalsePositiveRate(_predictions, _actualLabels);
    }

    [Benchmark]
    public double Interpretability16_ComputePrecision()
    {
        return InterpretabilityMetricsHelper<double>.ComputePrecision(_predictions, _actualLabels);
    }

    #endregion

    /// <summary>
    /// Simple test model for benchmarking
    /// </summary>
    private class SimpleTestModel : IFullModel<double, Matrix<double>, Vector<double>>
    {
        public Vector<double> Predict(Matrix<double> input)
        {
            var random = new Random(42);
            var result = new Vector<double>(input.Rows);
            for (int i = 0; i < input.Rows; i++)
            {
                result[i] = random.NextDouble() > 0.5 ? 1.0 : 0.0;
            }
            return result;
        }

        public void Train(Matrix<double> inputs, Vector<double> outputs)
        {
            // No-op for benchmark
        }
    }
}
