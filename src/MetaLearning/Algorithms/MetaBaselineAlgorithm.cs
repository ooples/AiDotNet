using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-Baseline (simple pre-train then meta-train with cosine classifier).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-Baseline trains a feature extractor with standard classification, then fine-tunes
/// with episodic training using cosine similarity nearest-centroid classification.
/// </para>
/// <para><b>For Beginners:</b> Meta-Baseline shows that the simplest approach can be the best:
/// 1. Train normally on many classes to get good features
/// 2. Switch to episodic training with cosine-distance centroids
/// 3. At test time, classify by nearest centroid (cosine distance)
/// </para>
/// <para>
/// Reference: Chen, Y., Liu, Z., Xu, H., Darrell, T., &amp; Wang, X. (2021).
/// Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning. ICLR 2021.
/// </para>
/// </remarks>
public class MetaBaselineAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaBaselineOptions<T, TInput, TOutput> _metaBaselineOptions;
    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaBaseline;

    /// <summary>Initializes a new Meta-Baseline meta-learner.</summary>
    public MetaBaselineAlgorithm(MetaBaselineOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    { _metaBaselineOptions = options; }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _metaBaselineOptions.OuterLearningRate));
        }
        return ComputeMean(losses);
    }

    /// <summary>L2-normalizes a feature vector for cosine similarity computation.</summary>
    private Vector<T>? L2Normalize(Vector<T>? features)
    {
        if (features == null || features.Length == 0) return features;
        double normSq = 0;
        for (int i = 0; i < features.Length; i++)
        {
            double v = NumOps.ToDouble(features[i]);
            normSq += v * v;
        }
        double norm = Math.Sqrt(normSq);
        if (norm < 1e-10) return features;
        var normalized = new Vector<T>(features.Length);
        for (int i = 0; i < features.Length; i++)
            normalized[i] = NumOps.Divide(features[i], NumOps.FromDouble(norm));
        return normalized;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract and L2-normalize support features for cosine-similarity classification
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var normalizedSupport = L2Normalize(supportFeatures);

        // Compute modulation: cosine normalization changes feature magnitudes
        double[]? modulationFactors = null;
        if (supportFeatures != null && normalizedSupport != null)
        {
            double sumRatio = 0;
            int count = 0;
            for (int i = 0; i < Math.Min(supportFeatures.Length, normalizedSupport.Length); i++)
            {
                double rawVal = NumOps.ToDouble(supportFeatures[i]);
                double adaptedVal = NumOps.ToDouble(normalizedSupport[i]);
                if (Math.Abs(rawVal) > 1e-10)
                {
                    sumRatio += Math.Max(0.5, Math.Min(2.0, adaptedVal / rawVal));
                    count++;
                }
            }
            if (count > 0)
                modulationFactors = [sumRatio / count];
        }

        return new MetaBaselineModel<T, TInput, TOutput>(
            MetaModel, currentParams, normalizedSupport,
            _metaBaselineOptions.Temperature, modulationFactors);
    }

}

/// <summary>Adapted model wrapper for Meta-Baseline with cosine-similarity classification.</summary>
internal class MetaBaselineModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    private readonly Vector<T>? _supportPrototypes;
    private readonly double _temperature;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _supportPrototypes;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public MetaBaselineModel(IFullModel<T, TInput, TOutput> model, Vector<T> p,
        Vector<T>? supportPrototypes, double temperature, double[]? modulationFactors)
    {
        _model = model;
        _params = p;
        _supportPrototypes = supportPrototypes;
        _temperature = Math.Max(temperature, 1e-10);
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = new Vector<T>(_params.Length);
            for (int i = 0; i < _params.Length; i++)
                modulated[i] = NumOps.Multiply(_params[i],
                    NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
            _model.SetParameters(modulated);
        }
        else
        {
            _model.SetParameters(_params);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
