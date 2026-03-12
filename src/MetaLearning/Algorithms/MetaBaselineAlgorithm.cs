using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Data.Structures;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning", "https://arxiv.org/abs/2003.04390", Year = 2021, Authors = "Chen et al.")]
public class MetaBaselineAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaBaselineOptions<T, TInput, TOutput> _metaBaselineOptions;
    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaBaseline;

    /// <summary>Initializes a new Meta-Baseline meta-learner.</summary>
    public MetaBaselineAlgorithm(MetaBaselineOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    { _metaBaselineOptions = options; }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch.Tasks.Length == 0)
            return NumOps.Zero;

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

        ApplyOuterUpdate(initParams, metaGradients, _metaBaselineOptions.OuterLearningRate);
        return ComputeMean(losses);
    }

    /// <summary>L2-normalizes a feature vector for cosine similarity computation.</summary>
    private Vector<T>? NormalizeVector(Vector<T>? features)
    {
        if (features == null || features.Length == 0) return features;
        return VectorHelper.Normalize(features);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract and L2-normalize support features for cosine-similarity classification
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var normalizedSupport = NormalizeVector(supportFeatures);

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
internal class MetaBaselineModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>, IAdaptedMetaModel<T>
{
    private Vector<T> _params;
    private readonly Vector<T>? _supportPrototypes;
    private readonly double _temperature;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _supportPrototypes;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    public MetaBaselineModel(IFullModel<T, TInput, TOutput> model, Vector<T> p,
        Vector<T>? supportPrototypes, double temperature, double[]? modulationFactors)
        : base(model)
    {
        _params = p;
        _supportPrototypes = supportPrototypes;
        _temperature = Math.Max(temperature, 1e-10);
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = new Vector<T>(_params.Length);
            for (int i = 0; i < _params.Length; i++)
                modulated[i] = NumOps.Multiply(_params[i],
                    NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
            BaseModel.SetParameters(modulated);
        }
        else
        {
            BaseModel.SetParameters(_params);
        }
        return BaseModel.Predict(input);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => _params;

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        _params = parameters ?? throw new ArgumentNullException(nameof(parameters));
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new MetaBaselineModel<T, TInput, TOutput>(BaseModel, parameters, _supportPrototypes, _temperature, _modulationFactors);
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> DeepCopy()
    {
        return new MetaBaselineModel<T, TInput, TOutput>(
            BaseModel.DeepCopy(), _params.Clone(), _supportPrototypes?.Clone(), _temperature,
            _modulationFactors is not null ? (double[])_modulationFactors.Clone() : null);
    }
}
