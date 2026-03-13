using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Base class for the Neural Process family of meta-learning algorithms.
/// Provides shared infrastructure for encoding context sets, aggregating representations,
/// and decoding target predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Neural Processes (NPs) are a family of models that define a distribution over functions.
/// They encode a context set (observed points) into a representation and use it to make
/// predictions at target points. Unlike gradient-based meta-learners (MAML), NPs perform
/// adaptation through a single forward pass via amortized inference.
/// </para>
/// <para><b>For Beginners:</b> Neural Processes are like learning a recipe for making predictions:
///
/// 1. **Context set** = examples you've already seen (like support set in few-shot learning)
/// 2. **Encoder** = summarizes what you've seen into a compact representation
/// 3. **Aggregator** = combines individual summaries into one global summary
/// 4. **Decoder** = uses the summary to make predictions at new points
///
/// Different NP variants differ in how they encode, aggregate, and decode:
/// - CNP: Simple mean aggregation, point predictions
/// - NP: Adds a latent variable for uncertainty
/// - ANP: Adds attention for better predictions
/// </para>
/// </remarks>
public abstract class NeuralProcessBase<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    /// <summary>Learned encoder parameters for mapping context pairs to representations.</summary>
    protected Vector<T> EncoderParams;

    /// <summary>Learned decoder parameters for mapping representations to predictions.</summary>
    protected Vector<T> DecoderParams;

    /// <summary>Dimensionality of the representation space.</summary>
    protected readonly int RepresentationDim;

    /// <summary>Initializes shared NP infrastructure.</summary>
    protected NeuralProcessBase(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IMetaLearnerOptions<T> options,
        IEpisodicDataLoader<T, TInput, TOutput>? dataLoader,
        IGradientBasedOptimizer<T, TInput, TOutput>? metaOptimizer,
        IGradientBasedOptimizer<T, TInput, TOutput>? innerOptimizer,
        int representationDim = 128)
        : base(metaModel, lossFunction, options, dataLoader, metaOptimizer, innerOptimizer)
    {
        if (representationDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(representationDim), "Representation dimension must be positive.");

        RepresentationDim = representationDim;
        EncoderParams = InitializeParams(representationDim * 2);
        DecoderParams = InitializeParams(representationDim * 2);
    }

    /// <summary>Base value for the modulation scale sigmoid.</summary>
    private const double ModScaleBase = 0.5;

    /// <summary>Range of the modulation scale sigmoid output.</summary>
    private const double ModScaleRange = 0.5;

    /// <summary>Center of the sigmoid for modulation scale computation.</summary>
    private const double ModScaleSigmoidCenter = 1.0;

    /// <summary>
    /// Builds context representations from support features and labels by encoding each
    /// example as a context pair via <see cref="EncodeContextPair"/>.
    /// </summary>
    protected List<Vector<T>> BuildContextRepresentations(Vector<T>? supportFeatures, Vector<T>? supportLabels)
    {
        var contextReps = new List<Vector<T>>();
        if (supportFeatures != null && supportLabels != null && supportFeatures.Length > 0 && supportLabels.Length > 0)
        {
            int numEx = Math.Max(1, supportLabels.Length);
            int fDim = Math.Max(1, supportFeatures.Length / numEx);
            for (int i = 0; i < numEx; i++)
            {
                int fStart = i * fDim;
                int fLen = Math.Min(fDim, supportFeatures.Length - fStart);
                if (fLen <= 0) break;
                var f = new Vector<T>(fLen);
                for (int j = 0; j < fLen; j++) f[j] = supportFeatures[fStart + j];
                var l = new Vector<T>(1);
                l[0] = supportLabels[Math.Min(i, supportLabels.Length - 1)];
                contextReps.Add(EncodeContextPair(f, l));
            }
        }
        return contextReps;
    }

    /// <summary>
    /// Computes a modulation scale from a representation vector using a shifted sigmoid
    /// that maps representation norm to [ModScaleBase, ModScaleBase + ModScaleRange].
    /// </summary>
    protected double ComputeModScale(Vector<T> rep)
    {
        double norm = 0;
        for (int i = 0; i < rep.Length; i++) norm += NumOps.ToDouble(rep[i]) * NumOps.ToDouble(rep[i]);
        norm = Math.Sqrt(norm / Math.Max(rep.Length, 1));
        return ModScaleBase + ModScaleRange / (1.0 + Math.Exp(-norm + ModScaleSigmoidCenter));
    }

    /// <summary>Initializes parameter vector with small random values.</summary>
    protected Vector<T> InitializeParams(int size)
    {
        var rng = RandomGenerator;
        var p = new Vector<T>(size);
        for (int i = 0; i < size; i++)
            p[i] = NumOps.FromDouble(rng.NextDouble() * 0.1 - 0.05);
        return p;
    }

    /// <summary>
    /// Encodes a single context pair (x, y) into a representation vector.
    /// Uses a simple linear projection: r = tanh(W * concat(features, label_features)).
    /// </summary>
    protected Vector<T> EncodeContextPair(Vector<T> features, Vector<T> labelFeatures)
    {
        var result = new Vector<T>(RepresentationDim);
        int inputDim = features.Length + labelFeatures.Length;

        for (int i = 0; i < RepresentationDim; i++)
        {
            double sum = 0;
            for (int j = 0; j < features.Length && j < RepresentationDim; j++)
            {
                int paramIdx = (i * inputDim + j) % EncoderParams.Length;
                sum += NumOps.ToDouble(features[j]) * NumOps.ToDouble(EncoderParams[paramIdx]);
            }
            for (int j = 0; j < labelFeatures.Length && j < RepresentationDim; j++)
            {
                int paramIdx = (i * inputDim + features.Length + j) % EncoderParams.Length;
                sum += NumOps.ToDouble(labelFeatures[j]) * NumOps.ToDouble(EncoderParams[paramIdx]);
            }
            result[i] = NumOps.FromDouble(Math.Tanh(sum));
        }

        return result;
    }

    /// <summary>
    /// Aggregates multiple context representations into a single representation via mean pooling.
    /// </summary>
    protected Vector<T> AggregateRepresentations(List<Vector<T>> representations)
    {
        if (representations.Count == 0)
            return new Vector<T>(RepresentationDim);

        var agg = new Vector<T>(RepresentationDim);
        foreach (var r in representations)
        {
            for (int i = 0; i < Math.Min(RepresentationDim, r.Length); i++)
                agg[i] = NumOps.Add(agg[i], r[i]);
        }

        double invCount = 1.0 / representations.Count;
        for (int i = 0; i < RepresentationDim; i++)
            agg[i] = NumOps.Multiply(agg[i], NumOps.FromDouble(invCount));

        return agg;
    }

    /// <summary>
    /// Decodes a representation + target features into a prediction.
    /// Uses a simple linear projection: output = W * concat(representation, target_features).
    /// </summary>
    protected Vector<T> DecodeTarget(Vector<T> representation, Vector<T> targetFeatures)
    {
        int outputDim = Math.Max(1, targetFeatures.Length);
        var result = new Vector<T>(outputDim);
        int inputDim = representation.Length + targetFeatures.Length;

        for (int i = 0; i < outputDim; i++)
        {
            double sum = 0;
            for (int j = 0; j < representation.Length; j++)
            {
                int paramIdx = (i * inputDim + j) % DecoderParams.Length;
                sum += NumOps.ToDouble(representation[j]) * NumOps.ToDouble(DecoderParams[paramIdx]);
            }
            for (int j = 0; j < targetFeatures.Length; j++)
            {
                int paramIdx = (i * inputDim + representation.Length + j) % DecoderParams.Length;
                sum += NumOps.ToDouble(targetFeatures[j]) * NumOps.ToDouble(DecoderParams[paramIdx]);
            }
            result[i] = NumOps.FromDouble(sum);
        }

        return result;
    }

    /// <summary>
    /// Computes the KL divergence between two Gaussian distributions parameterized by
    /// (mean1, logvar1) and (mean2, logvar2).
    /// </summary>
    protected double KLDivergenceGaussian(Vector<T> mean1, Vector<T> logvar1, Vector<T> mean2, Vector<T> logvar2)
    {
        double kl = 0;
        int dim = Math.Min(mean1.Length, mean2.Length);

        for (int i = 0; i < dim; i++)
        {
            double m1 = NumOps.ToDouble(mean1[i]);
            double lv1 = NumOps.ToDouble(logvar1[i]);
            double m2 = NumOps.ToDouble(mean2[i]);
            double lv2 = NumOps.ToDouble(logvar2[i]);

            double v1 = Math.Exp(lv1);
            double v2 = Math.Exp(lv2);

            kl += lv2 - lv1 + (v1 + (m1 - m2) * (m1 - m2)) / (2 * v2) - 0.5;
        }

        return Math.Max(kl, 0);
    }

    /// <summary>
    /// Samples from a Gaussian distribution using the reparameterization trick:
    /// z = mean + exp(0.5 * logvar) * epsilon, where epsilon ~ N(0,1).
    /// </summary>
    protected Vector<T> ReparameterizeSample(Vector<T> mean, Vector<T> logvar)
    {
        var z = new Vector<T>(mean.Length);

        for (int i = 0; i < mean.Length; i++)
        {
            double eps = SampleNormal();
            double std = Math.Exp(0.5 * NumOps.ToDouble(logvar[i]));
            z[i] = NumOps.FromDouble(NumOps.ToDouble(mean[i]) + std * eps);
        }

        return z;
    }

    /// <summary>
    /// Modulates backbone parameters by a scale derived from the context representation,
    /// then sets them on the model.
    /// </summary>
    protected void ModulateParameters(Vector<T> initParams, double scale)
    {
        MetaModel.SetParameters(ScaleVector(initParams, scale));
    }

    /// <summary>
    /// Standard NP meta-training loop shared by simple NP variants (EquivCNP, SwinTNP, TNP, etc.).
    /// For each task: build context reps, aggregate, modulate backbone, compute loss and gradients.
    /// Then apply outer update and update encoder params via SPSA.
    /// </summary>
    protected T StandardNPMetaTrain(TaskBatch<T, TInput, TOutput> taskBatch, double outerLR)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var supportLabels = ConvertToVector(task.SupportOutput);
            var contextReps = BuildContextRepresentations(supportFeatures, supportLabels);

            var aggRep = AggregateRepresentations(contextReps);
            double scale = ComputeModScale(aggRep);
            ModulateParameters(initParams, scale);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, outerLR);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref EncoderParams, outerLR, StandardAuxLoss);
        return ComputeMean(losses);
    }

    /// <summary>
    /// Standard NP adaptation: build context reps, aggregate, modulate, return adapted model.
    /// </summary>
    protected IModel<TInput, TOutput, ModelMetadata<T>> StandardNPAdapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);
        var contextReps = BuildContextRepresentations(supportFeatures, supportLabels);

        var aggRep = AggregateRepresentations(contextReps);
        double sc = ComputeModScale(aggRep);

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, ScaleVector(currentParams, sc), aggRep);
    }
}

/// <summary>Adapted model for Neural Process family algorithms.</summary>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
public class NeuralProcessModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>, IAdaptedMetaModel<T>
{
    private Vector<T> _params;
    private readonly Vector<T>? _contextRepresentation;

    public Vector<T>? AdaptedSupportFeatures => _contextRepresentation;
    public double[]? ParameterModulationFactors => null;

    public NeuralProcessModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> parameters,
        Vector<T>? contextRepresentation)
        : base(model)
    {
        _params = parameters;
        _contextRepresentation = contextRepresentation;
    }

    public override TOutput Predict(TInput input)
    {
        BaseModel.SetParameters(_params);
        return BaseModel.Predict(input);
    }

    public override Vector<T> GetParameters() => _params;

    public override void SetParameters(Vector<T> parameters)
    {
        _params = parameters ?? throw new ArgumentNullException(nameof(parameters));
    }

    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new NeuralProcessModel<T, TInput, TOutput>(BaseModel, parameters, _contextRepresentation);
    }

    public override IFullModel<T, TInput, TOutput> DeepCopy()
    {
        return new NeuralProcessModel<T, TInput, TOutput>(
            BaseModel.DeepCopy(), _params.Clone(), _contextRepresentation?.Clone());
    }
}
