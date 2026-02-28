using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;

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
        RepresentationDim = representationDim;
        EncoderParams = InitializeParams(representationDim * 2);
        DecoderParams = InitializeParams(representationDim * 2);
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
        var rng = RandomGenerator;

        for (int i = 0; i < mean.Length; i++)
        {
            double eps = NormalSample(rng);
            double std = Math.Exp(0.5 * NumOps.ToDouble(logvar[i]));
            z[i] = NumOps.FromDouble(NumOps.ToDouble(mean[i]) + std * eps);
        }

        return z;
    }

    /// <summary>Samples from a standard normal distribution using Box-Muller transform.</summary>
    private static double NormalSample(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

/// <summary>Adapted model for Neural Process family algorithms.</summary>
internal class NeuralProcessModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    private readonly Vector<T>? _contextRepresentation;

    public Vector<T>? AdaptedSupportFeatures => _contextRepresentation;
    public double[]? ParameterModulationFactors => null;
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public NeuralProcessModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> parameters,
        Vector<T>? contextRepresentation)
    {
        _model = model;
        _params = parameters;
        _contextRepresentation = contextRepresentation;
    }

    public TOutput Predict(TInput input)
    {
        _model.SetParameters(_params);
        return _model.Predict(input);
    }

    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training.");

    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
