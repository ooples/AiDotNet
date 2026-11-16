using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Modules;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// Relation Network model for few-shot classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// The adapted state of a Relation Network for one task: its own copy of the embedding module, the support set's
/// embeddings and labels, and the learned comparison. <see cref="Predict"/> returns each example's relation score
/// with every class, <c>[rows, NumClasses]</c>, for Tensor and Matrix outputs - a class with no support example
/// scores zero - and the class with the highest score for a Vector output.
/// </para>
/// <para><b>For Beginners:</b> After the Relation Network sees the support examples (the few
/// labeled examples for each class), this model remembers them and uses them to classify
/// new query examples. It does this by computing how "related" the query is to each class.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Learning to Compare: Relation Network for Few-Shot Learning",
    "https://arxiv.org/abs/1711.06025",
    Year = 2018,
    Authors = "Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H. S., & Hospedales, T. M.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public class RelationNetworkModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{

    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly RelationNetworkOptions<T, TInput, TOutput> _options;
    private readonly Vector<T> _relationWeights;
    private readonly RelationScorer<T> _scorer;
    private readonly Tensor<T> _support;
    private readonly Tensor<T> _membership;
    private readonly int[] _shotSlots;
    private readonly int[] _classSlots;
    private readonly Tensor<T> _classColumns;

    /// <summary>
    /// Initializes a new instance of the RelationNetworkModel around one relation module.
    /// </summary>
    /// <param name="featureEncoder">The feature encoder network; the model keeps its own copy.</param>
    /// <param name="relationModule">The relation module; sized to the embedding width if it has not been yet.</param>
    /// <param name="supportInputs">The support set inputs.</param>
    /// <param name="supportOutputs">The support set labels.</param>
    /// <param name="options">The Relation Network options.</param>
    /// <remarks>
    /// The other learned parts take their untrained values: no feature map, and uniform pooling over shots.
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public RelationNetworkModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        RelationModule<T> relationModule,
        TInput supportInputs,
        TOutput supportOutputs,
        RelationNetworkOptions<T, TInput, TOutput> options)
        : this(Parts.FromModule(featureEncoder, relationModule, supportInputs, supportOutputs, options))
    {
    }

    /// <summary>Initializes the model with the comparison a Relation Network learned.</summary>
    internal RelationNetworkModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        TInput supportInputs,
        TOutput supportOutputs,
        RelationNetworkOptions<T, TInput, TOutput> options,
        int heads,
        Vector<T> relationWeights,
        Vector<T> featureTransform,
        Vector<T> poolingWeights,
        Vector<T> shotWeights)
        : this(Parts.FromLearned(featureEncoder, supportInputs, supportOutputs, options, heads,
            relationWeights, featureTransform, poolingWeights, shotWeights))
    {
    }

    private RelationNetworkModel(Parts parts)
    {
        _featureEncoder = parts.Encoder;
        _options = parts.Options;
        _relationWeights = parts.RelationWeights;
        _support = parts.Support;

        var episode = PrototypeEpisode<T>.Build(parts.Labels, Array.Empty<int>());
        _membership = episode.Membership;
        _classSlots = episode.ClassSlots;
        _shotSlots = RelationScorer<T>.ShotSlots(parts.Labels);
        int numClasses = parts.Options.NumClasses;
        // classColumns[c, slot_c] = 1: each episode column moves to its class label's column.
        _classColumns = AiDotNetEngine.Current.TensorOneHot<T>(PrototypeEpisode<T>.IndexTensor(_classSlots), numClasses);

        _scorer = new RelationScorer<T>(
            parts.RelationType, parts.Hidden, parts.Options.AggregationMethod, parts.Heads, parts.RelationWeights,
            parts.FeatureTransform, parts.PoolingWeights, parts.ShotWeights, parts.Support.Shape[1], null, 0.0);
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var query = ClassifierOutputs<T>.AsRows(_featureEncoder.Predict(input));
        var scores = _scorer.Scores(_support, query, _membership, _shotSlots);

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            // The highest-scoring column of each row (the first on a tie), read out as its class label.
            int rows = scores.Shape[0];
            if (rows == 0) return (TOutput)(object)new Vector<T>(0);
            var best = engine.TensorArgMax(scores, 1);
            var slots = PrototypeEpisode<T>.IndexTensor(_classSlots).Cast<T>();
            return (TOutput)(object)engine.TensorIndexSelect(slots, best, 0).ToVector();
        }

        return ClassifierOutputs<T>.ToOutput<TOutput>(engine.TensorMatMul(scores, _classColumns));
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the training algorithm to train Relation Networks.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("Relation Network parameters are updated during training.");
    }

    /// <inheritdoc/>
    /// <remarks>The embedding module's parameters, then the relation heads' weights.</remarks>
    public Vector<T> GetParameters()
    {
        var encoderParams = InterfaceGuard.Parameterizable(_featureEncoder).GetParameters();
        return Vector<T>.Concatenate(encoderParams, _relationWeights);
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }

    /// <summary>Everything a model is built from, resolved before its fields are set.</summary>
    private sealed class Parts
    {
        private Parts(
            IFullModel<T, TInput, TOutput> encoder, RelationNetworkOptions<T, TInput, TOutput> options, Tensor<T> support,
            int[] labels, RelationModuleType relationType, int hidden, int heads, Vector<T> relationWeights,
            Vector<T> featureTransform, Vector<T> poolingWeights, Vector<T> shotWeights)
        {
            Encoder = encoder;
            Options = options;
            Support = support;
            Labels = labels;
            RelationType = relationType;
            Hidden = hidden;
            Heads = heads;
            RelationWeights = relationWeights;
            FeatureTransform = featureTransform;
            PoolingWeights = poolingWeights;
            ShotWeights = shotWeights;
        }

        public IFullModel<T, TInput, TOutput> Encoder { get; }
        public RelationNetworkOptions<T, TInput, TOutput> Options { get; }
        public Tensor<T> Support { get; }
        public int[] Labels { get; }
        public RelationModuleType RelationType { get; }
        public int Hidden { get; }
        public int Heads { get; }
        public Vector<T> RelationWeights { get; }
        public Vector<T> FeatureTransform { get; }
        public Vector<T> PoolingWeights { get; }
        public Vector<T> ShotWeights { get; }

        public static Parts FromModule(
            IFullModel<T, TInput, TOutput> featureEncoder, RelationModule<T> relationModule, TInput supportInputs,
            TOutput supportOutputs, RelationNetworkOptions<T, TInput, TOutput> options)
        {
            Guard.NotNull(featureEncoder);
            Guard.NotNull(relationModule);
            Guard.NotNull(options);
            var encoder = featureEncoder.DeepCopy();
            var (support, labels) = Embed(encoder, supportInputs, supportOutputs, options);
            relationModule.EnsureInitialized(support.Shape[1], RandomHelper.CreateSecureRandom());
            var weights = relationModule.Weights.Clone();
            return new Parts(encoder, options, support, labels, relationModule.RelationType, relationModule.HiddenDimension, 1,
                weights, new Vector<T>(0), new Vector<T>(0), new Vector<T>(0));
        }

        public static Parts FromLearned(
            IFullModel<T, TInput, TOutput> featureEncoder, TInput supportInputs, TOutput supportOutputs,
            RelationNetworkOptions<T, TInput, TOutput> options, int heads, Vector<T> relationWeights,
            Vector<T> featureTransform, Vector<T> poolingWeights, Vector<T> shotWeights)
        {
            Guard.NotNull(featureEncoder);
            Guard.NotNull(options);
            var encoder = featureEncoder.DeepCopy();
            var (support, labels) = Embed(encoder, supportInputs, supportOutputs, options);
            return new Parts(encoder, options, support, labels, options.RelationType, options.RelationHiddenDimension, heads,
                relationWeights, featureTransform, poolingWeights, shotWeights);
        }

        private static (Tensor<T> Support, int[] Labels) Embed(
            IFullModel<T, TInput, TOutput> encoder, TInput supportInputs, TOutput supportOutputs,
            RelationNetworkOptions<T, TInput, TOutput> options)
        {
            var labelTensor = ClassifierOutputs<T>.Labels(supportOutputs, options.NumClasses);
            var labels = AiDotNetEngine.Current.TensorRound(labelTensor).Cast<int>().ToArray();
            if (labels.Length == 0)
            {
                throw new ArgumentException("The support set is empty, so there is nothing to compare with.", nameof(supportOutputs));
            }

            using var noGrad = new NoGradScope<T>();
            return (ClassifierOutputs<T>.AsRows(encoder.Predict(supportInputs)), labels);
        }
    }
}
