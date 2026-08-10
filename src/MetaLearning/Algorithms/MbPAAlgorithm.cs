using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Memory-based Parameter Adaptation (MbPA): an episodic memory of <c>(embedding, target)</c> pairs
/// whose nearest neighbours are used to locally, transiently re-fit the output network before each
/// prediction.
/// </summary>
/// <remarks>
/// <para>
/// Sprechmann, Jayakumar, Rae, Pritzel, Puigdomènech Badia, Uria, Vinyals, Hassabis, Pascanu and
/// Blundell, "Memory-based Parameter Adaptation" (ICLR 2018, arXiv:1802.10542). The problem it
/// addresses: "Neural networks very gradually incorporate information into weights as they process
/// data, requiring very low learning rates. If the training distribution shifts, the network is slow
/// to adapt, and when it does adapt, it typically performs badly on the training distribution before
/// the shift."
/// </para>
/// <para>
/// The remedy is to stop trying to fix that with the weights at all. The network keeps an episodic
/// memory and, at prediction time, uses a context-based lookup to modify its weights DIRECTLY, for
/// that one prediction:
/// </para>
/// <code>
///   M = {(h_i, v_i)}          h_i = f_gamma(x_i)  (embedding network)   v_i = y_i
///
///   q = f_gamma(x)                                        query key
///   {(h_k, v_k)} = KNN(q, M)                              K nearest by Euclidean distance
///   w_k(x) ~ kern(h_k, q),   kern(h, q) = 1 / (eps + ||h - q||^2)
///
///   max_(theta_x)  log p(theta_x | theta) + sum_k w_k(x) log p(v_k | h_k, theta_x, x)
///        where     log p(theta_x | theta) ~ -||theta_x - theta||^2 / (2 alpha_M)
///
///   solved by T steps of
///        theta_x &lt;- theta_x - alpha_M grad sum_k w_k L_k(theta_x) - beta (theta_x - theta)
/// </code>
/// <para>
/// <b>Three properties do the work, and all three are implemented rather than described.</b>
/// </para>
/// <list type="number">
/// <item><description><b>The adaptation is TRANSIENT.</b> "These adapted parameters are used for
/// output computation but discarded thereafter." Nothing accumulates, which is precisely why
/// absorbing new evidence cannot damage old knowledge — the mechanism sidesteps catastrophic
/// forgetting rather than resisting it.</description></item>
/// <item><description><b>Only the OUTPUT network is adapted.</b> The embedding network stays fixed,
/// so keys written earlier remain comparable with the query computed now. Adapting the embedding
/// would silently invalidate every key already in memory.</description></item>
/// <item><description><b>The local learning rate is LARGE.</b> "Much higher learning rates can be
/// used for this local adaptation, reneging the need for many iterations over similar data before
/// good predictions can be made." The default here is 0.15, over a hundred times a typical training
/// rate — safe only because of the first two properties.</description></item>
/// </list>
/// <para>
/// <b>MbPA is not a bi-level meta-learner.</b> Unlike MAML and its relatives there is no inner/outer
/// loop and no meta-gradient: <see cref="MetaTrain"/> is ordinary parametric training that also
/// writes observed examples into memory. What makes MbPA a meta-algorithm lives entirely in
/// <see cref="Adapt"/>, and specifically in the model it returns, which re-adapts per input.
/// </para>
/// <para>
/// <b>Structure.</b> Following this library's ANIL convention, the configured meta-model is the
/// embedding network f_gamma and the algorithm owns the output network g_theta as an explicit linear
/// head. That is exactly the split MbPA requires.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine a doctor who has read every textbook but is now seeing a rare
/// condition. Rather than re-reading the textbooks, they pull the few most similar cases from their
/// notes, think about those for a moment, make the call, and then go back to being their normal
/// self. The notes are the episodic memory; the moment of thinking is the local adaptation; going
/// back to normal is what stops one rare case from distorting everything else they know.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Memory-based Parameter Adaptation",
    "https://arxiv.org/abs/1802.10542",
    Year = 2018,
    Authors = "Pablo Sprechmann, Siddhant M. Jayakumar, Jack W. Rae, Alexander Pritzel, " +
              "Adria Puigdomenech Badia, Benigno Uria, Oriol Vinyals, Demis Hassabis, " +
              "Razvan Pascanu, Charles Blundell")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MbPAAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private IParameterizable<T, TInput, TOutput>? _cachedParamModel;
    private IParameterizable<T, TInput, TOutput> ParamModel => _cachedParamModel ??= InterfaceGuard.Parameterizable(MetaModel);

    private readonly MbPAOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>
    /// The episodic memory M = {(h_i, v_i)}. Keys are embeddings, values are observed targets.
    /// </summary>
    private readonly MbPAEpisodicMemory<T> _memory;

    /// <summary>
    /// The output network g_theta as a flat vector of
    /// [OutputDimension * FeatureDimension weights | OutputDimension biases]. This — and only this —
    /// is what local adaptation modifies.
    /// </summary>
    private Vector<T> _outputParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MbPA;

    /// <summary>Gets the number of <c>(embedding, target)</c> pairs currently held in memory.</summary>
    public int MemoryCount => _memory.Count;

    /// <summary>Gets the trained output-network parameters, before any local adaptation.</summary>
    public Vector<T> OutputParameters => _outputParams;

    /// <summary>
    /// Initializes MbPA over an embedding network.
    /// </summary>
    /// <param name="options">MbPA options; the defaults sit inside the paper's reported ranges.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a dimension or count is not positive.</exception>
    public MbPAAlgorithm(MbPAOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.FeatureDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "FeatureDimension must be positive.");
        if (options.OutputDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "OutputDimension must be positive.");
        if (options.NumNeighbors <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumNeighbors (K) must be positive.");
        if (options.MemorySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "MemorySize must be positive.");
        if (options.LocalAdaptationSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "LocalAdaptationSteps (T) must be positive.");

        _algoOptions = options;
        _memory = new MbPAEpisodicMemory<T>(options.MemorySize);
        _outputParams = InitializeOutputNetwork();
    }

    #region Meta-training

    /// <inheritdoc/>
    /// <remarks>
    /// Ordinary parametric training of the embedding network and the output head, plus memory
    /// writes. MbPA has no meta-gradient — see the class remarks. The query loss is reported so the
    /// generic meta-learning harness has a comparable number; nothing about the update is bi-level.
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var embeddingGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);

            // Write the support examples into the episodic memory. This is what the memory is FOR:
            // the network's own record of what it has seen, keyed by how it currently sees it.
            if (_algoOptions.WriteMemoryDuringTraining)
            {
                WriteToMemory(task.SupportInput, task.SupportOutput);
            }

            // Train the output network on the support set by ordinary gradient descent. Note the
            // rate: InnerLearningRate, NOT LocalLearningRate. Training has to be slow and
            // cumulative; only the transient adaptation in Adapt() is allowed to be fast.
            var supportKeys = EmbedBatch(task.SupportInput);
            var supportTargets = ExtractTargets(task.SupportOutput, supportKeys.Count);
            for (int i = 0; i < supportKeys.Count; i++)
            {
                var grad = OutputNetworkGradient(_outputParams, supportKeys[i], supportTargets[i], weight: 1.0);
                for (int d = 0; d < _outputParams.Length; d++)
                {
                    _outputParams[d] = NumOps.Subtract(_outputParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(grad[d])));
                }
            }

            // THE QUERY LOSS GOES THROUGH THE HEAD, and this is the whole invariant of the method.
            // MetaModel emits a KEY, not a prediction. Comparing that key with the label directly --
            // which ComputeLossFromOutput(MetaModel.Predict(...), QueryOutput) did -- trains f_gamma to
            // output the label itself, leaves _outputParams out of the meta-objective entirely, and
            // then adapts locally on top of an embedding that was optimized for a different function.
            // MbPAHeadLoss composes g_theta onto the embedding so both the reported loss and the
            // gradient describe the same, correct objective; theta is a constant inside it, so only
            // the embedding is trained here and only the support loop above moves the head.
            var headLoss = new MbPAHeadLoss<T>(
                _outputParams, _algoOptions.FeatureDimension, _algoOptions.OutputDimension,
                _algoOptions.OutputDistribution);

            // ConvertToVector returns null for an output shape it cannot flatten. Feeding that to the
            // loss would surface as a null reference inside the loss function, naming neither the task
            // nor the conversion that actually failed.
            var queryPrediction = ConvertToVector(MetaModel.Predict(task.QueryInput))
                ?? throw new InvalidOperationException(
                    "MbPA could not convert the model's query prediction to a vector.");
            var queryTarget = ConvertToVector(task.QueryOutput)
                ?? throw new InvalidOperationException(
                    "MbPA could not convert the task's query target to a vector.");

            losses.Add(headLoss.CalculateLoss(queryPrediction, queryTarget));
            embeddingGradients.Add(ClipGradients(
                ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput, headLoss)));
        }

        // AN EMPTY TASK BATCH PRODUCES NO GRADIENTS AND NO LOSSES. ComputeMean over an empty
        // list has no defined value, and ApplyOuterUpdate would step on nothing.
        // LFTAlgorithm.MetaTrain already guards this exact case; this override was the odd one out.
        if (losses.Count == 0)
        {
            return NumOps.Zero;
        }

        ApplyOuterUpdate(initParams, embeddingGradients, _algoOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    #endregion

    #region Adaptation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Writes the task's support set into the episodic memory and returns a model that performs
    /// MbPA's local adaptation ON EVERY PREDICTION. The adaptation cannot be computed once here and
    /// reused: it is conditioned on the query's own embedding, so a different input retrieves
    /// different neighbours and needs different adapted parameters.
    /// </para>
    /// <para>
    /// This algorithm's own parameters are left exactly as they were, and the returned model holds
    /// no adapted state either — it recomputes and discards on each call, which is the paper's
    /// stated behaviour.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Adapt is public API, and both sibling algorithms (LFT, SparseMAML) guard this. Without
        // it the very next line dereferences task and reports a NullReferenceException, which
        // names neither the parameter nor the caller's mistake.
        if (task is null) throw new ArgumentNullException(nameof(task));

        // The new task's support set becomes memory. This is how MbPA absorbs a distribution shift:
        // by remembering it, not by retraining on it.
        WriteToMemory(task.SupportInput, task.SupportOutput);

        return new MbPAAdaptedModel<T, TInput, TOutput>(
            MetaModel, _memory, _outputParams.Clone(), _algoOptions);
    }

    #endregion

    #region Episodic memory

    /// <summary>
    /// Embeds each example of a batch and stores <c>(h_i, v_i)</c>.
    /// </summary>
    public void WriteToMemory(TInput inputs, TOutput targets)
    {
        var keys = EmbedBatch(inputs);
        var values = ExtractTargets(targets, keys.Count);
        for (int i = 0; i < keys.Count; i++)
        {
            _memory.Write(keys[i], values[i]);
        }
    }

    /// <summary>Clears the episodic memory.</summary>
    public void ClearMemory() => _memory.Clear();

    /// <summary>
    /// Embeds every example in a batch through the embedding network f_gamma.
    /// </summary>
    private List<Vector<T>> EmbedBatch(TInput inputs)
    {
        int batchSize = MbPAConversions<T>.GetBatchSize(inputs);
        var keys = new List<Vector<T>>(batchSize);

        // A per-example forward is what gives per-example keys. Embedding the batch as a unit would
        // store one key covering many examples, and K-nearest-neighbour retrieval over batch-level
        // keys is not the paper's mechanism.
        for (int i = 0; i < batchSize; i++)
        {
            var single = MbPAConversions<T>.SliceExample(inputs, i);
            var embedding = ConvertToVector(MetaModel.Predict(single));
            keys.Add(MbPAConversions<T>.ResizeTo(embedding, _algoOptions.FeatureDimension));
        }
        return keys;
    }

    private List<Vector<T>> ExtractTargets(TOutput targets, int count)
    {
        var values = new List<Vector<T>>(count);
        for (int i = 0; i < count; i++)
        {
            values.Add(MbPAConversions<T>.ResizeTo(
                MbPAConversions<T>.SliceTargetRow(targets, i), _algoOptions.OutputDimension));
        }
        return values;
    }

    #endregion

    #region Output network

    private Vector<T> InitializeOutputNetwork()
    {
        int weightCount = _algoOptions.OutputDimension * _algoOptions.FeatureDimension;
        var parameters = new Vector<T>(weightCount + _algoOptions.OutputDimension);

        // Xavier/Glorot scaling for the linear head; biases stay zero.
        double scale = Math.Sqrt(2.0 / (_algoOptions.FeatureDimension + _algoOptions.OutputDimension));
        for (int i = 0; i < weightCount; i++)
        {
            parameters[i] = NumOps.FromDouble((RandomGenerator.NextDouble() * 2.0 - 1.0) * scale);
        }
        return parameters;
    }

    /// <summary>
    /// The exact gradient of <c>-w * log p(v | h, theta)</c> with respect to the linear output
    /// network's parameters.
    /// </summary>
    /// <remarks>
    /// For both supported distributions the gradient is <c>w * (prediction - target) (x) h</c> for
    /// the weights and <c>w * (prediction - target)</c> for the biases — softmax composed with cross
    /// entropy, and a unit-variance Gaussian composed with squared error, share that form. It is
    /// therefore computed in closed form rather than by finite differences.
    /// </remarks>
    internal Vector<T> OutputNetworkGradient(Vector<T> parameters, Vector<T> key, Vector<T> target, double weight)
        => MbPAOutputNetwork<T>.Gradient(
            parameters, key, target, weight,
            _algoOptions.FeatureDimension, _algoOptions.OutputDimension, _algoOptions.OutputDistribution);

    #endregion
}
