using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// The model MbPA hands back from adaptation: it re-runs the local adaptation on EVERY prediction
/// and discards the adapted parameters immediately afterwards.
/// </summary>
/// <remarks>
/// <para>
/// This is where MbPA actually happens, and the per-prediction recomputation is not an
/// implementation detail — it is the method. The adapted parameters theta_x are a function of the
/// query's own embedding, because that embedding is what selects the neighbours to fit. Computing
/// them once at adaptation time and reusing them for every input would collapse MbPA into ordinary
/// fine-tuning on the support set, which is the thing the paper is arguing against.
/// </para>
/// <para>
/// Equally load-bearing: nothing is retained. Sprechmann et al.: "These adapted parameters are used
/// for output computation but discarded thereafter." Retaining them across calls would accumulate
/// drift and reintroduce exactly the catastrophic forgetting the memory exists to avoid.
/// </para>
/// <para>
/// <b>For Beginners:</b> Every time you ask this model a question it briefly looks up similar
/// remembered cases, nudges its last layer to agree with them, answers, and then undoes the nudge.
/// The next question starts from the same clean slate.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
/// <example>
/// <code>
/// // Produced by MbPAAlgorithm.Adapt rather than constructed directly: the returned model
/// // carries the episodic memory and the locally adapted output head for one task.
/// var metaModel = new NeuralNetwork&lt;double&gt;(
///     new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 8, outputSize: 4));
/// var options = new MbPAOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(metaModel);
/// var mbpa = new MbPAAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
///
/// var task = new MetaLearningTask&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;();
/// var taskBatch = new TaskBatch&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     new IMetaLearningTask&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;[] { task });
///
/// mbpa.MetaTrain(taskBatch);
/// var adapted = mbpa.Adapt(task);          // an MbPAAdaptedModel
/// var prediction = adapted.Predict(task.QueryInput);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Memory-based Parameter Adaptation", "https://arxiv.org/abs/1802.10542", Year = 2018)]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Evaluation)]
public partial class MbPAAdaptedModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>
{
    /// <summary>
    /// The embedding network this model was built with, read back from the base that holds it.
    /// </summary>
    /// <remarks>
    /// The constructor calls the argument <c>embeddingNetwork</c> and hands it to the base, which
    /// keeps it as <c>BaseModel</c>. That rename is invisible to the clone plan -- no name or type
    /// rule can tell that two differently named members are the same value -- so the model could not
    /// be rebuilt from its own state. Reading it back beats storing a second reference, which could
    /// drift from the one the base actually uses.
    /// </remarks>
    private IFullModel<T, TInput, TOutput> _embeddingNetwork => BaseModel;

    private readonly MbPAEpisodicMemory<T> _memory;
    private readonly MbPAOptions<T, TInput, TOutput> _options;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _trainedOutputParams;

    /// <summary>
    /// Gets the number of local gradient steps taken per prediction (T).
    /// </summary>
    public int LocalAdaptationSteps => _options.LocalAdaptationSteps;

    /// <summary>
    /// Gets the number of <c>(embedding, target)</c> pairs available to retrieve from.
    /// </summary>
    public int MemoryCount => _memory.Count;

    /// <summary>
    /// Initializes the adapting model.
    /// </summary>
    /// <param name="embeddingNetwork">f_gamma; held fixed, so memory keys stay comparable.</param>
    /// <param name="memory">The shared episodic memory to retrieve from.</param>
    /// <param name="trainedOutputParams">theta, the trained head. Never modified in place.</param>
    /// <param name="options">MbPA options.</param>
    public MbPAAdaptedModel(
        IFullModel<T, TInput, TOutput> embeddingNetwork,
        MbPAEpisodicMemory<T> memory,
        Vector<T> trainedOutputParams,
        MbPAOptions<T, TInput, TOutput> options)
        : base(embeddingNetwork)
    {
        _memory = memory ?? throw new ArgumentNullException(nameof(memory));
        _trainedOutputParams = trainedOutputParams ?? throw new ArgumentNullException(nameof(trainedOutputParams));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Embed, retrieve, locally adapt, emit, discard. The embedding network is never modified, so
    /// this is safe to call concurrently with respect to the base model's parameters — unlike the
    /// parameter-swapping adaptation used by the MAML-family models.
    /// </remarks>
    public override TOutput Predict(TInput input)
    {
        int batchSize = MbPAConversions<T>.GetBatchSize(input);
        if (batchSize > 1 && typeof(TOutput) == typeof(Vector<T>) && _options.OutputDimension > 1)
        {
            throw new NotSupportedException(
                $"MbPA cannot represent {batchSize} predictions with {_options.OutputDimension} components each " +
                "as Vector<T>. Use Matrix<T> or Tensor<T> for batched multi-component outputs.");
        }

        var rows = new List<Vector<T>>(batchSize);

        for (int i = 0; i < batchSize; i++)
        {
            var single = MbPAConversions<T>.SliceExample(input, i);
            rows.Add(PredictSingle(single));
        }

        return AssembleOutput(rows);
    }

    /// <summary>
    /// One query's worth of MbPA: q = f_gamma(x), retrieve, T local steps, g_(theta_x)(q).
    /// </summary>
    private Vector<T> PredictSingle(TInput single)
    {
        // FAIL AT THE REAL CAUSE. ConvertOutputToVector returns null for any TOutput that is not
        // Vector/Matrix/Tensor, and that null went straight into ResizeTo -- so an unsupported output
        // type surfaced as a failure inside a conversion helper. AssembleOutput already produces a
        // precise message for exactly this condition, but it never runs: PredictSingle fails first.
        var embedded = ConvertOutputToVector(BaseModel.Predict(single));
        if (embedded is null)
        {
            throw new NotSupportedException(
                $"MbPA cannot read an embedding out of {typeof(TOutput).Name}. " +
                "Use Vector<T>, Matrix<T> or Tensor<T> as the meta-learning output type.");
        }

        var query = MbPAConversions<T>.ResizeTo(embedded, _options.FeatureDimension);

        var neighbors = _memory.Retrieve(
            query, _options.NumNeighbors, _options.KernelEpsilon, NumOps.ToDouble);

        // theta_x — local, and about to be thrown away.
        var adapted = MbPAOutputNetwork<T>.LocallyAdapt(
            _trainedOutputParams, neighbors,
            _options.LocalAdaptationSteps, _options.LocalLearningRate, _options.RegularizationBeta,
            _options.FeatureDimension, _options.OutputDimension, _options.OutputDistribution);

        return MbPAOutputNetwork<T>.Forward(
            adapted, query, _options.FeatureDimension, _options.OutputDimension, _options.OutputDistribution);
        // `adapted` goes out of scope here. That is the discard.
    }

    private Vector<T>? ConvertOutputToVector(TOutput output) => output switch
    {
        Vector<T> vector => vector,
        Tensor<T> tensor => TensorToVector(tensor),
        Matrix<T> matrix => MatrixFirstRow(matrix),
        _ => null,
    };

    private static Vector<T> TensorToVector(Tensor<T> tensor)
    {
        var result = new Vector<T>(tensor.Length);
        for (int i = 0; i < tensor.Length; i++) result[i] = tensor[i];
        return result;
    }

    private static Vector<T> MatrixFirstRow(Matrix<T> matrix)
    {
        var result = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns && matrix.Rows > 0; j++) result[j] = matrix[0, j];
        return result;
    }

    /// <summary>
    /// Packs per-example predictions back into the caller's <c>TOutput</c> shape.
    /// </summary>
    private TOutput AssembleOutput(List<Vector<T>> rows)
    {
        int outputDim = _options.OutputDimension;

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            // THE PACKING RULE COMES FROM OutputDimension, NOT FROM THE BATCH SIZE. This branch used
            // to return the full OutputDimension-length prediction for one example and one SCALAR per
            // example for two or more -- so the same model on a batch of 1 and a batch of 2 returned
            // vectors with different meanings, and multi-class output was silently truncated to
            // component 0 as soon as a second example appeared.
            //
            // A Vector<T> can carry a batch of scalar predictions OR one multi-component prediction,
            // and OutputDimension is what distinguishes them: with a scalar head every example
            // contributes one component; with a multi-component head a flat vector cannot represent a
            // batch at all, so that combination is refused rather than silently truncated.
            if (outputDim <= 1)
            {
                var packed = new Vector<T>(rows.Count);
                for (int i = 0; i < rows.Count; i++) packed[i] = rows[i].Length > 0 ? rows[i][0] : NumOps.Zero;
                return (TOutput)(object)packed;
            }

            if (rows.Count == 1) return (TOutput)(object)rows[0];

            // A flat Vector<T> has no row boundary. Although all components could be copied into it,
            // the meta-learning consumers treat a vector as one prediction: ComputeAccuracy takes one
            // global argmax and ComputeLossFromOutput passes it directly to the configured loss. Until
            // those consumers become OutputDimension-aware, returning row-major data here would make a
            // valid representation locally and give it the wrong meaning everywhere downstream.
            throw new NotSupportedException(
                $"MbPA cannot represent {rows.Count} predictions with {outputDim} components each " +
                "as Vector<T>. Use Matrix<T> or Tensor<T> for batched multi-component outputs.");
        }

        if (typeof(TOutput) == typeof(Matrix<T>))
        {
            var packed = new Matrix<T>(rows.Count, outputDim);
            for (int i = 0; i < rows.Count; i++)
            {
                for (int j = 0; j < outputDim && j < rows[i].Length; j++) packed[i, j] = rows[i][j];
            }
            return (TOutput)(object)packed;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var packed = new Tensor<T>([rows.Count, outputDim]);
            for (int i = 0; i < rows.Count; i++)
            {
                for (int j = 0; j < outputDim && j < rows[i].Length; j++) packed[i, j] = rows[i][j];
            }
            return (TOutput)(object)packed;
        }

        // An output type this model cannot pack is a configuration error, not something to paper
        // over by returning the embedding network's own prediction — that would silently bypass the
        // local adaptation and make MbPA look like it was running when it was not.
        throw new NotSupportedException(
            $"MbPA cannot assemble predictions into {typeof(TOutput).Name}. " +
            "Use Vector<T>, Matrix<T> or Tensor<T> as the meta-learning output type.");
    }

    // GetParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    // SetParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
        => new MbPAAdaptedModel<T, TInput, TOutput>(BaseModel, _memory, parameters, _options);
}
