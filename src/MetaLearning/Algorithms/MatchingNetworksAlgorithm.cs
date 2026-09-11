using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Matching Networks for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Matching Networks (Vinyals et al. 2016) classify a query <c>x</c> by attention over the labelled support set:
/// <c>P(y | x, S) = sum_i a(x, x_i) y_i</c> with <c>a(x, x_i) = softmax_i c(f(x), g(x_i))</c>, where <c>c</c> is
/// the cosine similarity and <c>y_i</c> the one-hot support labels (eq. 1 and section 2.1.1). Training maximises
/// <c>log P(y | x, S)</c> of the query labels over episodes (eq. 2).
/// </para>
/// <para>
/// <b>Full context embeddings</b> (section 2.1.2 and appendix A) make both embeddings depend on the support set.
/// <c>g(x_i, S) = h_fwd_i + h_bwd_i + g'(x_i)</c> runs a bidirectional LSTM over the support set (A.2); and
/// <c>f(x, S) = attLSTM(f'(x), g(S), K)</c> runs <c>K</c> steps of an LSTM that reads <c>g(S)</c> by content-based
/// attention, <c>h_k = LSTM(f'(x), [h_(k-1), r_(k-1)], c_(k-1)) + f'(x)</c> with
/// <c>r_(k-1) = sum_i softmax(h_(k-1)' g(x_i)) g(x_i)</c> (A.1). Here <c>f'</c> and <c>g'</c> are the one embedding
/// network the options hold.
/// </para>
/// <para>
/// <b>Exact gradients.</b> Support and query examples go through the embedding network together and the whole
/// episode - context embeddings, attention and class probabilities - is rebuilt from those embeddings on the
/// tape, so the embedding network, the context LSTMs and a learned kernel all receive the exact gradient of the
/// episode loss. This replaced a forward-difference estimate over at most 100 evenly spaced parameters, scaled up
/// as if it were an unbiased estimate of the rest.
/// </para>
/// <para><b>For Beginners:</b> Matching Networks label a new example by comparing it with every labelled example
/// of the task and letting the most similar ones vote. Training shapes the feature space so that the vote is
/// right.
/// </para>
/// <para>
/// Reference: Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., &amp; Wierstra, D. (2016).
/// Matching Networks for One Shot Learning. NeurIPS.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Matching Networks for One Shot Learning",
    "https://arxiv.org/abs/1606.04080",
    Year = 2016,
    Authors = "Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MatchingNetworksAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MatchingNetworksOptions<T, TInput, TOutput> _matchingOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _matchingOptions;

    /// <summary>
    /// The learned bilinear kernel <c>W</c>, <c>[width, width]</c> row-major, for
    /// <see cref="MatchingNetworksAttentionFunction.Learned"/>; empty otherwise and until the first episode shows the
    /// embedding width. It starts at the identity, where the kernel is the paper's cosine.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _kernelWeights = new Vector<T>(0);

    /// <summary>
    /// The forward then the backward LSTM of the support-set context embedding <c>g(x_i, S)</c> (appendix A.2);
    /// empty while full context embeddings are off.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _supportContextWeights = new Vector<T>(0);

    /// <summary>
    /// The attention LSTM of the query context embedding <c>f(x, S)</c> (appendix A.1); empty unless
    /// <see cref="MatchingNetworksOptions{T,TInput,TOutput}.UseFullContextEmbedding"/> is on.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _queryContextWeights = new Vector<T>(0);

    /// <summary>
    /// Initializes a new instance of the MatchingNetworksAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for Matching Networks.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a Matching Network ready for few-shot learning. The only required
    /// piece is the embedding network; the default loss is cross-entropy, which on the attention probabilities is
    /// the paper's <c>-log P(y | x, S)</c>.
    /// </para>
    /// </remarks>
    public MatchingNetworksAlgorithm(MatchingNetworksOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? new CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            null) // Matching Networks doesn't use inner optimizer
    {
        _matchingOptions = options;

        if (!_matchingOptions.IsValid())
        {
            throw new ArgumentException("Matching Networks configuration is invalid. Check all parameters.", nameof(options));
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.MatchingNetworks"/>.</value>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MatchingNetworks;

    /// <summary>
    /// Performs one meta-training step: for each episode, the attention loss of its query examples, differentiated
    /// exactly into the embedding network and every learned part of the matching function.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on.</param>
    /// <returns>The average query loss across the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        EnsureMatchingShapes(taskBatch.Tasks);

        var body = ParamModel.GetParameters();
        Vector<T>? bodyGradient = null, kernelGradient = null, supportGradient = null, queryGradient = null;
        T totalLoss = NumOps.Zero;
        foreach (var task in taskBatch.Tasks)
        {
            var (loss, taskBody, kernel, support, query) = EpisodeGradient(task);
            totalLoss = NumOps.Add(totalLoss, loss);
            bodyGradient = Accumulate(bodyGradient, taskBody);
            kernelGradient = Accumulate(kernelGradient, kernel);
            supportGradient = Accumulate(supportGradient, support);
            queryGradient = Accumulate(queryGradient, query);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        bodyGradient = Scale(bodyGradient ?? new Vector<T>(body.Length), batchSize);
        kernelGradient = Scale(kernelGradient ?? new Vector<T>(_kernelWeights.Length), batchSize);
        supportGradient = Scale(supportGradient ?? new Vector<T>(_supportContextWeights.Length), batchSize);
        queryGradient = Scale(queryGradient ?? new Vector<T>(_queryContextWeights.Length), batchSize);

        if (_matchingOptions.GradientClipThreshold.HasValue && _matchingOptions.GradientClipThreshold.Value > 0)
        {
            double threshold = _matchingOptions.GradientClipThreshold.Value;
            bodyGradient = ClipGradients(bodyGradient, threshold);
            if (kernelGradient.Length > 0) kernelGradient = ClipGradients(kernelGradient, threshold);
            if (supportGradient.Length > 0) supportGradient = ClipGradients(supportGradient, threshold);
            if (queryGradient.Length > 0) queryGradient = ClipGradients(queryGradient, threshold);
        }

        if (_matchingOptions.L2Regularization > 0.0)
        {
            T decay = NumOps.FromDouble(2 * _matchingOptions.L2Regularization);
            for (int i = 0; i < bodyGradient.Length; i++)
            {
                bodyGradient[i] = NumOps.Add(bodyGradient[i], NumOps.Multiply(body[i], decay));
            }
        }

        double beta = _matchingOptions.OuterLearningRate;
        ParamModel.SetParameters(ApplyGradients(body, bodyGradient, beta));
        if (_kernelWeights.Length > 0) _kernelWeights = ApplyGradients(_kernelWeights, kernelGradient, beta);
        if (_supportContextWeights.Length > 0) _supportContextWeights = ApplyGradients(_supportContextWeights, supportGradient, beta);
        if (_queryContextWeights.Length > 0) _queryContextWeights = ApplyGradients(_queryContextWeights, queryGradient, beta);

        return NumOps.Divide(totalLoss, batchSize);
    }

    /// <summary>
    /// Adapts to a new task by embedding its support set.
    /// </summary>
    /// <param name="task">The new task containing support set examples.</param>
    /// <returns>A MatchingNetworksModel that classifies by attention over the support examples.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        EnsureMatchingShapes(new[] { task });
        return new MatchingNetworksModel<T, TInput, TOutput>(
            MetaModel, task.SupportInput, task.SupportOutput, _matchingOptions, NumOps,
            CloneVector(_kernelWeights), CloneVector(_supportContextWeights), CloneVector(_queryContextWeights));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The adapted model returns class probabilities, so this is the configured loss of their logarithm against the
    /// class indices - with the default cross-entropy, the paper's <c>-log P(y | x, S)</c>. A Vector output carries
    /// one predicted class per example instead, and its loss is the classification error rate.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ProbabilityLoss(LossFunction, predictions, expectedOutput);

    #region Episode

    /// <summary>
    /// One episode's query loss and its exact gradient with respect to the embedding network, the learned kernel and
    /// both context embeddings.
    /// </summary>
    private (T Loss, Vector<T> Body, Vector<T> Kernel, Vector<T> SupportContext, Vector<T> QueryContext) EpisodeGradient(
        IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = PrototypeEpisode<T>.Build(ReadLabels(task.SupportOutput), ReadLabels(task.QueryOutput));
        var stackedInput = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { episode.Rows, 1 }));
        var metric = CreateMetric();

        // The embedding network's gradient: the episode rebuilt from the embeddings on the tape, scored against the
        // query labels. The metric's weights are constants here.
        var composed = new EmbeddingClassificationLoss<T>(
            embeddings => EpisodeLogits(embeddings, episode, metric), LossFunction, episode.QueryTarget);
        var bodyGradient = ComputeGradients(MetaModel, stackedInput, stackedTarget, composed);

        // The matching function's gradient: the same logits from fixed embeddings, against its own weights.
        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = ClassifierOutputs<T>.AsRows(MetaModel.Predict(stackedInput));
        }

        var kernelGradient = new Vector<T>(_kernelWeights.Length);
        var supportGradient = new Vector<T>(_supportContextWeights.Length);
        var queryGradient = new Vector<T>(_queryContextWeights.Length);
        T loss;
        if (metric.Leaves.Count == 0)
        {
            using var noGrad = new NoGradScope<T>();
            loss = LossFunction.ComputeTapeLoss(EpisodeLogits(embeddings, episode, metric), episode.QueryTarget)[0];
        }
        else
        {
            using var tape = new GradientTape<T>();
            var episodeLoss = LossFunction.ComputeTapeLoss(EpisodeLogits(embeddings, episode, metric), episode.QueryTarget);
            loss = episodeLoss[0];
            var gradients = tape.ComputeGradients(episodeLoss, metric.Leaves.ToList());
            metric.CopyGradients(gradients, kernelGradient, supportGradient, queryGradient);
        }

        return (loss, bodyGradient, kernelGradient, supportGradient, queryGradient);
    }

    /// <summary>Class logits of an episode's query rows from the stacked support-then-query embeddings.</summary>
    private static Tensor<T> EpisodeLogits(Tensor<T> embeddings, PrototypeEpisode<T> episode, MatchingMetric<T> metric)
    {
        var engine = AiDotNetEngine.Current;
        var support = engine.TensorMatMul(episode.SupportSelector, embeddings);
        var query = engine.TensorMatMul(episode.QuerySelector, embeddings);
        return metric.Logits(support, query, episode.Membership);
    }

    private MatchingMetric<T> CreateMetric()
        => new MatchingMetric<T>(_matchingOptions.AttentionFunction, _matchingOptions.Temperature,
            _matchingOptions.ProcessingSteps, _kernelWeights, _supportContextWeights, _queryContextWeights,
            EmbeddingWidth());

    /// <summary>The embedding width the learned weights were sized for; zero before any are.</summary>
    private int EmbeddingWidth()
    {
        if (_kernelWeights.Length > 0) return (int)Math.Round(Math.Sqrt(_kernelWeights.Length));
        if (_supportContextWeights.Length > 0) return MatchingMetric<T>.WidthOfSupportContext(_supportContextWeights.Length);
        return 0;
    }

    /// <summary>
    /// Sizes the learned parts of the matching function to the embedding width: the kernel at the identity, where it
    /// is the paper's cosine, and the context LSTMs at PyTorch's default LSTM initialisation.
    /// </summary>
    private void EnsureMatchingShapes(IEnumerable<IMetaLearningTask<T, TInput, TOutput>> tasks)
    {
        bool learnedKernel = _matchingOptions.AttentionFunction == MatchingNetworksAttentionFunction.Learned;
        bool supportContext = _matchingOptions.UseBidirectionalEncoding || _matchingOptions.UseFullContextEmbedding;
        bool queryContext = _matchingOptions.UseFullContextEmbedding;
        bool needed = (learnedKernel && _kernelWeights.Length == 0)
            || (supportContext && _supportContextWeights.Length == 0)
            || (queryContext && _queryContextWeights.Length == 0);
        var first = tasks.FirstOrDefault();
        if (!needed || first is null) return;

        int width;
        using (new NoGradScope<T>())
        {
            width = ClassifierOutputs<T>.AsRows(MetaModel.Predict(first.SupportInput)).Shape[1];
        }

        if (learnedKernel && _kernelWeights.Length == 0)
        {
            _kernelWeights = new Vector<T>(width * width);
            for (int i = 0; i < width; i++) _kernelWeights[i * width + i] = NumOps.One;
        }

        if (supportContext && _supportContextWeights.Length == 0)
        {
            int count = 2 * TapeLstmCell<T>.ParameterCount(width, width, 1);
            _supportContextWeights = new Vector<T>(count);
            TapeLstmCell<T>.InitializeUniform(_supportContextWeights, 0, count, width, RandomGenerator);
        }

        if (queryContext && _queryContextWeights.Length == 0)
        {
            int count = TapeLstmCell<T>.ParameterCount(width, width, 2);
            _queryContextWeights = new Vector<T>(count);
            TapeLstmCell<T>.InitializeUniform(_queryContextWeights, 0, count, width, RandomGenerator);
        }
    }

    private int[] ReadLabels(TOutput labels)
    {
        var tensor = ClassifierOutputs<T>.Labels(labels, _matchingOptions.NumClasses);
        var indices = new int[tensor.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
        return indices;
    }

    #endregion

    #region Test hooks

    /// <summary>One episode's query loss and exact gradient from the current state, for gradient checks.</summary>
    internal (T Loss, Vector<T> Body, Vector<T> Kernel, Vector<T> SupportContext, Vector<T> QueryContext) EpisodeGradientForTesting(
        IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureMatchingShapes(new[] { task });
        return EpisodeGradient(task);
    }

    /// <summary>One episode's query loss from the current state.</summary>
    internal T EpisodeLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureMatchingShapes(new[] { task });
        var episode = PrototypeEpisode<T>.Build(ReadLabels(task.SupportOutput), ReadLabels(task.QueryOutput));
        using var noGrad = new NoGradScope<T>();
        var embeddings = ClassifierOutputs<T>.AsRows(
            MetaModel.Predict(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput)));
        return LossFunction.ComputeTapeLoss(EpisodeLogits(embeddings, episode, CreateMetric()), episode.QueryTarget)[0];
    }

    /// <summary>Gets or sets a copy of the learned bilinear kernel (for tests).</summary>
    internal Vector<T> KernelWeightsForTesting { get => CloneVector(_kernelWeights); set => _kernelWeights = CloneVector(value); }

    /// <summary>Gets or sets a copy of the support context LSTMs' weights (for tests).</summary>
    internal Vector<T> SupportContextWeightsForTesting
    {
        get => CloneVector(_supportContextWeights);
        set => _supportContextWeights = CloneVector(value);
    }

    /// <summary>Gets or sets a copy of the query attention LSTM's weights (for tests).</summary>
    internal Vector<T> QueryContextWeightsForTesting
    {
        get => CloneVector(_queryContextWeights);
        set => _queryContextWeights = CloneVector(value);
    }

    #endregion

    #region Helpers

    private static Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
    {
        if (sum is null) return CloneVector(values);
        for (int i = 0; i < sum.Length; i++) sum[i] = NumOps.Add(sum[i], values[i]);
        return sum;
    }

    private static Vector<T> Scale(Vector<T> values, T divisor)
    {
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.Divide(values[i], divisor);
        return values;
    }

    private static Vector<T> CloneVector(Vector<T> source)
    {
        var clone = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) clone[i] = source[i];
        return clone;
    }

    #endregion
}

/// <summary>
/// The matching function of Matching Networks in engine tensor ops: context embeddings, the similarity kernel and the
/// attention class probabilities, differentiable by a live tape.
/// </summary>
internal sealed class MatchingMetric<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly MatchingNetworksAttentionFunction _kernelKind;
    private readonly double _temperature;
    private readonly int _processingSteps;
    private readonly Tensor<T>? _kernel;
    private readonly TapeLstmCell<T>? _forward;
    private readonly TapeLstmCell<T>? _backward;
    private readonly TapeLstmCell<T>? _attention;
    private readonly List<Tensor<T>> _leaves = new List<Tensor<T>>();

    /// <summary>Unpacks the learned parts; an empty vector leaves that part out.</summary>
    internal MatchingMetric(
        MatchingNetworksAttentionFunction kernelKind, double temperature, int processingSteps,
        Vector<T> kernel, Vector<T> supportContext, Vector<T> queryContext, int width)
    {
        _kernelKind = kernelKind;
        _temperature = temperature;
        _processingSteps = processingSteps;
        if (kernel.Length > 0)
        {
            _kernel = Tensor<T>.FromVector(kernel);
            _leaves.Add(_kernel);
        }

        if (supportContext.Length > 0)
        {
            _forward = new TapeLstmCell<T>(supportContext, 0, width, width, 1);
            _backward = new TapeLstmCell<T>(supportContext, TapeLstmCell<T>.ParameterCount(width, width, 1), width, width, 1);
            _leaves.AddRange(_forward.Leaves);
            _leaves.AddRange(_backward.Leaves);
        }

        if (queryContext.Length > 0)
        {
            _attention = new TapeLstmCell<T>(queryContext, 0, width, width, 2);
            _leaves.AddRange(_attention.Leaves);
        }
    }

    /// <summary>Every learned tensor, in the order <see cref="CopyGradients"/> reads them.</summary>
    internal IReadOnlyList<Tensor<T>> Leaves => _leaves;

    /// <summary>The embedding width a support context vector of this length was sized for.</summary>
    internal static int WidthOfSupportContext(int length)
    {
        for (int width = 1; ; width++)
        {
            int count = 2 * TapeLstmCell<T>.ParameterCount(width, width, 1);
            if (count == length) return width;
            if (count > length) throw new ArgumentException($"{length} weights are not a support context LSTM pair.");
        }
    }

    /// <summary>
    /// The support embeddings the query attends over: <c>g(x_i, S)</c> of appendix A.2 when the context LSTMs are
    /// present, otherwise the embeddings as they are.
    /// </summary>
    internal Tensor<T> EmbedSupport(Tensor<T> support)
    {
        if (_forward is null || _backward is null) return support;
        var engine = AiDotNetEngine.Current;
        int rows = support.Shape[0], width = support.Shape[1];

        Tensor<T> Run(TapeLstmCell<T> cell, bool reverse)
        {
            var output = new Tensor<T>(new[] { 1, width });
            var state = new Tensor<T>(new[] { 1, width });
            Tensor<T>? outputs = null;
            for (int step = 0; step < rows; step++)
            {
                int i = reverse ? rows - 1 - step : step;
                var x = engine.TensorMatMul(Selector(1, rows, 0, i), support);
                (output, state) = cell.Step(x, new[] { output }, state);
                var placed = engine.TensorMatMul(Selector(rows, 1, i, 0), output);
                outputs = outputs is null ? placed : engine.TensorAdd(outputs, placed);
            }

            return outputs ?? new Tensor<T>(new[] { rows, width });
        }

        // g(x_i, S) = h_fwd_i + h_bwd_i + g'(x_i): the backward recursion starts from i = |S|.
        return engine.TensorAdd(engine.TensorAdd(Run(_forward, reverse: false), Run(_backward, reverse: true)), support);
    }

    /// <summary>
    /// The query embeddings: <c>f(x, S) = attLSTM(f'(x), g(S), K)</c> of appendix A.1 when the attention LSTM is
    /// present, otherwise the embeddings as they are.
    /// </summary>
    internal Tensor<T> EmbedQueries(Tensor<T> queries, Tensor<T> support)
    {
        if (_attention is null) return queries;
        var engine = AiDotNetEngine.Current;
        int rows = queries.Shape[0], width = queries.Shape[1];
        var hidden = new Tensor<T>(new[] { rows, width });
        var cell = new Tensor<T>(new[] { rows, width });
        for (int k = 0; k < _processingSteps; k++)
        {
            // r_(k-1) = sum_i softmax(h_(k-1)' g(x_i)) g(x_i); h_k = LSTM(f'(x), [h_(k-1), r_(k-1)], c_(k-1)) + f'(x).
            var read = engine.TensorMatMul(SoftmaxRows(engine.TensorMatMul(hidden, engine.TensorTranspose(support))), support);
            var (output, nextCell) = _attention.Step(queries, new[] { hidden, read }, cell);
            hidden = engine.TensorAdd(output, queries);
            cell = nextCell;
        }

        return hidden;
    }

    /// <summary>The kernel <c>c(f(x), g(x_i))</c> over temperature, <c>[queries, support]</c>.</summary>
    internal Tensor<T> Similarities(Tensor<T> queries, Tensor<T> support)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> scores;
        switch (_kernelKind)
        {
            case MatchingNetworksAttentionFunction.DotProduct:
                scores = engine.TensorMatMul(queries, engine.TensorTranspose(support));
                break;
            case MatchingNetworksAttentionFunction.Euclidean:
            {
                var queryNorm = engine.ReduceSum(engine.TensorMultiply(queries, queries), new[] { 1 }, keepDims: true);
                var supportNorm = engine.ReduceSum(engine.TensorMultiply(support, support), new[] { 1 }, keepDims: true);
                var cross = engine.TensorMatMul(queries, engine.TensorTranspose(support));
                var squared = engine.TensorAdd(
                    engine.TensorAdd(queryNorm, engine.TensorTranspose(supportNorm)),
                    engine.TensorMultiplyScalar(cross, Ops.FromDouble(-2.0)));
                var distance = engine.TensorSqrt(
                    engine.TensorAddScalar(engine.TensorClampMin(squared, Ops.Zero), Ops.FromDouble(1e-12)));
                scores = engine.TensorNegate(distance);
                break;
            }
            case MatchingNetworksAttentionFunction.Learned:
            {
                var q = PrototypeMetric<T>.Normalized(queries, normalize: true);
                var s = PrototypeMetric<T>.Normalized(support, normalize: true);
                int width = queries.Shape[1];
                var w = _kernel is null ? null : engine.Reshape(_kernel, new[] { width, width });
                var projected = w is null ? q : engine.TensorMatMul(q, w);
                scores = engine.TensorMatMul(projected, engine.TensorTranspose(s));
                break;
            }
            default:
            {
                // The paper's kernel: cosine similarity (section 2.1.1).
                var q = PrototypeMetric<T>.Normalized(queries, normalize: true);
                var s = PrototypeMetric<T>.Normalized(support, normalize: true);
                scores = engine.TensorMatMul(q, engine.TensorTranspose(s));
                break;
            }
        }

        return Math.Abs(_temperature - 1.0) < 1e-12
            ? scores
            : engine.TensorMultiplyScalar(scores, Ops.FromDouble(1.0 / _temperature));
    }

    /// <summary>
    /// Logits whose softmax is <c>P(y | x, S) = sum_i a(x, x_i) y_i</c>, <c>[queries, classes]</c>: the log of each
    /// class's attention mass. The softmax over classes renormalises nothing, since the masses already sum to one,
    /// so cross-entropy on these logits is exactly <c>-log P(y | x, S)</c>.
    /// </summary>
    /// <param name="similarities">The kernel values, <c>[queries, support]</c>.</param>
    /// <param name="membership"><c>[classes, support]</c>: 1 where a support example has the class.</param>
    internal static Tensor<T> ClassLogits(Tensor<T> similarities, Tensor<T> membership)
    {
        var engine = AiDotNetEngine.Current;
        var max = engine.ReduceMax(similarities, new[] { 1 }, keepDims: true, out _);
        var weights = engine.TensorExp(engine.TensorAdd(similarities, engine.TensorNegate(engine.StopGradient(max))));
        var mass = engine.TensorMatMul(weights, engine.TensorTranspose(membership));
        return engine.TensorLog(engine.TensorClampMin(mass, Ops.FromDouble(1e-30)));
    }

    /// <summary>The whole matching function: context embeddings, kernel and class logits.</summary>
    internal Tensor<T> Logits(Tensor<T> support, Tensor<T> queries, Tensor<T> membership)
    {
        var g = EmbedSupport(support);
        var f = EmbedQueries(queries, g);
        return ClassLogits(Similarities(f, g), membership);
    }

    /// <summary>Row-wise softmax from ops a tape records.</summary>
    internal static Tensor<T> SoftmaxRows(Tensor<T> scores)
    {
        var engine = AiDotNetEngine.Current;
        var max = engine.ReduceMax(scores, new[] { 1 }, keepDims: true, out _);
        var exp = engine.TensorExp(engine.TensorAdd(scores, engine.TensorNegate(engine.StopGradient(max))));
        return engine.TensorDivide(exp, engine.ReduceSum(exp, new[] { 1 }, keepDims: true));
    }

    /// <summary>Scatters a tape's gradients back into the three flat weight vectors.</summary>
    internal void CopyGradients(Dictionary<Tensor<T>, Tensor<T>> gradients, Vector<T> kernel, Vector<T> supportContext,
        Vector<T> queryContext)
    {
        if (_kernel is not null && gradients.TryGetValue(_kernel, out var kernelGradient))
        {
            for (int i = 0; i < kernel.Length; i++) kernel[i] = kernelGradient[i];
        }

        if (_forward is not null && _backward is not null)
        {
            _forward.CopyGradients(gradients, supportContext, 0);
            _backward.CopyGradients(gradients, supportContext, TapeLstmCell<T>.ParameterCount(_forward.Input, _forward.Hidden, 1));
        }

        _attention?.CopyGradients(gradients, queryContext, 0);
    }

    /// <summary>A <c>[rows, columns]</c> tensor with a single one at <c>(row, column)</c>.</summary>
    private static Tensor<T> Selector(int rows, int columns, int row, int column)
    {
        var selector = new Tensor<T>(new[] { rows, columns });
        selector[row * columns + column] = Ops.One;
        return selector;
    }
}

/// <summary>
/// Matching Networks model for inference.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// The adapted state of Matching Networks for one task: its own copy of the embedding network, the support set's
/// (context) embeddings and labels, and the learned matching function. <see cref="Predict"/> returns
/// <c>P(y | x, S)</c> per example, <c>[rows, NumClasses]</c>, for Tensor and Matrix outputs - a class with no
/// support example gets probability zero, exactly as eq. 1 gives it - and the most probable class of each example
/// for a Vector output.
/// </para>
/// </remarks>
public class MatchingNetworksModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _encoder;
    private readonly MatchingNetworksOptions<T, TInput, TOutput> _options;
    private readonly INumericOperations<T> _numOps;
    private readonly MatchingMetric<T> _metric;
    private readonly Tensor<T> _support;
    private readonly Tensor<T> _membership;
    private readonly Tensor<T> _classColumns;
    private readonly int[] _classSlots;

    /// <summary>
    /// Initializes a new instance of the MatchingNetworksModel with an untrained matching function.
    /// </summary>
    /// <remarks>
    /// A learned kernel starts at the identity, which is the cosine kernel. Full context embeddings have no untrained
    /// value to start from, so this constructor refuses them: adapt through
    /// <see cref="MatchingNetworksAlgorithm{T, TInput, TOutput}.Adapt"/> instead.
    /// </remarks>
    /// <exception cref="ArgumentException">The options ask for full context embeddings.</exception>
    public MatchingNetworksModel(
        IFullModel<T, TInput, TOutput> encoder,
        TInput supportInputs,
        TOutput supportLabels,
        MatchingNetworksOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps)
        : this(encoder, supportInputs, supportLabels, options, numOps,
            new Vector<T>(0), new Vector<T>(0), new Vector<T>(0))
    {
    }

    /// <summary>Initializes the model with the matching function Matching Networks learned.</summary>
    internal MatchingNetworksModel(
        IFullModel<T, TInput, TOutput> encoder,
        TInput supportInputs,
        TOutput supportLabels,
        MatchingNetworksOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps,
        Vector<T> kernel,
        Vector<T> supportContext,
        Vector<T> queryContext)
    {
        Guard.NotNull(encoder);
        Guard.NotNull(options);
        Guard.NotNull(numOps);
        _encoder = encoder.DeepCopy();
        _options = options;
        _numOps = numOps;

        bool wantsSupportContext = options.UseBidirectionalEncoding || options.UseFullContextEmbedding;
        if ((wantsSupportContext && supportContext.Length == 0) || (options.UseFullContextEmbedding && queryContext.Length == 0))
        {
            throw new ArgumentException(
                "Full context embeddings are learned during meta-training; build this model through "
                + "MatchingNetworksAlgorithm.Adapt rather than directly.", nameof(options));
        }

        var labels = ClassifierOutputs<T>.Labels(supportLabels, options.NumClasses);
        var supportLabelIndices = new int[labels.Length];
        for (int i = 0; i < supportLabelIndices.Length; i++) supportLabelIndices[i] = (int)Math.Round(numOps.ToDouble(labels[i]));
        if (supportLabelIndices.Length == 0)
            throw new ArgumentException("The support set is empty, so there is nothing to match against.", nameof(supportLabels));

        var episode = PrototypeEpisode<T>.Build(supportLabelIndices, Array.Empty<int>());
        _membership = episode.Membership;
        _classSlots = episode.ClassSlots;
        _classColumns = new Tensor<T>(new[] { _classSlots.Length, options.NumClasses });
        for (int c = 0; c < _classSlots.Length; c++) _classColumns[c * options.NumClasses + _classSlots[c]] = numOps.One;

        using var noGrad = new NoGradScope<T>();
        var rows = ClassifierOutputs<T>.AsRows(_encoder.Predict(supportInputs));
        int width = rows.Shape[1];
        if (options.AttentionFunction == MatchingNetworksAttentionFunction.Learned && kernel.Length == 0)
        {
            kernel = new Vector<T>(width * width);
            for (int i = 0; i < width; i++) kernel[i * width + i] = numOps.One;
        }

        _metric = new MatchingMetric<T>(options.AttentionFunction, options.Temperature, options.ProcessingSteps,
            kernel, supportContext, queryContext, width);
        _support = _metric.EmbedSupport(rows);
    }

    /// <summary>Gets the model metadata.</summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Classifies by attention over the support examples.
    /// </summary>
    /// <param name="input">The examples to classify.</param>
    /// <returns>Class probabilities per example, or the predicted class per example for a Vector output.</returns>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var queries = _metric.EmbedQueries(ClassifierOutputs<T>.AsRows(_encoder.Predict(input)), _support);
        var probabilities = MatchingMetric<T>.SoftmaxRows(
            MatchingMetric<T>.ClassLogits(_metric.Similarities(queries, _support), _membership));

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            int rows = probabilities.Shape[0], classes = probabilities.Shape[1];
            var predicted = new Vector<T>(rows);
            for (int r = 0; r < rows; r++)
            {
                int best = 0;
                for (int c = 1; c < classes; c++)
                {
                    if (_numOps.GreaterThan(probabilities[r * classes + c], probabilities[r * classes + best])) best = c;
                }

                predicted[r] = _numOps.FromDouble(_classSlots[best]);
            }

            return (TOutput)(object)predicted;
        }

        return ClassifierOutputs<T>.ToOutput<TOutput>(engine.TensorMatMul(probabilities, _classColumns));
    }

    /// <summary>
    /// Training is not supported for inference models.
    /// </summary>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the Matching Networks algorithm to train.");
    }

    /// <summary>
    /// Parameter updates are not supported for inference models.
    /// </summary>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("Matching Networks parameters are updated during training.");
    }

    /// <summary>
    /// Gets the parameters of the model's copy of the embedding network.
    /// </summary>
    public Vector<T> GetParameters() => InterfaceGuard.Parameterizable(_encoder).GetParameters();

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
