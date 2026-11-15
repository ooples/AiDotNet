using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Memory-Augmented Neural Networks (MANN) for meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Memory-Augmented Neural Networks combine a neural network with an external memory
/// matrix. The network can read from and write to this memory, enabling rapid
/// learning by storing new information directly in memory during adaptation.
/// </para>
/// <para><b>For Beginners:</b> MANN uses an external memory like a human's working memory:
///
/// **How it works:**
/// 1. Neural network processes input and produces a key
/// 2. Uses key to read relevant information from external memory
/// 3. Combines input with memory content to make prediction
/// 4. During learning, writes new information to memory
/// 5. Memory persists across episodes for lifelong learning
///
/// **Analogy:** Like a student with a notebook who can:
/// - Look up previous examples (reading memory)
/// - Write down new important information (writing memory)
/// - Use both to solve new problems
/// </para>
/// <para><b>Algorithm - Memory-Augmented Neural Networks:</b>
/// <code>
/// # Components
/// controller = NeuralNetwork()        # Processes inputs and generates keys
/// memory = MemoryMatrix(size=N)      # External memory matrix
/// read_head = AttentionMechanism()   # Reads from memory
/// write_head = WriteMechanism()      # Writes to memory
///
/// # Episode training
/// for each episode:
///     # Sample N-way K-shot task
///     support_set = {examples_from_N_classes, K_examples_each}
///     query_set = {examples_from_same_N_classes}
///
///     # Process support set and write to memory
///     for each support example (x, y):
///         key = controller.encode(x)
///         value = one_hot_encode(y)
///         memory.write(key, value)     # Store in external memory
///
///     # Process query set using memory
///     for each query example x_q:
///         # Generate key for query
///         query_key = controller.encode(x_q)
///
///         # Read from memory using attention
///         read_content = read_head(query_key, memory)
///
///         # Combine with controller output
///         controller_output = controller.forward(x_q)
///         combined = concatenate(controller_output, read_content)
///
///         # Make prediction
///         prediction = output_layer(combined)
///
///     # Train end-to-end with cross-entropy loss
/// </code>
/// </para>
/// <para><b>Key Insights:</b>
///
/// 1. **Rapid Learning**: New information can be stored in memory with a single
///    write operation, enabling one-shot learning.
///
/// 2. **Continuous Learning**: Memory persists across episodes, allowing the
///    model to accumulate knowledge over time.
///
/// 3. **Differentiable Memory**: Both reading and writing operations are
///    differentiable, enabling end-to-end training.
/// </para>
/// <para>
/// Reference: Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., &amp; Lillicrap, T. (2016).
/// Meta-Learning with Memory-Augmented Neural Networks. ICML.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-Learning with Memory-Augmented Neural Networks",
    "https://arxiv.org/abs/1605.06065",
    Year = 2016,
    Authors = "Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MANNAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MANNOptions<T, TInput, TOutput> _mannOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _mannOptions;
    private readonly ExternalMemory<T> _memory;
    private readonly MANNMemoryStatistics _memoryStats;

    // Santoro et al. 2016's learned parameters beside the controller: the output layer over
    // [controller output; read vector] and the write gate logit alpha. Both are trained by MetaTrain, so both are
    // part of the parameter surface; leaving them out made GetParameters/SetParameters cover only the controller,
    // and a restore silently kept the untrained classifier and gate.

    /// <summary>
    /// The output layer packed as its row-major weights <c>[inputs x NumClasses]</c> followed by its bias
    /// <c>[NumClasses]</c>, where inputs = controller width + MemoryValueSize. Empty until the controller's width
    /// is known (the first episode) or a restore supplies it.
    /// </summary>
    /// <remarks>
    /// One field rather than two so the layer is a single variable-width slot: a fresh instance learns its width
    /// from the restored slice (Availability = Fit gives a resizable source), and inputs follows from the length.
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Fit)]
    private Vector<T> _outputLayer = new Vector<T>(0);

    /// <summary>The write gate logit alpha, one element; eq. 7's gate is sigmoid(alpha).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _writeGateLogit;

    /// <summary>
    /// Initializes a new instance of the MANNAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for MANN.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a MANN ready for meta-learning with external memory.
    ///
    /// <b>What MANN needs:</b>
    /// - <b>controller:</b> Neural network that processes inputs (the MetaModel)
    /// - <b>memorySize:</b> Size of external memory matrix
    /// - <b>memoryKeySize:</b> Dimension of memory keys
    /// - <b>memoryValueSize:</b> Dimension of memory values
    /// - <b>numReadHeads:</b> How many read heads to use
    ///
    /// <b>What makes it different from other meta-learning:</b>
    /// - Has external memory that persists across tasks
    /// - Can learn new patterns in one shot by writing to memory
    /// - Combines neural processing with explicit memory storage
    /// </para>
    /// </remarks>
    public MANNAlgorithm(MANNOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            // Santoro et al. 2016 eq. 25: the controller output is softmaxed and the episode loss is
            // -sum_t y_t^T log p_t, cross-entropy against the one-hot label. This used to inherit the inner
            // model's DefaultLossFunction, which is squared error - a regression objective for a classifier.
            // The caller's own LossFunction still wins when supplied.
            options.LossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            null,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _mannOptions = options;

        // Validate configuration
        if (!_mannOptions.IsValid())
        {
            throw new ArgumentException("MANN configuration is invalid. Check all parameters.", nameof(options));
        }

        // Initialize the LRUA external memory (Santoro et al. 2016): usage decay gamma, the initial write gate
        // sigmoid(alpha) (alpha is learned), and one least-used slot per read head.
        _memory = new ExternalMemory<T>(
            options.MemorySize,
            options.MemoryKeySize,
            options.MemoryValueSize,
            NumOps,
            options.UsageDecay,
            1.0 / (1.0 + Math.Exp(-options.WriteGateLogit)),
            options.NumReadHeads);

        // Initialize memory statistics
        _memoryStats = new MANNMemoryStatistics();

        // The learned write gate starts at sigmoid(WriteGateLogit).
        _writeGateLogit = new Vector<T>(new[] { NumOps.FromDouble(options.WriteGateLogit) });

        // Pre-initialize memory if configured
        if (_mannOptions.UseMemoryPreInitialization)
        {
            InitializeMemory();
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.MANN"/>.</value>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MANN;

    /// <summary>
    /// Performs one meta-training step using MANN's episodic training with memory.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on.</param>
    /// <returns>The average meta-loss across all tasks in the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For each task in the batch:</b>
    /// 1. Optionally clear recent memories (based on configuration)
    /// 2. Write the support examples to the external memory one LRUA time step at a time
    /// 3. Read the memory for each query and classify <c>[controller output; read vector]</c> with the output layer
    /// 4. Differentiate the query cross-entropy through the reads and writes, to the controller, the output layer
    ///    and the write gate logit alpha (Santoro et al. 2016 train the whole system end to end)
    /// </para>
    /// <para>The batch-mean gradients update all three with the outer learning rate.</para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var losses = new List<T>();
        var sums = new Vector<T>?[4];
        foreach (var task in taskBatch.Tasks)
        {
            // Clear recent memory for new task (if configured)
            if (_mannOptions.ClearMemoryBetweenTasks)
            {
                _memory.ClearRecent(_mannOptions.MemoryRetentionRatio);
            }

            var gradient = EpisodeGradient(task, commit: true);
            losses.Add(gradient.Loss);
            sums[0] = Accumulate(sums[0], gradient.Body);
            sums[1] = Accumulate(sums[1], gradient.OutputWeights);
            sums[2] = Accumulate(sums[2], gradient.OutputBias);
            sums[3] = Accumulate(sums[3], gradient.GateLogit);
        }

        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        Vector<T> BatchMean(Vector<T>? sum, int length)
        {
            var mean = sum is null ? new Vector<T>(length) : Engine.Divide(sum, batchSizeT);
            if (_mannOptions.GradientClipThreshold.HasValue && _mannOptions.GradientClipThreshold.Value > 0 && mean.Length > 0)
            {
                mean = ClipGradients(mean, _mannOptions.GradientClipThreshold.Value);
            }

            return mean;
        }

        var layer = OutputLayer();
        var body = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
        var updated = ApplyOuterGradients(_mannOptions.OuterLearningRate,
            (body, BatchMean(sums[0], body.Length)),
            (layer.Weights, BatchMean(sums[1], layer.Weights.Length)),
            (layer.Bias, BatchMean(sums[2], layer.Bias.Length)),
            (_writeGateLogit, BatchMean(sums[3], _writeGateLogit.Length)));
        InterfaceGuard.Parameterizable(MetaModel).SetParameters(updated[0]);
        _outputLayer = Vector<T>.Concatenate(updated[1], updated[2]);
        _writeGateLogit = updated[3];
        SyncWriteGate();

        // Apply memory consolidation if configured
        if (_mannOptions.UseMemoryConsolidation)
        {
            ConsolidateMemory();
        }

        // Return average loss
        T totalLoss = Engine.TensorSum(new Tensor<T>(losses.ToArray(), new[] { losses.Count }));
        return NumOps.Divide(totalLoss, batchSizeT);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The adapted model returns one score per class for each example, so this is the configured loss of their
    /// logarithm against the class indices - cross-entropy by default. A Vector output carries the predicted
    /// class of each example instead, and its loss is the classification error rate. The base's default compared
    /// the whole score block against a vector of labels, which cannot be lined up at all.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ProbabilityLoss(LossFunction, predictions, expectedOutput);

    /// <summary>
    /// Adapts to a new task by writing support examples to a copy of the memory.
    /// </summary>
    /// <param name="task">The new task containing support set examples.</param>
    /// <returns>A MANNModel that can classify using the populated memory.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// MANN adaptation is fast - it just writes support examples to memory with the learned write gate.
    /// The returned model classifies new examples by reading from memory through the learned output layer.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When you have a new task with labeled examples,
    /// MANN stores them in its "notebook" (memory). Then it can classify new
    /// examples by looking up similar stored examples.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Clone memory for this adaptation
        var adaptedMemory = _memory.Clone();

        // Write support examples to memory
        var features = ClassifierOutputs<T>.AsRows(MetaModel.Predict(task.SupportInput));
        EnsureOutputLayer(features.Shape[1]);
        int count = features.Shape[0];
        var keys = MannRows<T>.Keys(features, _mannOptions.MemoryKeySize);
        var values = MannRows<T>.OneHot(MannRows<T>.Labels(task.SupportOutput, count), _mannOptions.MemoryValueSize);
        for (int i = 0; i < count; i++)
        {
            adaptedMemory.WriteAndRead(keys.GetSlice(i).ToVector(), values.GetSlice(i).ToVector());
            _memoryStats.RecordWrite();
        }

        var layer = OutputLayer();
        return new MANNModel<T, TInput, TOutput>(
            MetaModel,
            adaptedMemory,
            _mannOptions,
            NumOps,
            MannRows<T>.Owned(layer.Weights, layer.Inputs, _mannOptions.NumClasses),
            MannRows<T>.Owned(layer.Bias, 1, _mannOptions.NumClasses));
    }

    /// <summary>One task's constants: its support and query sizes, the written label rows and the query targets.</summary>
    private sealed class Episode
    {
        public Episode(int support, int query, Tensor<T> supportValues, Tensor<T> queryTargets, TInput stacked,
            TOutput stackedTarget)
        {
            Support = support;
            Query = query;
            SupportValues = supportValues;
            QueryTargets = queryTargets;
            Stacked = stacked;
            StackedTarget = stackedTarget;
        }

        public int Support { get; }
        public int Query { get; }

        /// <summary>The one-hot support labels written as memory values, <c>[support, MemoryValueSize]</c>.</summary>
        public Tensor<T> SupportValues { get; }

        /// <summary>The one-hot query labels, <c>[query, NumClasses]</c>.</summary>
        public Tensor<T> QueryTargets { get; }

        /// <summary>The support rows followed by the query rows, embedded in one controller pass.</summary>
        public TInput Stacked { get; }

        /// <summary>A placeholder target with one row per stacked example (its values are not read).</summary>
        public TOutput StackedTarget { get; }
    }

    private Episode EpisodeOf(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // The episode is read again after ComputeGradients resets the step arena, which reissues arena tensors by
        // element count under other shapes ([4, 2] targets came back as [8]). Build it on the heap.
        using var heapOwned = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        int support = GetBatchSize(task.SupportInput), query = GetBatchSize(task.QueryInput);
        var supportValues = MannRows<T>.OneHot(MannRows<T>.Labels(task.SupportOutput, support), _mannOptions.MemoryValueSize);
        var queryTargets = MannRows<T>.OneHot(MannRows<T>.Labels(task.QueryOutput, query), _mannOptions.NumClasses);
        var stacked = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { support + query, 1 }));
        return new Episode(support, query, supportValues, queryTargets, stacked, stackedTarget);
    }

    /// <summary>The controller's features for the stacked support and query rows, <c>[support + query, width]</c>.</summary>
    private Tensor<T> Embed(Episode episode)
    {
        var embeddings = ClassifierOutputs<T>.AsRows(MetaModel.Predict(episode.Stacked));
        if (embeddings.Shape[0] != episode.Support + episode.Query)
        {
            throw new InvalidOperationException(
                $"The controller produced {embeddings.Shape[0]} rows for {episode.Support + episode.Query} examples.");
        }

        return embeddings;
    }

    /// <summary>The trainable output layer and write gate logit as tape leaves.</summary>
    private sealed class MannWeights
    {
        public MannWeights(Tensor<T> output, Tensor<T> bias, Tensor<T> gateLogit)
        {
            Output = output;
            Bias = bias;
            GateLogit = gateLogit;
            Leaves = new[] { output, bias, gateLogit };
        }

        /// <summary>The output layer's weights, <c>[controller width + MemoryValueSize, NumClasses]</c>.</summary>
        public Tensor<T> Output { get; }

        /// <summary>The output layer's bias, <c>[1, NumClasses]</c>.</summary>
        public Tensor<T> Bias { get; }

        /// <summary>The write gate logit alpha, <c>[1, 1]</c>.</summary>
        public Tensor<T> GateLogit { get; }

        public IReadOnlyList<Tensor<T>> Leaves { get; }
    }

    private MannWeights Weights()
    {
        var layer = OutputLayer();
        return new MannWeights(
            MannRows<T>.Owned(layer.Weights, layer.Inputs, _mannOptions.NumClasses),
            MannRows<T>.Owned(layer.Bias, 1, _mannOptions.NumClasses),
            MannRows<T>.Owned(_writeGateLogit, 1, 1));
    }

    private sealed class EpisodeGradients
    {
        public EpisodeGradients(T loss, Vector<T> body, Vector<T> outputWeights, Vector<T> outputBias, Vector<T> gateLogit)
        {
            Loss = loss;
            Body = body;
            OutputWeights = outputWeights;
            OutputBias = outputBias;
            GateLogit = gateLogit;
        }

        public T Loss { get; }
        public Vector<T> Body { get; }
        public Vector<T> OutputWeights { get; }
        public Vector<T> OutputBias { get; }
        public Vector<T> GateLogit { get; }
    }

    /// <summary>
    /// One episode's loss and its exact gradients. The controller's gradient comes from the whole episode rebuilt
    /// from its features on the tape (output layer and gate held constant); the output layer's and the gate's from
    /// the same episode on fixed features. With <paramref name="commit"/> the memory keeps the episode's final
    /// state, as a recurrent memory does.
    /// </summary>
    private EpisodeGradients EpisodeGradient(IMetaLearningTask<T, TInput, TOutput> task, bool commit)
    {
        var episode = EpisodeOf(task);
        var start = _memory.Snapshot();
        var embeddings = Embed(episode);
        EnsureOutputLayer(embeddings.Shape[1]);

        var frozen = Weights();
        var composed = new EmbeddingObjectiveLoss<T>(rows => Objective(start, rows, episode, frozen, out _));
        var body = ComputeGradients(MetaModel, episode.Stacked, episode.StackedTarget, composed);

        var weights = Weights();
        var output = new Vector<T>(weights.Output.Length);
        var bias = new Vector<T>(weights.Bias.Length);
        var gate = new Vector<T>(weights.GateLogit.Length);
        ExternalMemory<T>.State? end = null;
        T loss = AiDotNet.Training.TapeGradients<T>.Of(
            () =>
            {
                var objective = Objective(start, embeddings, episode, weights, out var final);
                end = final;
                return objective;
            },
            weights.Leaves,
            gradients =>
            {
                CopyGradient(gradients, weights.Output, output);
                CopyGradient(gradients, weights.Bias, bias);
                CopyGradient(gradients, weights.GateLogit, gate);
                if (commit && end is not null)
                {
                    _memory.Commit(end);
                }
            });

        for (int i = 0; i < episode.Support; i++) _memoryStats.RecordWrite();
        for (int q = 0; q < episode.Query; q++) _memoryStats.RecordRead();
        return new EpisodeGradients(loss, body, output, bias, gate);
    }

    /// <summary>
    /// The episode on the tape: each support example is one LRUA step (write its key and label through the gated
    /// write weights, then read with the key), each query reads the memory, and the output layer classifies
    /// <c>[controller output; read vector]</c>. Returns <c>-mean_q y_q^T log softmax(o_q)</c> plus the usage penalty.
    /// </summary>
    private Tensor<T> Objective(ExternalMemory<T>.State start, Tensor<T> embeddings, Episode episode, MannWeights weights,
        out ExternalMemory<T>.State end)
    {
        int width = embeddings.Shape[1], keySize = _mannOptions.MemoryKeySize, valueSize = _mannOptions.MemoryValueSize;
        var gate = Engine.TensorSigmoid(weights.GateLogit);
        var state = start;

        if (episode.Support > 0)
        {
            var supportKeys = MannRows<T>.Keys(
                Engine.TensorSlice(embeddings, new[] { 0, 0 }, new[] { episode.Support, width }), keySize);
            for (int i = 0; i < episode.Support; i++)
            {
                var key = Engine.Reshape(Engine.TensorSlice(supportKeys, new[] { i, 0 }, new[] { 1, keySize }), new[] { keySize });
                var value = Engine.Reshape(
                    Engine.TensorSlice(episode.SupportValues, new[] { i, 0 }, new[] { 1, valueSize }), new[] { valueSize });
                state = _memory.Step(state, key, value, gate, out _);
            }
        }

        var objective = new Tensor<T>(new[] { 1 });
        if (episode.Query > 0)
        {
            var query = Engine.TensorSlice(embeddings, new[] { episode.Support, 0 }, new[] { episode.Query, width });
            var readWeights = _memory.ContentWeightRows(state.Keys, MannRows<T>.Keys(query, keySize));
            state = _memory.AfterReads(state, readWeights, out var reads);
            var logits = MannRows<T>.Logits(Engine.TensorConcatenate(new[] { query, reads }, 1), weights.Output, weights.Bias);
            var picked = Engine.TensorMultiply(episode.QueryTargets, Engine.TensorLogSoftmax(logits, 1));
            objective = Engine.TensorMultiplyScalar(Sum(picked), NumOps.FromDouble(-1.0 / episode.Query));
        }

        if (_mannOptions.MemoryRegularization > 0.0)
        {
            var penalty = Engine.TensorMultiplyScalar(Sum(state.Usage), NumOps.FromDouble(1.0 / _memory.Size));
            objective = Engine.TensorAdd(objective,
                Engine.TensorMultiplyScalar(penalty, NumOps.FromDouble(_mannOptions.MemoryRegularization)));
        }

        end = state;
        return objective;
    }

    private Tensor<T> Sum(Tensor<T> value) => Engine.Reshape(Engine.ReduceSum(value, null), new[] { 1 });

    private static void CopyGradient(Dictionary<Tensor<T>, Tensor<T>> gradients, Tensor<T> leaf, Vector<T> into)
    {
        if (!gradients.TryGetValue(leaf, out var gradient)) return;
        gradient.ToVector().AsSpan().Slice(0, leaf.Length).CopyTo(into.AsWritableSpan());
    }

    private Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
        => sum is null ? new Vector<T>(values.AsSpan().ToArray()) : Engine.Add(sum, values);

    /// <summary>
    /// Creates the output layer the first time the controller's width is known: Glorot-uniform weights over
    /// <c>[controller output; read vector]</c> and a zero bias.
    /// </summary>
    private void EnsureOutputLayer(int controllerWidth)
    {
        int inputs = controllerWidth + _mannOptions.MemoryValueSize;
        if (_outputLayer.Length > 0)
        {
            int existing = OutputLayer().Inputs;
            if (inputs != existing)
            {
                throw new InvalidOperationException(
                    $"The controller's output width changed from {existing - _mannOptions.MemoryValueSize} to {controllerWidth}.");
            }

            return;
        }

        var (weights, bias) = MannRows<T>.GlorotLayer(inputs, _mannOptions.NumClasses, _mannOptions.RandomSeed);
        _outputLayer = Vector<T>.Concatenate(weights.ToVector(), bias.ToVector());
    }

    /// <summary>
    /// Unpacks <see cref="_outputLayer"/> into its weights <c>[inputs x NumClasses]</c> and bias <c>[NumClasses]</c>.
    /// </summary>
    private (Vector<T> Weights, Vector<T> Bias, int Inputs) OutputLayer()
    {
        if (_outputLayer.Length == 0)
        {
            throw new InvalidOperationException("The output layer is created on the first episode.");
        }

        int classes = _mannOptions.NumClasses;
        int inputs = _outputLayer.Length / classes - 1;
        if (_outputLayer.Length % classes != 0 || inputs <= _mannOptions.MemoryValueSize)
        {
            throw new InvalidOperationException(
                $"The output layer holds {_outputLayer.Length} values, which is not (controller width + " +
                $"{_mannOptions.MemoryValueSize}) x {classes} weights plus {classes} biases.");
        }

        int weightCount = inputs * classes;
        return (_outputLayer.GetSubVector(0, weightCount), _outputLayer.GetSubVector(weightCount, classes), inputs);
    }

    /// <summary>Hands the memory the current gate <c>sigmoid(alpha)</c> for the writes outside training.</summary>
    private void SyncWriteGate()
        => _memory.SetWriteGate(1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(_writeGateLogit[0]))));

    /// <summary>
    /// Applies memory consolidation to prune rarely-used memories.
    /// </summary>
    private void ConsolidateMemory()
    {
        var rarelyUsedSlots = _memory.FindRarelyUsedSlots(_mannOptions.MemoryUsageThreshold);
        foreach (var slot in rarelyUsedSlots)
        {
            _memory.ClearSlot(slot);
        }
    }

    /// <summary>
    /// Initializes memory with random or pre-trained values.
    /// </summary>
    private void InitializeMemory()
    {
        var random = _mannOptions.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_mannOptions.RandomSeed.Value)
            : RandomHelper.CreateSeededRandom(0);

        for (int i = 0; i < _mannOptions.MemorySize; i++)
        {
            var key = new Vector<T>(_mannOptions.MemoryKeySize);
            for (int j = 0; j < _mannOptions.MemoryKeySize; j++)
            {
                key[j] = NumOps.FromDouble((random.NextDouble() - 0.5) * 0.01);
            }

            var value = new Vector<T>(_mannOptions.MemoryValueSize);
            _memory.Write(i, key, value);
        }
    }

    /// <summary>
    /// Gets batch size from input.
    /// </summary>
    private int GetBatchSize(TInput input)
    {
        if (input is Matrix<T> matrix)
            return matrix.Rows;
        if (input is Tensor<T> tensor && tensor.Shape.Length >= 1)
            return tensor.Shape[0];
        if (input is Vector<T>)
            return 1;
        return 1;
    }

    /// <summary>One episode's loss and gradients, without changing the memory, for tests.</summary>
    internal (T Loss, Vector<T> Body, Vector<T> OutputWeights, Vector<T> OutputBias, Vector<T> GateLogit)
        EpisodeGradientForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var g = EpisodeGradient(task, commit: false);
        return (g.Loss, g.Body, g.OutputWeights, g.OutputBias, g.GateLogit);
    }

    /// <summary>
    /// One episode's loss from the memory as the next <see cref="MetaTrain"/> would start it (cleared first when
    /// configured), without changing the memory, for tests.
    /// </summary>
    internal T EpisodeLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var memory = _memory.Clone();
        if (_mannOptions.ClearMemoryBetweenTasks)
        {
            memory.ClearRecent(_mannOptions.MemoryRetentionRatio);
        }

        var episode = EpisodeOf(task);
        var embeddings = Embed(episode);
        EnsureOutputLayer(embeddings.Shape[1]);
        return Objective(memory.Snapshot(), embeddings, episode, Weights(), out _)[0];
    }

    /// <summary>The output layer's weights, row-major <c>[inputs, classes]</c>, for tests.</summary>
    // A view over state the registry already persists, not state of its own: the generator must not serialize it.
    [AiDotNet.Attributes.Scratch]
    internal Vector<T> OutputWeightsForTesting
    {
        get => OutputLayer().Weights;
        set => _outputLayer = Vector<T>.Concatenate(value, OutputLayer().Bias);
    }

    /// <summary>The output layer's bias, for tests.</summary>
    // A view over state the registry already persists, not state of its own: the generator must not serialize it.
    [AiDotNet.Attributes.Scratch]
    internal Vector<T> OutputBiasForTesting
    {
        get => OutputLayer().Bias;
        set => _outputLayer = Vector<T>.Concatenate(OutputLayer().Weights, value);
    }

    /// <summary>The write gate logit alpha, for tests.</summary>
    // A view over state the registry already persists, not state of its own: the generator must not serialize it.
    [AiDotNet.Attributes.Scratch]
    internal Vector<T> GateLogitForTesting
    {
        get => _writeGateLogit;
        set
        {
            _writeGateLogit = value;
            SyncWriteGate();
        }
    }
}

/// <summary>
/// The row arithmetic MANN's learner and adapted model share, as whole-tensor engine ops.
/// </summary>
internal static class MannRows<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Rows <c>0 .. count - 1</c> of <paramref name="rows"/> taken cyclically, <c>[count, width]</c>.</summary>
    private static Tensor<T> CycleRows(Tensor<T> rows, int count)
    {
        int available = rows.Shape[0], width = rows.Shape[1];
        if (available == count) return rows;
        if (available == 0) return new Tensor<T>(new[] { count, width });
        int repeats = (count + available - 1) / available;
        var tiled = repeats > 1 ? Engine.TensorTile(rows, new[] { repeats, 1 }) : rows;
        return Engine.TensorSlice(tiled, new[] { 0, 0 }, new[] { count, width });
    }

    /// <summary>Each feature row truncated or zero-padded to the memory's key width, <c>[count, keySize]</c>.</summary>
    internal static Tensor<T> Keys(Tensor<T> features, int keySize)
    {
        int count = features.Shape[0], width = features.Shape[1];
        if (width == keySize) return features;
        if (width > keySize) return Engine.TensorSlice(features, new[] { 0, 0 }, new[] { count, keySize });
        return Engine.TensorConcatenate(new[] { features, new Tensor<T>(new[] { count, keySize - width }) }, 1);
    }

    /// <summary>
    /// The class label of each of <paramref name="count"/> examples, <c>[count]</c>, truncated toward zero as an
    /// integer cast would. A vector or rank-1 tensor holds one label per example (0 past its end); a one-column
    /// matrix holds labels, a wider one one-hot rows whose first maximum is the label; rows repeat cyclically.
    /// </summary>
    internal static Tensor<T> Labels(object? output, int count)
    {
        switch (output)
        {
            case Vector<T> vector:
                return Truncate(Prefix(Tensor<T>.FromVector(vector), count));
            case Tensor<T> tensor when tensor.Shape.Length == 1:
                return Truncate(Prefix(tensor, count));
            case Matrix<T> matrix:
                return Truncate(CycleRows(RowLabels(Tensor<T>.FromMatrix(matrix)), count).Reshape(count));
            case Tensor<T> tensor when tensor.Shape.Length >= 2:
                return Truncate(CycleRows(RowLabels(ClassifierOutputs<T>.AsRows(tensor)), count).Reshape(count));
            default:
                return new Tensor<T>(new[] { count });
        }
    }

    /// <summary>The first <paramref name="count"/> values, zero-padded past the end.</summary>
    private static Tensor<T> Prefix(Tensor<T> values, int count)
    {
        if (values.Length == count) return values;
        if (values.Length > count) return Engine.TensorSlice(values, new[] { 0 }, new[] { count });
        return Engine.TensorConcatenate(new[] { values, new Tensor<T>(new[] { count - values.Length }) }, 0);
    }

    /// <summary>A one-column matrix's values, or each wider row's first maximum's column, <c>[rows, 1]</c>.</summary>
    private static Tensor<T> RowLabels(Tensor<T> rows)
    {
        int count = rows.Shape[0], width = rows.Shape[1];
        if (width == 1 || count == 0) return width == 1 ? rows : new Tensor<T>(new[] { 0, 1 });

        var maximum = Engine.TensorBroadcastTo(Engine.ReduceMax(rows, new[] { 1 }, true), new[] { count, width });
        var isMaximum = Engine.TensorEquals(rows, maximum);
        var first = Engine.TensorMultiply(isMaximum, Engine.TensorEquals(Engine.TensorCumSum(isMaximum, 1), Ops.One));
        var columns = Engine.Reshape(Engine.TensorArange<T>(0, width, 1), new[] { width, 1 });
        return Engine.TensorMatMul(first, columns);
    }

    private static Tensor<T> Truncate(Tensor<T> values) => Engine.TensorTrunc(values);

    /// <summary>One-hot rows of <paramref name="labels"/>, <c>[count, width]</c>; a label outside [0, width) is a zero row.</summary>
    internal static Tensor<T> OneHot(Tensor<T> labels, int width)
    {
        int count = labels.Length;
        var labelColumn = Engine.TensorBroadcastTo(Engine.Reshape(labels, new[] { count, 1 }), new[] { count, width });
        var classRow = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<T>(0, width, 1), new[] { 1, width }), new[] { count, width });
        return Engine.TensorEquals(labelColumn, classRow);
    }

    /// <summary>
    /// Santoro et al. 2016's output layer: the logits <c>o = [h; r] W + b</c> of each combined row, <c>[count, classes]</c>.
    /// </summary>
    internal static Tensor<T> Logits(Tensor<T> combined, Tensor<T> weights, Tensor<T> bias)
    {
        var ones = new Tensor<T>(new[] { combined.Shape[0], 1 });
        ones.Fill(Ops.One);
        return Engine.TensorAdd(Engine.TensorMatMul(combined, weights), Engine.TensorMatMul(ones, bias));
    }

    /// <summary>
    /// A Glorot-uniform weight matrix <c>[inputs, classes]</c>, limit <c>sqrt(6 / (inputs + classes))</c>, and a zero
    /// bias <c>[1, classes]</c>. Seeded draws are reproducible; without a seed the draw seed is secure-random.
    /// </summary>
    internal static (Tensor<T> Weights, Tensor<T> Bias) GlorotLayer(int inputs, int classes, int? seed)
    {
        double limit = Math.Sqrt(6.0 / (inputs + classes));
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var weights = Engine.TensorRandomUniformRange<T>(new[] { inputs, classes },
            Ops.FromDouble(-limit), Ops.FromDouble(limit), random.Next());
        return (Owned(weights.ToVector(), inputs, classes), new Tensor<T>(new[] { 1, classes }));
    }

    /// <summary>An owned <c>[rows, columns]</c> copy of a row-major vector.</summary>
    internal static Tensor<T> Owned(Vector<T> values, int rows, int columns)
    {
        var matrix = new Tensor<T>(new[] { rows, columns });
        values.AsSpan().Slice(0, matrix.Length).CopyTo(matrix.AsWritableSpan());
        return matrix;
    }
}

/// <summary>
/// The external memory of a MANN with least-recently-used access (LRUA), Santoro et al. 2016, section 3.2.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// The memory <c>M</c> holds one key row per slot; alongside it, a value row per slot carries what was written with
/// that key (the label of the example in MANN's classification setting), written with the same weights. Per time
/// step, with key <c>k_t</c>:
/// </para>
/// <code>
///   w^lu_{t-1}(i) = 1 for the n smallest usages (n = number of reads), else 0
///   w^w_t        = sigmoid(alpha) w^r_{t-1} + (1 - sigmoid(alpha)) w^lu_{t-1}
///   M_t(i)       = M_{t-1}(i) + w^w_t(i) k_t                     after the least-used slot is zeroed
///   w^r_t(i)     = softmax_i(K(k_t, M_t(i))),  K the cosine similarity
///   r_t          = sum_i w^r_t(i) M_t(i)       (here: the value rows)
///   w^u_t        = gamma w^u_{t-1} + w^r_t + w^w_t
/// </code>
/// <para>
/// Ties in usage go to the lower slot index, so the least-used slot and the n least used are deterministic. Every
/// step is made of engine ops, so on a gradient tape the reads and writes are differentiable in the keys, the
/// values and the gate; the least-used selection is piecewise constant, as in the paper.
/// </para>
/// </remarks>
public class ExternalMemory<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    // Owned buffers, updated in place: engine results may live in a step arena, so state is copied into these.
    private readonly Tensor<T> _memoryKeys;
    private readonly Tensor<T> _memoryValues;
    private readonly Tensor<T> _usage;
    private readonly Tensor<T> _previousReadWeights;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the ExternalMemory with the default LRUA settings: usage decay 0.95, write gate
    /// 0.5 and one read.
    /// </summary>
    public ExternalMemory(int size, int keySize, int valueSize, INumericOperations<T> numOps)
        : this(size, keySize, valueSize, numOps, usageDecay: 0.95, writeGate: 0.5, numReads: 1)
    {
    }

    /// <summary>
    /// Initializes a new instance of the ExternalMemory.
    /// </summary>
    /// <param name="size">The number of slots.</param>
    /// <param name="keySize">The key width.</param>
    /// <param name="valueSize">The value width.</param>
    /// <param name="numOps">The numeric operations for <typeparamref name="T"/>.</param>
    /// <param name="usageDecay">The usage decay gamma, in [0, 1].</param>
    /// <param name="writeGate">The write gate sigmoid(alpha), in [0, 1].</param>
    /// <param name="numReads">The number of reads n: how many least-used slots w^lu marks.</param>
    public ExternalMemory(int size, int keySize, int valueSize, INumericOperations<T> numOps,
        double usageDecay, double writeGate, int numReads)
    {
        if (usageDecay < 0 || usageDecay > 1) throw new ArgumentOutOfRangeException(nameof(usageDecay));
        if (writeGate < 0 || writeGate > 1) throw new ArgumentOutOfRangeException(nameof(writeGate));
        if (numReads <= 0) throw new ArgumentOutOfRangeException(nameof(numReads));

        Size = size;
        KeySize = keySize;
        ValueSize = valueSize;
        UsageDecay = usageDecay;
        WriteGate = writeGate;
        NumReads = Math.Min(numReads, size);
        _numOps = numOps;

        _memoryKeys = new Tensor<T>(new[] { size, keySize });
        _memoryValues = new Tensor<T>(new[] { size, valueSize });
        _usage = new Tensor<T>(new[] { size });
        _previousReadWeights = new Tensor<T>(new[] { size });

        InitializeMemory();
    }

    /// <summary>Gets the memory size.</summary>
    public int Size { get; }

    /// <summary>Gets the key dimension.</summary>
    public int KeySize { get; }

    /// <summary>Gets the value dimension.</summary>
    public int ValueSize { get; }

    /// <summary>Gets the usage decay gamma.</summary>
    public double UsageDecay { get; }

    /// <summary>Gets the write gate sigmoid(alpha) used by <see cref="WriteAndRead"/>; MANN keeps it at its learned value.</summary>
    public double WriteGate { get; private set; }

    /// <summary>Gets the number of reads n, the number of slots the least-used weights mark.</summary>
    public int NumReads { get; }

    /// <summary>
    /// One memory state: the key and value rows, the usage weights and the last read weights. Steps build new
    /// states from engine ops, so a state built on a gradient tape is differentiable.
    /// </summary>
    internal sealed class State
    {
        public State(Tensor<T> keys, Tensor<T> values, Tensor<T> usage, Tensor<T> previousRead)
        {
            Keys = keys;
            Values = values;
            Usage = usage;
            PreviousRead = previousRead;
        }

        /// <summary>The key rows, <c>[Size, KeySize]</c>.</summary>
        public Tensor<T> Keys { get; }

        /// <summary>The value rows, <c>[Size, ValueSize]</c>.</summary>
        public Tensor<T> Values { get; }

        /// <summary>The usage weights w^u, <c>[Size]</c>.</summary>
        public Tensor<T> Usage { get; }

        /// <summary>The last read weights w^r, <c>[Size]</c>.</summary>
        public Tensor<T> PreviousRead { get; }
    }

    /// <summary>Sets the write gate sigmoid(alpha), in [0, 1].</summary>
    internal void SetWriteGate(double writeGate)
    {
        if (writeGate < 0 || writeGate > 1 || double.IsNaN(writeGate))
            throw new ArgumentOutOfRangeException(nameof(writeGate));
        WriteGate = writeGate;
    }

    /// <summary>A copy of the current state, independent of later changes to the memory.</summary>
    internal State Snapshot()
        => new State(_memoryKeys.Clone(), _memoryValues.Clone(), _usage.Clone(), _previousReadWeights.Clone());

    /// <summary>Makes <paramref name="state"/>'s values the memory's own.</summary>
    internal void Commit(State state)
    {
        Assign(_memoryKeys, state.Keys);
        Assign(_memoryValues, state.Values);
        Assign(_usage, state.Usage);
        Assign(_previousReadWeights, state.PreviousRead);
    }

    private State Current => new State(_memoryKeys, _memoryValues, _usage, _previousReadWeights);

    /// <summary>
    /// Writes a key and value straight into one slot, as a write whose weight is 1 at <paramref name="location"/>:
    /// the slot's leading entries are replaced and its usage grows by 1.
    /// </summary>
    public void Write(int location, Vector<T> key, Vector<T> value)
    {
        if (location < 0 || location >= Size)
            return;

        int keyCount = Math.Min(KeySize, key.Length);
        key.AsSpan().Slice(0, keyCount).CopyTo(_memoryKeys.AsWritableSpan().Slice(location * KeySize, keyCount));
        int valueCount = Math.Min(ValueSize, value.Length);
        value.AsSpan().Slice(0, valueCount).CopyTo(_memoryValues.AsWritableSpan().Slice(location * ValueSize, valueCount));

        _usage[location] = _numOps.Add(_usage[location], _numOps.One);
    }

    /// <summary>
    /// One LRUA time step: writes <paramref name="key"/> and <paramref name="value"/> through the write weights,
    /// then reads with the same key. Returns the read content.
    /// </summary>
    public Vector<T> WriteAndRead(Vector<T> key, Vector<T> value)
    {
        var gate = new Tensor<T>(new[] { 1, 1 });
        gate[0] = _numOps.FromDouble(WriteGate);
        var next = Step(Current, Fit(key, KeySize), Fit(value, ValueSize), gate, out var read);
        var content = read.ToVector();
        Commit(next);
        return content;
    }

    /// <summary>
    /// One LRUA read without a write: content weights for <paramref name="key"/>, which update the usage and become
    /// the next write's read weights. Returns the read content.
    /// </summary>
    public Vector<T> ReadAndUpdate(Vector<T> key)
    {
        var keys = Engine.Reshape(Fit(key, KeySize), new[] { 1, KeySize });
        var next = AfterReads(Current, ContentWeightRows(_memoryKeys, keys), out var reads);
        var content = Engine.Reshape(reads, new[] { ValueSize }).ToVector();
        Commit(next);
        return content;
    }

    /// <summary>
    /// One LRUA step on <paramref name="state"/>: the write weights <c>gate w^r_{t-1} + (1 - gate) w^lu_{t-1}</c>,
    /// the least-used slot zeroed, the key and value rows added with the write weights, then the read with the same
    /// key and the usage update. <paramref name="key"/> is <c>[KeySize]</c>, <paramref name="value"/>
    /// <c>[ValueSize]</c> and <paramref name="gate"/> <c>[1, 1]</c>; <paramref name="read"/> is <c>[ValueSize]</c>.
    /// </summary>
    internal State Step(State state, Tensor<T> key, Tensor<T> value, Tensor<T> gate, out Tensor<T> read)
    {
        Tensor<T> leastUsed, keep;
        using (new NoGradScope<T>())
        {
            leastUsed = Engine.Reshape(LeastUsed(state.Usage, NumReads), new[] { Size, 1 });
            keep = Engine.Reshape(
                Engine.TensorAddScalar(Engine.TensorNegate(LeastUsed(state.Usage, 1)), _numOps.One), new[] { Size, 1 });
        }

        var closedGate = Engine.TensorAddScalar(Engine.TensorNegate(gate), _numOps.One);
        var write = Engine.TensorAdd(
            Engine.TensorMatMul(Engine.Reshape(state.PreviousRead, new[] { Size, 1 }), gate),
            Engine.TensorMatMul(leastUsed, closedGate));

        var keyRow = Engine.Reshape(key, new[] { 1, KeySize });
        var keys = Engine.TensorAdd(Engine.TensorMultiply(state.Keys, keep), Engine.TensorMatMul(write, keyRow));
        var values = Engine.TensorAdd(Engine.TensorMultiply(state.Values, keep),
            Engine.TensorMatMul(write, Engine.Reshape(value, new[] { 1, ValueSize })));

        var readWeights = ContentWeightRows(keys, keyRow);
        read = Engine.Reshape(Engine.TensorMatMul(readWeights, values), new[] { ValueSize });
        var readColumn = Engine.Reshape(readWeights, new[] { Size });
        var usage = Engine.TensorAdd(
            Engine.TensorAdd(Engine.TensorMultiplyScalar(state.Usage, _numOps.FromDouble(UsageDecay)), readColumn),
            Engine.Reshape(write, new[] { Size }));
        return new State(keys, values, usage, readColumn);
    }

    /// <summary>
    /// Reads in sequence with the read weight rows <paramref name="readWeights"/> <c>[n, Size]</c>, each read updating
    /// the usage: <c>w^u = gamma^n w^u + sum_q gamma^(n-1-q) w^r_q</c>, and the last row becomes the previous read.
    /// <paramref name="reads"/> are the read contents, <c>[n, ValueSize]</c>.
    /// </summary>
    internal State AfterReads(State state, Tensor<T> readWeights, out Tensor<T> reads)
    {
        int count = readWeights.Shape[0];
        reads = Engine.TensorMatMul(readWeights, state.Values);
        if (count == 0) return state;

        Tensor<T> decay;
        using (new NoGradScope<T>())
        {
            // gamma^(n-1-q) for q = 0 .. n-1.
            var steps = Engine.TensorAddScalar(Engine.TensorNegate(Engine.TensorArange<T>(0, count, 1)),
                _numOps.FromDouble(count - 1));
            decay = UsageDecay <= 0.0
                ? Engine.TensorEquals(steps, _numOps.Zero)
                : Engine.TensorExp(Engine.TensorMultiplyScalar(steps, _numOps.FromDouble(Math.Log(UsageDecay))));
            decay = Engine.Reshape(decay, new[] { 1, count });
        }

        var usage = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(state.Usage, _numOps.FromDouble(Math.Pow(UsageDecay, count))),
            Engine.Reshape(Engine.TensorMatMul(decay, readWeights), new[] { Size }));
        var last = Engine.Reshape(Engine.TensorSlice(readWeights, new[] { count - 1, 0 }, new[] { 1, Size }), new[] { Size });
        return new State(state.Keys, state.Values, usage, last);
    }

    /// <summary>The read weights <c>softmax_i(cosine(key, M(i)))</c> for a key, without changing the memory.</summary>
    public Vector<T> ContentWeights(Vector<T> key)
        => ContentWeightRows(_memoryKeys, Engine.Reshape(Fit(key, KeySize), new[] { 1, KeySize })).ToVector();

    /// <summary>The least-used weights <c>w^lu</c>: 1 for the <see cref="NumReads"/> least-used slots, else 0.</summary>
    public Vector<T> LeastUsedWeights() => LeastUsed(_usage, NumReads).ToVector();

    /// <summary>A copy of the usage weights <c>w^u</c>.</summary>
    public Vector<T> GetUsage() => _usage.ToVector();

    /// <summary>
    /// Reads from memory using attention weights: <c>sum_i w(i) V(i)</c> over the first slots the weights cover.
    /// </summary>
    public Vector<T> Read(Vector<T> attentionWeights)
    {
        int slots = Math.Min(Size, attentionWeights.Length);
        if (slots == 0) return new Vector<T>(ValueSize);

        var weights = Engine.Reshape(
            Engine.TensorSlice(Tensor<T>.FromVector(attentionWeights), new[] { 0 }, new[] { slots }), new[] { 1, slots });
        var values = Engine.TensorSlice(_memoryValues, new[] { 0, 0 }, new[] { slots, ValueSize });
        return Engine.Reshape(Engine.TensorMatMul(weights, values), new[] { ValueSize }).ToVector();
    }

    /// <summary>
    /// Reads for many keys at once without changing the memory: content weights per key row, then the weighted
    /// value rows, <c>[keys, ValueSize]</c>.
    /// </summary>
    internal Tensor<T> ReadRows(Tensor<T> keys)
        => Engine.TensorMatMul(ContentWeightRows(_memoryKeys, keys), _memoryValues);

    /// <summary>
    /// Gets memory key at specified index.
    /// </summary>
    public Vector<T> GetKey(int index)
    {
        var key = new Vector<T>(KeySize);
        if (index >= 0 && index < Size)
        {
            _memoryKeys.AsSpan().Slice(index * KeySize, KeySize).CopyTo(key.AsWritableSpan());
        }
        return key;
    }

    /// <summary>
    /// Finds the least-used memory slot (the lowest index among equal usages).
    /// </summary>
    public int FindLeastRecentlyUsedSlot()
        => Engine.TensorArgMax(LeastUsed(_usage, 1), 0).GetFlat(0);

    /// <summary>
    /// Finds the slots whose usage is below <paramref name="threshold"/>.
    /// </summary>
    public List<int> FindRarelyUsedSlots(double threshold)
    {
        var flags = Engine.TensorLessThan(_usage, _numOps.FromDouble(threshold)).Cast<int>().ToArray();
        return Enumerable.Range(0, Size).Where(slot => flags[slot] != 0).ToList();
    }

    /// <summary>
    /// Clears a specific memory slot.
    /// </summary>
    public void ClearSlot(int location)
    {
        if (location < 0 || location >= Size)
            return;

        var mask = new Tensor<T>(new[] { Size });
        mask.Fill(_numOps.One);
        mask[location] = _numOps.Zero;
        ClearWhere(mask);
    }

    /// <summary>
    /// Keeps the <paramref name="retentionRatio"/> most-used slots and clears the rest (ties keep the lower index).
    /// </summary>
    public void ClearRecent(double retentionRatio)
    {
        int numToKeep = (int)(Size * retentionRatio);
        var keep = Engine.TensorLessThan(Ranks(_usage, descending: true), _numOps.FromDouble(numToKeep));
        ClearWhere(keep);
    }

    /// <summary>
    /// Computes memory usage penalty for regularization: the mean usage weight.
    /// </summary>
    public T ComputeUsagePenalty()
        => _numOps.Divide(Engine.TensorSum(_usage), _numOps.FromDouble(Size));

    /// <summary>
    /// Clones the external memory.
    /// </summary>
    public ExternalMemory<T> Clone()
    {
        var cloned = new ExternalMemory<T>(Size, KeySize, ValueSize, _numOps, UsageDecay, WriteGate, NumReads);
        cloned.Commit(Current);
        return cloned;
    }

    /// <summary>Small uniform keys in [-0.005, 0.005), one seeded engine draw; zero values and usage.</summary>
    private void InitializeMemory()
    {
        int seed = RandomHelper.CreateSeededRandom(0).Next();
        Assign(_memoryKeys, Engine.TensorRandomUniformRange<T>(new[] { Size, KeySize },
            _numOps.FromDouble(-0.005), _numOps.FromDouble(0.005), seed));
    }

    /// <summary>Copies <paramref name="source"/>'s values into the owned buffer <paramref name="target"/>.</summary>
    private static void Assign(Tensor<T> target, Tensor<T> source)
        => source.ToVector().AsSpan().CopyTo(target.AsWritableSpan());

    /// <summary>
    /// Zeroes every slot where <paramref name="keep"/> is 0: its key, value, usage and last read weight.
    /// </summary>
    private void ClearWhere(Tensor<T> keep)
    {
        var rows = Engine.Reshape(keep, new[] { Size, 1 });
        Assign(_memoryKeys, Engine.TensorMultiply(_memoryKeys, rows));
        Assign(_memoryValues, Engine.TensorMultiply(_memoryValues, rows));
        Assign(_usage, Engine.TensorMultiply(_usage, keep));
        Assign(_previousReadWeights, Engine.TensorMultiply(_previousReadWeights, keep));
    }

    /// <summary>
    /// <c>softmax_i(cosine(key, M(i)))</c> per key row of <paramref name="keys"/> against <paramref name="memoryKeys"/>,
    /// <c>[keys, Size]</c>. A pair whose norm product is at most 1e-20 (norms' product at most 1e-10) has similarity
    /// 0, and similarities are clamped to [-1, 1]. The undefined pairs are kept off the square root and the division,
    /// so their gradient is 0 rather than NaN.
    /// </summary>
    internal Tensor<T> ContentWeightRows(Tensor<T> memoryKeys, Tensor<T> keys)
    {
        var dots = Engine.TensorMatMul(keys, Engine.TensorTranspose(memoryKeys));
        var keyNorms = Engine.ReduceSum(Engine.TensorMultiply(keys, keys), new[] { 1 }, true);
        var slotNorms = Engine.ReduceSum(Engine.TensorMultiply(memoryKeys, memoryKeys), new[] { 1 }, true);
        var normProducts = Engine.TensorMatMul(keyNorms, Engine.TensorTranspose(slotNorms));

        Tensor<T> defined, guard;
        using (new NoGradScope<T>())
        {
            defined = Engine.TensorGreaterThan(normProducts, _numOps.FromDouble(1e-20));
            guard = Engine.TensorAddScalar(Engine.TensorNegate(defined), _numOps.One);
        }

        var denominator = Engine.TensorSqrt(Engine.TensorAdd(normProducts, guard));
        var similarity = Engine.TensorClamp(Engine.TensorDivide(dots, denominator), _numOps.FromDouble(-1.0), _numOps.One);
        return Engine.TensorSoftmax(Engine.TensorMultiply(similarity, defined), 1);
    }

    /// <summary>
    /// 1 for the <paramref name="count"/> least-used slots of <paramref name="usage"/> (ascending usage, the lower
    /// index first among equal usages), else 0.
    /// </summary>
    private Tensor<T> LeastUsed(Tensor<T> usage, int count)
        => Engine.TensorLessThan(Ranks(usage, descending: false), _numOps.FromDouble(count));

    /// <summary>
    /// Each slot's position when the values are sorted (ascending, or descending), ties broken by the lower index:
    /// the number of values strictly before it plus the equal values at lower indices.
    /// </summary>
    private Tensor<T> Ranks(Tensor<T> values, bool descending)
    {
        var square = new[] { Size, Size };

        // others[i, j] = values[j], own[i, j] = values[i].
        var others = Engine.TensorBroadcastTo(Engine.Reshape(values, new[] { 1, Size }), square);
        var own = Engine.TensorBroadcastTo(Engine.Reshape(values, new[] { Size, 1 }), square);
        var before = descending ? Engine.TensorGreaterThan(others, own) : Engine.TensorLessThan(others, own);

        var ones = new Tensor<T>(square);
        ones.Fill(_numOps.One);
        var lowerIndex = Engine.TensorTril(ones, -1);
        var tiedBefore = Engine.TensorMultiply(Engine.TensorEquals(others, own), lowerIndex);
        return Engine.ReduceSum(Engine.TensorAdd(before, tiedBefore), new[] { 1 }, false);
    }

    /// <summary>A vector's first entries as a tensor of exactly <paramref name="width"/>, zero-padded.</summary>
    private static Tensor<T> Fit(Vector<T> vector, int width)
    {
        var tensor = Tensor<T>.FromVector(vector);
        if (vector.Length == width) return tensor;
        if (vector.Length > width) return Engine.TensorSlice(tensor, new[] { 0 }, new[] { width });
        return Engine.TensorConcatenate(new[] { tensor, new Tensor<T>(new[] { width - vector.Length }) }, 0);
    }
}

/// <summary>
/// Memory statistics tracking for MANN.
/// </summary>
public class MANNMemoryStatistics
{
    /// <summary>Gets the number of reads.</summary>
    public int NumReads { get; private set; }

    /// <summary>Gets the number of writes.</summary>
    public int NumWrites { get; private set; }

    /// <summary>Records a read operation.</summary>
    public void RecordRead() => NumReads++;

    /// <summary>Records a write operation.</summary>
    public void RecordWrite() => NumWrites++;
}

/// <summary>
/// MANN model for inference with external memory.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class MANNModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _controller;
    private readonly ExternalMemory<T> _memory;
    private readonly MANNOptions<T, TInput, TOutput> _options;
    private Tensor<T>? _outputWeights;
    private Tensor<T>? _outputBias;

    /// <summary>
    /// Initializes a new instance of the MANNModel. Its output layer is created Glorot-uniform (seeded by
    /// <c>options.RandomSeed</c>) on the first prediction; a model adapted by <see cref="MANNAlgorithm{T, TInput, TOutput}"/>
    /// carries the learned one instead.
    /// </summary>
    public MANNModel(
        IFullModel<T, TInput, TOutput> controller,
        ExternalMemory<T> memory,
        MANNOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps)
    {
        Guard.NotNull(controller);
        _controller = controller;
        Guard.NotNull(memory);
        _memory = memory;
        Guard.NotNull(options);
        _options = options;
        Guard.NotNull(numOps);
    }

    /// <summary>Initializes a MANNModel with a learned output layer, <c>[inputs, classes]</c> weights and <c>[1, classes]</c> bias.</summary>
    internal MANNModel(
        IFullModel<T, TInput, TOutput> controller,
        ExternalMemory<T> memory,
        MANNOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps,
        Tensor<T> outputWeights,
        Tensor<T> outputBias)
        : this(controller, memory, options, numOps)
    {
        Guard.NotNull(outputWeights);
        Guard.NotNull(outputBias);
        _outputWeights = outputWeights;
        _outputBias = outputBias;
    }

    /// <summary>Gets the model metadata.</summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Makes predictions using memory-augmented classification: one row per example, the output layer applied to
    /// the example's own controller features and a read of the memory with them as the key - softmax
    /// probabilities, or logits when <c>UseOutputSoftmax</c> is off. Prediction does not change the memory.
    /// </summary>
    public TOutput Predict(TInput input)
    {
        var engine = AiDotNetEngine.Current;
        var features = ClassifierOutputs<T>.AsRows(_controller.Predict(input));
        var reads = _memory.ReadRows(MannRows<T>.Keys(features, _options.MemoryKeySize));
        var combined = engine.TensorConcatenate(new[] { features, reads }, 1);
        var (weights, bias) = Layer(combined.Shape[1]);
        var logits = MannRows<T>.Logits(combined, weights, bias);
        var scores = _options.UseOutputSoftmax && logits.Shape[0] > 0 ? engine.TensorSoftmax(logits, 1) : logits;
        return ClassifierOutputs<T>.ToOutput<TOutput>(scores);
    }

    private (Tensor<T> Weights, Tensor<T> Bias) Layer(int inputs)
    {
        if (_outputWeights is null || _outputBias is null)
        {
            var (weights, bias) = MannRows<T>.GlorotLayer(inputs, _options.NumClasses, _options.RandomSeed);
            _outputWeights = weights;
            _outputBias = bias;
        }

        if (_outputWeights.Shape[0] != inputs)
        {
            throw new InvalidOperationException(
                $"The output layer takes {_outputWeights.Shape[0]} inputs, but the controller and memory give {inputs}.");
        }

        return (_outputWeights, _outputBias);
    }

    /// <summary>
    /// Training is not supported for adapted models.
    /// </summary>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the MANN algorithm to train.");
    }

    /// <summary>
    /// Parameter updates are not supported for adapted models.
    /// </summary>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("MANN parameters are updated during training.");
    }

    /// <summary>
    /// Gets the controller's parameters followed by the output layer's weights and bias (once the layer exists).
    /// </summary>
    public Vector<T> GetParameters()
    {
        var controller = InterfaceGuard.Parameterizable(_controller).GetParameters();
        if (_outputWeights is null || _outputBias is null) return controller;
        return AiDotNetEngine.Current.TensorConcatenate(new[]
        {
            Tensor<T>.FromVector(controller),
            _outputWeights.Reshape(_outputWeights.Length),
            _outputBias.Reshape(_outputBias.Length),
        }, 0).ToVector();
    }

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
