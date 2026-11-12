using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
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

        // Initialize the LRUA external memory (Santoro et al. 2016): usage decay gamma, write gate sigmoid(alpha),
        // and one least-used slot per read head.
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
    /// MANN training involves writing support examples to memory and using memory
    /// reads to classify query examples:
    /// </para>
    /// <para>
    /// <b>For each task in the batch:</b>
    /// 1. Optionally clear recent memories (based on configuration)
    /// 2. Write support set examples to external memory
    /// 3. Process query set using memory-augmented predictions
    /// 4. Compute cross-entropy loss
    /// 5. Update controller network
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var losses = new List<T>();
        Vector<T>? accumulatedGradients = null;

        foreach (var task in taskBatch.Tasks)
        {
            // Clear recent memory for new task (if configured)
            if (_mannOptions.ClearMemoryBetweenTasks)
            {
                _memory.ClearRecent(_mannOptions.MemoryRetentionRatio);
            }

            // Compute episode loss and gradients
            var (episodeLoss, episodeGradients) = TrainEpisode(task);
            losses.Add(episodeLoss);

            // Accumulate gradients
            accumulatedGradients = accumulatedGradients == null
                ? episodeGradients
                : Engine.Add(accumulatedGradients, episodeGradients);
        }

        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        if (accumulatedGradients != null)
        {
            // Average gradients
            accumulatedGradients = Engine.Divide(accumulatedGradients, batchSizeT);

            // Apply gradient clipping if configured
            if (_mannOptions.GradientClipThreshold.HasValue && _mannOptions.GradientClipThreshold.Value > 0)
            {
                accumulatedGradients = ClipGradients(accumulatedGradients, _mannOptions.GradientClipThreshold.Value);
            }

            // Update controller parameters
            var currentParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
            var updatedParams = ApplyOuterGradients(currentParams, accumulatedGradients, _mannOptions.OuterLearningRate);
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(updatedParams);
        }

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
    /// MANN adaptation is fast - it just writes support examples to memory.
    /// The returned model can then classify new examples by reading from memory.
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
        WriteSupportSetToMemory(task.SupportInput, task.SupportOutput, adaptedMemory);

        // Return adapted model
        return new MANNModel<T, TInput, TOutput>(
            MetaModel,
            adaptedMemory,
            _mannOptions,
            NumOps);
    }

    /// <summary>
    /// Trains the controller on a single episode.
    /// </summary>
    private (T loss, Vector<T> gradients) TrainEpisode(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Step 1: Write support set to memory
        WriteSupportSetToMemory(task.SupportInput, task.SupportOutput, _memory);

        // Step 2: Process query set using memory
        var queryPredictions = ProcessQuerySetWithMemory(task.QueryInput);

        // Step 3: Compute loss
        T episodeLoss = ComputeLoss(queryPredictions, task.QueryOutput);

        // Step 4: Add memory regularization
        if (_mannOptions.MemoryRegularization > 0.0)
        {
            T memoryPenalty = _memory.ComputeUsagePenalty();
            T regWeight = NumOps.FromDouble(_mannOptions.MemoryRegularization);
            episodeLoss = NumOps.Add(episodeLoss, NumOps.Multiply(memoryPenalty, regWeight));
        }

        // Step 5: Compute gradients
        var gradients = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);

        return (episodeLoss, gradients);
    }

    /// <summary>
    /// Writes the support examples to memory one time step at a time: each step writes the example's key and
    /// one-hot label through the LRUA write weights, then reads with the same key (Santoro et al. 2016).
    /// </summary>
    private void WriteSupportSetToMemory(TInput supportInputs, TOutput supportOutputs, ExternalMemory<T> memory)
    {
        int numSupport = GetBatchSize(supportInputs);
        var features = MannRows<T>.Features(MetaModel.Predict(supportInputs), numSupport);
        var keys = MannRows<T>.Keys(features, _mannOptions.MemoryKeySize);
        var values = MannRows<T>.OneHot(MannRows<T>.Labels(supportOutputs, numSupport), _mannOptions.MemoryValueSize);

        for (int i = 0; i < numSupport; i++)
        {
            memory.WriteAndRead(keys.GetSlice(i).ToVector(), values.GetSlice(i).ToVector());
            _memoryStats.RecordWrite();
        }
    }

    /// <summary>
    /// Processes the query set: each query reads the memory with its key (the read updates the LRUA usage), the
    /// controller features and the read content are concatenated and scored.
    /// </summary>
    private Tensor<T> ProcessQuerySetWithMemory(TInput queryInputs)
    {
        int numQueries = GetBatchSize(queryInputs);
        var features = MannRows<T>.Features(MetaModel.Predict(queryInputs), numQueries);
        var keys = MannRows<T>.Keys(features, _mannOptions.MemoryKeySize);

        var reads = new List<Tensor<T>>(numQueries);
        for (int q = 0; q < numQueries; q++)
        {
            reads.Add(Tensor<T>.FromVector(_memory.ReadAndUpdate(keys.GetSlice(q).ToVector()))
                .Reshape(1, _mannOptions.MemoryValueSize));
            _memoryStats.RecordRead();
        }

        var readRows = numQueries == 0
            ? new Tensor<T>(new[] { 0, _mannOptions.MemoryValueSize })
            : Engine.TensorConcatenate(reads.ToArray(), 0);
        return MannRows<T>.Predictions(Engine.TensorConcatenate(new[] { features, readRows }, 1),
            _mannOptions.NumClasses, _mannOptions.UseOutputSoftmax);
    }

    /// <summary>
    /// Computes the cross-entropy of the true classes, <c>-mean log(p + 1e-8)</c>; a label outside the class range
    /// adds nothing but still counts toward the mean.
    /// </summary>
    private T ComputeLoss(Tensor<T> predictions, TOutput trueLabels)
    {
        int numExamples = predictions.Shape[0], classes = predictions.Shape[1];
        var target = MannRows<T>.OneHot(MannRows<T>.Labels(trueLabels, numExamples), classes);
        var logProbabilities = Engine.TensorLog(Engine.TensorAddScalar(predictions, NumOps.FromDouble(1e-8)));
        var picked = Engine.TensorWhere(target, logProbabilities, new Tensor<T>(new[] { numExamples, classes }));
        return NumOps.Divide(NumOps.Negate(Engine.TensorSum(picked)), NumOps.FromDouble(numExamples));
    }

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
}

/// <summary>
/// The row arithmetic MANN's learner and adapted model share, as whole-tensor engine ops.
/// </summary>
internal static class MannRows<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The controller's features for <paramref name="count"/> examples, <c>[count, width]</c>. A vector or rank-1
    /// output is one feature vector shared by every example; a matrix or higher-rank output gives example
    /// <c>i</c> row <c>i mod rows</c>.
    /// </summary>
    internal static Tensor<T> Features(object? output, int count)
    {
        switch (output)
        {
            case Vector<T> vector:
                return Engine.TensorTile(Tensor<T>.FromVector(vector).Reshape(1, vector.Length), new[] { count, 1 });
            case Tensor<T> tensor when tensor.Shape.Length == 1:
                return Engine.TensorTile(tensor.Reshape(1, tensor.Length), new[] { count, 1 });
            case Matrix<T> matrix:
                return CycleRows(Tensor<T>.FromMatrix(matrix), count);
            case Tensor<T> tensor when tensor.Shape.Length >= 2:
                return CycleRows(ClassifierOutputs<T>.AsRows(tensor), count);
            default:
                return new Tensor<T>(new[] { count, 1 });
        }
    }

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
    /// Class scores from combined rows: class <c>c</c> sums the <c>c</c>-th block of <c>width / classes</c>
    /// entries, then a row softmax when requested.
    /// </summary>
    internal static Tensor<T> Predictions(Tensor<T> combined, int classes, bool softmax)
    {
        int count = combined.Shape[0], width = combined.Shape[1], stride = width / classes;
        Tensor<T> scores = stride == 0
            ? new Tensor<T>(new[] { count, classes })
            : Engine.ReduceSum(
                Engine.Reshape(Engine.TensorSlice(combined, new[] { 0, 0 }, new[] { count, classes * stride }),
                    new[] { count, classes, stride }),
                new[] { 2 }, false);
        return softmax && count > 0 ? Engine.TensorSoftmax(scores, 1) : scores;
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
/// Ties in usage go to the lower slot index, so the least-used slot and the n least used are deterministic.
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
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Memory size must be positive.");
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

    /// <summary>Gets the write gate sigmoid(alpha).</summary>
    public double WriteGate { get; }

    /// <summary>Gets the number of reads n, the number of slots the least-used weights mark.</summary>
    public int NumReads { get; }

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
        var keyRow = Fit(key, KeySize);
        var valueRow = Fit(value, ValueSize);

        // w^w_t from the previous step's read and least-used weights.
        T gate = _numOps.FromDouble(WriteGate);
        var writeWeights = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(_previousReadWeights, gate),
            Engine.TensorMultiplyScalar(LeastUsed(NumReads), _numOps.FromDouble(1.0 - WriteGate)));

        // The least-used slot is zeroed before the write.
        var keep = Engine.Reshape(Engine.TensorAddScalar(Engine.TensorNegate(LeastUsed(1)), _numOps.One), new[] { Size, 1 });
        var column = Engine.Reshape(writeWeights, new[] { Size, 1 });
        Assign(_memoryKeys, Engine.TensorAdd(Engine.TensorMultiply(_memoryKeys, keep),
            Engine.TensorMatMul(column, Engine.Reshape(keyRow, new[] { 1, KeySize }))));
        Assign(_memoryValues, Engine.TensorAdd(Engine.TensorMultiply(_memoryValues, keep),
            Engine.TensorMatMul(column, Engine.Reshape(valueRow, new[] { 1, ValueSize }))));

        var readWeights = ContentWeightsOf(keyRow);
        Assign(_usage, Engine.TensorAdd(Engine.TensorAdd(DecayedUsage(), readWeights), writeWeights));
        Assign(_previousReadWeights, readWeights);
        return ReadWith(readWeights);
    }

    /// <summary>
    /// One LRUA read without a write: content weights for <paramref name="key"/>, which update the usage and become
    /// the next write's read weights. Returns the read content.
    /// </summary>
    public Vector<T> ReadAndUpdate(Vector<T> key)
    {
        var readWeights = ContentWeightsOf(Fit(key, KeySize));
        Assign(_usage, Engine.TensorAdd(DecayedUsage(), readWeights));
        Assign(_previousReadWeights, readWeights);
        return ReadWith(readWeights);
    }

    /// <summary>The read weights <c>softmax_i(cosine(key, M(i)))</c> for a key, without changing the memory.</summary>
    public Vector<T> ContentWeights(Vector<T> key) => ContentWeightsOf(Fit(key, KeySize)).ToVector();

    /// <summary>The least-used weights <c>w^lu</c>: 1 for the <see cref="NumReads"/> least-used slots, else 0.</summary>
    public Vector<T> LeastUsedWeights() => LeastUsed(NumReads).ToVector();

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
    {
        return Engine.TensorMatMul(ContentWeightRows(keys), _memoryValues);
    }

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
        => Engine.TensorArgMax(LeastUsed(1), 0).GetFlat(0);

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
        Assign(cloned._memoryKeys, _memoryKeys);
        Assign(cloned._memoryValues, _memoryValues);
        Assign(cloned._usage, _usage);
        Assign(cloned._previousReadWeights, _previousReadWeights);
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

    /// <summary>Zeroes every slot where <paramref name="keep"/> is 0: its key, value and usage.</summary>
    private void ClearWhere(Tensor<T> keep)
    {
        var rows = Engine.Reshape(keep, new[] { Size, 1 });
        Assign(_memoryKeys, Engine.TensorMultiply(_memoryKeys, rows));
        Assign(_memoryValues, Engine.TensorMultiply(_memoryValues, rows));
        Assign(_usage, Engine.TensorMultiply(_usage, keep));
    }

    private Tensor<T> DecayedUsage() => Engine.TensorMultiplyScalar(_usage, _numOps.FromDouble(UsageDecay));

    private Tensor<T> ReadWith(Tensor<T> readWeights)
        => Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(readWeights, new[] { 1, Size }), _memoryValues),
            new[] { ValueSize }).ToVector();

    private Tensor<T> ContentWeightsOf(Tensor<T> key)
        => Engine.Reshape(ContentWeightRows(Engine.Reshape(key, new[] { 1, KeySize })), new[] { Size });

    /// <summary>
    /// <c>softmax_i(cosine(key, M(i)))</c> per key row, <c>[keys, Size]</c>. A pair whose norm product is at most
    /// 1e-10 has similarity 0, and similarities are clamped to [-1, 1].
    /// </summary>
    private Tensor<T> ContentWeightRows(Tensor<T> keys)
    {
        int count = keys.Shape[0];
        var dots = Engine.TensorMatMul(keys, Engine.TensorTranspose(_memoryKeys));
        var keyNorms = Engine.ReduceSum(Engine.TensorMultiply(keys, keys), new[] { 1 }, true);
        var slotNorms = Engine.ReduceSum(Engine.TensorMultiply(_memoryKeys, _memoryKeys), new[] { 1 }, true);
        var denominator = Engine.TensorSqrt(Engine.TensorMatMul(keyNorms, Engine.TensorTranspose(slotNorms)));
        var similarity = Engine.TensorClamp(Engine.TensorDivide(dots, denominator), _numOps.FromDouble(-1.0), _numOps.One);
        var defined = Engine.TensorGreaterThan(denominator, _numOps.FromDouble(1e-10));
        var cosine = Engine.TensorWhere(defined, similarity, new Tensor<T>(new[] { count, Size }));
        return Engine.TensorSoftmax(cosine, 1);
    }

    /// <summary>
    /// 1 for the <paramref name="count"/> least-used slots (ascending usage, the lower index first among equal
    /// usages), else 0.
    /// </summary>
    private Tensor<T> LeastUsed(int count)
        => Engine.TensorLessThan(Ranks(_usage, descending: false), _numOps.FromDouble(count));

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

    /// <summary>
    /// Initializes a new instance of the MANNModel.
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

    /// <summary>Gets the model metadata.</summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Makes predictions using memory-augmented classification: one score row per example, from the example's
    /// own controller features and a read of the memory with them as the key. Prediction does not change the
    /// memory.
    /// </summary>
    public TOutput Predict(TInput input)
    {
        var features = ClassifierOutputs<T>.AsRows(_controller.Predict(input));
        var reads = _memory.ReadRows(MannRows<T>.Keys(features, _options.MemoryKeySize));
        var combined = AiDotNetEngine.Current.TensorConcatenate(new[] { features, reads }, 1);
        var scores = MannRows<T>.Predictions(combined, _options.NumClasses, _options.UseOutputSoftmax);
        return ClassifierOutputs<T>.ToOutput<TOutput>(scores);
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
    /// Gets controller parameters.
    /// </summary>
    public Vector<T> GetParameters() => InterfaceGuard.Parameterizable(_controller).GetParameters();

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
