using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

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
public class MANNAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
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
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            null) // MANN doesn't use inner optimizer (memory-based adaptation)
    {
        _mannOptions = options;

        // Validate configuration
        if (!_mannOptions.IsValid())
        {
            throw new ArgumentException("MANN configuration is invalid. Check all parameters.", nameof(options));
        }

        // Initialize external memory
        _memory = new ExternalMemory<T>(
            options.MemorySize,
            options.MemoryKeySize,
            options.MemoryValueSize,
            NumOps);

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

        T totalLoss = NumOps.Zero;
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
            totalLoss = NumOps.Add(totalLoss, episodeLoss);

            // Accumulate gradients
            if (accumulatedGradients == null)
            {
                accumulatedGradients = episodeGradients;
            }
            else
            {
                for (int i = 0; i < accumulatedGradients.Length; i++)
                {
                    accumulatedGradients[i] = NumOps.Add(accumulatedGradients[i], episodeGradients[i]);
                }
            }
        }

        if (accumulatedGradients != null)
        {
            // Average gradients
            T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
            for (int i = 0; i < accumulatedGradients.Length; i++)
            {
                accumulatedGradients[i] = NumOps.Divide(accumulatedGradients[i], batchSizeT);
            }

            // Apply gradient clipping if configured
            if (_mannOptions.GradientClipThreshold.HasValue && _mannOptions.GradientClipThreshold.Value > 0)
            {
                accumulatedGradients = ClipGradients(accumulatedGradients, _mannOptions.GradientClipThreshold.Value);
            }

            // Update controller parameters
            var currentParams = MetaModel.GetParameters();
            var updatedParams = ApplyGradients(currentParams, accumulatedGradients, _mannOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        // Apply memory consolidation if configured
        if (_mannOptions.UseMemoryConsolidation)
        {
            ConsolidateMemory();
        }

        // Return average loss
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <summary>
    /// Adapts to a new task by writing support examples to memory.
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
    /// Writes support set examples to external memory.
    /// </summary>
    private void WriteSupportSetToMemory(TInput supportInputs, TOutput supportOutputs, ExternalMemory<T> memory)
    {
        int numSupport = GetBatchSize(supportInputs);

        for (int i = 0; i < numSupport; i++)
        {
            // Generate memory key from input
            var memoryKey = GenerateMemoryKey(supportInputs, i);

            // Generate memory value from label
            var memoryValue = GenerateMemoryValue(supportOutputs, i);

            // Find where to write (least recently used)
            int writeLocation = memory.FindLeastRecentlyUsedSlot();

            // Write to memory
            memory.Write(writeLocation, memoryKey, memoryValue);

            // Update statistics
            _memoryStats.RecordWrite();
        }
    }

    /// <summary>
    /// Processes query set using memory-augmented computation.
    /// </summary>
    private Matrix<T> ProcessQuerySetWithMemory(TInput queryInputs)
    {
        int numQueries = GetBatchSize(queryInputs);
        var predictions = new List<Vector<T>>();

        for (int q = 0; q < numQueries; q++)
        {
            // Generate memory key for reading
            var readKey = GenerateMemoryKey(queryInputs, q);

            // Read from memory using attention
            var readContent = ReadFromMemory(readKey);

            // Process through controller
            var controllerOutput = ProcessWithController(queryInputs, q);

            // Combine controller output with memory content
            var combined = CombineWithMemory(controllerOutput, readContent);

            // Generate prediction
            var prediction = GeneratePrediction(combined);
            predictions.Add(prediction);
        }

        // Convert to matrix
        return StackPredictions(predictions);
    }

    /// <summary>
    /// Reads relevant content from external memory using attention.
    /// </summary>
    private Vector<T> ReadFromMemory(Vector<T> readKey)
    {
        // Compute attention weights over memory
        var attentionWeights = ComputeAttentionWeights(readKey);

        // Read weighted sum of memory values
        var readContent = _memory.Read(attentionWeights);

        // Update statistics
        _memoryStats.RecordRead();

        return readContent;
    }

    /// <summary>
    /// Computes attention weights using cosine similarity.
    /// </summary>
    private Vector<T> ComputeAttentionWeights(Vector<T> queryKey)
    {
        var weights = new Vector<T>(_mannOptions.MemorySize);

        // Compute cosine similarity with each memory key
        for (int i = 0; i < _mannOptions.MemorySize; i++)
        {
            var memoryKey = _memory.GetKey(i);
            T similarity = ComputeCosineSimilarity(queryKey, memoryKey);
            weights[i] = similarity;
        }

        // Apply softmax
        return ApplySoftmax(weights);
    }

    /// <summary>
    /// Computes cosine similarity between two vectors.
    /// </summary>
    private T ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        T dotProduct = NumOps.Zero;
        T normASq = NumOps.Zero;
        T normBSq = NumOps.Zero;

        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(a[i], b[i]));
            normASq = NumOps.Add(normASq, NumOps.Multiply(a[i], a[i]));
            normBSq = NumOps.Add(normBSq, NumOps.Multiply(b[i], b[i]));
        }

        T normA = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(normASq)));
        T normB = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(normBSq)));

        T denominator = NumOps.Multiply(normA, normB);
        if (NumOps.ToDouble(denominator) < 1e-8)
        {
            return NumOps.Zero;
        }

        return NumOps.Divide(dotProduct, denominator);
    }

    /// <summary>
    /// Applies softmax to a vector.
    /// </summary>
    private Vector<T> ApplySoftmax(Vector<T> values)
    {
        var result = new Vector<T>(values.Length);

        // Find max for numerical stability
        T maxVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.ToDouble(values[i]) > NumOps.ToDouble(maxVal))
            {
                maxVal = values[i];
            }
        }

        // Compute exp values and sum
        T sumExp = NumOps.Zero;
        var expValues = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            T shifted = NumOps.Subtract(values[i], maxVal);
            expValues[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(shifted)));
            sumExp = NumOps.Add(sumExp, expValues[i]);
        }

        // Normalize
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = NumOps.Divide(expValues[i], sumExp);
        }

        return result;
    }

    /// <summary>
    /// Generates a memory key from input at specified index.
    /// </summary>
    private Vector<T> GenerateMemoryKey(TInput input, int index)
    {
        // Process input through controller to get features
        var features = ProcessWithController(input, index);

        // Project to key dimension
        var key = new Vector<T>(_mannOptions.MemoryKeySize);
        int copyLen = Math.Min(features.Length, _mannOptions.MemoryKeySize);
        for (int i = 0; i < copyLen; i++)
        {
            key[i] = features[i];
        }

        return key;
    }

    /// <summary>
    /// Generates a memory value from output label at specified index.
    /// </summary>
    private Vector<T> GenerateMemoryValue(TOutput output, int index)
    {
        int classLabel = GetClassLabel(output, index);

        // Create one-hot encoding
        var value = new Vector<T>(_mannOptions.MemoryValueSize);
        if (classLabel >= 0 && classLabel < _mannOptions.MemoryValueSize)
        {
            value[classLabel] = NumOps.One;
        }

        return value;
    }

    /// <summary>
    /// Processes input through the controller network.
    /// </summary>
    private Vector<T> ProcessWithController(TInput input, int index)
    {
        // Get prediction from model
        var prediction = MetaModel.Predict(input);

        // Extract features for the specific index
        return ExtractFeatureVector(prediction, index);
    }

    /// <summary>
    /// Combines controller output with memory content.
    /// </summary>
    private Vector<T> CombineWithMemory(Vector<T> controllerOutput, Vector<T> memoryContent)
    {
        // Concatenate controller output and memory content
        var combined = new Vector<T>(controllerOutput.Length + memoryContent.Length);

        for (int i = 0; i < controllerOutput.Length; i++)
        {
            combined[i] = controllerOutput[i];
        }

        for (int i = 0; i < memoryContent.Length; i++)
        {
            combined[controllerOutput.Length + i] = memoryContent[i];
        }

        return combined;
    }

    /// <summary>
    /// Generates final prediction from combined features.
    /// </summary>
    private Vector<T> GeneratePrediction(Vector<T> combinedFeatures)
    {
        // Simple linear projection to num classes
        var prediction = new Vector<T>(_mannOptions.NumClasses);

        int stride = combinedFeatures.Length / _mannOptions.NumClasses;
        for (int c = 0; c < _mannOptions.NumClasses; c++)
        {
            T sum = NumOps.Zero;
            for (int i = c * stride; i < Math.Min((c + 1) * stride, combinedFeatures.Length); i++)
            {
                sum = NumOps.Add(sum, combinedFeatures[i]);
            }
            prediction[c] = sum;
        }

        // Apply softmax if configured
        if (_mannOptions.UseOutputSoftmax)
        {
            prediction = ApplySoftmax(prediction);
        }

        return prediction;
    }

    /// <summary>
    /// Computes cross-entropy loss.
    /// </summary>
    private T ComputeLoss(Matrix<T> predictions, TOutput trueLabels)
    {
        T totalLoss = NumOps.Zero;
        int numExamples = predictions.Rows;

        for (int i = 0; i < numExamples; i++)
        {
            int trueClass = GetClassLabel(trueLabels, i);
            if (trueClass >= 0 && trueClass < predictions.Columns)
            {
                T predictedProb = predictions[i, trueClass];
                predictedProb = NumOps.Add(predictedProb, NumOps.FromDouble(1e-8));
                T logProb = NumOps.FromDouble(Math.Log(NumOps.ToDouble(predictedProb)));
                totalLoss = NumOps.Subtract(totalLoss, logProb);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(numExamples));
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
    /// Stacks prediction vectors into a matrix.
    /// </summary>
    private Matrix<T> StackPredictions(List<Vector<T>> predictions)
    {
        if (predictions.Count == 0)
            return new Matrix<T>(0, 0);

        int numRows = predictions.Count;
        int numCols = predictions[0].Length;
        var matrix = new Matrix<T>(numRows, numCols);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                matrix[i, j] = predictions[i][j];
            }
        }

        return matrix;
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

    /// <summary>
    /// Extracts feature vector from output at specified index.
    /// </summary>
    private Vector<T> ExtractFeatureVector(TOutput output, int index)
    {
        if (output is Vector<T> vector)
            return vector;

        if (output is Matrix<T> matrix)
        {
            var row = new Vector<T>(matrix.Columns);
            for (int j = 0; j < matrix.Columns; j++)
            {
                row[j] = matrix[index % matrix.Rows, j];
            }
            return row;
        }

        if (output is Tensor<T> tensor)
        {
            if (tensor.Shape.Length == 1)
            {
                var result = new Vector<T>(tensor.Shape[0]);
                for (int j = 0; j < tensor.Shape[0]; j++)
                {
                    result[j] = tensor[new int[] { j }];
                }
                return result;
            }
            else if (tensor.Shape.Length >= 2)
            {
                int cols = tensor.Shape[1];
                var result = new Vector<T>(cols);
                for (int j = 0; j < cols; j++)
                {
                    result[j] = tensor[new int[] { index % tensor.Shape[0], j }];
                }
                return result;
            }
        }

        return new Vector<T>(1);
    }

    /// <summary>
    /// Gets class label from output at specified index.
    /// </summary>
    private int GetClassLabel(TOutput output, int index)
    {
        if (output is Vector<T> vector)
        {
            if (index < vector.Length)
                return (int)NumOps.ToDouble(vector[index]);
            return 0;
        }

        if (output is Matrix<T> matrix)
        {
            if (matrix.Columns == 1)
                return (int)NumOps.ToDouble(matrix[index % matrix.Rows, 0]);

            // One-hot: find argmax
            int maxIdx = 0;
            T maxVal = matrix[index % matrix.Rows, 0];
            for (int c = 1; c < matrix.Columns; c++)
            {
                if (NumOps.ToDouble(matrix[index % matrix.Rows, c]) > NumOps.ToDouble(maxVal))
                {
                    maxVal = matrix[index % matrix.Rows, c];
                    maxIdx = c;
                }
            }
            return maxIdx;
        }

        if (output is Tensor<T> tensor)
        {
            if (tensor.Shape.Length >= 1 && index < tensor.Shape[0])
            {
                return (int)NumOps.ToDouble(tensor[new int[] { index }]);
            }
        }

        return 0;
    }
}

/// <summary>
/// External memory matrix for MANN.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Stores key-value pairs that can be read and written by the MANN controller.
/// Uses attention-based reading and LRU-based writing.
/// </para>
/// </remarks>
public class ExternalMemory<T>
{
    private readonly Matrix<T> _memoryKeys;
    private readonly Matrix<T> _memoryValues;
    private readonly Vector<T> _usageCounts;
    private readonly double[] _accessTimes;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the ExternalMemory.
    /// </summary>
    public ExternalMemory(int size, int keySize, int valueSize, INumericOperations<T> numOps)
    {
        Size = size;
        KeySize = keySize;
        ValueSize = valueSize;
        _numOps = numOps;

        _memoryKeys = new Matrix<T>(size, keySize);
        _memoryValues = new Matrix<T>(size, valueSize);
        _usageCounts = new Vector<T>(size);
        _accessTimes = new double[size];

        InitializeMemory();
    }

    /// <summary>Gets the memory size.</summary>
    public int Size { get; }

    /// <summary>Gets the key dimension.</summary>
    public int KeySize { get; }

    /// <summary>Gets the value dimension.</summary>
    public int ValueSize { get; }

    /// <summary>
    /// Writes to memory at specified location.
    /// </summary>
    public void Write(int location, Vector<T> key, Vector<T> value)
    {
        if (location < 0 || location >= Size)
            return;

        for (int i = 0; i < KeySize && i < key.Length; i++)
        {
            _memoryKeys[location, i] = key[i];
        }

        for (int i = 0; i < ValueSize && i < value.Length; i++)
        {
            _memoryValues[location, i] = value[i];
        }

        _usageCounts[location] = _numOps.Add(_usageCounts[location], _numOps.One);
        _accessTimes[location] = DateTime.UtcNow.Ticks;
    }

    /// <summary>
    /// Reads from memory using attention weights.
    /// </summary>
    public Vector<T> Read(Vector<T> attentionWeights)
    {
        var readContent = new Vector<T>(ValueSize);

        for (int v = 0; v < ValueSize; v++)
        {
            T weightedSum = _numOps.Zero;
            for (int m = 0; m < Size && m < attentionWeights.Length; m++)
            {
                T weightedValue = _numOps.Multiply(_memoryValues[m, v], attentionWeights[m]);
                weightedSum = _numOps.Add(weightedSum, weightedValue);
            }
            readContent[v] = weightedSum;
        }

        return readContent;
    }

    /// <summary>
    /// Gets memory key at specified index.
    /// </summary>
    public Vector<T> GetKey(int index)
    {
        var key = new Vector<T>(KeySize);
        if (index >= 0 && index < Size)
        {
            for (int i = 0; i < KeySize; i++)
            {
                key[i] = _memoryKeys[index, i];
            }
        }
        return key;
    }

    /// <summary>
    /// Finds the least recently used memory slot.
    /// </summary>
    public int FindLeastRecentlyUsedSlot()
    {
        int lruIndex = 0;
        double minTime = _accessTimes[0];

        for (int i = 1; i < Size; i++)
        {
            if (_accessTimes[i] < minTime)
            {
                minTime = _accessTimes[i];
                lruIndex = i;
            }
        }

        return lruIndex;
    }

    /// <summary>
    /// Finds rarely used memory slots.
    /// </summary>
    public List<int> FindRarelyUsedSlots(double threshold)
    {
        var rarelyUsed = new List<int>();
        for (int i = 0; i < Size; i++)
        {
            if (_numOps.ToDouble(_usageCounts[i]) < threshold)
            {
                rarelyUsed.Add(i);
            }
        }
        return rarelyUsed;
    }

    /// <summary>
    /// Clears a specific memory slot.
    /// </summary>
    public void ClearSlot(int location)
    {
        if (location < 0 || location >= Size)
            return;

        for (int i = 0; i < KeySize; i++)
        {
            _memoryKeys[location, i] = _numOps.Zero;
        }

        for (int i = 0; i < ValueSize; i++)
        {
            _memoryValues[location, i] = _numOps.Zero;
        }

        _usageCounts[location] = _numOps.Zero;
        _accessTimes[location] = 0;
    }

    /// <summary>
    /// Clears recent memories based on retention ratio.
    /// </summary>
    public void ClearRecent(double retentionRatio)
    {
        var sortedSlots = Enumerable.Range(0, Size)
            .OrderByDescending(i => _accessTimes[i])
            .ToList();

        int numToKeep = (int)(Size * retentionRatio);
        for (int i = numToKeep; i < sortedSlots.Count; i++)
        {
            ClearSlot(sortedSlots[i]);
        }
    }

    /// <summary>
    /// Computes memory usage penalty for regularization.
    /// </summary>
    public T ComputeUsagePenalty()
    {
        T totalUsage = _numOps.Zero;
        for (int i = 0; i < Size; i++)
        {
            totalUsage = _numOps.Add(totalUsage, _usageCounts[i]);
        }
        return _numOps.Divide(totalUsage, _numOps.FromDouble(Size));
    }

    /// <summary>
    /// Clones the external memory.
    /// </summary>
    public ExternalMemory<T> Clone()
    {
        var cloned = new ExternalMemory<T>(Size, KeySize, ValueSize, _numOps);

        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < KeySize; j++)
            {
                cloned._memoryKeys[i, j] = _memoryKeys[i, j];
            }

            for (int j = 0; j < ValueSize; j++)
            {
                cloned._memoryValues[i, j] = _memoryValues[i, j];
            }

            cloned._usageCounts[i] = _usageCounts[i];
            cloned._accessTimes[i] = _accessTimes[i];
        }

        return cloned;
    }

    private void InitializeMemory()
    {
        var random = RandomHelper.CreateSeededRandom(0);
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < KeySize; j++)
            {
                _memoryKeys[i, j] = _numOps.FromDouble((random.NextDouble() - 0.5) * 0.01);
            }

            for (int j = 0; j < ValueSize; j++)
            {
                _memoryValues[i, j] = _numOps.Zero;
            }

            _usageCounts[i] = _numOps.Zero;
            _accessTimes[i] = 0;
        }
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
    private readonly INumericOperations<T> _numOps;

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
        _numOps = numOps;
    }

    /// <summary>Gets the model metadata.</summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Makes predictions using memory-augmented classification.
    /// </summary>
    public TOutput Predict(TInput input)
    {
        // Process input through controller
        var controllerOutput = _controller.Predict(input);

        // Generate read key
        var readKey = GenerateReadKey(controllerOutput);

        // Read from memory
        var attentionWeights = ComputeAttentionWeights(readKey);
        var readContent = _memory.Read(attentionWeights);

        // Combine and generate prediction
        var combined = CombineWithMemory(ExtractVector(controllerOutput), readContent);
        var prediction = GeneratePrediction(combined);

        // Convert to output type
        return ConvertToOutput(prediction);
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
    public Vector<T> GetParameters() => _controller.GetParameters();

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public ModelMetadata<T> GetModelMetadata() => Metadata;

    private Vector<T> GenerateReadKey(TOutput output)
    {
        var features = ExtractVector(output);
        var key = new Vector<T>(_options.MemoryKeySize);
        int copyLen = Math.Min(features.Length, _options.MemoryKeySize);
        for (int i = 0; i < copyLen; i++)
        {
            key[i] = features[i];
        }
        return key;
    }

    private Vector<T> ExtractVector(TOutput output)
    {
        if (output is Vector<T> vector)
            return vector;

        if (output is Matrix<T> matrix)
        {
            var row = new Vector<T>(matrix.Columns);
            for (int j = 0; j < matrix.Columns; j++)
            {
                row[j] = matrix[0, j];
            }
            return row;
        }

        if (output is Tensor<T> tensor)
        {
            return tensor.ToVector();
        }

        return new Vector<T>(1);
    }

    private Vector<T> ComputeAttentionWeights(Vector<T> queryKey)
    {
        var weights = new Vector<T>(_options.MemorySize);

        for (int i = 0; i < _options.MemorySize; i++)
        {
            var memoryKey = _memory.GetKey(i);
            T similarity = ComputeCosineSimilarity(queryKey, memoryKey);
            weights[i] = similarity;
        }

        return ApplySoftmax(weights);
    }

    private T ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        T dotProduct = _numOps.Zero;
        T normASq = _numOps.Zero;
        T normBSq = _numOps.Zero;

        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(a[i], b[i]));
            normASq = _numOps.Add(normASq, _numOps.Multiply(a[i], a[i]));
            normBSq = _numOps.Add(normBSq, _numOps.Multiply(b[i], b[i]));
        }

        T normA = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(normASq)));
        T normB = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(normBSq)));

        T denominator = _numOps.Multiply(normA, normB);
        if (_numOps.ToDouble(denominator) < 1e-8)
        {
            return _numOps.Zero;
        }

        return _numOps.Divide(dotProduct, denominator);
    }

    private Vector<T> ApplySoftmax(Vector<T> values)
    {
        var result = new Vector<T>(values.Length);

        T maxVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (_numOps.ToDouble(values[i]) > _numOps.ToDouble(maxVal))
            {
                maxVal = values[i];
            }
        }

        T sumExp = _numOps.Zero;
        var expValues = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            T shifted = _numOps.Subtract(values[i], maxVal);
            expValues[i] = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(shifted)));
            sumExp = _numOps.Add(sumExp, expValues[i]);
        }

        for (int i = 0; i < values.Length; i++)
        {
            result[i] = _numOps.Divide(expValues[i], sumExp);
        }

        return result;
    }

    private Vector<T> CombineWithMemory(Vector<T> controllerOutput, Vector<T> memoryContent)
    {
        var combined = new Vector<T>(controllerOutput.Length + memoryContent.Length);

        for (int i = 0; i < controllerOutput.Length; i++)
        {
            combined[i] = controllerOutput[i];
        }

        for (int i = 0; i < memoryContent.Length; i++)
        {
            combined[controllerOutput.Length + i] = memoryContent[i];
        }

        return combined;
    }

    private Vector<T> GeneratePrediction(Vector<T> combinedFeatures)
    {
        var prediction = new Vector<T>(_options.NumClasses);

        int stride = combinedFeatures.Length / _options.NumClasses;
        for (int c = 0; c < _options.NumClasses; c++)
        {
            T sum = _numOps.Zero;
            for (int i = c * stride; i < Math.Min((c + 1) * stride, combinedFeatures.Length); i++)
            {
                sum = _numOps.Add(sum, combinedFeatures[i]);
            }
            prediction[c] = sum;
        }

        if (_options.UseOutputSoftmax)
        {
            prediction = ApplySoftmax(prediction);
        }

        return prediction;
    }

    private TOutput ConvertToOutput(Vector<T> prediction)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)prediction;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var tensor = new Tensor<T>(new int[] { prediction.Length });
            for (int i = 0; i < prediction.Length; i++)
            {
                tensor[new int[] { i }] = prediction[i];
            }
            return (TOutput)(object)tensor;
        }

        throw new NotSupportedException($"Output type {typeof(TOutput).Name} is not supported.");
    }
}
