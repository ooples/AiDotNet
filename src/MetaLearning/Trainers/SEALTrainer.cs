using AiDotNet.Data.Abstractions;

using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Config;
using AiDotNet.Models.Results;
using System.Diagnostics;

namespace AiDotNet.MetaLearning.Trainers;

/// <summary>
/// Implementation of SEAL (Self-supervised and Episodic Active Learning) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// SEAL (Hao et al., 2019) extends meta-learning by leveraging unlabeled data in the query set
/// through self-supervised learning and active learning. This enables better few-shot performance
/// by utilizing all available data, not just the labeled support set.
/// </para>
/// <para><b>Algorithm - SEAL Three-Phase Training:</b>
/// <code>
/// Initialize: θ (meta-parameters)
///
/// for iteration = 1 to N:
///     # Sample task with support (labeled) and query (treat as unlabeled)
///     task = SampleTask()
///     θ_task = Clone(θ)
///
///     # Phase 1: Self-Supervised Pre-Training on Query Set
///     for step = 1 to K_self:
///         # Create self-supervised task (e.g., rotation prediction)
///         X_rotated, Y_rotation = ApplyRotations(task.QuerySetX)
///         θ_task = θ_task - α * ∇L_self(θ_task, X_rotated, Y_rotation)
///
///     # Phase 2: Active Learning - Select Confident Examples
///     predictions = θ_task(task.QuerySetX)
///     confidence = max(predictions, axis=1)  # Max probability per example
///     top_k_indices = argsort(confidence)[-K_active:]
///     X_pseudo = task.QuerySetX[top_k_indices]
///     Y_pseudo = argmax(predictions[top_k_indices])  # Pseudo-labels
///
///     # Phase 3: Supervised Fine-Tuning on Support + Pseudo-Labeled
///     X_combined = Concatenate(task.SupportSetX, X_pseudo)
///     Y_combined = Concatenate(task.SupportSetY, Y_pseudo)
///
///     for step = 1 to K_supervised:
///         θ_task = θ_task - α * ∇L_supervised(θ_task, X_combined, Y_combined)
///
///     # Outer Loop: Reptile-style Meta-Update
///     θ = θ + ε * (θ_task - θ)
///
/// return θ
/// </code>
/// </para>
/// <para><b>Why SEAL Works:</b>
///
/// 1. **Self-Supervised Learning**: Helps model learn useful representations without labels
///    - Example: Rotation prediction teaches spatial understanding
///    - Better initialization than random parameters
///    - Particularly effective for image data
///
/// 2. **Active Learning**: Leverages model's confident predictions
///    - High-confidence predictions are usually correct
///    - Effectively increases training data size
///    - Reduces label requirements
///
/// 3. **Combined Approach**: Synergistic effects
///    - Self-supervision → better representations
///    - Better representations → more confident predictions
///    - Confident predictions → more training data
///    - More training data → better final model
/// </para>
/// <para><b>For Beginners:</b> SEAL is like learning from a textbook with some answers missing:
///
/// Traditional approach:
/// - You only study the 5 example problems with solutions (support set)
/// - Ignore the 50 practice problems without solutions (query set)
/// - Result: Limited practice
///
/// SEAL approach:
/// - Phase 1: Figure out patterns in ALL problems (self-supervised on query set)
/// - Phase 2: Solve the problems you're confident about (active learning)
/// - Phase 3: Study both example problems AND your confident solutions (supervised)
/// - Result: More practice, better performance!
///
/// The key insight: Even unlabeled data contains valuable information!
/// </para>
/// </remarks>
public class SEALTrainer<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ISelfSupervisedLoss<T> _selfSupervisedLoss;

    /// <summary>
    /// Initializes a new instance of the SEALTrainer.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for supervised training.</param>
    /// <param name="selfSupervisedLoss">Loss function for self-supervised pre-training.</param>
    /// <param name="dataLoader">Episodic data loader for sampling meta-learning tasks.</param>
    /// <param name="config">Configuration object containing all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a SEAL trainer for few-shot learning.
    ///
    /// SEAL improves on MAML/Reptile by using unlabeled query set data through:
    /// 1. Self-supervised pre-training (learn patterns without labels)
    /// 2. Active learning (select confident predictions)
    /// 3. Pseudo-labeling (use predictions as additional training data)
    ///
    /// <b>Parameters explained:</b>
    /// - <b>metaModel:</b> Your neural network to meta-train
    /// - <b>lossFunction:</b> For supervised learning (e.g., CrossEntropyLoss)
    /// - <b>selfSupervisedLoss:</b> For self-supervised pre-training (e.g., RotationPredictionLoss)
    /// - <b>dataLoader:</b> Provides N-way K-shot tasks
    /// - <b>config:</b> All hyperparameters (learning rates, steps, etc.)
    ///
    /// <b>Typical configuration:</b>
    /// - Self-supervised steps: 10-20 (learn representations)
    /// - Active learning K: 10-30 (number of pseudo-labels)
    /// - Supervised steps: 5-10 (fine-tune on combined data)
    /// </para>
    /// </remarks>
    public SEALTrainer(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        ISelfSupervisedLoss<T> selfSupervisedLoss,
        IEpisodicDataLoader<T, TInput, TOutput> dataLoader,
        SEALTrainerConfig<T>? config = null)
        : base(metaModel, lossFunction, dataLoader, config ?? new SEALTrainerConfig<T>())
    {
        _selfSupervisedLoss = selfSupervisedLoss ?? throw new ArgumentNullException(nameof(selfSupervisedLoss));
    }

    /// <inheritdoc/>
    public override MetaTrainingStepResult<T> MetaTrainStep(int batchSize)
    {
        if (batchSize < 1)
            throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

        var startTime = Stopwatch.StartNew();
        var config = (SEALTrainerConfig<T>)Configuration;

        // Save original meta-parameters
        Vector<T> originalParameters = MetaModel.GetParameters();

        var parameterUpdates = new List<Vector<T>>();
        var taskLosses = new List<T>();
        var taskAccuracies = new List<T>();

        // Process each task in the batch
        for (int taskIdx = 0; taskIdx < batchSize; taskIdx++)
        {
            // Sample task
            MetaLearningTask<T, TInput, TOutput> task = DataLoader.GetNextTask();

            // Reset model to original meta-parameters
            MetaModel.SetParameters(originalParameters.Clone());

            // ===================================================================
            // PHASE 1: Self-Supervised Pre-Training on Query Set
            // ===================================================================
            SelfSupervisedPreTraining(task, config.SelfSupervisedSteps);

            // ===================================================================
            // PHASE 2: Active Learning - Select Confident Examples
            // ===================================================================
            var (pseudoX, pseudoY) = ActiveLearningSelection(task, config.ActiveLearningK);

            // ===================================================================
            // PHASE 3: Supervised Fine-Tuning on Support + Pseudo-Labeled
            // ===================================================================
            SupervisedFineTuning(task, pseudoX, pseudoY, config.SupervisedSteps);

            // Get adapted parameters
            Vector<T> adaptedParameters = MetaModel.GetParameters();
            Vector<T> parameterUpdate = adaptedParameters.Subtract(originalParameters);
            parameterUpdates.Add(parameterUpdate);

            // Evaluate on query set
            T queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
            T queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

            taskLosses.Add(queryLoss);
            taskAccuracies.Add(queryAccuracy);
        }

        // Outer loop: Meta-update (Reptile-style)
        Vector<T> averageUpdate = AverageVectors(parameterUpdates);
        Vector<T> scaledUpdate = averageUpdate.Multiply(config.MetaLearningRate);
        Vector<T> newMetaParameters = originalParameters.Add(scaledUpdate);
        MetaModel.SetParameters(newMetaParameters);

        _currentIteration++;
        startTime.Stop();

        // Return metrics
        var lossVector = new Vector<T>(taskLosses.ToArray());
        var accuracyVector = new Vector<T>(taskAccuracies.ToArray());
        T meanLoss = StatisticsHelper<T>.CalculateMean(lossVector);
        T meanAccuracy = StatisticsHelper<T>.CalculateMean(accuracyVector);

        return new MetaTrainingStepResult<T>(
            metaLoss: meanLoss,
            taskLoss: meanLoss,
            accuracy: meanAccuracy,
            numTasks: batchSize,
            iteration: _currentIteration,
            timeMs: startTime.Elapsed.TotalMilliseconds);
    }

    /// <inheritdoc/>
    public override MetaAdaptationResult<T> AdaptAndEvaluate(MetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
            throw new ArgumentNullException(nameof(task));

        var startTime = Stopwatch.StartNew();
        var config = (SEALTrainerConfig<T>)Configuration;

        // Save original meta-parameters
        Vector<T> originalParameters = MetaModel.GetParameters();

        // Evaluate before adaptation (baseline)
        T initialQueryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
        var perStepLosses = new List<T> { initialQueryLoss };

        // Three-phase adaptation
        SelfSupervisedPreTraining(task, config.SelfSupervisedSteps);
        var (pseudoX, pseudoY) = ActiveLearningSelection(task, config.ActiveLearningK);
        SupervisedFineTuning(task, pseudoX, pseudoY, config.SupervisedSteps);

        // Evaluate after adaptation
        T queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
        T queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);
        T supportLoss = ComputeLoss(MetaModel, task.SupportSetX, task.SupportSetY);
        T supportAccuracy = ComputeAccuracy(MetaModel, task.SupportSetX, task.SupportSetY);

        startTime.Stop();

        // Restore original meta-parameters
        MetaModel.SetParameters(originalParameters);

        // Calculate additional metrics
        var additionalMetrics = new Dictionary<string, T>
        {
            ["initial_query_loss"] = initialQueryLoss,
            ["loss_improvement"] = NumOps.Subtract(initialQueryLoss, queryLoss),
            ["support_query_accuracy_gap"] = NumOps.Subtract(supportAccuracy, queryAccuracy)
        };

        return new MetaAdaptationResult<T>(
            queryAccuracy: queryAccuracy,
            queryLoss: queryLoss,
            supportAccuracy: supportAccuracy,
            supportLoss: supportLoss,
            adaptationSteps: config.SelfSupervisedSteps + config.SupervisedSteps,
            adaptationTimeMs: startTime.Elapsed.TotalMilliseconds,
            perStepLosses: perStepLosses,
            additionalMetrics: additionalMetrics);
    }

    /// <summary>
    /// Performs self-supervised pre-training on the query set.
    /// </summary>
    /// <param name="task">The meta-learning task.</param>
    /// <param name="steps">Number of self-supervised training steps.</param>
    /// <remarks>
    /// <para>
    /// Self-supervised learning creates artificial tasks from unlabeled data.
    /// For images, a common approach is rotation prediction:
    /// - Rotate each image by 0°, 90°, 180°, 270°
    /// - Train model to predict which rotation was applied
    /// - Model learns spatial relationships and features
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as learning to recognize patterns without knowing what they mean.
    ///
    /// Example: Looking at 75 unlabeled animal photos
    /// - You can learn: ears, tails, fur patterns, body shapes
    /// - You don't know: which animal is which (no labels)
    /// - But learning these features helps later classification!
    /// </para>
    /// </remarks>
    private void SelfSupervisedPreTraining(
        MetaLearningTask<T, TInput, TOutput> task,
        int steps)
    {
        for (int step = 0; step < steps; step++)
        {
            // Create self-supervised task from query set
            var (augmentedX, augmentedY) = _selfSupervisedLoss.CreateTask<TInput, TOutput>(task.QuerySetX);

            // Train on self-supervised task
            MetaModel.Train(augmentedX, augmentedY);
        }
    }

    /// <summary>
    /// Selects the most confident examples from query set and pseudo-labels them.
    /// </summary>
    /// <param name="task">The meta-learning task.</param>
    /// <param name="k">Number of examples to select.</param>
    /// <returns>Pseudo-labeled examples (input, output pairs).</returns>
    /// <remarks>
    /// <para>
    /// Active learning selects examples where the model is most confident:
    /// 1. Make predictions on entire query set
    /// 2. Calculate confidence (max probability) for each prediction
    /// 3. Select top-K most confident predictions
    /// 4. Use predicted class as "pseudo-label"
    /// </para>
    /// <para><b>For Beginners:</b> This is like solving problems you're confident about.
    ///
    /// After self-supervised learning, model looks at 75 unlabeled images:
    /// - Image 1: 95% sure it's a cat → High confidence, select it
    /// - Image 2: 92% sure it's a dog → High confidence, select it
    /// - Image 3: 60% sure it's a bird → Low confidence, skip it
    ///
    /// Result: 20 examples with high-confidence predictions to use as training data
    ///
    /// Why this works: High-confidence predictions are usually correct!
    /// </para>
    /// </remarks>
    private (TInput pseudoX, TOutput pseudoY) ActiveLearningSelection(
        MetaLearningTask<T, TInput, TOutput> task,
        int k)
    {
        // Make predictions on query set
        TOutput predictions = MetaModel.Predict(task.QuerySetX);

        // Calculate confidence for each prediction (max probability)
        var confidences = CalculateConfidences(predictions);

        // Select top-K most confident indices
        var topKIndices = SelectTopK(confidences, k);

        // Extract selected examples
        TInput pseudoX = ExtractExamples(task.QuerySetX, topKIndices);

        // Create pseudo-labels from predictions (argmax for classification)
        TOutput pseudoY = ExtractPseudoLabels(predictions, topKIndices);

        return (pseudoX, pseudoY);
    }

    /// <summary>
    /// Performs supervised fine-tuning on combined support set and pseudo-labeled examples.
    /// </summary>
    /// <param name="task">The meta-learning task.</param>
    /// <param name="pseudoX">Pseudo-labeled input examples.</param>
    /// <param name="pseudoY">Pseudo-labels.</param>
    /// <param name="steps">Number of supervised training steps.</param>
    /// <remarks>
    /// <para>
    /// Supervised fine-tuning combines real labeled data (support set) with high-confidence
    /// pseudo-labeled data from active learning. This effectively increases the training set size.
    /// </para>
    /// <para><b>For Beginners:</b> Now you have MORE training data to learn from:
    ///
    /// Original support set (real labels): 25 examples (5 classes × 5 shots)
    /// Pseudo-labeled set (confident predictions): 20 examples (selected in Phase 2)
    /// Combined training set: 45 examples total (25 real + 20 pseudo)
    ///
    /// Training on 45 examples gives better performance than just 25!
    /// </para>
    /// </remarks>
    private void SupervisedFineTuning(
        MetaLearningTask<T, TInput, TOutput> task,
        TInput pseudoX,
        TOutput pseudoY,
        int steps)
    {
        // Combine support set and pseudo-labeled set
        TInput combinedX = CombineInputs(task.SupportSetX, pseudoX);
        TOutput combinedY = CombineOutputs(task.SupportSetY, pseudoY);

        // Standard supervised training on combined set
        for (int step = 0; step < steps; step++)
        {
            MetaModel.Train(combinedX, combinedY);
        }
    }

    /// <summary>
    /// Calculates confidence scores (max probability) for each prediction.
    /// </summary>
    private Vector<T> CalculateConfidences(TOutput predictions)
    {
        if (predictions is not Tensor<T> tensor)
        {
            throw new NotSupportedException($"Confidence calculation only supports Tensor<T>, got {typeof(TOutput)}");
        }

        // For classification: confidence = max probability per example
        // predictions shape: (N, C) where N = examples, C = classes
        int numExamples = tensor.Shape[0];
        int numClasses = tensor.Shape.Length > 1 ? tensor.Shape[1] : 1;
        var confidences = new Vector<T>(numExamples);

        for (int i = 0; i < numExamples; i++)
        {
            T maxProb = NumOps.Zero;
            for (int j = 0; j < numClasses; j++)
            {
                T prob = tensor[i, j];
                if (NumOps.GreaterThan(prob, maxProb))
                {
                    maxProb = prob;
                }
            }
            confidences[i] = maxProb;
        }

        return confidences;
    }

    /// <summary>
    /// Selects indices of top-K most confident examples using a min-heap for O(n log k) performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If k exceeds the number of available examples, all examples are used (k is clamped to confidences.Length)
    /// and a warning is issued via Debug.WriteLine. This may include low-confidence examples, which can affect active learning quality.
    /// </para>
    /// <para>
    /// Uses a min-heap (SortedSet) to maintain top-K elements efficiently. Time complexity is O(n log k) instead of O(n log n),
    /// which provides significant performance benefits when k is much smaller than n.
    /// </para>
    /// </remarks>
    private List<int> SelectTopK(Vector<T> confidences, int k)
    {
        // Limit k to the number of available examples
        if (k > confidences.Length)
        {
            System.Diagnostics.Debug.WriteLine(
                $"[SEALTrainer] Warning: Requested top-{k} examples, but only {confidences.Length} available. Using all available examples. " +
                "This may include low-confidence examples and affect active learning quality.");
        }
        k = Math.Min(k, confidences.Length);

        // Use a min-heap to keep track of top-K elements
        // The heap stores (confidence, index) so that the smallest confidence is at the top
        var heap = new SortedSet<(T confidence, int index)>(Comparer<(T confidence, int index)>.Create((a, b) =>
        {
            // Compare by confidence ascending (min-heap)
            if (NumOps.LessThan(a.confidence, b.confidence))
                return -1;
            else if (NumOps.GreaterThan(a.confidence, b.confidence))
                return 1;
            else
                return a.index.CompareTo(b.index); // Ensure uniqueness in SortedSet
        }));

        for (int i = 0; i < confidences.Length; i++)
        {
            var item = (confidences[i], i);
            if (heap.Count < k)
            {
                heap.Add(item);
            }
            else if (NumOps.GreaterThan(confidences[i], heap.Min.confidence))
            {
                heap.Remove(heap.Min);
                heap.Add(item);
            }
        }

        // Extract indices and sort them by descending confidence
        var result = heap.OrderByDescending(x => x.confidence, Comparer<T>.Create((a, b) =>
        {
            if (NumOps.GreaterThan(a, b))
                return 1;
            else if (NumOps.LessThan(a, b))
                return -1;
            else
                return 0;
        })).Select(x => x.index).ToList();

        return result;
    }

    /// <summary>
    /// Extracts selected examples from input tensor.
    /// </summary>
    private TInput ExtractExamples(TInput input, List<int> indices)
    {
        if (input is not Tensor<T> tensor)
        {
            throw new NotSupportedException($"Example extraction only supports Tensor<T>, got {typeof(TInput)}");
        }

        int[] newShape = (int[])tensor.Shape.Clone();
        newShape[0] = indices.Count;

        var extracted = new Tensor<T>(newShape);

        for (int i = 0; i < indices.Count; i++)
        {
            int srcIdx = indices[i];
            CopyTensorSlice(tensor, extracted, srcIdx, i);
        }

        return (TInput)(object)extracted;
    }

    /// <summary>
    /// Extracts pseudo-labels (argmax of predictions) for selected examples.
    /// </summary>
    private TOutput ExtractPseudoLabels(TOutput predictions, List<int> indices)
    {
        if (predictions is not Tensor<T> tensor)
        {
            throw new NotSupportedException($"Pseudo-label extraction only supports Tensor<T>, got {typeof(TOutput)}");
        }

        int numClasses = tensor.Shape.Length > 1 ? tensor.Shape[1] : 1;
        var pseudoLabels = new Tensor<T>(new[] { indices.Count, numClasses });

        for (int i = 0; i < indices.Count; i++)
        {
            int srcIdx = indices[i];

            // Find argmax (predicted class)
            int predictedClass = 0;
            T maxProb = NumOps.Zero;
            for (int j = 0; j < numClasses; j++)
            {
                T prob = tensor[srcIdx, j];
                if (NumOps.GreaterThan(prob, maxProb))
                {
                    maxProb = prob;
                    predictedClass = j;
                }
            }

            // Create one-hot encoding
            for (int j = 0; j < numClasses; j++)
            {
                pseudoLabels[i, j] = (j == predictedClass) ? NumOps.One : NumOps.Zero;
            }
        }

        return (TOutput)(object)pseudoLabels;
    }

    /// <summary>
    /// Combines two input tensors along the first dimension.
    /// </summary>
    private TInput CombineInputs(TInput input1, TInput input2)
    {
        if (input1 is not Tensor<T> tensor1 || input2 is not Tensor<T> tensor2)
        {
            throw new NotSupportedException($"Combining only supports Tensor<T>, got {typeof(TInput)}");
        }

        // Validate tensors are not empty
        if (tensor1.Shape.Length == 0 || tensor2.Shape.Length == 0)
        {
            throw new ArgumentException("Cannot combine empty tensors");
        }

        // Validate shape compatibility (all dimensions except first must match)
        if (tensor1.Shape.Length != tensor2.Shape.Length)
        {
            throw new ArgumentException($"Tensors must have same number of dimensions. Got {tensor1.Shape.Length} and {tensor2.Shape.Length}");
        }

        for (int i = 1; i < tensor1.Shape.Length; i++)
        {
            if (tensor1.Shape[i] != tensor2.Shape[i])
            {
                throw new ArgumentException($"Tensor shapes must match in dimension {i}. Got {tensor1.Shape[i]} and {tensor2.Shape[i]}");
            }
        }

        int[] newShape = (int[])tensor1.Shape.Clone();
        newShape[0] = tensor1.Shape[0] + tensor2.Shape[0];

        var combined = new Tensor<T>(newShape);

        // Copy first tensor
        for (int i = 0; i < tensor1.Shape[0]; i++)
        {
            CopyTensorSlice(tensor1, combined, i, i);
        }

        // Copy second tensor
        for (int i = 0; i < tensor2.Shape[0]; i++)
        {
            CopyTensorSlice(tensor2, combined, i, tensor1.Shape[0] + i);
        }

        return (TInput)(object)combined;
    }

    /// <summary>
    /// Combines two output tensors along the first dimension.
    /// </summary>
    private TOutput CombineOutputs(TOutput output1, TOutput output2)
    {
        if (output1 is not Tensor<T> tensor1 || output2 is not Tensor<T> tensor2)
        {
            throw new NotSupportedException($"Combining only supports Tensor<T>, got {typeof(TOutput)}");
        }

        // Validate tensors are not empty
        if (tensor1.Shape.Length == 0 || tensor2.Shape.Length == 0)
        {
            throw new ArgumentException("Cannot combine empty tensors");
        }

        // Validate shape compatibility (all dimensions except first must match)
        if (tensor1.Shape.Length != tensor2.Shape.Length)
        {
            throw new ArgumentException($"Tensors must have same number of dimensions. Got {tensor1.Shape.Length} and {tensor2.Shape.Length}");
        }

        for (int i = 1; i < tensor1.Shape.Length; i++)
        {
            if (tensor1.Shape[i] != tensor2.Shape[i])
            {
                throw new ArgumentException($"Tensor shapes must match in dimension {i}. Got {tensor1.Shape[i]} and {tensor2.Shape[i]}");
            }
        }

        int[] newShape = (int[])tensor1.Shape.Clone();
        newShape[0] = tensor1.Shape[0] + tensor2.Shape[0];

        var combined = new Tensor<T>(newShape);

        // Copy first tensor
        for (int i = 0; i < tensor1.Shape[0]; i++)
        {
            CopyTensorSlice(tensor1, combined, i, i);
        }

        // Copy second tensor
        for (int i = 0; i < tensor2.Shape[0]; i++)
        {
            CopyTensorSlice(tensor2, combined, i, tensor1.Shape[0] + i);
        }

        return (TOutput)(object)combined;
    }

    /// <summary>
    /// Copies a slice (first dimension) from source to destination tensor.
    /// </summary>
    private void CopyTensorSlice(Tensor<T> source, Tensor<T> dest, int srcIdx, int destIdx)
    {
        if (source.Shape.Length < 2)
        {
            throw new NotSupportedException($"Tensor copy only supports 2D or higher tensors, got {source.Shape.Length}D");
        }

        if (source.Shape.Length != dest.Shape.Length)
        {
            throw new ArgumentException("Source and destination tensors must have the same number of dimensions");
        }

        for (int d = 1; d < source.Shape.Length; d++)
        {
            if (source.Shape[d] != dest.Shape[d])
            {
                throw new ArgumentException("Source and destination tensor shapes must match in all but the first dimension");
            }
        }

        int dims = source.Shape.Length;
        int[] srcIndices = new int[dims];
        int[] destIndices = new int[dims];
        srcIndices[0] = srcIdx;
        destIndices[0] = destIdx;

        CopyTensorSliceRecursive(source, dest, srcIndices, destIndices, 1);
    }

    /// <summary>
    /// Recursively copies a slice from source to destination tensor for arbitrary dimensions.
    /// </summary>
    private void CopyTensorSliceRecursive(Tensor<T> source, Tensor<T> dest, int[] srcIndices, int[] destIndices, int dim)
    {
        if (dim == source.Shape.Length)
        {
            // At the innermost element, copy value
            dest[destIndices] = source[srcIndices];
            return;
        }

        for (int i = 0; i < source.Shape[dim]; i++)
        {
            srcIndices[dim] = i;
            destIndices[dim] = i;
            CopyTensorSliceRecursive(source, dest, srcIndices, destIndices, dim + 1);
        }
    }

    /// <summary>
    /// Averages a list of vectors element-wise.
    /// </summary>
    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
            throw new ArgumentException("Cannot average empty list of vectors");

        int dimension = vectors[0].Length;
        var result = new Vector<T>(dimension);

        // Sum all vectors
        foreach (var vector in vectors)
        {
            if (vector.Length != dimension)
                throw new ArgumentException("All vectors must have the same dimension");

            result = result.Add(vector);
        }

        // Divide by count to get average
        T divisor = NumOps.FromDouble(vectors.Count);
        result = result.Divide(divisor);

        return result;
    }
}
