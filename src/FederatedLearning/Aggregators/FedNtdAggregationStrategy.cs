using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedNTD (Not-True Distillation) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> FedNTD addresses the problem of local models "forgetting" the
/// global model's knowledge during local training. It adds a distillation loss that only
/// penalizes changes to non-true class logits (the classes that are NOT the correct answer),
/// preserving local knowledge about the true class while keeping the rest aligned with the
/// global model.</para>
///
/// <para>Local training objective:</para>
/// <code>L = L_CE + beta * KL(softmax(z_global / tau)[not-true] || softmax(z_local / tau)[not-true])</code>
///
/// <para>The key innovation is masking out the true class before computing the KL divergence,
/// so the local model can freely adapt its prediction for the correct class while maintaining
/// the global model's knowledge about other classes.</para>
///
/// <para>Reference: Lee, G., Shin, M., and Hwang, S. J. (2022). "Preservation of the Global Knowledge
/// by Not-True Distillation in Federated Learning." NeurIPS 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
internal class FedNtdAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _distillationWeight;
    private readonly double _temperature;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedNtdAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="distillationWeight">Weight of the not-true distillation loss (beta). Default: 1.0 per paper.</param>
    /// <param name="temperature">Softmax temperature for distillation. Default: 3.0 per paper.</param>
    public FedNtdAggregationStrategy(double distillationWeight = 1.0, double temperature = 3.0)
    {
        if (distillationWeight < 0)
        {
            throw new ArgumentException("Distillation weight must be non-negative.", nameof(distillationWeight));
        }

        if (temperature <= 0)
        {
            throw new ArgumentException("Temperature must be positive.", nameof(temperature));
        }

        _distillationWeight = distillationWeight;
        _temperature = temperature;
    }

    /// <summary>
    /// Aggregates client models using standard weighted averaging.
    /// </summary>
    /// <remarks>
    /// <para><b>By design</b>, FedNTD uses standard FedAvg for aggregation. The not-true
    /// distillation behavior is applied during <em>local training</em> via <see cref="ComputeNTDLoss"/>
    /// and <see cref="ComputeTotalLoss"/>. The aggregation step is unchanged from FedAvg.</para>
    /// </remarks>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Computes the Not-True Distillation loss for a single sample during local training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method computes KL divergence between the global model's
    /// soft predictions and the local model's soft predictions, but ONLY for the non-true classes.
    /// The true class is masked out so the local model can freely learn from its local data
    /// for that class, while still preserving global knowledge about all other classes.</para>
    /// </remarks>
    /// <param name="localLogits">Raw logits from the local model for a single sample.</param>
    /// <param name="globalLogits">Raw logits from the global model for the same sample.</param>
    /// <param name="trueClassIndex">Index of the true (correct) class to mask out.</param>
    /// <returns>The NTD loss value: beta * KL(q_global_NT || p_local_NT) * tau^2.</returns>
    public T ComputeNTDLoss(Vector<T> localLogits, Vector<T> globalLogits, int trueClassIndex)
    {
        Guard.NotNull(localLogits);
        Guard.NotNull(globalLogits);
        int numClasses = localLogits.Length;

        if (globalLogits.Length != numClasses)
        {
            throw new ArgumentException("Local and global logits must have the same number of classes.");
        }

        if (trueClassIndex < 0 || trueClassIndex >= numClasses)
        {
            throw new ArgumentOutOfRangeException(nameof(trueClassIndex),
                $"True class index must be in [0, {numClasses}).");
        }

        if (numClasses <= 1)
        {
            return NumOps.Zero;
        }

        var local = localLogits.AsTensor().Reshape(1, numClasses);
        var global = globalLogits.AsTensor().Reshape(1, numClasses);
        return NumOps.Multiply(NotTrueDistillationScale(), NotTrueKlSum(local, global, new[] { trueClassIndex }));
    }

    /// <summary>
    /// Computes the complete local training loss including task loss and NTD loss.
    /// </summary>
    /// <param name="taskLoss">The base task loss (e.g., cross-entropy).</param>
    /// <param name="localLogits">Raw logits from the local model.</param>
    /// <param name="globalLogits">Raw logits from the cached global model.</param>
    /// <param name="trueClassIndex">Index of the true class.</param>
    /// <returns>L_total = L_CE + beta * NTD_loss.</returns>
    public T ComputeTotalLoss(T taskLoss, Vector<T> localLogits, Vector<T> globalLogits, int trueClassIndex)
    {
        var ntdLoss = ComputeNTDLoss(localLogits, globalLogits, trueClassIndex);
        return NumOps.Add(taskLoss, ntdLoss);
    }

    /// <summary>
    /// Computes the batch-averaged NTD loss across multiple samples.
    /// </summary>
    /// <param name="localLogitsBatch">Local logits per sample (rows = samples, columns = classes).</param>
    /// <param name="globalLogitsBatch">Global logits per sample.</param>
    /// <param name="trueClassIndices">True class index per sample.</param>
    /// <returns>Average NTD loss across the batch.</returns>
    public T ComputeBatchNTDLoss(
        Matrix<T> localLogitsBatch,
        Matrix<T> globalLogitsBatch,
        int[] trueClassIndices)
    {
        Guard.NotNull(localLogitsBatch);
        Guard.NotNull(globalLogitsBatch);
        Guard.NotNull(trueClassIndices);
        int batchSize = localLogitsBatch.Rows;

        if (batchSize == 0)
        {
            return NumOps.Zero;
        }

        if (globalLogitsBatch.Rows != batchSize)
        {
            throw new ArgumentException("Batch sizes must match between local and global logits.");
        }

        if (trueClassIndices.Length != batchSize)
        {
            throw new ArgumentException("Must provide one true class index per sample.");
        }

        if (localLogitsBatch.Columns != globalLogitsBatch.Columns)
        {
            throw new ArgumentException("Local and global logits must have the same number of classes.");
        }

        int numClasses = localLogitsBatch.Columns;
        foreach (int trueClassIndex in trueClassIndices)
        {
            if (trueClassIndex < 0 || trueClassIndex >= numClasses)
            {
                throw new ArgumentOutOfRangeException(nameof(trueClassIndices),
                    $"True class index must be in [0, {numClasses}).");
            }
        }

        if (numClasses <= 1)
        {
            return NumOps.Zero;
        }

        var klSum = NotTrueKlSum(
            Tensor<T>.FromMatrix(localLogitsBatch), Tensor<T>.FromMatrix(globalLogitsBatch), trueClassIndices);
        return NumOps.Divide(NumOps.Multiply(NotTrueDistillationScale(), klSum), NumOps.FromDouble(batchSize));
    }

    /// <summary>beta * tau^2 (standard KD scaling).</summary>
    private T NotTrueDistillationScale() => NumOps.FromDouble(_distillationWeight * _temperature * _temperature);

    /// <summary>
    /// Sum over rows of KL(q_global || p_local) between the temperature-softened softmaxes over each row's
    /// not-true classes (Lee et al. 2022, FedNTD). The true class is excluded by giving it a -infinity logit.
    /// </summary>
    private T NotTrueKlSum(Tensor<T> local, Tensor<T> global, int[] trueClassIndices)
    {
        int rows = local.Shape[0];
        int classes = local.Shape[1];
        var invTemperature = NumOps.FromDouble(1.0 / _temperature);
        var localScaled = Engine.TensorMultiplyScalar(local, invTemperature);
        var globalScaled = Engine.TensorMultiplyScalar(global, invTemperature);

        var localTotal = Engine.TensorSum(localScaled);
        var globalTotal = Engine.TensorSum(globalScaled);
        if (NumOps.IsNaN(localTotal) || NumOps.IsInfinity(localTotal)
            || NumOps.IsNaN(globalTotal) || NumOps.IsInfinity(globalTotal))
        {
            throw new InvalidOperationException(
                $"Non-finite temperature-scaled logit: local sum={localTotal}, global sum={globalTotal}.");
        }

        var trueMask = new int[rows * classes];
        for (int r = 0; r < rows; r++)
        {
            trueMask[(r * classes) + trueClassIndices[r]] = 1;
        }

        var isTrue = Vector<int>.Wrap(trueMask).AsTensor().Reshape(rows, classes).Cast<T>();
        var excluded = new Tensor<T>(new[] { rows, classes });
        Engine.TensorFill(excluded, NumOps.FromDouble(double.NegativeInfinity));
        var pLocal = Engine.TensorSoftmax(Engine.TensorWhere(isTrue, excluded, localScaled), 1);
        var qGlobal = Engine.TensorSoftmax(Engine.TensorWhere(isTrue, excluded, globalScaled), 1);

        // KL(q || p) = sum q * log(q / max(p, 1e-10)), skipping terms with q <= 1e-10.
        var floor = NumOps.FromDouble(1e-10);
        var logRatio = Engine.TensorSubtract(
            Engine.TensorLog(Engine.TensorClampMin(qGlobal, floor)),
            Engine.TensorLog(Engine.TensorClampMin(pLocal, floor)));
        var terms = Engine.TensorMultiply(qGlobal, logRatio);
        var kept = Engine.TensorWhere(
            Engine.TensorGreaterThan(qGlobal, floor), terms, new Tensor<T>(new[] { rows, classes }));
        return Engine.TensorSum(kept);
    }

    /// <summary>Gets the not-true distillation weight (beta).</summary>
    public double DistillationWeight => _distillationWeight;

    /// <summary>Gets the softmax temperature for distillation.</summary>
    public double Temperature => _temperature;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedNTD(\u03b2={_distillationWeight},\u03c4={_temperature})";
}
