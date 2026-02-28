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

        // Step 1: Extract non-true class logits and apply temperature scaling.
        int ntCount = numClasses - 1;
        var localNT = new double[ntCount];
        var globalNT = new double[ntCount];
        int idx = 0;

        for (int c = 0; c < numClasses; c++)
        {
            if (c == trueClassIndex)
            {
                continue;
            }

            localNT[idx] = NumOps.ToDouble(localLogits[c]) / _temperature;
            globalNT[idx] = NumOps.ToDouble(globalLogits[c]) / _temperature;
            idx++;
        }

        // Step 2: Compute softmax for both (numerically stable).
        var pLocal = SoftmaxInPlace(localNT);
        var qGlobal = SoftmaxInPlace(globalNT);

        // Step 3: KL divergence: KL(q || p) = sum(q * log(q / p))
        double kl = 0;
        for (int i = 0; i < ntCount; i++)
        {
            if (qGlobal[i] > 1e-10)
            {
                double pClamped = Math.Max(pLocal[i], 1e-10);
                kl += qGlobal[i] * Math.Log(qGlobal[i] / pClamped);
            }
        }

        // Step 4: Scale by beta * tau^2 (standard KD scaling).
        double loss = _distillationWeight * _temperature * _temperature * kl;
        return NumOps.FromDouble(loss);
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

        double totalLoss = 0;

        for (int s = 0; s < batchSize; s++)
        {
            var localRow = localLogitsBatch.GetRow(s);
            var globalRow = globalLogitsBatch.GetRow(s);
            totalLoss += NumOps.ToDouble(ComputeNTDLoss(localRow, globalRow, trueClassIndices[s]));
        }

        return NumOps.FromDouble(totalLoss / batchSize);
    }

    private static double[] SoftmaxInPlace(double[] logits)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
            }
        }

        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] = Math.Exp(logits[i] - max);
            sum += logits[i];
        }

        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] /= sum;
        }

        return logits;
    }

    /// <summary>Gets the not-true distillation weight (beta).</summary>
    public double DistillationWeight => _distillationWeight;

    /// <summary>Gets the softmax temperature for distillation.</summary>
    public double Temperature => _temperature;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedNTD(\u03b2={_distillationWeight},\u03c4={_temperature})";
}
