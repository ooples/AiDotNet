namespace AiDotNet.FederatedLearning.Distillation;

/// <summary>
/// FedMD — Model-agnostic federated learning via mutual distillation on a public dataset.
/// </summary>
/// <remarks>
/// <para>
/// FedMD (Li &amp; Wang, 2019) enables heterogeneous model architectures across clients.
/// Each client computes logits on a shared public dataset, the server averages the logits,
/// and clients distill the consensus logits into their local models.
/// </para>
/// <para>
/// <b>For Beginners:</b> All clients are given the same "test questions" (public dataset).
/// Each client submits their answers (predicted probabilities). The server averages all
/// answers to get a "consensus answer," then each client learns from this consensus.
/// This way, a simple model on a phone can learn from a powerful model on a server.
/// </para>
/// <para>
/// Reference: Li &amp; Wang (2019), "FedMD: Heterogeneous Federated Learning via Model Distillation".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FedMDDistillation<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedDistillationStrategy<T>
{
    private readonly double _temperature;
    private readonly double _distillationAlpha;
    private readonly int _distillationEpochs;
    private readonly double _learningRate;

    /// <summary>
    /// Creates a new FedMD distillation strategy.
    /// </summary>
    /// <param name="temperature">Softmax temperature for soft labels. Default: 3.0.</param>
    /// <param name="distillationAlpha">Weight for distillation loss vs task loss. Default: 0.5.</param>
    /// <param name="distillationEpochs">Number of local distillation epochs. Default: 5.</param>
    /// <param name="learningRate">Learning rate for distillation updates. Default: 0.01.</param>
    public FedMDDistillation(double temperature = 3.0, double distillationAlpha = 0.5,
        int distillationEpochs = 5, double learningRate = 0.01)
    {
        _temperature = temperature;
        _distillationAlpha = distillationAlpha;
        _distillationEpochs = distillationEpochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData)
    {
        if (publicData is null)
            throw new ArgumentException("FedMD requires a public dataset for logit extraction.", nameof(publicData));

        int n = publicData.Rows;
        int d = localModelParameters.Length;
        int outputDim = Math.Max(1, d / 10); // Heuristic: output dim is a fraction of params

        // Compute logits: simple linear projection of public data through model parameters
        // In production, this would use the actual model forward pass
        var logits = new Matrix<T>(n, outputDim);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < outputDim; j++)
            {
                T sum = NumOps.Zero;
                int paramCols = Math.Min(publicData.Columns, d - j * publicData.Columns);
                for (int k = 0; k < Math.Min(publicData.Columns, paramCols); k++)
                {
                    int paramIdx = (j * publicData.Columns + k) % d;
                    sum = NumOps.Add(sum, NumOps.Multiply(publicData[i, k], localModelParameters[paramIdx]));
                }
                logits[i, j] = sum;
            }
        }

        // Apply temperature scaling
        ApplyTemperatureScaling(logits, _temperature);
        return logits;
    }

    /// <inheritdoc/>
    public Matrix<T> AggregateKnowledge(Dictionary<int, Matrix<T>> clientKnowledge, Dictionary<int, double>? clientWeights)
    {
        if (clientKnowledge.Count == 0)
            throw new ArgumentException("No client knowledge provided.", nameof(clientKnowledge));

        // Get dimensions from first client
        var first = clientKnowledge.Values.First();
        int n = first.Rows;
        int outputDim = first.Columns;

        // Weighted average of all client logits
        double totalWeight = 0;
        var aggregated = new Matrix<T>(n, outputDim);

        foreach (var (clientId, knowledge) in clientKnowledge)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            T weight = NumOps.FromDouble(w);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    aggregated[i, j] = NumOps.Add(aggregated[i, j], NumOps.Multiply(knowledge[i, j], weight));
                }
            }
        }

        // Normalize
        T invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < outputDim; j++)
            {
                aggregated[i, j] = NumOps.Multiply(aggregated[i, j], invTotal);
            }
        }

        return aggregated;
    }

    /// <inheritdoc/>
    public Vector<T> ApplyKnowledge(Vector<T> localModelParameters, Matrix<T> globalKnowledge,
        Matrix<T>? publicData, double temperature)
    {
        if (publicData is null)
            throw new ArgumentException("FedMD requires a public dataset.", nameof(publicData));

        int d = localModelParameters.Length;
        var updatedParams = new T[d];
        Array.Copy(localModelParameters.ToArray(), updatedParams, d);

        // Distillation: minimize KL divergence between local logits and global consensus
        for (int epoch = 0; epoch < _distillationEpochs; epoch++)
        {
            var localLogits = ExtractKnowledge(new Vector<T>(updatedParams), publicData);
            int outputDim = Math.Min(localLogits.Columns, globalKnowledge.Columns);

            // Compute gradient: difference between local and global soft labels
            for (int j = 0; j < outputDim; j++)
            {
                T gradSum = NumOps.Zero;
                for (int i = 0; i < publicData.Rows; i++)
                {
                    T diff = NumOps.Subtract(localLogits[i, j], globalKnowledge[i, j]);
                    gradSum = NumOps.Add(gradSum, diff);
                }

                // Scale gradient
                T scaledGrad = NumOps.Multiply(gradSum, NumOps.FromDouble(_learningRate * _distillationAlpha / publicData.Rows));

                // Update parameters that contribute to this output dimension
                for (int k = 0; k < Math.Min(publicData.Columns, d); k++)
                {
                    int paramIdx = (j * publicData.Columns + k) % d;
                    updatedParams[paramIdx] = NumOps.Subtract(updatedParams[paramIdx], scaledGrad);
                }
            }
        }

        return new Vector<T>(updatedParams);
    }

    private void ApplyTemperatureScaling(Matrix<T> logits, double temp)
    {
        T invTemp = NumOps.FromDouble(1.0 / temp);
        for (int i = 0; i < logits.Rows; i++)
        {
            for (int j = 0; j < logits.Columns; j++)
            {
                logits[i, j] = NumOps.Multiply(logits[i, j], invTemp);
            }
        }
    }
}
