namespace AiDotNet.FederatedLearning.Distillation;

/// <summary>
/// FedDF — Federated ensemble distillation using model averaging on unlabeled public data.
/// </summary>
/// <remarks>
/// <para>
/// FedDF (Lin et al., 2020) performs federated distillation by treating each client model
/// as an ensemble member. The server distills the ensemble's collective predictions on
/// unlabeled data into a single global model. Unlike FedMD, FedDF works with completely
/// different model architectures and does not require labeled public data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as a "committee vote." Each client model votes on
/// what the answer should be for some shared unlabeled examples. The server then trains
/// a fresh model to match the committee's averaged predictions. This means each client
/// can use a completely different type of model.
/// </para>
/// <para>
/// Reference: Lin et al. (2020), "Ensemble Distillation for Robust Model Fusion in Federated Learning".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FedDFDistillation<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedDistillationStrategy<T>
{
    private readonly double _temperature;
    private readonly int _serverDistillationEpochs;
    private readonly double _serverLearningRate;

    /// <summary>
    /// Creates a new FedDF distillation strategy.
    /// </summary>
    /// <param name="temperature">Softmax temperature for ensemble averaging. Default: 3.0.</param>
    /// <param name="serverDistillationEpochs">Number of server-side distillation epochs. Default: 10.</param>
    /// <param name="serverLearningRate">Server-side learning rate. Default: 0.01.</param>
    public FedDFDistillation(double temperature = 3.0, int serverDistillationEpochs = 10,
        double serverLearningRate = 0.01)
    {
        _temperature = temperature;
        _serverDistillationEpochs = serverDistillationEpochs;
        _serverLearningRate = serverLearningRate;
    }

    /// <inheritdoc/>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData)
    {
        if (publicData is null)
            throw new ArgumentException("FedDF requires unlabeled public data.", nameof(publicData));

        int n = publicData.Rows;
        int d = localModelParameters.Length;
        int outputDim = Math.Max(2, (int)Math.Sqrt(d));

        // Compute soft predictions via linear projection
        var predictions = new Matrix<T>(n, outputDim);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < outputDim; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < publicData.Columns; k++)
                {
                    int paramIdx = (j * publicData.Columns + k) % d;
                    sum = NumOps.Add(sum, NumOps.Multiply(publicData[i, k], localModelParameters[paramIdx]));
                }
                predictions[i, j] = sum;
            }
        }

        // Softmax with temperature
        ApplySoftmax(predictions, _temperature);
        return predictions;
    }

    /// <inheritdoc/>
    public Matrix<T> AggregateKnowledge(Dictionary<int, Matrix<T>> clientKnowledge, Dictionary<int, double>? clientWeights)
    {
        if (clientKnowledge.Count == 0)
            throw new ArgumentException("No client knowledge provided.", nameof(clientKnowledge));

        var first = clientKnowledge.Values.First();
        int n = first.Rows;
        int outputDim = first.Columns;

        // Ensemble average: uniform or weighted combination of soft predictions
        double totalWeight = 0;
        var ensemble = new Matrix<T>(n, outputDim);

        foreach (var (clientId, knowledge) in clientKnowledge)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            T weight = NumOps.FromDouble(w);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    ensemble[i, j] = NumOps.Add(ensemble[i, j], NumOps.Multiply(knowledge[i, j], weight));
                }
            }
        }

        T invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < outputDim; j++)
            {
                ensemble[i, j] = NumOps.Multiply(ensemble[i, j], invTotal);
            }
        }

        return ensemble;
    }

    /// <inheritdoc/>
    public Vector<T> ApplyKnowledge(Vector<T> localModelParameters, Matrix<T> globalKnowledge,
        Matrix<T>? publicData, double temperature)
    {
        if (publicData is null)
            throw new ArgumentException("FedDF requires unlabeled public data.", nameof(publicData));

        int d = localModelParameters.Length;
        var updatedParams = new T[d];
        Array.Copy(localModelParameters.ToArray(), updatedParams, d);

        // Server-side distillation: train local model to match ensemble predictions
        for (int epoch = 0; epoch < _serverDistillationEpochs; epoch++)
        {
            var localPredictions = ExtractKnowledge(new Vector<T>(updatedParams), publicData);
            int outputDim = Math.Min(localPredictions.Columns, globalKnowledge.Columns);

            // Cross-entropy gradient between local predictions and ensemble target
            for (int j = 0; j < outputDim; j++)
            {
                T gradSum = NumOps.Zero;
                for (int i = 0; i < publicData.Rows; i++)
                {
                    // KL divergence gradient: local - target
                    T diff = NumOps.Subtract(localPredictions[i, j], globalKnowledge[i, j]);
                    gradSum = NumOps.Add(gradSum, diff);
                }

                T scaledGrad = NumOps.Multiply(gradSum, NumOps.FromDouble(_serverLearningRate / publicData.Rows));

                for (int k = 0; k < publicData.Columns; k++)
                {
                    int paramIdx = (j * publicData.Columns + k) % d;
                    updatedParams[paramIdx] = NumOps.Subtract(updatedParams[paramIdx], scaledGrad);
                }
            }
        }

        return new Vector<T>(updatedParams);
    }

    private void ApplySoftmax(Matrix<T> logits, double temp)
    {
        for (int i = 0; i < logits.Rows; i++)
        {
            // Find max for numerical stability
            double maxVal = double.NegativeInfinity;
            for (int j = 0; j < logits.Columns; j++)
            {
                double val = NumOps.ToDouble(logits[i, j]) / temp;
                if (val > maxVal) maxVal = val;
            }

            // Compute exp and sum
            double sumExp = 0;
            var expVals = new double[logits.Columns];
            for (int j = 0; j < logits.Columns; j++)
            {
                expVals[j] = Math.Exp(NumOps.ToDouble(logits[i, j]) / temp - maxVal);
                sumExp += expVals[j];
            }

            // Normalize
            for (int j = 0; j < logits.Columns; j++)
            {
                logits[i, j] = NumOps.FromDouble(expVals[j] / sumExp);
            }
        }
    }
}
