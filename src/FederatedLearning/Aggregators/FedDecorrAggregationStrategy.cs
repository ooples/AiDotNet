namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedDecorr (Decorrelation) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In federated learning with non-IID data, local models tend to learn
/// redundant (correlated) features — a phenomenon called "dimensional collapse." FedDecorr adds a
/// decorrelation regularizer that encourages each client's feature representations to be diverse
/// and complementary, improving the quality of the aggregated global model.</para>
///
/// <para>Local training objective:</para>
/// <code>
/// L = L_task + lambda * ||C - I||_F^2
///
/// where:
///   C = (X^T * X) / N       (correlation matrix of batch features)
///   X is the [N x D] feature matrix (N samples, D feature dimensions)
///   I is the D x D identity matrix
///   ||.||_F^2 is the squared Frobenius norm
/// </code>
///
/// <para>The decorrelation loss penalizes off-diagonal elements of the correlation matrix,
/// encouraging each feature dimension to capture unique information.</para>
///
/// <para>Reference: Shi, Y., et al. (2023). "Towards Understanding and Mitigating Dimensional Collapse
/// in Heterogeneous Federated Learning." ICML 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedDecorrAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _decorrelationWeight;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedDecorrAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="decorrelationWeight">Weight of the decorrelation loss (lambda). Default: 0.1 per paper.</param>
    public FedDecorrAggregationStrategy(double decorrelationWeight = 0.1)
    {
        if (decorrelationWeight < 0)
        {
            throw new ArgumentException("Decorrelation weight must be non-negative.", nameof(decorrelationWeight));
        }

        _decorrelationWeight = decorrelationWeight;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Computes the decorrelation loss for a batch of feature representations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Given a batch of feature vectors (one per sample), this method
    /// first normalizes each feature dimension to zero mean and unit variance. Then it computes
    /// how correlated the features are with each other. Ideally, each feature should capture
    /// independent information, so the correlation matrix should be close to the identity matrix.
    /// The loss penalizes deviations from this ideal.</para>
    /// </remarks>
    /// <param name="features">Feature matrix [N x D] where N is batch size and D is feature dimension.
    /// Each row is one sample's feature representation from the model's penultimate layer.</param>
    /// <returns>The decorrelation loss: lambda * ||C - I||_F^2.</returns>
    public T ComputeDecorrelationLoss(Matrix<T> features)
    {
        int n = features.Rows;    // batch size
        int d = features.Columns; // feature dimension

        if (n < 2 || d < 2)
        {
            return NumOps.Zero;
        }

        // Step 1: Center features (subtract column means).
        var means = new double[d];
        for (int j = 0; j < d; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(features[i, j]);
            }

            means[j] = sum / n;
        }

        // Step 2: Compute standard deviations for normalization.
        var stds = new double[d];
        for (int j = 0; j < d; j++)
        {
            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(features[i, j]) - means[j];
                sumSq += diff * diff;
            }

            stds[j] = Math.Sqrt(sumSq / n);
            if (stds[j] < 1e-8)
            {
                stds[j] = 1e-8; // Prevent division by zero.
            }
        }

        // Step 3: Compute correlation matrix C[i,j] = sum_k((x_k_i - mean_i)(x_k_j - mean_j)) / (N * std_i * std_j)
        // Then compute ||C - I||_F^2 = sum_ij (C[i,j] - delta_ij)^2
        double frobeniusNormSq = 0;

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double corr = 0;
                for (int k = 0; k < n; k++)
                {
                    double xi = (NumOps.ToDouble(features[k, i]) - means[i]) / stds[i];
                    double xj = (NumOps.ToDouble(features[k, j]) - means[j]) / stds[j];
                    corr += xi * xj;
                }

                corr /= n;

                // C[i,j] - I[i,j]
                double target = (i == j) ? 1.0 : 0.0;
                double diff = corr - target;
                frobeniusNormSq += diff * diff;
            }
        }

        return NumOps.FromDouble(_decorrelationWeight * frobeniusNormSq);
    }

    /// <summary>
    /// Computes the complete local training loss including task loss and decorrelation loss.
    /// </summary>
    /// <param name="taskLoss">The base task loss.</param>
    /// <param name="features">Feature matrix [N x D] from the model's penultimate layer.</param>
    /// <returns>L_total = L_task + lambda * ||C - I||_F^2.</returns>
    public T ComputeTotalLoss(T taskLoss, Matrix<T> features)
    {
        var decorrelationLoss = ComputeDecorrelationLoss(features);
        return NumOps.Add(taskLoss, decorrelationLoss);
    }

    /// <summary>
    /// Computes the correlation matrix for diagnostic purposes.
    /// </summary>
    /// <param name="features">Feature matrix [N x D].</param>
    /// <returns>The D x D correlation matrix.</returns>
    public Matrix<T> ComputeCorrelationMatrix(Matrix<T> features)
    {
        int n = features.Rows;
        int d = features.Columns;

        var means = new double[d];
        var stds = new double[d];

        for (int j = 0; j < d; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(features[i, j]);
            }

            means[j] = sum / n;

            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(features[i, j]) - means[j];
                sumSq += diff * diff;
            }

            stds[j] = Math.Sqrt(sumSq / n);
            if (stds[j] < 1e-8)
            {
                stds[j] = 1e-8;
            }
        }

        var corrMatrix = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double corr = 0;
                for (int k = 0; k < n; k++)
                {
                    double xi = (NumOps.ToDouble(features[k, i]) - means[i]) / stds[i];
                    double xj = (NumOps.ToDouble(features[k, j]) - means[j]) / stds[j];
                    corr += xi * xj;
                }

                corrMatrix[i, j] = NumOps.FromDouble(corr / n);
            }
        }

        return corrMatrix;
    }

    /// <summary>Gets the decorrelation regularization weight (lambda).</summary>
    public double DecorrelationWeight => _decorrelationWeight;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedDecorr(\u03bb={_decorrelationWeight})";
}
