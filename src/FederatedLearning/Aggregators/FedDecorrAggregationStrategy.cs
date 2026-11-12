using AiDotNet.Attributes;
using AiDotNet.Enums;

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
/// L = L_task + lambda * (1 / D^2) * ||C||_F^2
///
/// where:
///   C = (Z^T * Z) / N       (correlation matrix of the column-standardised batch features Z)
///   X is the [N x D] feature matrix (N samples, D feature dimensions)
///   ||.||_F^2 is the squared Frobenius norm
/// </code>
///
/// <para>The decorrelation loss penalizes off-diagonal elements of the correlation matrix,
/// encouraging each feature dimension to capture unique information.</para>
///
/// <para>Reference: Shi, Y., et al. (2023). "Towards Understanding and Mitigating Dimensional Collapse
/// in Heterogeneous Federated Learning." ICLR 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
internal class FedDecorrAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _decorrelationWeight;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedDecorrAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="decorrelationWeight">Weight of the decorrelation loss (lambda). Default: 0.1 per paper.</param>
    public FedDecorrAggregationStrategy(double decorrelationWeight = 0.1)
    {
        if (double.IsNaN(decorrelationWeight) || double.IsInfinity(decorrelationWeight) || decorrelationWeight < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(decorrelationWeight),
                "Decorrelation weight must be a finite non-negative number.");
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
    /// <returns>The decorrelation loss: lambda * (1 / D^2) * ||C||_F^2.</returns>
    public T ComputeDecorrelationLoss(Matrix<T> features)
    {
        Guard.NotNull(features);
        int n = features.Rows;    // batch size
        int d = features.Columns; // feature dimension

        if (n < 2 || d < 2)
        {
            return NumOps.Zero;
        }

        // Shi et al. 2023: L_FedDecorr = (1 / d^2) * ||K||_F^2, K the correlation matrix of the batch.
        var correlation = ComputeCorrelationTensor(features);
        var normalized = NumOps.Divide(Engine.TensorSumOfSquares(correlation), NumOps.FromDouble((double)d * d));
        return NumOps.Multiply(NumOps.FromDouble(_decorrelationWeight), normalized);
    }

    /// <summary>
    /// Computes the complete local training loss including task loss and decorrelation loss.
    /// </summary>
    /// <param name="taskLoss">The base task loss.</param>
    /// <param name="features">Feature matrix [N x D] from the model's penultimate layer.</param>
    /// <returns>L_total = L_task + lambda * (1 / D^2) * ||C||_F^2.</returns>
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
        Guard.NotNull(features);
        int n = features.Rows;
        int d = features.Columns;

        if (n == 0 || d == 0)
        {
            return new Matrix<T>(d, d);
        }

        var corrMatrix = ComputeCorrelationTensor(features).ToMatrix();
        return corrMatrix;
    }

    /// <summary>
    /// K = Z^T Z / N where Z is the batch standardised per column (population std, floored at 1e-8).
    /// </summary>
    private static Tensor<T> ComputeCorrelationTensor(Matrix<T> features)
    {
        int n = features.Rows;
        int d = features.Columns;
        var x = Tensor<T>.FromMatrix(features);
        var centered = Engine.TensorSubtract(x, Engine.TensorBroadcastTo(Engine.ReduceMean(x, new[] { 0 }, true), new[] { n, d }));
        var std = Engine.TensorClampMin(
            Engine.TensorSqrt(Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, true)),
            NumOps.FromDouble(1e-8));
        var z = Engine.TensorDivide(centered, Engine.TensorBroadcastTo(std, new[] { n, d }));
        return Engine.TensorDivideScalar(Engine.TensorMatMul(Engine.TensorTranspose(z), z), NumOps.FromDouble(n));
    }

    /// <summary>Gets the decorrelation regularization weight (lambda).</summary>
    public double DecorrelationWeight => _decorrelationWeight;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedDecorr(\u03bb={_decorrelationWeight})";
}
