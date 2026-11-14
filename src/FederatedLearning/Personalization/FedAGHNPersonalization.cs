
using AiDotNet.Attributes;
using AiDotNet.Enums;
namespace AiDotNet.FederatedLearning.Personalization;

/// <summary>
/// Implements FedAGHN (Adaptive Gradient-based Heterogeneous Networks) personalization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In standard FL, all clients must use the same model architecture.
/// FedAGHN relaxes this: each client can have a differently-sized model (e.g., a phone uses
/// a small model, a workstation uses a large one). It works by defining a shared "knowledge
/// representation" space and learning adapter layers that project each client's heterogeneous
/// model into this shared space for aggregation. Gradient similarity across the shared space
/// determines aggregation weights adaptively.</para>
///
/// <para>Architecture:</para>
/// <code>
/// Client k (model size d_k):
///   local_params → ProjectToShared(d_k → d_shared) → aggregation → ProjectBack(d_shared → d_k)
/// </code>
///
/// <para>Reference: FedAGHN: Adaptive Gradient Heterogeneous Networks for FL (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FedAGHNPersonalization<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _sharedDimension;
    private readonly double _adaptiveWeightMomentum;

    /// <summary>
    /// Creates a new FedAGHN personalization strategy.
    /// </summary>
    /// <param name="sharedDimension">Dimension of the shared knowledge space. Default: 256.</param>
    /// <param name="adaptiveWeightMomentum">Momentum for adaptive weight updates. Default: 0.9.</param>
    public FedAGHNPersonalization(int sharedDimension = 256, double adaptiveWeightMomentum = 0.9)
    {
        if (sharedDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sharedDimension), "Shared dimension must be positive.");
        }

        if (adaptiveWeightMomentum < 0 || adaptiveWeightMomentum > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(adaptiveWeightMomentum), "Momentum must be in [0, 1].");
        }

        _sharedDimension = sharedDimension;
        _adaptiveWeightMomentum = adaptiveWeightMomentum;
    }

    /// <summary>
    /// Projects client parameters from local dimension to shared space for aggregation.
    /// </summary>
    /// <remarks>
    /// <para><b>Implementation Note:</b> This reference implementation uses simple truncation/zero-padding
    /// as the projection. Production deployments should replace this with a learned linear projection
    /// (e.g., via <see cref="ProjectWithMatrix"/>) for better information preservation.</para>
    /// </remarks>
    /// <param name="localParams">Client's local parameter vector.</param>
    /// <returns>Projected parameters in shared space.</returns>
    public T[] ProjectToShared(T[] localParams)
    {
        Guard.NotNull(localParams);
        var projected = new T[_sharedDimension];
        int copyLen = Math.Min(localParams.Length, _sharedDimension);
        for (int i = 0; i < copyLen; i++)
        {
            projected[i] = localParams[i];
        }

        for (int i = copyLen; i < _sharedDimension; i++)
        {
            projected[i] = NumOps.Zero;
        }

        return projected;
    }

    /// <summary>
    /// Projects shared-space parameters back to client's local dimension.
    /// </summary>
    /// <param name="sharedParams">Parameters in shared space.</param>
    /// <param name="localDimension">Target local dimension.</param>
    /// <returns>Parameters in local space.</returns>
    public T[] ProjectToLocal(T[] sharedParams, int localDimension)
    {
        Guard.NotNull(sharedParams);
        if (localDimension < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(localDimension), "Must be at least 1.");
        }

        var local = new T[localDimension];
        int copyLen = Math.Min(sharedParams.Length, localDimension);
        for (int i = 0; i < copyLen; i++)
        {
            local[i] = sharedParams[i];
        }

        for (int i = copyLen; i < localDimension; i++)
        {
            local[i] = NumOps.Zero;
        }

        return local;
    }

    /// <summary>
    /// Projects client parameters using a learned linear projection matrix rather than truncation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Simple truncation/zero-padding loses information when the local
    /// model is larger than the shared space. A learned projection matrix (like PCA) compresses the
    /// information more efficiently by learning which directions in parameter space are most important.</para>
    /// </remarks>
    /// <param name="localParams">Client's local parameter vector.</param>
    /// <param name="projectionMatrix">Projection matrix of shape [sharedDim x localDim]. Row-major.</param>
    /// <returns>Projected parameters in shared space.</returns>
    public T[] ProjectWithMatrix(T[] localParams, T[] projectionMatrix)
    {
        Guard.NotNull(localParams);
        Guard.NotNull(projectionMatrix);
        int localDim = localParams.Length;
        if (projectionMatrix.Length != _sharedDimension * localDim)
        {
            throw new ArgumentException(
                $"Projection matrix must have {_sharedDimension * localDim} elements (shared={_sharedDimension} x local={localDim}). Got {projectionMatrix.Length}.");
        }

        // projected = P (sharedDim x localDim, row-major) * localParams.
        var projection = Vector<T>.Wrap(projectionMatrix).AsTensor().Reshape(_sharedDimension, localDim);
        var local = Vector<T>.Wrap(localParams).AsTensor().Reshape(localDim, 1);
        return Engine.TensorMatMul(projection, local).ToArray();
    }

    /// <summary>
    /// Computes gradient similarity between two clients in the shared space using cosine similarity.
    /// </summary>
    /// <param name="gradA">Client A's projected gradient in shared space.</param>
    /// <param name="gradB">Client B's projected gradient in shared space.</param>
    /// <returns>Cosine similarity in [-1, 1]; zero when either gradient has (near-)zero norm.</returns>
    public T ComputeGradientSimilarity(T[] gradA, T[] gradB)
    {
        Guard.NotNull(gradA);
        Guard.NotNull(gradB);
        if (gradA.Length != gradB.Length)
        {
            throw new ArgumentException(
                $"Gradient lengths must match. Got gradA={gradA.Length}, gradB={gradB.Length}.");
        }

        // Formed in double (float for the Half / BFloat16 sums): a Half dot product or norm product overflows and
        // the 1e-10 guard rounds to 0.
        return NumOps.FromDouble(AiDotNet.FederatedLearning.Infrastructure.FederatedReductions.Cosine(Vector<T>.Wrap(gradA), Vector<T>.Wrap(gradB), 1e-10));
    }

    /// <summary>
    /// Computes adaptive aggregation weights for a target client based on gradient similarity with all other clients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of giving all clients equal weight, FedAGHN weighs each
    /// client's contribution based on how similar their gradients are to the target client's gradient.
    /// Clients with more similar gradients (learning similar patterns) get higher weights.
    /// This prevents "negative transfer" from clients with very different data.</para>
    /// </remarks>
    /// <param name="targetGradient">Target client's projected gradient in shared space.</param>
    /// <param name="clientGradients">All clients' projected gradients (clientId → gradient).</param>
    /// <param name="previousWeights">Previous round's weights for momentum smoothing. Can be null.</param>
    /// <returns>Adaptive aggregation weights per client (non-negative, sum to 1).</returns>
    public Dictionary<int, T> ComputeAdaptiveWeights(
        T[] targetGradient,
        Dictionary<int, T[]> clientGradients,
        Dictionary<int, T>? previousWeights = null)
    {
        Guard.NotNull(targetGradient);
        Guard.NotNull(clientGradients);
        int[] clientIds = clientGradients.Keys.ToArray();
        int n = clientIds.Length;
        if (n == 0)
        {
            return new Dictionary<int, T>();
        }

        var similarities = new Tensor<T>(
            clientIds.Select(id => ComputeGradientSimilarity(targetGradient, clientGradients[id])).ToArray(),
            new[] { n });

        // ReLU: only positive similarity contributes.
        var weights = Engine.TensorMax(similarities, NumOps.Zero);
        T totalWeight = Engine.TensorSum(weights);
        weights = NumOps.GreaterThan(totalWeight, NumOps.Zero)
            ? Engine.TensorDivideScalar(weights, totalWeight)
            : new Tensor<T>(Enumerable.Repeat(NumOps.FromDouble(1.0 / n), n).ToArray(), new[] { n });

        // Apply momentum smoothing with previous weights if available; a client without a previous
        // weight keeps its current one.
        if (previousWeights != null)
        {
            T[] current = weights.ToArray();
            var previous = new Tensor<T>(
                clientIds.Select((id, i) => previousWeights.TryGetValue(id, out var prev) ? prev : current[i]).ToArray(),
                new[] { n });
            weights = Engine.TensorAddScaled(
                previous,
                weights,
                NumOps.FromDouble(_adaptiveWeightMomentum),
                NumOps.FromDouble(1 - _adaptiveWeightMomentum));

            // Renormalize after momentum blending to ensure weights sum to 1.
            T blendedTotal = Engine.TensorSum(weights);
            if (NumOps.GreaterThan(blendedTotal, NumOps.Zero))
            {
                weights = Engine.TensorDivideScalar(weights, blendedTotal);
            }
        }

        T[] result = weights.ToArray();
        return clientIds
            .Select((id, i) => (Id: id, Weight: result[i]))
            .ToDictionary(entry => entry.Id, entry => entry.Weight);
    }

    /// <summary>Gets the shared knowledge space dimension.</summary>
    public int SharedDimension => _sharedDimension;

    /// <summary>Gets the adaptive weight momentum.</summary>
    public double AdaptiveWeightMomentum => _adaptiveWeightMomentum;
}
