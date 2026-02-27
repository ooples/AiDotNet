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
    /// Uses a simple linear projection (truncation/zero-padding for now).
    /// </summary>
    /// <param name="localParams">Client's local parameter vector.</param>
    /// <returns>Projected parameters in shared space.</returns>
    public T[] ProjectToShared(T[] localParams)
    {
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
        int localDim = localParams.Length;
        if (projectionMatrix.Length != _sharedDimension * localDim)
        {
            throw new ArgumentException(
                $"Projection matrix must have {_sharedDimension * localDim} elements (shared={_sharedDimension} x local={localDim}). Got {projectionMatrix.Length}.");
        }

        var projected = new T[_sharedDimension];
        for (int s = 0; s < _sharedDimension; s++)
        {
            T sum = NumOps.Zero;
            for (int l = 0; l < localDim; l++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(projectionMatrix[s * localDim + l], localParams[l]));
            }

            projected[s] = sum;
        }

        return projected;
    }

    /// <summary>
    /// Computes gradient similarity between two clients in the shared space using cosine similarity.
    /// </summary>
    /// <param name="gradA">Client A's projected gradient in shared space.</param>
    /// <param name="gradB">Client B's projected gradient in shared space.</param>
    /// <returns>Cosine similarity in [-1, 1].</returns>
    public double ComputeGradientSimilarity(T[] gradA, T[] gradB)
    {
        int len = Math.Min(gradA.Length, gradB.Length);
        double dot = 0, normA = 0, normB = 0;

        for (int i = 0; i < len; i++)
        {
            double a = NumOps.ToDouble(gradA[i]);
            double b = NumOps.ToDouble(gradB[i]);
            dot += a * b;
            normA += a * a;
            normB += b * b;
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 1e-10 ? dot / denom : 0;
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
    public Dictionary<int, double> ComputeAdaptiveWeights(
        T[] targetGradient,
        Dictionary<int, T[]> clientGradients,
        Dictionary<int, double>? previousWeights = null)
    {
        var weights = new Dictionary<int, double>();
        double totalWeight = 0;

        foreach (var (clientId, grad) in clientGradients)
        {
            double sim = ComputeGradientSimilarity(targetGradient, grad);
            // ReLU: only positive similarity contributes.
            double w = Math.Max(0, sim);
            weights[clientId] = w;
            totalWeight += w;
        }

        // Normalize.
        if (totalWeight > 0)
        {
            foreach (var key in weights.Keys.ToArray())
            {
                weights[key] /= totalWeight;
            }
        }
        else
        {
            // Fallback to uniform weights if no positive similarity.
            double uniform = 1.0 / Math.Max(1, clientGradients.Count);
            foreach (var key in clientGradients.Keys)
            {
                weights[key] = uniform;
            }
        }

        // Apply momentum smoothing with previous weights if available.
        if (previousWeights != null)
        {
            foreach (var key in weights.Keys.ToArray())
            {
                double prev = previousWeights.GetValueOrDefault(key, weights[key]);
                weights[key] = _adaptiveWeightMomentum * prev + (1 - _adaptiveWeightMomentum) * weights[key];
            }
        }

        return weights;
    }

    /// <summary>Gets the shared knowledge space dimension.</summary>
    public int SharedDimension => _sharedDimension;

    /// <summary>Gets the adaptive weight momentum.</summary>
    public double AdaptiveWeightMomentum => _adaptiveWeightMomentum;
}
