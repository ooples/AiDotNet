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

    /// <summary>Gets the shared knowledge space dimension.</summary>
    public int SharedDimension => _sharedDimension;

    /// <summary>Gets the adaptive weight momentum.</summary>
    public double AdaptiveWeightMomentum => _adaptiveWeightMomentum;
}
