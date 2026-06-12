using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Standard (z-score) FEATURE scaling for <see cref="Tensor{T}"/> data — the tensor-shaped counterpart
/// of <see cref="StandardScaler{T}"/> for facade pipelines whose input type is <c>Tensor&lt;T&gt;</c>
/// (e.g. neural-network builders): <c>ConfigurePreprocessing(new TensorStandardScaler&lt;T&gt;())</c>.
/// </summary>
/// <remarks>
/// <para><b>Why:</b> <see cref="StandardScaler{T}"/> implements <c>IDataTransformer</c> over
/// <c>Matrix&lt;T&gt;</c> only, so Tensor-input builders previously had NO way to scale features through
/// the facade and consumers hand-rolled scaling outside it. This adapter scales each column of a rank-2
/// <c>[n, k]</c> tensor independently (rank-1 treated as a single column), reusing the shape adapters
/// from <see cref="TargetStandardScaler{T, TOutput}"/>.</para>
/// </remarks>
public sealed class TensorStandardScaler<T> : TargetStandardScaler<T, Tensor<T>>
{
    public TensorStandardScaler(StandardScaler<T>? scaler = null)
        : base(scaler)
    {
    }
}
