using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Transforms an attention block's projected queries, keys and values BEFORE they are split into
/// heads and scored.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This exists for methods that steer attention by rewriting Q/K/V rather than by rewriting the
/// block's output. <see cref="IAttentionBlockDecorator{T}"/> cannot express those: it only sees
/// <c>PostProcess(innerOutput)</c>, i.e. the tensor produced AFTER the softmax and the output
/// projection. Applying a Q/K/V-shaped operation there would transform a different tensor and
/// silently implement a different method.
/// </para>
/// <para>
/// UniVST (arXiv:2410.20084) is the motivating case: it blends the query with a content query at
/// every timestep and AdaIN-aligns the key/value against a style key/value on a schedule. Both act
/// on the projections, so neither is reachable from the output.
/// </para>
/// <para>
/// <b>Opt-in and shape-preserving.</b> Attention layers consult this only when one is attached, so a
/// null transform leaves their behaviour bit-identical. Each method MUST return a tensor with the same
/// shape it was given — the caller reshapes the result into
/// <c>[batch, heads, sequence, headDimension]</c> immediately afterwards and a changed element count
/// makes that reshape fail.
/// </para>
/// <para>
/// Implementations should compose from <c>Engine</c> operations so the transform is recorded on the
/// caller's gradient tape. Building a fresh tensor and writing into it by index severs the tape, which
/// leaves the projection weights with no gradient.
/// </para>
/// <para><b>For Beginners:</b> Attention turns each input into three tensors — a query, a key and a
/// value — before deciding what to pay attention to. This lets a technique adjust those three first,
/// which is how some style-transfer methods steer a frozen model without retraining it.</para>
/// </remarks>
public interface IQkvTransform<T>
{
    /// <summary>
    /// Transforms the projected queries, shaped <c>[batch * sequenceQ, embeddingDimension]</c>.
    /// Must return the same shape.
    /// </summary>
    Tensor<T> TransformQuery(Tensor<T> query);

    /// <summary>
    /// Transforms the projected keys, shaped <c>[batch * sequenceKV, embeddingDimension]</c>.
    /// Must return the same shape.
    /// </summary>
    Tensor<T> TransformKey(Tensor<T> key);

    /// <summary>
    /// Transforms the projected values, shaped <c>[batch * sequenceKV, embeddingDimension]</c>.
    /// Must return the same shape.
    /// </summary>
    Tensor<T> TransformValue(Tensor<T> value);
}
