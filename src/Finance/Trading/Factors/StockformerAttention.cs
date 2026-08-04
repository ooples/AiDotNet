using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// Scaled dot-product attention as Stockformer defines it, over caller-owned projection layers.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Ma, Xue, Lu and Chen, arXiv:2401.06139. The paper's general form, used by all three of its
/// attention sites:
/// </para>
/// <code>
///   Att(Q, K, V) = softmax( (Q W^Q)(K^T W^K) / sqrt(d) ) (V W^V)
/// </code>
/// <para>
/// <b>This replaces a fixed averaging operator.</b> An earlier revision approximated the temporal
/// stage with a lower-triangular causal mean, which produces plausible activations and gives the model
/// almost NO capacity to learn which timesteps matter — the only trainable weights were in the
/// projection afterwards. The visible symptom was that longer training made the objective WORSE
/// (200-step loss above 50-step), because there was nothing for the extra steps to fit. Real attention
/// is not an optimisation here, it is the mechanism.
/// </para>
/// <para>
/// Q, K and V are separate parameters so the same type serves self-attention (all three the same
/// tensor, Eq. 7 and 10) and cross-attention (query from one band, key/value from the other, Eq. 11).
/// </para>
/// <para><b>For Beginners:</b> Attention lets each position decide how much to draw from every other
/// position, and those decisions are learned. Averaging everything equally — which is what this
/// replaced — throws that choice away.</para>
/// </remarks>
public sealed class StockformerAttention<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly ILayer<T> _query;
    private readonly ILayer<T> _key;
    private readonly ILayer<T> _value;
    private readonly int _features;
    private readonly bool _causal;

    /// <summary>
    /// Creates an attention block.
    /// </summary>
    /// <param name="features">Model width <c>d</c>; also the scaling denominator's basis.</param>
    /// <param name="query">W^Q.</param>
    /// <param name="key">W^K.</param>
    /// <param name="value">W^V.</param>
    /// <param name="causal">
    /// When true, a position may not attend to later positions. Used for the TEMPORAL axis, where
    /// letting a timestep see its own future would leak the answer; false for the ASSET axis, where
    /// there is no ordering to respect.
    /// </param>
    public StockformerAttention(int features, ILayer<T> query, ILayer<T> key, ILayer<T> value, bool causal)
    {
        if (features <= 0)
            throw new ArgumentOutOfRangeException(nameof(features), features, "Width must be positive.");

        _features = features;
        _query = query ?? throw new ArgumentNullException(nameof(query));
        _key = key ?? throw new ArgumentNullException(nameof(key));
        _value = value ?? throw new ArgumentNullException(nameof(value));
        _causal = causal;
    }

    /// <summary>
    /// Computes <c>Att(q, k, v)</c> over the LEADING axis of each input.
    /// </summary>
    /// <param name="q">Query source, <c>[positions, features]</c>.</param>
    /// <param name="k">Key source, <c>[positions, features]</c>.</param>
    /// <param name="v">Value source, <c>[positions, features]</c>.</param>
    /// <returns><c>[positions, features]</c>.</returns>
    /// <remarks>
    /// Every step is an Engine op or a layer Forward, so the whole thing is on the gradient tape.
    /// </remarks>
    public Tensor<T> Apply(Tensor<T> q, Tensor<T> k, Tensor<T> v)
    {
        if (q is null) throw new ArgumentNullException(nameof(q));
        if (k is null) throw new ArgumentNullException(nameof(k));
        if (v is null) throw new ArgumentNullException(nameof(v));

        int positions = q.Shape[0];
        if (k.Shape[0] != positions || v.Shape[0] != positions)
        {
            throw new ArgumentException(
                $"Query has {positions} positions but key has {k.Shape[0]} and value has {v.Shape[0]}. " +
                "All three must align along the attended axis.", nameof(k));
        }

        var projectedQuery = _query.Forward(q);
        var projectedKey = _key.Forward(k);
        var projectedValue = _value.Forward(v);

        // scores = Q K^T / sqrt(d)
        var scores = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(projectedQuery, Engine.TensorPermute(projectedKey, new[] { 1, 0 })),
            Ops.FromDouble(1.0 / Math.Sqrt(_features)));

        // The causal mask is ADDED as a large negative bias before the softmax rather than applied by
        // zeroing entries afterwards. Zeroing would mean writing into the weight tensor by index, which
        // is invisible to the tape and would cut the gradient to W^Q and W^K entirely — leaving only
        // W^V trainable while the model still looked like it had attention.
        if (_causal) scores = Engine.TensorAdd(scores, CausalBias(positions));

        var weights = Engine.Softmax(scores, axis: 1);
        return Engine.TensorMatMul(weights, projectedValue);
    }

    /// <summary>
    /// A <c>[positions, positions]</c> additive mask: zero where attention is allowed, a large negative
    /// value where it is not, so softmax drives those weights to ~0.
    /// </summary>
    /// <remarks>
    /// Constant DATA, so building it by index costs no gradient; the add is a recorded op. A finite
    /// sentinel rather than negative infinity, because inf - inf produces NaN if an entire row is
    /// masked.
    /// </remarks>
    private static Tensor<T> CausalBias(int positions)
    {
        var bias = new Tensor<T>(new[] { positions, positions });
        var blocked = Ops.FromDouble(-1e9);
        for (int i = 0; i < positions; i++)
        {
            for (int j = i + 1; j < positions; j++) bias[(i * positions) + j] = blocked;
        }
        return bias;
    }
}
