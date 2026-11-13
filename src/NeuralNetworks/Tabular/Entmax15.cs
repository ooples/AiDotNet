using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Differentiable 1.5-entmax and 1.5-entmoid (Peters et al. 2019; Popov et al. 2019, NODE) on the
/// active tape.
/// </summary>
/// <remarks>
/// <para>
/// <c>entmax15(x) = [x/2 - tau]_+^2</c>, with <c>tau</c> chosen per slice of the last axis so the
/// slice sums to one. The threshold uses the exact sort-based algorithm of the reference
/// <c>entmax</c> package (<c>_entmax_threshold_and_support</c>). <c>entmoid15(x)</c> is the first
/// entry of <c>entmax15([x, 0])</c>, which is NODE's split function.
/// </para>
/// <para>
/// <see cref="EntmaxAttention{T}"/> and <see cref="EntmoidActivation{T}"/> compute detached outputs, so
/// a layer trained on the tape gets no gradient through them. This helper keeps the gradient. The
/// threshold is solved off the tape. It then re-enters as <c>tau + sum(w (z - stop(z)))</c> with
/// <c>w = sqrt(p) / sum(sqrt(p))</c>. The added term is zero in value, and its gradient is the exact
/// implicit derivative <c>d tau / d z</c>. The resulting Jacobian is the one the reference
/// <c>Entmax15Function.backward</c> applies.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class Entmax15<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>1.5-entmax over the last axis.</summary>
    public static Tensor<T> Forward(IEngine engine, Tensor<T> logits)
    {
        int axis = logits.Rank - 1;
        int width = logits.Shape[axis];
        var shape = logits._shape;
        var z = engine.TensorMultiplyScalar(logits, NumOps.FromDouble(0.5));   // (alpha - 1) x
        // Keep the threshold relative to a detached maximum. Adding a large offset back to tau
        // rounds away the threshold and can make equal finite logits produce probabilities summing to two.
        Tensor<T> center;
        using (new NoGradScope<T>()) center = engine.ReduceMax(z, new[] { axis }, true);
        z = engine.TensorSubtract(z, engine.TensorBroadcastTo(center, shape));


        Tensor<T> tau;
        Tensor<T> weights;
        Tensor<T> frozen;
        using (new NoGradScope<T>())
        {
            frozen = z.Clone();
            var max = engine.ReduceMax(frozen, new[] { axis }, true);
            var shifted = engine.TensorSubtract(frozen, engine.TensorBroadcastTo(max, shape));
            // The largest probability is (max - tau)^2 <= 1, so tau >= max - 1 and no entry below
            // max - 1 is ever in the support. Raising those entries to max - 2 leaves them outside the
            // support, so the exact threshold is unchanged, while every value the moment computation
            // squares stays in [0, 4]. Without the bound a finite Half spread such as [0, 1000, -1000]
            // squares to 250000 > Half.MaxValue and the threshold becomes inf - inf = NaN.
            shifted = engine.TensorClampMin(shifted, NumOps.FromDouble(-2.0));
            var sorted = engine.TensorSort(shifted, axis, true).Values;

            var rhoShape = new int[shape.Length];
            for (int d = 0; d < rhoShape.Length; d++) rhoShape[d] = 1;
            rhoShape[axis] = width;
            var rho = engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorArange<T>(1, width + 1, 1.0), rhoShape), shape);

            var mean = engine.TensorDivide(engine.TensorCumSum(sorted, axis), rho);
            var meanSquare = engine.TensorDivide(engine.TensorCumSum(engine.TensorSquare(sorted), axis), rho);
            var spread = engine.TensorMultiply(rho, engine.TensorSubtract(meanSquare, engine.TensorSquare(mean)));
            var delta = engine.TensorDivide(engine.ScalarMinusTensor(NumOps.One, spread), rho);
            var candidates = engine.TensorSubtract(mean, engine.TensorSqrt(engine.TensorClampMin(delta, NumOps.Zero)));

            // support = #{k : tau_k <= sorted_k}; tau* = tau_(support).
            var inSupport = engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(candidates, sorted));
            var supportSize = engine.ReduceSum(inSupport, new[] { axis }, true);
            var pick = engine.TensorEquals(rho, engine.TensorBroadcastTo(supportSize, shape));
            var tauShifted = engine.ReduceSum(engine.TensorMultiply(candidates, pick), new[] { axis }, true);
            tau = engine.TensorAdd(tauShifted, max);

            var root = engine.TensorReLU(engine.TensorSubtract(frozen, engine.TensorBroadcastTo(tau, shape)));
            var rootSum = engine.TensorBroadcastTo(engine.ReduceSum(root, new[] { axis }, true), shape);
            weights = engine.TensorDivide(root, rootSum);
        }

        var tauTraced = engine.TensorAdd(
            tau,
            engine.ReduceSum(engine.TensorMultiply(weights, engine.TensorSubtract(z, frozen)), new[] { axis }, true));
        var difference = engine.TensorSubtract(z, engine.TensorBroadcastTo(tauTraced, shape));
        return engine.TensorSquare(engine.TensorReLU(difference));
    }

    /// <summary>
    /// 1.5-entmoid, element-wise: <c>entmax15([x, 0])[0]</c>. It is 0.5 at 0, exactly 1 for x &gt;= 2 and
    /// exactly 0 for x &lt;= -2.
    /// </summary>
    public static Tensor<T> Entmoid(IEngine engine, Tensor<T> logits)
    {
        int rank = logits.Rank;
        var pair = engine.TensorStack(new[] { logits, new Tensor<T>(logits._shape) }, rank);
        var probabilities = Forward(engine, pair);
        return engine.Reshape(engine.TensorNarrow(probabilities, rank, 0, 1), logits._shape);
    }
}
