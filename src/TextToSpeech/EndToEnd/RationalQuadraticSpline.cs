namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// The monotonic rational-quadratic spline transform with linear tails (Durkan et al. 2019; reference
/// <c>transforms.unconstrained_rational_quadratic_spline</c>, as VITS's <c>ConvFlow</c> uses it), on the autodiff tape.
/// </summary>
/// <remarks>
/// <para>
/// For N inputs with K bins each: widths and heights are softmaxed, floored at 1e-3 and accumulated to knots on
/// [−B, B]; derivatives are 1e-3 + softplus, with the outer knots fixed so the spline joins the identity tails. Each
/// input's bin is found by a search over the current knots (a constant selection), the bin quantities are read by a
/// one-hot product, and the forward map is θ ↦ y_k + h_k (s_k θ² + d_k θ(1−θ)) / (s_k + (d_k + d_{k+1} − 2 s_k) θ(1−θ))
/// with its log-derivative; the inverse solves the quadratic for θ. Inputs outside [−B, B] pass through with zero
/// log-determinant.
/// </para>
/// </remarks>
internal static class RationalQuadraticSpline
{
    private const double MinBinWidth = 1e-3;
    private const double MinBinHeight = 1e-3;
    private const double MinDerivative = 1e-3;

    /// <summary>Transforms <paramref name="inputs"/> <c>[N]</c> with spline parameters <c>[N, K]</c>, <c>[N, K]</c> and
    /// <c>[N, K − 1]</c>; returns the outputs and the log |dy/dx| (negated for the inverse), both <c>[N]</c>.</summary>
    public static (Tensor<T> Outputs, Tensor<T> LogAbsDet) Apply<T>(IEngine engine, Tensor<T> inputs, Tensor<T> unnormalizedWidths,
        Tensor<T> unnormalizedHeights, Tensor<T> unnormalizedDerivatives, bool inverse, double tailBound)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = inputs.Length, k = unnormalizedWidths.Shape[1];
        T C(double v) => ops.FromDouble(v);

        // Inside mask (constant) and a safe input for the elements outside, whose spline value is discarded.
        var inside = new Tensor<T>(new[] { n });
        for (int i = 0; i < n; i++)
        {
            double x = ops.ToDouble(inputs[i]);
            inside[i] = x >= -tailBound && x <= tailBound ? ops.One : ops.Zero;
        }
        var outside = engine.TensorAddScalar(engine.TensorNegate(inside), ops.One);
        var safe = engine.TensorMultiply(inputs, inside);

        // Linear tails: the outer derivatives are the constant whose softplus is 1 − min_derivative.
        double tailDerivative = Math.Log(Math.Exp(1 - MinDerivative) - 1);
        var edge = new Tensor<T>(new[] { n, 1 });
        for (int i = 0; i < n; i++) edge[i, 0] = C(tailDerivative);
        var derivatives = engine.TensorAddScalar(engine.Softplus(engine.TensorConcatenate(new[] { edge, unnormalizedDerivatives, edge }, 1)), C(MinDerivative));

        Tensor<T> Knots(Tensor<T> unnormalized, double floor)
        {
            var sizes = engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorSoftmax(unnormalized, axis: 1), C(1 - floor * k)), C(floor));
            // cumsum with a leading zero: sizes · U, U[j, i] = 1 for j < i, of shape [K, K + 1]
            var upper = new Tensor<T>(new[] { k, k + 1 });
            for (int j = 0; j < k; j++)
                for (int i = j + 1; i <= k; i++) upper[j, i] = ops.One;
            return engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorMatMul(sizes, upper), C(2 * tailBound)), C(-tailBound));
        }
        var cumWidths = Knots(unnormalizedWidths, MinBinWidth);                 // [N, K + 1]
        var cumHeights = Knots(unnormalizedHeights, MinBinHeight);

        // Bin search over the current knots (searchsorted with the last knot nudged up).
        var search = inverse ? cumHeights : cumWidths;
        var oneHot = new Tensor<T>(new[] { n, k });
        for (int i = 0; i < n; i++)
        {
            double x = ops.ToDouble(safe[i]);
            int bin = 0;
            for (int j = 0; j <= k; j++)
            {
                double knot = ops.ToDouble(search[i, j]) + (j == k ? 1e-6 : 0);
                if (x >= knot) bin = j;
            }
            oneHot[i, Math.Min(bin, k - 1)] = ops.One;
        }
        Tensor<T> Pick(Tensor<T> values, int from) =>
            engine.ReduceSum(engine.TensorMultiply(engine.TensorSlice(values, new[] { 0, from }, new[] { n, k }), oneHot), new[] { 1 }, keepDims: false);

        var x0 = Pick(cumWidths, 0);
        var binWidth = engine.TensorSubtract(Pick(cumWidths, 1), x0);
        var y0 = Pick(cumHeights, 0);
        var binHeight = engine.TensorSubtract(Pick(cumHeights, 1), y0);
        var delta = engine.TensorDivide(binHeight, binWidth);
        var d0 = Pick(derivatives, 0);
        var d1 = Pick(derivatives, 1);
        var curvature = engine.TensorSubtract(engine.TensorAdd(d0, d1), engine.TensorMultiplyScalar(delta, C(2)));

        Tensor<T> outputs, logAbsDet;
        if (!inverse)
        {
            var theta = engine.TensorDivide(engine.TensorSubtract(safe, x0), binWidth);
            var oneMinus = engine.TensorAddScalar(engine.TensorNegate(theta), ops.One);
            var tt = engine.TensorMultiply(theta, oneMinus);
            var numerator = engine.TensorMultiply(binHeight, engine.TensorAdd(engine.TensorMultiply(delta, engine.TensorMultiply(theta, theta)),
                engine.TensorMultiply(d0, tt)));
            var denominator = engine.TensorAdd(delta, engine.TensorMultiply(curvature, tt));
            outputs = engine.TensorAdd(y0, engine.TensorDivide(numerator, denominator));
            var derivativeNumerator = engine.TensorMultiply(engine.TensorMultiply(delta, delta),
                engine.TensorAdd(engine.TensorAdd(engine.TensorMultiply(d1, engine.TensorMultiply(theta, theta)),
                    engine.TensorMultiplyScalar(engine.TensorMultiply(delta, tt), C(2))),
                    engine.TensorMultiply(d0, engine.TensorMultiply(oneMinus, oneMinus))));
            logAbsDet = engine.TensorSubtract(engine.TensorLog(derivativeNumerator), engine.TensorMultiplyScalar(engine.TensorLog(denominator), C(2)));
        }
        else
        {
            var shifted = engine.TensorSubtract(safe, y0);
            var a = engine.TensorAdd(engine.TensorMultiply(shifted, curvature), engine.TensorMultiply(binHeight, engine.TensorSubtract(delta, d0)));
            var b = engine.TensorSubtract(engine.TensorMultiply(binHeight, d0), engine.TensorMultiply(shifted, curvature));
            var c = engine.TensorNegate(engine.TensorMultiply(delta, shifted));
            var discriminant = engine.TensorSubtract(engine.TensorMultiply(b, b), engine.TensorMultiplyScalar(engine.TensorMultiply(a, c), C(4)));
            var root = engine.TensorDivide(engine.TensorMultiplyScalar(c, C(2)),
                engine.TensorSubtract(engine.TensorNegate(b), engine.TensorPow(engine.ReLU(discriminant), C(0.5))));
            outputs = engine.TensorAdd(engine.TensorMultiply(root, binWidth), x0);
            var oneMinus = engine.TensorAddScalar(engine.TensorNegate(root), ops.One);
            var tt = engine.TensorMultiply(root, oneMinus);
            var denominator = engine.TensorAdd(delta, engine.TensorMultiply(curvature, tt));
            var derivativeNumerator = engine.TensorMultiply(engine.TensorMultiply(delta, delta),
                engine.TensorAdd(engine.TensorAdd(engine.TensorMultiply(d1, engine.TensorMultiply(root, root)),
                    engine.TensorMultiplyScalar(engine.TensorMultiply(delta, tt), C(2))),
                    engine.TensorMultiply(d0, engine.TensorMultiply(oneMinus, oneMinus))));
            logAbsDet = engine.TensorNegate(engine.TensorSubtract(engine.TensorLog(derivativeNumerator),
                engine.TensorMultiplyScalar(engine.TensorLog(denominator), C(2))));
        }
        return (engine.TensorAdd(engine.TensorMultiply(outputs, inside), engine.TensorMultiply(inputs, outside)),
            engine.TensorMultiply(logAbsDet, inside));
    }
}
