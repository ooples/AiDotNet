using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Engine-op FastICA (Hyvarinen 1999) for the ICA-based feature selectors: PCA whitening and the deflation
/// scheme of one-unit fixed-point iterations with the tanh contrast.
/// </summary>
internal static class FastIcaOps
{
    /// <summary>
    /// PCA whitening of the centered data [n, p] to <paramref name="dimensions"/> dimensions: z = x K with
    /// K = E D^(-1/2) [p, d] from the top eigenpairs of the covariance x^T x / n, so z has identity covariance.
    /// Directions without variance are dropped (numpy.linalg.matrix_rank's cutoff, largest eigenvalue * p * eps):
    /// flooring their eigenvalue instead gave a constant feature a whitening gain of 1e5. d is therefore at most
    /// <paramref name="dimensions"/> and is the column count of the result; it is 0 for constant data.
    /// Half and BFloat16 whiten in float, since the covariance of representable data passes 65504.
    /// </summary>
    public static Tensor<T> Whiten<T>(Tensor<T> centered, int dimensions, out Tensor<T> whitening)
    {
        if (PreprocessingTensorOps.IsNarrowFloat<T>())
        {
            var wide = Whiten(centered.Cast<float>(), dimensions, out var wideWhitening);
            whitening = wideWhitening.Cast<T>();
            return wide.Cast<T>();
        }

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = centered.Shape[0];
        int p = centered.Shape[1];
        // divided by sqrt(n) before the product, which keeps x^T x / n finite where x^T x is not
        var scaled = engine.TensorDivideScalar(centered, ops.FromDouble(Math.Sqrt(n)));
        var covariance = engine.TensorMatMul(engine.TensorTranspose(scaled), scaled);
        var vectors = FeatureSelectionTensorOps.TopEigenpairs(covariance, dimensions, out var values);

        // values are in descending order, so the retained directions are a prefix
        double largest = values.Length > 0 ? Math.Max(ops.ToDouble(values[0]), 0.0) : 0.0;
        double epsilon = typeof(T) == typeof(float) ? 1.1920928955078125e-7 : 2.220446049250313e-16;
        var retained = PreprocessingTensorOps.MaskPositions(
            engine.TensorGreaterThan(values, ops.FromDouble(largest * Math.Max(p, 1) * epsilon)));
        int d = retained.Length;
        if (d == 0)
        {
            whitening = new Tensor<T>(new[] { p, 0 });
            return new Tensor<T>(new[] { n, 0 });
        }

        if (d < values.Length)
        {
            var keep = new Tensor<int>(retained, new[] { d });
            values = engine.TensorIndexSelect(values, keep, 0);
            vectors = engine.TensorIndexSelect(vectors, keep, 1);
        }

        var inverseRoot = engine.TensorDivide(
            PreprocessingTensorOps.Filled(new[] { d }, ops.One), engine.TensorSqrt(values));
        whitening = engine.TensorMultiply(
            vectors, engine.TensorBroadcastTo(engine.Reshape(inverseRoot, new[] { 1, d }), new[] { p, d }));
        return engine.TensorMatMul(centered, whitening);
    }

    /// <summary>
    /// Deflationary FastICA on whitened data z [n, d]: component c starts from column c of
    /// <paramref name="initial"/> [d, k] (normalized) and iterates w = E[z g(w^T z)] - E[g'(w^T z)] w with
    /// g = tanh, Gram-Schmidt against the earlier components and normalization, for at most
    /// <paramref name="maxIterations"/> steps or until sum_j | |w_new,j| - |w_j| | &lt; <paramref name="tolerance"/>
    /// (no early stop for a tolerance of 0). Returns the unmixing vectors as the columns of [d, k].
    /// </summary>
    public static Tensor<T> Deflation<T>(Tensor<T> z, Tensor<T> initial, int maxIterations, double tolerance)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = z.Shape[0];
        int d = z.Shape[1];
        int k = initial.Shape[1];
        T invN = ops.FromDouble(1.0 / n);
        var tol = PreprocessingTensorOps.Filled(new[] { 1 }, ops.FromDouble(tolerance));
        var zt = engine.TensorTranspose(z);

        var components = new List<Tensor<T>>(k);
        for (int c = 0; c < k; c++)
        {
            var w = Normalize(engine.Reshape(engine.TensorSlice(initial, new[] { 0, c }, new[] { d, 1 }), new[] { d }));
            var previous = c == 0
                ? null
                : engine.TensorConcatenate(components.Select(v => engine.Reshape(v, new[] { d, 1 })).ToArray(), 1);

            for (int iter = 0; iter < maxIterations; iter++)
            {
                var projection = engine.Reshape(engine.TensorMatMul(z, engine.Reshape(w, new[] { d, 1 })), new[] { n });
                var g = engine.TensorTanh(projection);
                var gPrime = PreprocessingTensorOps.OneMinus(engine.TensorSquare(g));
                var first = engine.TensorMultiplyScalar(
                    engine.Reshape(engine.TensorMatMul(zt, engine.Reshape(g, new[] { n, 1 })), new[] { d }), invN);
                var meanSlope = engine.ReduceMean(gPrime, new[] { 0 }, true);
                var next = engine.TensorSubtract(first, PopulationSearchOps.Scale(w, meanSlope));

                // Gram-Schmidt against the earlier (orthonormal) components
                if (previous is not null)
                {
                    var overlap = engine.TensorMatMul(engine.TensorTranspose(previous), engine.Reshape(next, new[] { d, 1 }));
                    next = engine.TensorSubtract(next, engine.Reshape(engine.TensorMatMul(previous, overlap), new[] { d }));
                }

                next = Normalize(next);
                var change = engine.ReduceSum(
                    engine.TensorAbs(engine.TensorSubtract(engine.TensorAbs(next), engine.TensorAbs(w))), new[] { 0 }, true);
                w = next;
                if (tolerance > 0 && PopulationSearchOps.Greater(tol, change))
                    break;
            }

            components.Add(w);
        }

        return engine.TensorConcatenate(components.Select(v => engine.Reshape(v, new[] { d, 1 })).ToArray(), 1);
    }

    /// <summary>The vector divided by its Euclidean norm, or unchanged when the norm is at most 1e-10.</summary>
    private static Tensor<T> Normalize<T>(Tensor<T> v)
    {
        var engine = AiDotNetEngine.Current;
        var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(v), new[] { 0 }, true));
        return PopulationSearchOps.Greater(norm, PreprocessingTensorOps.Filled(new[] { 1 }, PreprocessingTensorOps.Tolerance<T>(1e-10)))
            ? engine.TensorDivide(v, engine.TensorBroadcastTo(norm, new[] { v.Length }))
            : v;
    }
}
