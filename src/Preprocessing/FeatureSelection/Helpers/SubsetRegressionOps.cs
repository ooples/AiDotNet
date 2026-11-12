using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Engine-op least-squares helpers of the wrapper feature selectors: subset design matrices, ridge-stabilized
/// coefficients and the coefficient of determination.
/// </summary>
internal static class SubsetRegressionOps
{
    /// <summary>The given columns of <paramref name="x"/> [n, p], in order ([n, k]).</summary>
    public static Tensor<T> Columns<T>(Tensor<T> x, IReadOnlyList<int> columns) =>
        AiDotNetEngine.Current.TensorIndexSelect(x, new Tensor<int>(columns.ToArray(), new[] { columns.Count }), 1);

    /// <summary>The given rows of <paramref name="x"/> (rank 1 or 2), in order.</summary>
    public static Tensor<T> Rows<T>(Tensor<T> x, IReadOnlyList<int> rows) =>
        AiDotNetEngine.Current.TensorIndexSelect(x, new Tensor<int>(rows.ToArray(), new[] { rows.Count }), 0);

    /// <summary>A leading column of ones followed by <paramref name="design"/> [n, k] ([n, k + 1]).</summary>
    public static Tensor<T> WithIntercept<T>(Tensor<T> design)
    {
        var engine = AiDotNetEngine.Current;
        int n = design.Shape[0];
        var ones = FeatureSelectionTensorOps.Filled(new[] { n, 1 }, MathHelper.GetNumericOperations<T>().One);
        return engine.TensorConcatenate(new[] { ones, design }, 1);
    }

    /// <summary>The coefficients (D^T D + ridge I)^-1 D^T y of a design D [n, k] and response y [n] ([k]).</summary>
    public static Tensor<T> RidgeCoefficients<T>(Tensor<T> design, Tensor<T> y, double ridge)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = design.Shape[0];
        int k = design.Shape[1];
        var transposed = engine.TensorTranspose(design);
        var gram = engine.TensorAdd(
            engine.TensorMatMul(transposed, design), engine.TensorMultiplyScalar(engine.TensorEye<T>(k), ops.FromDouble(ridge)));
        var rhs = engine.TensorMatMul(transposed, engine.Reshape(y, new[] { n, 1 }));
        return engine.Reshape(LinalgDispatch<T>.Solve(gram, rhs), new[] { k });
    }

    /// <summary>The predictions D c of a design [n, k] and coefficients [k] ([n]).</summary>
    public static Tensor<T> Predict<T>(Tensor<T> design, Tensor<T> coefficients)
    {
        var engine = AiDotNetEngine.Current;
        int n = design.Shape[0];
        int k = design.Shape[1];
        return engine.Reshape(engine.TensorMatMul(design, engine.Reshape(coefficients, new[] { k, 1 })), new[] { n });
    }

    /// <summary>
    /// The intercept-first coefficients [p + 1] of a least-squares fit (ridge 1e-6) of y [n] on all columns of
    /// <paramref name="x"/> [n, p].
    /// </summary>
    public static Tensor<T> FitLinear<T>(Tensor<T> x, Tensor<T> y) => RidgeCoefficients(WithIntercept(x), y, 1e-6);

    /// <summary>
    /// The training R^2 (about the mean of y) of the linear model with intercept-first
    /// <paramref name="coefficients"/> on <paramref name="x"/> [n, p]; 0 for a constant target. Returns [1].
    /// </summary>
    public static Tensor<T> LinearRSquared<T>(Tensor<T> x, Tensor<T> y, Tensor<T> coefficients)
    {
        var engine = AiDotNetEngine.Current;
        var prediction = Predict(WithIntercept(x), coefficients);
        return RSquared(y, prediction, engine.ReduceMean(y, new[] { 0 }, true), out _);
    }

    /// <summary>
    /// 1 - SS_res / SS_tot as [1], with SS_tot = sum (y - <paramref name="center"/>)^2 for a [1] center, or 0 when
    /// SS_tot is at most 1e-10. <paramref name="valid"/> receives whether SS_tot exceeded 1e-10.
    /// </summary>
    public static Tensor<T> RSquared<T>(Tensor<T> y, Tensor<T> prediction, Tensor<T> center, out bool valid)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = y.Length;
        var ssRes = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(y, prediction)), new[] { 0 }, true);
        var ssTot = engine.ReduceSum(
            engine.TensorSquare(engine.TensorSubtract(y, engine.TensorBroadcastTo(center, new[] { n }))), new[] { 0 }, true);
        valid = PopulationSearchOps.Greater(ssTot, FeatureSelectionTensorOps.Filled(new[] { 1 }, ops.FromDouble(1e-10)));
        return valid
            ? FeatureSelectionTensorOps.OneMinus(engine.TensorDivide(ssRes, ssTot))
            : new Tensor<T>(new[] { 1 });
    }
}
