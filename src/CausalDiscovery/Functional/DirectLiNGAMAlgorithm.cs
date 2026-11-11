using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// DirectLiNGAM — direct method for LiNGAM without ICA.
/// </summary>
/// <remarks>
/// <para>
/// DirectLiNGAM avoids the ICA step entirely and instead uses a direct regression-based
/// approach to identify the causal ordering. It iteratively finds the "root" variable
/// (the one with the most independent residuals) and removes its effect.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Find the variable with minimum dependence on others (root cause)</item>
/// <item>Regress out the root's effect from all remaining variables</item>
/// <item>Repeat on the residuals until all variables are ordered</item>
/// <item>Estimate connection strengths via OLS in the causal order</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> DirectLiNGAM finds causal structure step by step. First, it finds
/// the variable that seems to cause others but isn't caused by anything (the "root").
/// Then it removes that variable's influence and repeats. This gives a causal ordering
/// from which the full structure follows naturally.
/// </para>
/// <para>
/// Reference: Shimizu et al. (2011), "DirectLiNGAM: A Direct Method for Learning a
/// Linear Non-Gaussian Structural Equation Model", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model", "https://jmlr.org/papers/v12/shimizu11a.html", Year = 2011, Authors = "Shohei Shimizu, Takanori Inazumi, Yasuhiro Sogawa, Aapo Hyvarinen, Yoshinobu Kawahara, Takashi Washio, Patrik O. Hoyer, Kenneth Bollen")]
public class DirectLiNGAMAlgorithm<T> : FunctionalBase<T>
{
    private double _threshold = 0.1;

    /// <summary>
    /// The structural coefficients from the most recent run, BEFORE the edge threshold is applied.
    /// Null until <c>DiscoverStructure</c> has run.
    /// </summary>
    /// <remarks>
    /// The returned graph is the thresholded view, which is the right thing to report. But an
    /// algorithm that does ALGEBRA with B_0 -- VARLiNGAM forms (I - B_0) M_tau -- needs the real
    /// signed values: treating every sub-threshold coefficient as an exact zero biases the product.
    /// Internal because it is a detail of that collaboration, not public API.
    /// </remarks>
    internal Matrix<T>? RawCoefficients { get; private set; }


    /// <inheritdoc/>
    public override string Name => "DirectLiNGAM";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes DirectLiNGAM with optional configuration.
    /// </summary>
    public DirectLiNGAMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        if (options?.EdgeThreshold.HasValue == true) _threshold = options.EdgeThreshold.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        var standardized = StandardizeData(data);

        var remaining = Enumerable.Range(0, d).ToList();
        var causalOrder = new List<int>();
        var currentData = standardized;

        // Iteratively find root and regress out
        while (remaining.Count > 0)
        {
            // Find the variable with minimum total dependence on remaining variables
            int root = -1;
            double minDep = double.MaxValue;

            foreach (int j in remaining)
            {
                double totalDep = 0;
                var col = currentData.GetColumn(j);

                foreach (int k in remaining)
                {
                    if (k == j) continue;
                    var resid = RegressOut(col, currentData.GetColumn(k));

                    // Measure non-Gaussianity of residuals (kurtosis-based independence)
                    double kurtRes = Math.Abs(ComputeKurtosis(resid) - 3.0);
                    double kurtOrig = Math.Abs(ComputeKurtosis(col) - 3.0);
                    totalDep += Math.Abs(kurtOrig - kurtRes);
                }

                if (totalDep < minDep)
                {
                    minDep = totalDep;
                    root = j;
                }
            }

            if (root < 0) root = remaining[0];

            causalOrder.Add(root);
            remaining.Remove(root);

            // Regress out root from remaining variables
            if (remaining.Count > 0)
            {
                // Every remaining column is replaced by its residual after regressing on the root
                // (the centered RegressOut residual, for all columns at once).
                var current = Tensor<T>.FromMatrix(currentData);
                var centered = CenterColumns(current);
                var rootCentered = Engine.TensorSliceAxis(centered, 1, root).Reshape(new[] { n, 1 });
                T sxx = Engine.TensorSum(Engine.TensorSquare(rootCentered));
                var sxy = Engine.TensorMatMul(Engine.TensorTranspose(rootCentered), centered);
                var beta = NumOps.GreaterThan(sxx, NumOps.FromDouble(1e-10))
                    ? Engine.TensorDivideScalar(sxy, sxx)
                    : Engine.TensorZerosLike(sxy);
                var residuals = Engine.TensorSubtract(centered, Engine.TensorMatMul(rootCentered, beta));

                var remainingIndex = new Tensor<int>(remaining.ToArray(), new[] { remaining.Count });
                var replace = Engine.TensorScatter(new Tensor<T>(new[] { d }), remainingIndex,
                    Engine.TensorOnesLike(new Tensor<T>(new[] { remaining.Count })), 0).Reshape(new[] { 1, d });
                currentData = Engine.TensorWhere(Engine.TensorBroadcastTo(replace, new[] { n, d }), residuals, current)
                    .ToMatrix();
            }
        }

        // Estimate B matrix using OLS in causal order
        // The UNTHRESHOLDED coefficients are kept alongside the thresholded graph. VARLiNGAM forms
        // (I - B_0) M_tau from B_0, and a matrix operation on a thresholded matrix biases its result:
        // every coefficient below the edge threshold is treated as an exact zero rather than as the
        // small value it is. Thresholding belongs to the reported GRAPH, not to the algebra.
        // Simple regression coefficient of every column j on every column p: G[p, j] / G[p, p] with
        // G the centered Gram matrix, 0 when G[p, p] is not above 1e-10.
        var standardCentered = CenterColumns(Tensor<T>.FromMatrix(standardized));
        var gram = Engine.TensorMatMul(Engine.TensorTranspose(standardCentered), standardCentered);
        var sxxColumn = Engine.TensorDiagonal(gram).Reshape(new[] { d, 1 });
        var validRow = Engine.TensorBroadcastTo(Engine.TensorGreaterThan(sxxColumn, NumOps.FromDouble(1e-10)), new[] { d, d });
        var safeSxx = Engine.TensorWhere(validRow, Engine.TensorBroadcastTo(sxxColumn, new[] { d, d }),
            Engine.TensorOnesLike(gram));
        var coefficients = Engine.TensorWhere(validRow, Engine.TensorDivide(gram, safeSxx), Engine.TensorZerosLike(gram));

        // Only parent -> j pairs with the parent earlier in the causal order carry a coefficient.
        var earlierPairs = new List<int>();
        for (int idx = 1; idx < d; idx++)
            for (int pidx = 0; pidx < idx; pidx++)
                earlierPairs.Add(causalOrder[pidx] * d + causalOrder[idx]);
        var raw = CausalWeightScatter.KeepPositions(coefficients, earlierPairs);

        var rawTensor = Tensor<T>.FromMatrix(raw);
        var keep = Engine.TensorGreaterEqual(Engine.TensorAbs(rawTensor),
            Engine.TensorFullLike(rawTensor, NumOps.FromDouble(_threshold)));
        RawCoefficients = raw;
        return Engine.TensorWhere(keep, rawTensor, Engine.TensorZerosLike(rawTensor)).ToMatrix();
    }
}
