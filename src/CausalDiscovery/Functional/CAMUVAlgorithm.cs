using AiDotNet.Extensions;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CAM-UV — Causal Additive Model with Unobserved Variables.
/// </summary>
/// <remarks>
/// <para>
/// CAM-UV extends CAM to handle latent (unobserved) confounders. It discovers the causal
/// structure among observed variables even when some common causes are hidden. The algorithm:
/// <list type="number">
/// <item>Fits pairwise additive noise models between all variable pairs.</item>
/// <item>Identifies potential latent confounders by detecting pairs where residuals in
/// both directions show high dependence (neither direction fits well).</item>
/// <item>Marks bidirectional edges for pairs with suspected latent confounders.</item>
/// <item>Orients remaining edges using the standard ANM asymmetry criterion.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Sometimes two variables appear related not because one causes the
/// other, but because a hidden third variable causes both. CAM-UV detects these situations
/// by checking: if neither direction X→Y nor Y→X fits cleanly, there might be a hidden
/// common cause. It marks such pairs as "confounded" rather than forcing a causal direction.
/// </para>
/// <para>Reference: Maeda and Shimizu (2021), "Causal Additive Models with Unobserved Variables".</para>
/// </remarks>
internal class CAMUVAlgorithm<T> : FunctionalBase<T>
{
    private double _threshold = 0.1;
    private double _confoundingThreshold = 0.3;

    /// <inheritdoc/>
    public override string Name => "CAM-UV";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CAMUVAlgorithm(CausalDiscoveryOptions? options = null)
    {
        if (options?.EdgeThreshold.HasValue == true) _threshold = options.EdgeThreshold.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int d = data.Columns;
        var standardized = StandardizeData(data);

        var W = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                var xi = standardized.GetColumn(i);
                var xj = standardized.GetColumn(j);

                double weight = Math.Abs(ComputeCorrelation(xi, xj));
                if (weight < _threshold) continue;

                // Test i → j
                var residIJ = KernelRegressOut(xi, xj);
                double depIJ = Math.Abs(GaussianMI(residIJ, xi));

                // Test j → i
                var residJI = KernelRegressOut(xj, xi);
                double depJI = Math.Abs(GaussianMI(residJI, xj));

                bool suspectConfounder = depIJ > _confoundingThreshold && depJI > _confoundingThreshold;

                T weightT = NumOps.FromDouble(weight);
                if (suspectConfounder)
                {
                    W[i, j] = weightT;
                    W[j, i] = weightT;
                }
                else
                {
                    double asymmetry = depJI - depIJ;
                    if (Math.Abs(asymmetry) > _threshold * 0.1)
                    {
                        if (asymmetry > 0)
                            W[i, j] = weightT;
                        else
                            W[j, i] = weightT;
                    }
                }
            }
        }

        return W;
    }
}
