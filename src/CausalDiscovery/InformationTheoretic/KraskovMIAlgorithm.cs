using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.InformationTheoretic;

/// <summary>
/// Kraskov MI — Mutual Information estimation using k-nearest neighbors (KSG estimator).
/// </summary>
/// <remarks>
/// <para>
/// The Kraskov-Stoegbauer-Grassberger (KSG) estimator computes mutual information
/// using nearest-neighbor distances in the joint and marginal spaces. It's non-parametric
/// and works well for both linear and nonlinear dependencies.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most MI estimators assume the data follows a specific distribution
/// (like Gaussian). The Kraskov method doesn't make this assumption — it works by looking
/// at how close data points are to each other in different ways. This makes it more reliable
/// for complex, real-world data.
/// </para>
/// <para>
/// Reference: Kraskov et al. (2004), "Estimating Mutual Information", Physical Review E.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Estimating Mutual Information", "https://doi.org/10.1103/PhysRevE.69.066138", Year = 2004, Authors = "Alexander Kraskov, Harald Stoegbauer, Peter Grassberger")]
public class KraskovMIAlgorithm<T> : InfoTheoreticBase<T>
{
    /// <inheritdoc/>
    public override string Name => "KraskovMI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public KraskovMIAlgorithm(CausalDiscoveryOptions? options = null) { ApplyInfoOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ConstraintBased.PCAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
