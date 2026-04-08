using AiDotNet.Interfaces;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class AffinityPropagationTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new AffinityPropagation<double>(new AffinityPropagationOptions<double>
        {
            // No explicit preference — auto-computed from median similarity
            // works better after normalization
            Damping = 0.9,
            MaxIterations = 500
        });

    // For single-cluster data: with tightly grouped points, median similarity is high
    // and every point looks like a potential exemplar, producing too many clusters.
    // Per Frey & Dueck 2007, use a very low preference (minimum similarity) to
    // encourage fewer exemplars. Also increase damping for convergence stability.
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateSingleClusterModel()
        => new AffinityPropagation<double>(new AffinityPropagationOptions<double>
        {
            Preference = -100.0, // Strong penalty for being an exemplar → fewer clusters
            Damping = 0.95,
            MaxIterations = 500
        });
}
