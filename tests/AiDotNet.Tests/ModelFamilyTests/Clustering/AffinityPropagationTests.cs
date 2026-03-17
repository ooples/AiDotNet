using AiDotNet.Interfaces;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class AffinityPropagationTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new AffinityPropagation<double>(new AiDotNet.Clustering.Options.AffinityPropagationOptions<double>
        {
            Preference = -500.0, // Strong preference for fewer clusters
            Damping = 0.9, // High damping for stability
            MaxIterations = 500
        });
}
