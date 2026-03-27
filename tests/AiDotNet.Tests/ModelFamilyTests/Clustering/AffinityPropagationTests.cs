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
            // No explicit preference — auto-computed from median similarity
            // works better after normalization
            Damping = 0.9,
            MaxIterations = 500
        });
}
