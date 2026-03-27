using AiDotNet.Interfaces;
using AiDotNet.Clustering.Density;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class DenclueTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new Denclue<double>(new AiDotNet.Clustering.Options.DenclueOptions<double>
        {
            Bandwidth = 0.8, // Tuned for normalized data (std=1, cluster spacing ~2-3 sigma)
            AttractorMergeThreshold = 1.0
        });

    // DENCLUE is density-based — doesn't have centroid parameters
    protected override bool HasFlatParameters => false;
}
