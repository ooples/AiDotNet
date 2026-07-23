using AiDotNet.Interfaces;
using AiDotNet.Clustering.Subspace;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class SUBCLUTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new SUBCLU<double>(new AiDotNet.Clustering.Options.SUBCLUOptions<double>
        {
            Epsilon = 0.5, // After normalization, data has std=1, clusters well-separated
            MinPoints = 3
        });

    // For single-cluster data: after SUBCLU's internal normalization to unit variance,
    // the 90 points span [-3,+3] in each dimension. Epsilon=0.5 is too small for 3D
    // nearest-neighbor distances (~1.7 expected). Use a larger epsilon so all points
    // are density-connected.
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateSingleClusterModel()
        => new SUBCLU<double>(new AiDotNet.Clustering.Options.SUBCLUOptions<double>
        {
            Epsilon = 2.0,
            MinPoints = 3
        });

    // SUBCLU uses fixed epsilon — not scale-invariant by design
    protected override bool HasFlatParameters => false;
}
