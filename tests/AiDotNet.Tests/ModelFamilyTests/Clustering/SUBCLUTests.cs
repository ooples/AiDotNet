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

    // SUBCLU uses fixed epsilon — not scale-invariant by design
    protected override bool HasFlatParameters => false;
}
