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
            Epsilon = 3.0, // Data has clusters at spacing=10 with std=1
            MinPoints = 3
        });

    // SUBCLU uses fixed epsilon — not scale-invariant by design
    protected override bool HasFlatParameters => false;
}
