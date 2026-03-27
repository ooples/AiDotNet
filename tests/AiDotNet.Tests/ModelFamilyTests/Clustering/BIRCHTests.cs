using AiDotNet.Interfaces;
using AiDotNet.Clustering.Hierarchical;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class BIRCHTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new BIRCH<double>(new AiDotNet.Clustering.Options.BIRCHOptions<double>
        {
            NumClusters = NumClusters,
            Threshold = 5.0 // Data has cluster spacing of 10, std=1
        });
}
