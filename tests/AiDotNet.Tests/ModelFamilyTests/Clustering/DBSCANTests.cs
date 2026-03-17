using AiDotNet.Interfaces;
using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class DBSCANTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new DBSCAN<double>(new DBSCANOptions<double>
        {
            Epsilon = 2.0,
            MinPoints = 3
        });

    // DBSCAN is density-based — doesn't have centroid parameters
    protected override bool HasFlatParameters => false;
}
