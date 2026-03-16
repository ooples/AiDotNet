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
            // Data has clusters at 0, 10, 20 with std=0.5. Points within a cluster
            // are within ~2 Euclidean distance (sqrt(3) * 0.5 ≈ 0.87 for 3 features).
            // Default epsilon=0.5 is too small for this data scale.
            Epsilon = 2.0,
            MinPoints = 3
        });

    // DBSCAN is density-based — doesn't have centroid parameters
    protected override bool HasFlatParameters => false;
}
