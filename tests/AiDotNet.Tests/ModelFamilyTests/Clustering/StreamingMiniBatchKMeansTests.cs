using AiDotNet.Interfaces;
using AiDotNet.Clustering.Streaming;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class StreamingMiniBatchKMeansTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new MiniBatchKMeans<double>(new AiDotNet.Clustering.Options.MiniBatchKMeansOptions<double>
        {
            NumClusters = NumClusters,
            BatchSize = 30, // Appropriate for 90 training samples
            Seed = 42
        });
}
