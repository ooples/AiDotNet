using AiDotNet.Interfaces;
using AiDotNet.Clustering.Streaming;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class OnlineKMeansTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new OnlineKMeans<double>(new AiDotNet.Clustering.Options.OnlineKMeansOptions<double>
        {
            NumClusters = NumClusters,
            LearningRate = 0.5, // Higher rate for better convergence in single pass
            DecayLearningRate = false, // Don't decay during single-pass batch training
            Seed = 42
        });
}
