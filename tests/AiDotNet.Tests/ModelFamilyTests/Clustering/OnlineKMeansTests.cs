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
            LearningRate = 0.2,
            DecayLearningRate = false,
            MaxIterations = 10, // Multiple passes for convergence
            Seed = 42
        });
}
