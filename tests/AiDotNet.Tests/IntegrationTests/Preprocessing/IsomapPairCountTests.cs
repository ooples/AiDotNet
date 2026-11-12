using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection.Manifold;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the Isomap selectors' connected-pair count, which is kept in int.
/// </summary>
public class IsomapPairCountTests
{
    // 260 points on a line (feature 1 = 0.001 i) with a constant feature 0: 260 * 259 = 67340 connected pairs. The
    // former count was a TensorSum of the pair mask in Half, which is +infinity past 65504, so every mean score was
    // finite / infinity = 0 and the tie went to feature 0. Feature 1 carries all the geodesic structure.
    [Fact(Timeout = 120000)]
    public async Task IsomapBasedSelector_Half_ScoresPastHalfsLargestPairCount()
    {
        await Task.CompletedTask;
        const int n = 260;
        var data = new Half[n, 2];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = (Half)0;
            data[i, 1] = (Half)(0.001 * i);
        }

        var selector = new IsomapBasedSelector<Half>(nFeaturesToSelect: 1, nNeighbors: 5);
        selector.Fit(new Matrix<Half>(data));

        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
        var scores = selector.FeatureScores;
        Assert.NotNull(scores);
        if (scores is not null)
        {
            Assert.Equal(0.0, (double)scores[0]);
            Assert.Equal(1.0, (double)scores[1]);
        }
    }
}
