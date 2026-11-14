using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// The hypernetwork of <see cref="TaskCondHyperNetAlgorithm{T, TInput, TOutput}"/> is trained by backpropagating the
/// first-order meta-gradient through delta = 0.01 W_c tanh(W_h e + b_h) (Ha et al., "HyperNetworks", 2016, trained
/// end to end). It used SPSA. This checks the analytic weight gradient against central finite differences.
/// </summary>
public class TaskCondHyperNetGradientTests
{
    [Fact]
    public void WeightGradient_MatchesFiniteDifferences()
    {
        // 4 target parameters, chunk size 3: two chunks, the second one partial.
        var algorithm = new TaskCondHyperNetAlgorithm<double, Matrix<double>, Vector<double>>(
            new TaskCondHyperNetOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3))
            {
                EmbeddingDim = 2, HyperHiddenDim = 3, ChunkSize = 3, Seed = 9,
            });
        var embedding = new Vector<double>(new[] { 0.7, -1.3 });
        var g = new Vector<double>(new[] { 0.5, -2.0, 1.0, 3.0 });

        var (_, gradient) = algorithm.HyperNetworkProbe(embedding, g);
        var weights = algorithm.HyperNetWeights.Clone();
        Assert.Equal(weights.Length, gradient.Length);

        const double h = 1e-6;
        for (int i = 0; i < weights.Length; i++)
        {
            double Objective(double shift)
            {
                var w = weights.Clone();
                w[i] += shift;
                algorithm.HyperNetWeights = w;
                var (delta, _) = algorithm.HyperNetworkProbe(embedding, g);
                double dot = 0;
                for (int d = 0; d < g.Length; d++) dot += g[d] * delta[d];
                return dot;
            }

            double numeric = (Objective(h) - Objective(-h)) / (2 * h);
            Assert.Equal(numeric, gradient[i], 7);
        }

        algorithm.HyperNetWeights = weights;
    }
}
