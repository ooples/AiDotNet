using AiDotNet.Augmentation;
using AiDotNet.Augmentation.Tabular;
using AiDotNet.Augmentation.Tabular.Undersampling;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Augmentation;

/// <summary>
/// Reference tests for the tabular resamplers against Zhang et al.'s mixup code and imbalanced-learn.
/// </summary>
public class TabularResamplingReferenceTests
{
    private static Matrix<double> Column(params double[] values)
    {
        var m = new Matrix<double>(values.Length, 1);
        for (int i = 0; i < values.Length; i++)
        {
            m[i, 0] = values[i];
        }

        return m;
    }

    [Fact(Timeout = 60000)]
    public async Task MixUp_PartnersFormAPermutation()
    {
        // Zhang et al.: x~ = lambda x + (1 - lambda) x[randperm(n)], so every row is a partner exactly once.
        // Rows are distinct powers of ten, so each partner can be recovered from the mixed row.
        const int rows = 6;
        var values = Enumerable.Range(0, rows).Select(i => Math.Pow(10, i)).ToArray();
        var data = Column(values);
        var mixup = new TabularMixUp<double>(alpha: 1.0, probability: 1.0);

        for (int seed = 0; seed < 20; seed++)
        {
            var mixed = mixup.Apply(data, new AugmentationContext<double>(seed: seed));
            double lambda = mixup.LastMixingLambda;
            if (Math.Abs(1 - lambda) < 1e-6)
            {
                continue;
            }

            var partners = new List<int>();
            for (int i = 0; i < rows; i++)
            {
                double partner = (mixed[i, 0] - lambda * values[i]) / (1 - lambda);
                int index = Array.FindIndex(values, v => Math.Abs(v - partner) < 1e-6 * Math.Max(1, v));
                Assert.True(index >= 0, $"seed {seed}: row {i} has no partner");
                partners.Add(index);
            }

            Assert.Equal(Enumerable.Range(0, rows), partners.OrderBy(p => p));
        }
    }

    [Fact(Timeout = 60000)]
    public async Task TomekLinks_LinksAnyTwoClassesNotOnlyMinorityVersusRest()
    {
        // imbalanced-learn: a Tomek link is a mutual-nearest-neighbour pair with different labels. Rows 0 and 1
        // (classes 0 and 1) form one; the minority class is 2, so 'auto' removes both linked samples.
        var data = Column(0.0, 0.1, 10.0, 10.3, 20.0, 20.5);
        var labels = new Vector<double>(new[] { 0.0, 1.0, 2.0, 2.0, 0.0, 0.0 });
        var tomek = new TomekLinksAugmenter<double>(TomekLinksAugmenter<double>.RemovalStrategy.RemoveMajority);

        Assert.Equal(1, tomek.CountTomekLinks(data, labels, 2.0));
        var (cleaned, cleanedLabels) = tomek.RemoveTomekLinks(data, labels, 2.0);

        Assert.Equal(new[] { 10.0, 10.3, 20.0, 20.5 }, Enumerable.Range(0, cleaned.Rows).Select(i => cleaned[i, 0]));
        Assert.Equal(new[] { 2.0, 2.0, 0.0, 0.0 }, cleanedLabels.ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task RandomUnderSampler_UndersamplesEachNonMinorityClassSeparately()
    {
        // imbalanced-learn 'auto': every class other than the minority is reduced to the minority size.
        var data = Column(Enumerable.Range(0, 12).Select(i => (double)i).ToArray());
        var labels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0 });
        var sampler = new RandomUnderSampler<double>(randomSeed: 5);

        var (_, resampledLabels) = sampler.Undersample(data, labels, 0.0);

        var counts = resampledLabels.ToArray().GroupBy(l => l).ToDictionary(g => g.Key, g => g.Count());
        Assert.Equal(2, counts[0.0]);
        Assert.Equal(2, counts[1.0]);
        Assert.Equal(2, counts[2.0]);
    }

    [Fact(Timeout = 60000)]
    public async Task NearMiss3_OrdersCandidatesByLargestMeanMinorityDistance()
    {
        // imbalanced-learn NearMiss-3: minority {0, 10, 20}, majority {1, 11, 15, 100}, k = 1, target 3.
        // Candidates (nearest majority of each minority): 0 -> 1, 10 -> 11,
        // 20 -> 15. Nearest-minority distances: 1 -> 1, 11 -> 1, 15 -> 5. Target 3 keeps all three
        // candidates ordered farthest first: 15, then 1 and 11 in their original order.
        var data = Column(0.0, 10.0, 20.0, 1.0, 11.0, 15.0, 100.0);
        var labels = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 });
        var sampler = new NearMissUnderSampler<double>(
            NearMissUnderSampler<double>.NearMissVersion.NearMiss3, nNeighbors: 1);

        var (resampled, _) = sampler.Undersample(data, labels, 1.0);

        var kept = Enumerable.Range(0, resampled.Rows).Select(i => resampled[i, 0]).ToArray();
        Assert.Equal(new[] { 0.0, 10.0, 20.0, 15.0, 1.0, 11.0 }, kept);
    }
}
