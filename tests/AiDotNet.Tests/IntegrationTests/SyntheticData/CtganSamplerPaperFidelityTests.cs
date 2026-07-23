using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.SyntheticData;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// Paper-fidelity guards for <see cref="CTGANDataSampler{T}"/>'s training-by-sampling
/// (Xu et al. 2019 §4.3). Pins: (1) the conditioned category is drawn from the
/// LOG-frequency distribution — not uniformly (the previous behaviour) and not
/// proportionally — so rare categories are lifted but frequent ones still dominate;
/// (2) the emitted mask is exactly the one selected column's category block; (3) the
/// conditional vector is one-hot inside that block.
/// </summary>
public class CtganSamplerPaperFidelityTests
{
    private const int Seed = 7;

    [Fact]
    public void Sampler_LogFrequencyCategorySampling_AndMaskShape()
    {
        // One categorical column with a heavily imbalanced 3-category distribution:
        // cat0 = 800 rows, cat1 = 150, cat2 = 50.
        int[] counts = { 800, 150, 50 };
        int rows = counts[0] + counts[1] + counts[2];
        var data = new Matrix<double>(rows, 1);
        int r = 0;
        for (int c = 0; c < 3; c++)
            for (int j = 0; j < counts[c]; j++) data[r++, 0] = c;

        var columns = new List<ColumnMetadata>
        {
            new("Cat", ColumnDataType.Categorical, new[] { "A", "B", "C" }, columnIndex: 0),
        };

        var sampler = new CTGANDataSampler<double>(new Random(Seed));
        sampler.Fit(data, columns);

        var drawn = new int[3];
        const int draws = 30000;
        for (int i = 0; i < draws; i++)
        {
            var (cond, mask, colIdx, catIdx, rowIdx) = sampler.SampleCondVecWithMask();

            Assert.Equal(0, colIdx);                       // only one discrete column
            Assert.InRange(catIdx, 0, 2);
            // Mask covers the whole 3-category block; cond is one-hot within it.
            int ones = 0, condOnes = 0;
            for (int k = 0; k < cond.Length; k++)
            {
                if (Math.Abs(mask[k] - 1.0) < 1e-9) ones++;
                if (Math.Abs(cond[k] - 1.0) < 1e-9) condOnes++;
            }
            Assert.Equal(3, ones);                          // exactly the selected column's block
            Assert.Equal(1, condOnes);                      // one-hot condition
            Assert.Equal(1.0, cond[catIdx], 9);             // the set bit matches the reported category
            // The sampled real row must actually have the conditioned category.
            Assert.Equal(catIdx, (int)Math.Round(data[rowIdx, 0]));
            drawn[catIdx]++;
        }

        // Log-frequency ordering: P(cat0) > P(cat1) > P(cat2), but the ratio is far
        // gentler than the raw 800:150:50 frequency (that is the whole point of
        // log-frequency vs proportional sampling). Expected ~ log(801):log(151):log(51)
        // = 6.69 : 5.02 : 3.93  →  0.43 : 0.32 : 0.25.
        double p0 = (double)drawn[0] / draws, p1 = (double)drawn[1] / draws, p2 = (double)drawn[2] / draws;
        Assert.True(p0 > p1 && p1 > p2, $"log-freq ordering violated: {p0:F3},{p1:F3},{p2:F3}");
        // Not proportional: cat2 would be ~5% under proportional sampling; log-freq
        // must lift it well above that.
        Assert.True(p2 > 0.15, $"rare category under-sampled ({p2:F3}); not log-frequency.");
        // Not uniform: cat0 must stay clearly above 1/3.
        Assert.True(p0 > 0.38, $"frequent category not favored ({p0:F3}); looks uniform.");
    }
}
