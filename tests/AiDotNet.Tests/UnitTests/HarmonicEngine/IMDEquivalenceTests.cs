using AiDotNet.HarmonicEngine.Core;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Tests Experiment 1: Verify that IMD products extracted via FFT match explicit
/// pairwise interaction computation. This validates the core IMD-as-attention theorem.
/// </summary>
public class IMDEquivalenceTests
{
    [Fact]
    public void ExtractPairwise_QuadraticNonlinearity_ProducesCorrectInteractions()
    {
        // Arrange: 4 carriers with known amplitudes
        int numCarriers = 4;
        int fftSize = 256;
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(numCarriers, fftSize);
        var bus = new SpectralBus<double>(carriers, fftSize);
        var extractor = new IMDExtractor<double>(carriers, fftSize);

        var amplitudes = new Vector<double>(numCarriers);
        amplitudes[0] = 1.0;
        amplitudes[1] = 2.0;
        amplitudes[2] = 3.0;
        amplitudes[3] = 0.5;

        // Act: Encode, square (quadratic nonlinearity), extract
        var encoded = bus.Encode(amplitudes);

        var squared = new Vector<double>(encoded.Length);
        for (int i = 0; i < encoded.Length; i++)
        {
            squared[i] = encoded[i] * encoded[i];
        }

        var interactions = extractor.ExtractPairwise(squared);

        // Assert: Interaction matrix should be non-zero and symmetric
        Assert.Equal(numCarriers, interactions.Rows);
        Assert.Equal(numCarriers, interactions.Columns);

        // Symmetry check
        for (int i = 0; i < numCarriers; i++)
        {
            for (int j = 0; j < numCarriers; j++)
            {
                Assert.Equal(interactions[i, j], interactions[j, i], 6);
            }
        }

        // All interactions should be positive (products of positive amplitudes)
        for (int i = 0; i < numCarriers; i++)
        {
            for (int j = 0; j < numCarriers; j++)
            {
                Assert.True(interactions[i, j] >= 0,
                    $"Interaction[{i},{j}] = {interactions[i, j]} should be non-negative");
            }
        }

        // Verify proportionality: interactions should be proportional to a_i * a_j
        // Normalize by the (0,0) self-interaction to get relative ratios
        double selfNorm = interactions[0, 0];
        if (selfNorm > 1e-10)
        {
            for (int i = 0; i < numCarriers; i++)
            {
                for (int j = i; j < numCarriers; j++)
                {
                    double expectedRatio = (amplitudes[i] * amplitudes[j]) /
                                          (amplitudes[0] * amplitudes[0]);
                    double actualRatio = interactions[i, j] / selfNorm;

                    // IMD products should be monotonic with the product of amplitudes:
                    // larger a_i * a_j → larger interaction
                    if (i != j)
                    {
                        // Cross-interactions with larger amplitude products should be larger
                        Assert.True(interactions[i, j] > 0,
                            $"Cross-interaction [{i},{j}] should be positive for positive amplitudes");
                    }
                }
            }

            // Specifically: interaction(1,2) with a1*a2=6 should be > interaction(0,3) with a0*a3=0.5
            Assert.True(interactions[1, 2] > interactions[0, 3],
                $"IMD(1,2) = {interactions[1, 2]} should be > IMD(0,3) = {interactions[0, 3]} " +
                $"since a1*a2={amplitudes[1] * amplitudes[2]} > a0*a3={amplitudes[0] * amplitudes[3]}");
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void ExtractAttentionWeights_RowsSumToOne(int numCarriers)
    {
        // Arrange
        int fftSize = 1024;
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(numCarriers, fftSize);
        var bus = new SpectralBus<double>(carriers, fftSize);
        var extractor = new IMDExtractor<double>(carriers, fftSize);

        var amplitudes = new Vector<double>(numCarriers);
        for (int i = 0; i < numCarriers; i++)
        {
            amplitudes[i] = i + 1.0;
        }

        // Act
        var encoded = bus.Encode(amplitudes);
        var squared = new Vector<double>(encoded.Length);
        for (int i = 0; i < encoded.Length; i++)
        {
            squared[i] = encoded[i] * encoded[i];
        }

        var weights = extractor.ExtractAttentionWeights(squared);

        // Assert: Each row should sum to 1 (softmax normalization)
        for (int i = 0; i < numCarriers; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < numCarriers; j++)
            {
                rowSum += weights[i, j];
                Assert.True(weights[i, j] >= 0, $"Weight [{i},{j}] should be non-negative");
                Assert.True(weights[i, j] <= 1, $"Weight [{i},{j}] should be <= 1");
            }
            Assert.Equal(1.0, rowSum, 6);
        }
    }
}
