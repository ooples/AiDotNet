using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Data;
using AiDotNet.Data.Audio.Benchmarks;
using AiDotNet.Data.Sampling;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Data;

/// <summary>
/// Regression tests for defects found while reviewing the element-wise sweep of the data loaders and samplers.
/// Each test compares the vectorized code with the scalar loop it replaced.
/// </summary>
public class ElementWiseSweepDataRegressionTests
{
    /// <summary>
    /// Gathering a batch of matrix rows must cost O(batch), not O(dataset). Converting the matrix to a tensor
    /// first copied the whole dataset on every batch: 100 batches of 32 rows from a 200,000 x 64 matrix
    /// allocated about 10 GB and took over a second, against about 1 ms for the loop it replaced.
    /// </summary>
    [Fact]
    public void BatchGather_MatrixRows_AllocatesForTheBatchNotTheDataset()
    {
        const int rows = 100_000;
        const int cols = 64;
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            matrix[i, 0] = i;
            matrix[i, cols - 1] = -i;
        }

        var random = new Random(1);
        var batches = Enumerable.Range(0, 20)
            .Select(_ => Enumerable.Range(0, 32).Select(__ => random.Next(rows)).ToArray())
            .ToArray();

        // Warm up so one-time allocations are not counted.
        BatchGather.Rows(matrix, batches[0]);

        long before = AllocatedBytes();
        var gathered = batches.Select(batch => BatchGather.Rows(matrix, batch)).ToArray();
        long allocated = AllocatedBytes() - before;

        for (int b = 0; b < batches.Length; b++)
        {
            Assert.Equal(32, gathered[b].Rows);
            Assert.Equal(cols, gathered[b].Columns);
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal((double)batches[b][i], gathered[b][i, 0]);
                Assert.Equal(-(double)batches[b][i], gathered[b][i, cols - 1]);
            }
        }

        // The 20 batches hold 20 * 32 * 64 doubles = 328 KB. One copy of the dataset is 51 MB.
        Assert.True(allocated < 4_000_000,
            $"Gathering 20 batches of 32 rows allocated {allocated / 1e6:F1} MB; the whole dataset is being copied per batch.");
    }

    [Fact]
    public void BatchGather_MatrixRows_PreservesOrderAndDuplicates_AndRejectsBadIndices()
    {
        var matrix = new Matrix<float>(4, 3);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                matrix[i, j] = i * 10 + j;
            }
        }

        var gathered = BatchGather.Rows(matrix, new[] { 3, 0, 3, 1 });

        Assert.Equal(new float[] { 30, 31, 32 }, new[] { gathered[0, 0], gathered[0, 1], gathered[0, 2] });
        Assert.Equal(new float[] { 0, 1, 2 }, new[] { gathered[1, 0], gathered[1, 1], gathered[1, 2] });
        Assert.Equal(new float[] { 30, 31, 32 }, new[] { gathered[2, 0], gathered[2, 1], gathered[2, 2] });
        Assert.Equal(new float[] { 10, 11, 12 }, new[] { gathered[3, 0], gathered[3, 1], gathered[3, 2] });

        // The result is a copy: changing it leaves the dataset alone.
        gathered[0, 0] = -1f;
        Assert.Equal(30f, matrix[3, 0]);

        var empty = BatchGather.Rows(matrix, Array.Empty<int>());
        Assert.Equal(0, empty.Rows);
        Assert.Equal(3, empty.Columns);

        Assert.Throws<ArgumentOutOfRangeException>(() => BatchGather.Rows(matrix, new[] { 0, 4 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => BatchGather.Rows(matrix, new[] { -1 }));
    }

    /// <summary>
    /// One weighted draw is a binary search. Going through the engine's search-sorted allocated two tensors
    /// per draw and, for single precision, scanned the whole cumulative table for NaN on every draw, so
    /// 100,000 draws over 3,000,000 weights took over a minute (about 10 ms with the loop it replaced).
    /// </summary>
    [Fact]
    public void WeightedSampler_SinglePrecision_DrawStaysLogarithmicInDatasetSize()
    {
        const int datasetSize = 2_000_000;
        const int draws = 20_000;
        var weights = Enumerable.Repeat(1f, datasetSize).ToArray();
        var sampler = new WeightedSampler<float>(weights, draws, replacement: true, seed: 3);

        // Warm up.
        Assert.Equal(draws, sampler.GetIndices().Count());

        long before = AllocatedBytes();
        var stopwatch = Stopwatch.StartNew();
        int count = 0;
        long sum = 0;
        int smallest = int.MaxValue;
        int largest = int.MinValue;
        foreach (int index in sampler.GetIndices())
        {
            smallest = Math.Min(smallest, index);
            largest = Math.Max(largest, index);
            sum += index;
            count++;
        }

        stopwatch.Stop();
        long allocated = AllocatedBytes() - before;

        Assert.Equal(draws, count);
        Assert.InRange(smallest, 0, datasetSize - 1);
        Assert.InRange(largest, 0, datasetSize - 1);
        // Uniform weights: the mean index is near the middle of the dataset.
        Assert.InRange(sum / (double)count, datasetSize * 0.45, datasetSize * 0.55);

        // A draw allocates no tensors (the engine search allocated about 1 KB per draw, 21 MB here).
        Assert.True(allocated < 4_000_000,
            $"{draws} draws allocated {allocated / 1e3:F0} KB; a draw should not allocate tensors.");
        // A full-table scan per draw takes several seconds here; the binary search takes milliseconds.
        Assert.True(stopwatch.Elapsed < TimeSpan.FromSeconds(3),
            $"{draws} draws over {datasetSize} weights took {stopwatch.Elapsed.TotalSeconds:F1} s.");
    }

    /// <summary>
    /// With double weights the seeded draws must be the ones the scalar implementation produced: same
    /// cumulative table, same search, same number and order of random draws.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void WeightedSampler_Double_SeededDrawsMatchTheScalarReference(bool replacement)
    {
        var random = new Random(5);
        // Zero weights, repeated weights and a wide dynamic range.
        var weights = Enumerable.Range(0, 2000)
            .Select(i => i % 7 == 0 ? 0.0 : (i % 5 == 0 ? 1.0 : random.NextDouble() * Math.Pow(10, random.Next(-3, 4))))
            .ToArray();
        int numSamples = replacement ? 20_000 : weights.Length;

        var expected = replacement
            ? ReferenceWithReplacement(weights, numSamples, seed: 42)
            : ReferenceWithoutReplacement(weights, numSamples, seed: 42);
        var actual = new WeightedSampler<double>(weights, numSamples, replacement, seed: 42).GetIndices().ToList();

        Assert.Equal(expected, actual);
        Assert.DoesNotContain(actual, index => weights[index] == 0.0);
        if (!replacement)
        {
            Assert.Equal(actual.Count, actual.Distinct().Count());
            Assert.Equal(weights.Count(w => w > 0), actual.Count);
        }
    }

    [Fact]
    public void ImportanceSampler_DrawsFollowTheScores()
    {
        var sampler = new ImportanceSampler<float>(4, seed: 9);
        sampler.SetImportances(new[] { 0f, 0f, 1f, 0f });

        var counts = new int[4];
        // One pass draws as many indices as the dataset holds (4), so draw 1000 passes.
        for (int pass = 0; pass < 1000; pass++)
        {
            foreach (int index in sampler.GetIndices())
            {
                counts[index]++;
            }
        }

        // With the default smoothing of 0.2 the probabilities are 0.05, 0.05, 0.85, 0.05.
        Assert.Equal(4000, counts.Sum());
        Assert.InRange(counts[2], 3200, 3600);
        Assert.InRange(counts[0], 100, 300);
        Assert.InRange(counts[1], 100, 300);
        Assert.InRange(counts[3], 100, 300);
    }

    /// <summary>
    /// 32-bit PCM in single precision must decode to the nearest float of sample / 2^31, as the scalar
    /// decoder did. Building the unsigned 32-bit value and then subtracting 2^32 cancelled the low bits of
    /// small negative samples: -1 and -100 decoded to 0, -1000 to -1024 / 2^31.
    /// </summary>
    [Fact]
    public void DecodePcmSamples_32Bit_SinglePrecision_MatchesTheScalarDecoder()
    {
        int[] samples =
        {
            0, 1, -1, 100, -100, 1000, -1000, 65535, -65536, 8388607, -8388608, 16777217, -16777217,
            123456789, -123456789, int.MaxValue, int.MinValue, int.MinValue + 1,
        };
        var bytes = samples.SelectMany(BitConverter.GetBytes).ToArray();

        var normalized = AudioLoaderHelper.DecodePcmSamples<float>(bytes, 0, samples.Length, 32, 4).AsSpan().ToArray();
        var raw = AudioLoaderHelper.DecodePcmSamples<float>(bytes, 0, samples.Length, 32, 4, normalize: false).AsSpan().ToArray();
        var normalizedDouble = AudioLoaderHelper.DecodePcmSamples<double>(bytes, 0, samples.Length, 32, 4).AsSpan().ToArray();

        for (int i = 0; i < samples.Length; i++)
        {
            Assert.Equal((float)(samples[i] / 2147483648.0), normalized[i]);
            Assert.Equal((float)samples[i], raw[i]);
            Assert.Equal(samples[i] / 2147483648.0, normalizedDouble[i]);
        }
    }

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(24)]
    public void DecodePcmSamples_NarrowDepths_MatchTheScalarDecoder(int bitsPerSample)
    {
        int bytesPerSample = bitsPerSample / 8;
        var random = new Random(bitsPerSample);
        var bytes = new byte[bytesPerSample * 64];
        random.NextBytes(bytes);
        // Extremes: most negative, -1, zero, most positive.
        var extremes = new[] { 1 << (bitsPerSample - 1), (1 << bitsPerSample) - 1, 0, (1 << (bitsPerSample - 1)) - 1 };
        for (int s = 0; s < extremes.Length; s++)
        {
            for (int k = 0; k < bytesPerSample; k++)
            {
                bytes[s * bytesPerSample + k] = (byte)(extremes[s] >> (8 * k));
            }
        }

        var decoded = AudioLoaderHelper.DecodePcmSamples<float>(bytes, 0, 64, bitsPerSample, bytesPerSample).AsSpan().ToArray();

        for (int s = 0; s < 64; s++)
        {
            int offset = s * bytesPerSample;
            double expected;
            if (bitsPerSample == 8)
            {
                expected = (bytes[offset] - 128.0) / 128.0;
            }
            else if (bitsPerSample == 16)
            {
                expected = (short)(bytes[offset] | (bytes[offset + 1] << 8)) / 32768.0;
            }
            else
            {
                int pcm24 = bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
                if ((pcm24 & 0x800000) != 0) pcm24 |= unchecked((int)0xFF000000);
                expected = pcm24 / 8388608.0;
            }

            Assert.Equal((float)expected, decoded[s]);
        }
    }

    /// <summary>
    /// Bytes allocated by the current thread. .NET Framework has no per-thread counter (and a process-wide
    /// one counts parallel tests too), so the allocation bounds are only checked on .NET Core 3.0 and later.
    /// </summary>
    private static long AllocatedBytes()
    {
#if NETCOREAPP3_0_OR_GREATER
        return GC.GetAllocatedBytesForCurrentThread();
#else
        return 0;
#endif
    }

    private static List<int> ReferenceWithReplacement(double[] weights, int count, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        double sum = 0;
        foreach (double w in weights) sum += w;

        var cumulative = new double[weights.Length];
        double running = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            running += weights[i] / sum;
            cumulative[i] = running;
        }

        cumulative[weights.Length - 1] = 1.0;

        var result = new List<int>(count);
        for (int s = 0; s < count; s++)
        {
            double r = random.NextDouble();
            int left = 0;
            int right = cumulative.Length - 1;
            while (left < right)
            {
                int mid = (left + right) / 2;
                if (cumulative[mid] < r) left = mid + 1; else right = mid;
            }

            result.Add(left);
        }

        return result;
    }

    private static List<int> ReferenceWithoutReplacement(double[] weights, int count, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var available = Enumerable.Range(0, weights.Length).ToList();
        var result = new List<int>();
        for (int i = 0; i < Math.Min(count, weights.Length); i++)
        {
            double sum = 0;
            for (int j = 0; j < available.Count; j++) sum += weights[available[j]];
            if (sum <= 0) break;

            double cumulative = 0;
            int selected = available.Count - 1;
            double u = random.NextDouble();
            for (int j = 0; j < available.Count; j++)
            {
                cumulative += weights[available[j]] / sum;
                if (u <= cumulative)
                {
                    selected = j;
                    break;
                }
            }

            result.Add(available[selected]);
            available.RemoveAt(selected);
        }

        return result;
    }
}
