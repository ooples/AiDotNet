using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data;
using AiDotNet.Data.Audio.Benchmarks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Graph;
using AiDotNet.Data.Loaders;
using AiDotNet.Data.Sampling;
using AiDotNet.Data.Transforms.Numeric;
using AiDotNet.Data.Vision.Benchmarks;
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

        var random = RandomHelper.CreateSeededRandom(1);
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
        var random = RandomHelper.CreateSeededRandom(5);
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
        var random = RandomHelper.CreateSeededRandom(bitsPerSample);
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
    /// The cumulative table must not drift: with 2,000,000 equal single-precision weights a running sum in float
    /// ended near 0.9814, so forcing the last entry to 1 gave about 1.9 % of all draws to the last index.
    /// </summary>
    [Fact]
    public void WeightedSampler_SinglePrecision_LargeUniformTable_DoesNotPileOntoTheLastIndex()
    {
        const int datasetSize = 2_000_000;
        const int draws = 20_000;
        var weights = Enumerable.Repeat(1f, datasetSize).ToArray();
        var sampler = new WeightedSampler<float>(weights, draws, replacement: true, seed: 11);

        int last = sampler.GetIndices().Count(index => index == datasetSize - 1);

        // Expected draws on one index: 20,000 / 2,000,000 = 0.01; the drifting table gave about 370.
        Assert.InRange(last, 0, 3);
    }

    /// <summary>
    /// In half precision the running sum of 4,096 equal probabilities stalls at 0.5, so every draw above 0.5
    /// returned the last index.
    /// </summary>
    [Fact]
    public void WeightedSampler_Half_UniformTable_DrawsCoverTheUpperHalf()
    {
        const int datasetSize = 4096;
        var weights = Enumerable.Repeat((Half)1.0, datasetSize).ToArray();
        var indices = new WeightedSampler<Half>(weights, datasetSize, replacement: true, seed: 2).GetIndices().ToList();

        Assert.InRange(indices.Count(index => index == datasetSize - 1), 0, 8);
        Assert.InRange(indices.Count(index => index >= datasetSize / 2 && index < datasetSize - 1), 1700, 2400);
    }

    /// <summary>
    /// Without replacement the remaining-weight total of 100 half-precision weights of 1,000 is 100,000, which
    /// overflows Half: every weight / total became 0 and the sampler always took the last available index. The
    /// draws must equal the double reference (Half 1,000 is exact).
    /// </summary>
    [Fact]
    public void WeightedSampler_Half_WithoutReplacement_MatchesTheDoubleReference()
    {
        var weights = Enumerable.Repeat(1000.0, 100).ToArray();
        var expected = ReferenceWithoutReplacement(weights, weights.Length, seed: 8);

        var actual = new WeightedSampler<Half>(weights.Select(w => (Half)w), weights.Length, replacement: false, seed: 8)
            .GetIndices().ToList();

        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// Ten classes of 7,000 samples: total / (numClasses * count) = 70,000 / 70,000 = 1, but 70,000 overflows Half,
    /// so the weights came out NaN.
    /// </summary>
    [Fact]
    public void CreateBalancedWeights_Half_LargeClasses_StayFinite()
    {
        var labels = Enumerable.Range(0, 70_000).Select(i => i % 10).ToArray();

        var weights = WeightedSampler<Half>.CreateBalancedWeights(labels, 10);

        Assert.Equal(labels.Length, weights.Length);
        Assert.All(weights, w => Assert.Equal((Half)1.0, w));
    }

    /// <summary>
    /// The importance distribution of 4,096 uniform half-precision scores: the cumulative table stalled at 0.5
    /// in Half, so half the draws went to the last index.
    /// </summary>
    [Fact]
    public void ImportanceSampler_Half_UniformScores_DoNotPileOntoTheLastIndex()
    {
        const int datasetSize = 4096;
        var sampler = new ImportanceSampler<Half>(datasetSize, seed: 4);

        var indices = sampler.GetIndices().ToList();

        Assert.Equal(datasetSize, indices.Count);
        Assert.InRange(indices.Count(index => index == datasetSize - 1), 0, 8);
    }

    /// <summary>
    /// 24- and 32-bit PCM in half precision: the place values 65,536 and 2^24 overflow Half, so the decoder
    /// returned infinities. It must return the Half nearest to sample / 2^(bits - 1), as the scalar decoder did.
    /// </summary>
    [Theory]
    [InlineData(24)]
    [InlineData(32)]
    public void DecodePcmSamples_Half_WideDepths_MatchTheScalarDecoder(int bitsPerSample)
    {
        int bytesPerSample = bitsPerSample / 8;
        long[] samples = bitsPerSample == 24
            ? new long[] { 0, 1, -1, 65535, -65536, 4_000_000, -4_000_000, 8_388_607, -8_388_608 }
            : new long[] { 0, 1, -1, 65535, -65536, 16_777_217, -123_456_789, int.MaxValue, int.MinValue };
        var bytes = new byte[samples.Length * bytesPerSample];
        for (int s = 0; s < samples.Length; s++)
        {
            for (int k = 0; k < bytesPerSample; k++)
            {
                bytes[s * bytesPerSample + k] = (byte)(samples[s] >> (8 * k));
            }
        }

        var decoded = AudioLoaderHelper.DecodePcmSamples<Half>(bytes, 0, samples.Length, bitsPerSample, bytesPerSample)
            .AsSpan().ToArray();

        double fullScale = Math.Pow(2, bitsPerSample - 1);
        for (int s = 0; s < samples.Length; s++)
        {
            Assert.Equal((Half)(samples[s] / fullScale), decoded[s]);
        }
    }

    /// <summary>
    /// FLAC mono mix of 24-bit samples in half precision: casting the raw samples to Half first overflowed, so
    /// every sample above 65,504 decoded to exactly 1. The mix must match the scalar (sum / channels) * norm.
    /// </summary>
    [Fact]
    public void FlacMixToMono_Half_24BitSamples_MatchTheScalarMix()
    {
        int[] left = { 8_388_607, -8_388_608, 100_000, -70_000, 1, 0 };
        int[] right = { 8_388_607, 8_388_607, 30_000, -1, -1, 5 };
        double normFactor = 1.0 / (1L << 23);

        var mono = FlacDecoder.MixToMono<Half>(
            new[] { new Tensor<int>(left, new[] { left.Length }) }, normFactor).AsSpan().ToArray();
        var stereo = FlacDecoder.MixToMono<Half>(
            new[] { new Tensor<int>(left, new[] { left.Length }), new Tensor<int>(right, new[] { right.Length }) },
            normFactor).AsSpan().ToArray();

        for (int i = 0; i < left.Length; i++)
        {
            Assert.Equal((Half)Math.Max(-1.0, Math.Min(1.0, left[i] * normFactor)), mono[i]);
            double mixed = (left[i] + (double)right[i]) / 2 * normFactor;
            Assert.Equal((Half)Math.Max(-1.0, Math.Min(1.0, mixed)), stereo[i]);
        }
    }

    /// <summary>
    /// Temporal-graph hash embedding sin(node * (f + 1)) in half precision: the product passed 65,504 and sin of
    /// infinity is NaN. It must be the Half nearest to the double Math.Sin, as before.
    /// </summary>
    [Fact]
    public void TemporalGraphHashEmbeddings_Half_LargeNodeIds_MatchMathSin()
    {
        int[] nodes = { 1, 2_049, 40_000, 1_000_000 };
        const int halfDim = 16;

        var table = TemporalGraphDataLoader<Half>.HashEmbeddings(nodes, halfDim);

        Assert.Equal(nodes.Length * halfDim, table.Length);
        for (int n = 0; n < nodes.Length; n++)
        {
            for (int f = 0; f < halfDim; f++)
            {
                Assert.Equal((Half)Math.Sin(nodes[n] * (f + 1.0)), table[n * halfDim + f]);
            }
        }
    }

    /// <summary>
    /// Wikidata5M hash embedding sin(id * (d + 1) / N): in half precision id * (d + 1) overflowed to infinity
    /// (NaN embeddings), and in single precision it rounded once it passed 2^24.
    /// </summary>
    [Fact]
    public void Wikidata5mHashEmbeddings_MatchTheDoubleFormula()
    {
        int[] heads = { 1, 70_000, 4_594_485 };
        int[] tails = { 3, 2_000_001, 4_000_000 };
        const int dim = 32;
        const int entities = 4_594_485;

        var half = Wikidata5mDataLoader<Half>.HashEmbeddings(heads, tails, dim, entities);
        var single = Wikidata5mDataLoader<float>.HashEmbeddings(heads, tails, dim, entities);

        Assert.Equal(new[] { heads.Length, 2 * dim }, half.Shape.ToArray());
        for (int i = 0; i < heads.Length; i++)
        {
            for (int d = 0; d < dim; d++)
            {
                double head = Math.Sin(heads[i] * (d + 1.0) / entities);
                double tail = Math.Sin(tails[i] * (d + 1.0) / entities);
                Assert.Equal((Half)head, half[i, d]);
                Assert.Equal((Half)tail, half[i, dim + d]);
                Assert.Equal((float)head, single[i, d]);
                Assert.Equal((float)tail, single[i, dim + d]);
            }
        }
    }

    /// <summary>
    /// Speech Commands resampling 16 kHz to 96 kHz in half precision: the target rate is above Half.MaxValue, so
    /// the interpolation weights were NaN. The output must equal the double blend v0 * (1 - frac) + v1 * frac.
    /// </summary>
    [Fact]
    public void SpeechCommandsResample_Half_HighTargetRate_MatchesTheDoubleBlend()
    {
        const int nativeRate = 16_000;
        const int targetRate = 96_000;
        var native = Enumerable.Range(0, 64).Select(i => (Half)Math.Sin(i * 0.37)).ToArray();
        const int targetLen = 300;
        var dst = new Half[targetLen];

        SpeechCommandsDataLoader<Half>.ResampleIntoDestination(native, 0, dst, 0, targetLen, nativeRate, targetRate);

        for (int t = 0; t < targetLen; t++)
        {
            double position = t * (double)nativeRate / targetRate;
            int f = Math.Min((int)Math.Floor(position), native.Length - 1);
            int c = Math.Min(f + 1, native.Length - 1);
            double frac = position - Math.Floor(position);
            double expected = (double)native[f] * (1 - frac) + (double)native[c] * frac;
            Assert.Equal((Half)expected, dst[t]);
        }
    }

    /// <summary>
    /// The 1e-12 zero guards of the numeric transforms are 0 in half precision, so a zero std or range was
    /// divided by: NaN or infinity instead of the scalar fallback (centre only, targetMin, std of 1).
    /// </summary>
    [Fact]
    public void NumericTransforms_Half_ZeroSpreadFeatures_UseTheScalarFallback()
    {
        var normalize = new NormalizeTransform<Half>(new[] { (Half)2.0, (Half)1.0 }, new[] { (Half)0.0, (Half)2.0 });
        Assert.Equal(new[] { (Half)3.0, (Half)1.5 }, normalize.Apply(new[] { (Half)5.0, (Half)4.0 }));

        var reference = new[] { new[] { (Half)7.0, (Half)0.0 }, new[] { (Half)7.0, (Half)4.0 } };
        var minMax = new MinMaxScaleTransform<Half>(reference, -1.0, 1.0);
        Assert.Equal(new[] { (Half)(-1.0), (Half)0.0 }, minMax.Apply(new[] { (Half)7.0, (Half)2.0 }));

        var standard = new StandardScaleTransform<Half>(reference);
        Assert.Equal(new[] { (Half)2.0, (Half)0.0 }, standard.Apply(new[] { (Half)9.0, (Half)2.0 }));
    }

    /// <summary>
    /// Fitting a standard scaler in half precision: deviations of 1,000 square to 1e6, which overflows Half, so the
    /// std was infinite and every scaled value 0. The fit must match the double statistics.
    /// </summary>
    [Fact]
    public void StandardScaleTransform_Half_LargeDeviations_MatchTheDoubleFit()
    {
        var reference = new[] { new[] { (Half)(-1000.0) }, new[] { (Half)1000.0 }, new[] { (Half)0.0 } };
        var standard = new StandardScaleTransform<Half>(reference);

        double std = Math.Sqrt((1000.0 * 1000.0 * 2) / 3);
        Assert.Equal((Half)(500.0 / (double)(Half)std), standard.Apply(new[] { (Half)500.0 })[0]);
    }

    /// <summary>
    /// CIFAR-10 normalisation must equal the scalar b / 255: multiplying by fl(1/255) instead rounded 126 of the
    /// 256 byte values to a different float.
    /// </summary>
    [Fact]
    public async Task Cifar10_Normalized_PixelsEqualTheScalarDivision()
    {
        string root = Path.Combine(Path.GetTempPath(), "adn-cifar10-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var record = new byte[1 + 3072];
            record[0] = 3;
            for (int p = 0; p < 3072; p++) record[1 + p] = (byte)(p % 256);
            File.WriteAllBytes(Path.Combine(root, "data_batch_1.bin"), new byte[0]);
            File.WriteAllBytes(Path.Combine(root, "test_batch.bin"), record);

            var loader = new Cifar10DataLoader<float>(new Cifar10DataLoaderOptions
            {
                DataPath = root,
                AutoDownload = false,
                Split = DatasetSplit.Test,
                Layout = ImageTensorLayout.NCHW,
                Normalize = true
            });
            await loader.LoadAsync();

            var pixels = loader.Features.AsSpan().ToArray();
            Assert.Equal(3072, pixels.Length);
            for (int p = 0; p < 3072; p++)
            {
                Assert.Equal((float)(record[1 + p] / 255.0), pixels[p]);
            }
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>
    /// The OGB loader assigned its class count after parsing, so the first load one-hot encoded node labels with 0
    /// classes and threw. ogbn-arxiv has 40 classes.
    /// </summary>
    [Fact]
    public async Task OgbNodeDataset_FirstLoad_OneHotEncodesWithTheDatasetClassCount()
    {
        string root = Path.Combine(Path.GetTempPath(), "adn-ogb-" + Guid.NewGuid().ToString("N"));
        string raw = Path.Combine(root, "ogbn-arxiv", "raw");
        Directory.CreateDirectory(raw);
        try
        {
            File.WriteAllText(Path.Combine(raw, "edge.csv"), "0,1\n1,2\n");
            File.WriteAllText(Path.Combine(raw, "node-feat.csv"), "0.5,1.0\n1.5,2.0\n2.5,3.0\n");
            File.WriteAllText(Path.Combine(raw, "node-label.csv"), "3\n39\n0\n");

            var loader = new OGBDatasetLoader<double>(
                "ogbn-arxiv", OGBDatasetLoader<double>.OGBTask.NodePrediction, dataPath: root, autoDownload: false);
            await loader.LoadAsync();

            var labels = loader.NodeLabels;
            Assert.NotNull(labels);
            if (labels is null) return;
            Assert.Equal(new[] { 3, 40 }, labels.Shape.ToArray());
            int[] expected = { 3, 39, 0 };
            for (int node = 0; node < 3; node++)
            {
                for (int c = 0; c < 40; c++)
                {
                    Assert.Equal(c == expected[node] ? 1.0 : 0.0, labels[node, c]);
                }
            }
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>
    /// When no sample is under the threshold the curriculum keeps the easiest ones (within 1e-9 of the minimum).
    /// In half precision 1e-9 is 0, so "&lt; 1e-9" excluded the minimum itself and the epoch was empty.
    /// </summary>
    [Fact]
    public void CurriculumSampler_Half_NothingEligible_FallsBackToTheEasiestSamples()
    {
        var difficulties = new[] { (Half)0.5, (Half)0.25, (Half)0.75, (Half)0.25 };
        var sampler = new CurriculumSampler<Half>(difficulties, totalEpochs: 10, CurriculumStrategy.CompetenceBased, seed: 1);
        sampler.SetCompetence(0.0);

        var indices = sampler.GetIndices().OrderBy(i => i).ToArray();

        Assert.Equal(new[] { 1, 3 }, indices);
    }

    /// <summary>
    /// Extracting a batch must not touch the source tensor: the shape "clone" was the tensor's own shape array, so
    /// the train split rewrote the dataset's first dimension to the train size.
    /// </summary>
    [Fact]
    public async Task InMemoryDataLoader_TensorSplit_LeavesTheSourceShapeAlone()
    {
        var features = new Tensor<double>(Enumerable.Range(0, 30).Select(i => (double)i).ToArray(), new[] { 10, 3 });
        var labels = new Tensor<double>(Enumerable.Range(0, 10).Select(i => (double)i).ToArray(), new[] { 10, 1 });
        var loader = new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(features, labels);
        await loader.LoadAsync();

        var (train, validation, test) = loader.Split(0.6, 0.2, seed: 3);

        Assert.Equal(new[] { 10, 3 }, features.Shape.ToArray());
        Assert.Equal(new[] { 10, 1 }, labels.Shape.ToArray());

        // Every row lands in exactly one split with its own label.
        var seen = new List<int>();
        foreach (var split in new[] { train, validation, test })
        {
            await split.LoadAsync();
            var f = split.Features;
            var l = split.Labels;
            for (int r = 0; r < f.Shape[0]; r++)
            {
                int row = (int)l[r, 0];
                Assert.Equal(row * 3.0, f[r, 0]);
                Assert.Equal(row * 3.0 + 2, f[r, 2]);
                seen.Add(row);
            }
        }

        Assert.Equal(Enumerable.Range(0, 10), seen.OrderBy(i => i));
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
