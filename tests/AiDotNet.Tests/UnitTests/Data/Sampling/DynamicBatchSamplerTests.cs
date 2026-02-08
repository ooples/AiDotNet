using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Data.Sampling;
using Xunit;

namespace AiDotNetTests.UnitTests.Data.Sampling
{
    /// <summary>
    /// Unit tests for <see cref="DynamicBatchSampler"/>, which creates batches
    /// that fit a maximum token/element budget rather than a fixed number of samples.
    /// </summary>
    public class DynamicBatchSamplerTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidParameters_Succeeds()
        {
            var sampler = new DynamicBatchSampler(
                sampleLengths: new[] { 10, 20, 30 },
                maxTokensPerBatch: 100);

            Assert.NotNull(sampler);
            Assert.Equal(3, sampler.Length);
        }

        [Fact]
        public void Constructor_WithNullSampleLengths_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() =>
                new DynamicBatchSampler(sampleLengths: null!, maxTokensPerBatch: 100));
        }

        [Fact]
        public void Constructor_WithEmptySampleLengths_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() =>
                new DynamicBatchSampler(sampleLengths: Array.Empty<int>(), maxTokensPerBatch: 100));
        }

        [Fact]
        public void Constructor_WithZeroMaxTokens_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new DynamicBatchSampler(sampleLengths: new[] { 10 }, maxTokensPerBatch: 0));
        }

        [Fact]
        public void Constructor_WithZeroMaxSamples_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new DynamicBatchSampler(
                    sampleLengths: new[] { 10 },
                    maxTokensPerBatch: 100,
                    maxSamplesPerBatch: 0));
        }

        [Fact]
        public void Constructor_WithNegativeSampleLength_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new DynamicBatchSampler(sampleLengths: new[] { 10, -5, 20 }, maxTokensPerBatch: 100));
        }

        [Fact]
        public void Constructor_WithZeroSampleLength_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new DynamicBatchSampler(sampleLengths: new[] { 10, 0, 20 }, maxTokensPerBatch: 100));
        }

        #endregion

        #region Token Budget Batching Tests

        [Fact]
        public void GetBatchIndices_RespectesTokenBudget()
        {
            // 5 samples of 30 tokens each, budget of 100
            // Should fit 3 per batch (90 <= 100), then 2 in the last batch
            var sampler = new DynamicBatchSampler(
                sampleLengths: new[] { 30, 30, 30, 30, 30 },
                maxTokensPerBatch: 100,
                shuffle: false);

            var batches = sampler.GetBatchIndices().ToList();

            Assert.Equal(2, batches.Count);
            Assert.Equal(3, batches[0].Length); // 90 tokens
            Assert.Equal(2, batches[1].Length); // 60 tokens
        }

        [Fact]
        public void GetBatchIndices_VariableLengths_PacksEfficiently()
        {
            // Samples: 10, 20, 30, 40, 50 with budget 60
            var sampler = new DynamicBatchSampler(
                sampleLengths: new[] { 10, 20, 30, 40, 50 },
                maxTokensPerBatch: 60,
                shuffle: false);

            var batches = sampler.GetBatchIndices().ToList();

            // Batch 1: 10+20+30 = 60 (fits exactly)
            // Batch 2: 40 (adding 50 would exceed 60)
            // Batch 3: 50
            Assert.Equal(3, batches.Count);
            Assert.Equal(new[] { 0, 1, 2 }, batches[0]); // 10+20+30 = 60
            Assert.Single(batches[1]); // 40
            Assert.Single(batches[2]); // 50
        }

        [Fact]
        public void GetBatchIndices_SingleSampleExceedsBudget_GetsOwnBatch()
        {
            // A sample longer than the budget still gets included in its own batch
            var sampler = new DynamicBatchSampler(
                sampleLengths: new[] { 10, 200, 10 },
                maxTokensPerBatch: 50,
                shuffle: false);

            var batches = sampler.GetBatchIndices().ToList();

            // Batch 1: [10] (first sample)
            // Batch 2: [200] (exceeds budget but gets its own batch)
            // Batch 3: [10] (third sample)
            Assert.Equal(3, batches.Count);
            Assert.Contains(1, batches.SelectMany(b => b)); // The oversized sample is still included
        }

        [Fact]
        public void GetBatchIndices_MaxSamplesPerBatch_LimitsBatchSize()
        {
            // All small samples, but limit batch size to 2
            var sampler = new DynamicBatchSampler(
                sampleLengths: new[] { 1, 1, 1, 1 },
                maxTokensPerBatch: 1000,
                maxSamplesPerBatch: 2,
                shuffle: false);

            var batches = sampler.GetBatchIndices().ToList();

            Assert.Equal(2, batches.Count);
            Assert.Equal(2, batches[0].Length);
            Assert.Equal(2, batches[1].Length);
        }

        #endregion

        #region DropLast Tests

        [Fact]
        public void GetBatchIndices_DropLastFalse_IncludesIncompleteBatch()
        {
            var sampler = new DynamicBatchSampler(
                sampleLengths: new[] { 50, 50, 50 },
                maxTokensPerBatch: 100,
                shuffle: false,
                dropLast: false);

            var batches = sampler.GetBatchIndices().ToList();

            // Batch 1: [50, 50] = 100, Batch 2: [50] (incomplete but included)
            Assert.Equal(2, batches.Count);
            Assert.Single(batches[1]); // Incomplete batch is included
        }

        [Fact]
        public void GetBatchIndices_DropLastTrue_ExcludesIncompleteBatch()
        {
            var sampler = new DynamicBatchSampler(
                sampleLengths: new[] { 50, 50, 50 },
                maxTokensPerBatch: 100,
                shuffle: false,
                dropLast: true);

            var batches = sampler.GetBatchIndices().ToList();

            // Batch 1: [50, 50] = 100, Batch 2 [50] is dropped
            Assert.Single(batches);
            Assert.Equal(2, batches[0].Length);
        }

        #endregion

        #region Shuffle Determinism Tests

        [Fact]
        public void GetBatchIndices_SeededShuffle_IsDeterministic()
        {
            var lengths = new[] { 10, 20, 30, 40, 50, 60, 70, 80 };

            var sampler1 = new DynamicBatchSampler(lengths, maxTokensPerBatch: 100, seed: 42);
            var sampler2 = new DynamicBatchSampler(lengths, maxTokensPerBatch: 100, seed: 42);

            var batches1 = sampler1.GetBatchIndices().SelectMany(b => b).ToList();
            var batches2 = sampler2.GetBatchIndices().SelectMany(b => b).ToList();

            Assert.Equal(batches1, batches2);
        }

        [Fact]
        public void GetBatchIndices_DifferentSeeds_ProduceDifferentOrder()
        {
            var lengths = Enumerable.Range(1, 50).Select(i => i * 5).ToArray();

            var sampler1 = new DynamicBatchSampler(lengths, maxTokensPerBatch: 200, seed: 1);
            var sampler2 = new DynamicBatchSampler(lengths, maxTokensPerBatch: 200, seed: 2);

            var batches1 = sampler1.GetBatchIndices().SelectMany(b => b).ToList();
            var batches2 = sampler2.GetBatchIndices().SelectMany(b => b).ToList();

            Assert.NotEqual(batches1, batches2);
        }

        [Fact]
        public void GetBatchIndices_NoShuffle_ReturnsSequentialOrder()
        {
            var sampler = new DynamicBatchSampler(
                sampleLengths: new[] { 10, 20, 30, 40, 50 },
                maxTokensPerBatch: 1000,
                shuffle: false);

            var indices = sampler.GetBatchIndices().SelectMany(b => b).ToList();

            Assert.Equal(new[] { 0, 1, 2, 3, 4 }, indices);
        }

        #endregion

        #region GetIndices Tests

        [Fact]
        public void GetIndices_ReturnsAllSampleIndices()
        {
            var sampler = new DynamicBatchSampler(
                sampleLengths: new[] { 10, 20, 30, 40 },
                maxTokensPerBatch: 100,
                shuffle: false);

            var indices = sampler.GetIndices().ToList();

            Assert.Equal(4, indices.Count);
            Assert.Equal(new HashSet<int> { 0, 1, 2, 3 }, new HashSet<int>(indices));
        }

        #endregion
    }
}
