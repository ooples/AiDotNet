using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using AiDotNet.Data.Datasets;
using AiDotNet.Data.Loaders;

namespace UnitTests.Data.Loaders
{
    public class DataLoaderTests
    {
        [Fact]
        public void Batches_Without_Shuffle()
        {
            var data = Enumerable.Range(0, 10).ToArray();
            var ds = new ArrayDataset<int>(data);
            var loader = new DataLoader<int>(ds, batchSize: 4, shuffle: false);
            var batches = loader.ToList();
            Assert.Equal(3, batches.Count);
            Assert.Equal(new[] {0,1,2,3}, batches[0]);
            Assert.Equal(new[] {4,5,6,7}, batches[1]);
            Assert.Equal(new[] {8,9}, batches[2]);
        }

        [Fact]
        public void Deterministic_Shuffle_With_Seed()
        {
            var data = Enumerable.Range(0, 8).ToArray();
            var ds = new ArrayDataset<int>(data);
            var a = new DataLoader<int>(ds, batchSize: 4, shuffle: true, seed: 42).SelectMany(x => x).ToArray();
            var b = new DataLoader<int>(ds, batchSize: 4, shuffle: true, seed: 42).SelectMany(x => x).ToArray();
            Assert.Equal(a, b);
        }

        [Fact]
        public void Throws_On_Invalid_BatchSize()
        {
            var ds = new ArrayDataset<int>(new[]{1,2,3});
            Assert.Throws<ArgumentOutOfRangeException>(() => new DataLoader<int>(ds, batchSize: 0));
        }

        [Fact]
        public void ArrayDataset_Guards_Index()
        {
            var ds = new ArrayDataset<int>(new[]{1,2});
            Assert.Equal(2, ds.Count);
            Assert.Throws<ArgumentOutOfRangeException>(() => ds.GetItem(-1));
            Assert.Throws<ArgumentOutOfRangeException>(() => ds.GetItem(2));
        }
    }
}

