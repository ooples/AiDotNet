using System;
using System.Linq;
using Xunit;
using AiDotNet.Data.Batching;
using AiDotNet.LinearAlgebra;

namespace UnitTests.Data.Loaders
{
    public class DataLoaderTests
    {
        [Fact]
        public void BatchProvider_GetBatch_Basic()
        {
            // Use Matrix<T> to validate InputHelper path via default IBatchProvider methods
            var data = new Matrix<int>(new int[,] { {0,1,2,3}, {4,5,6,7}, {8,9,10,11} });
            IBatchProvider<int, Matrix<int>> provider = new DefaultBatchProvider<int, Matrix<int>>();
            var size = provider.GetBatchSize(data);
            Assert.Equal(3, size);
            var batch = provider.GetBatch(data, new[] { 1 });
            Assert.Equal(1, batch.Rows);
            Assert.Equal(4, batch.Columns);
        }

        private sealed class DefaultBatchProviderLocal : IBatchProvider<int, Matrix<int>>
        {
            public int GetBatchSize(Matrix<int> input) => new DefaultBatchProvider<int, Matrix<int>>().GetBatchSize(input);
            public Matrix<int> GetBatch(Matrix<int> input, int[] indices) => new DefaultBatchProvider<int, Matrix<int>>().GetBatch(input, indices);
        }
    }
}
