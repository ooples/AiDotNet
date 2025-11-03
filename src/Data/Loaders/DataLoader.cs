using System;
using System.Collections;
using System.Collections.Generic;
using AiDotNet.Data.Datasets;

namespace AiDotNet.Data.Loaders
{
    public sealed class DataLoader<T> : IEnumerable<IList<T>>
    {
        private readonly IDataset<T> _dataset;
        private readonly int _batchSize;
        private readonly bool _shuffle;
        private readonly int? _seed;

        public DataLoader(IDataset<T> dataset, int batchSize = 32, bool shuffle = false, int? seed = null)
        {
            _dataset = dataset ?? throw new ArgumentNullException("dataset");
            if (batchSize <= 0) throw new ArgumentOutOfRangeException("batchSize");
            _batchSize = batchSize;
            _shuffle = shuffle;
            _seed = seed;
        }

        public IEnumerator<IList<T>> GetEnumerator()
        {
            int n = _dataset.Count;
            var indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;

            if (_shuffle)
            {
                var rng = _seed.HasValue ? new Random(_seed.Value) : new Random();
                for (int i = n - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }

            for (int i = 0; i < n; i += _batchSize)
            {
                int size = Math.Min(_batchSize, n - i);
                var batch = new List<T>(capacity: size);
                for (int k = 0; k < size; k++)
                {
                    batch.Add(_dataset.GetItem(indices[i + k]));
                }
                yield return batch;
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}

