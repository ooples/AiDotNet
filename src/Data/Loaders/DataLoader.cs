using System;
using System.Collections;
using System.Collections.Generic;
using AiDotNet.Data.Datasets;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Loaders
{
    public sealed class DataLoader<T> : IEnumerable<Vector<T>>
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

        public IEnumerator<Vector<T>> GetEnumerator()
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
                var arr = new T[size];
                for (int k = 0; k < size; k++)
                {
                    arr[k] = _dataset.GetItem(indices[i + k]);
                }
                yield return new Vector<T>(arr);
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
