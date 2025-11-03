using System;

namespace AiDotNet.Data.Datasets
{
    public sealed class ArrayDataset<T> : IDataset<T>
    {
        private readonly T[] _data;

        public ArrayDataset(T[] data)
        {
            _data = data ?? throw new ArgumentNullException("data");
        }

        public int Count => _data.Length;

        public T GetItem(int index)
        {
            if (index < 0 || index >= _data.Length) throw new ArgumentOutOfRangeException("index");
            return _data[index];
        }
    }
}

