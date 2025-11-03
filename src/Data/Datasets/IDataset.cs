using System;

namespace AiDotNet.Data.Datasets
{
    public interface IDataset<T>
    {
        int Count { get; }
        T GetItem(int index);
    }
}

