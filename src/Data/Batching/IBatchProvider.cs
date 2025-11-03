namespace AiDotNet.Data.Batching;

using AiDotNet.Helpers;

public interface IBatchProvider<T, TInput>
{
    int GetBatchSize(TInput input);
    TInput GetBatch(TInput input, int[] indices);
}

public sealed class DefaultBatchProvider<T, TInput> : IBatchProvider<T, TInput>
{
    public int GetBatchSize(TInput input) => InputHelper<T, TInput>.GetBatchSize(input);
    public TInput GetBatch(TInput input, int[] indices) => InputHelper<T, TInput>.GetBatch(input, indices);
}
