namespace AiDotNet.MetaLearning;

using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

public interface IMetaLearner<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    // Trains across tasks (episodes). Kept generic for integration with existing patterns.
    void FitEpisodes(Tensor<T> supportInputs, Tensor<T> supportLabels, Tensor<T> queryInputs, Tensor<T> queryLabels);
}

