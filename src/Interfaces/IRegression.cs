namespace AiDotNet.Interfaces;

public abstract class IRegression<TInput, TOutput>
{
    internal abstract (TInput[] trainingInputs, TOutput[] trainingOutputs, TInput[] oosInputs, TOutput[]
        oosOutputs)
        PrepareData(TInput[] inputs, TOutput[] outputs, int trainingSize, INormalizer? normalizer);

    internal abstract void Fit(TInput[] inputs, TOutput[] outputs);

    internal abstract TOutput[] Transform(TInput[] inputs);
}