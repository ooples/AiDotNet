namespace AiDotNet.Interfaces;

public interface ITreeBasedModel<T> : IFullModel<T>
{
    int NumberOfTrees { get; }
    int MaxDepth { get; }
    Vector<T> FeatureImportances { get; }
}