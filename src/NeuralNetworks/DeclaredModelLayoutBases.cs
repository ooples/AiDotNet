using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks;

/// <summary>Shared constructor surface for model families whose layouts are declared on a measured base.</summary>
public abstract class DeclaredModelLayoutBase<T> : NeuralNetworkBase<T>
{
    protected DeclaredModelLayoutBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction, maxGradNorm)
    {
    }

    protected DeclaredModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm)
    {
    }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class VectorModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected VectorModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected VectorModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
public abstract class SequenceModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected SequenceModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected SequenceModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Time,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Classes,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class TokenLanguageModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected TokenLanguageModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected TokenLanguageModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Time,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
public abstract class TextEmbeddingModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected TextEmbeddingModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected TextEmbeddingModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Classes,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class ImageClassifierModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected ImageClassifierModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected ImageClassifierModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class ImageTranslationModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected ImageTranslationModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected ImageTranslationModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class ImageGeneratorModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected ImageGeneratorModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected ImageGeneratorModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Classes,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class GraphModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected GraphModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected GraphModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class MultimodalModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected MultimodalModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected MultimodalModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}

[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Depth, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Depth, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class VolumetricModelLayoutBase<T> : DeclaredModelLayoutBase<T>
{
    protected VolumetricModelLayoutBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0) : base(architecture, lossFunction, maxGradNorm) { }
    protected VolumetricModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }
}
