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

    /// <summary>
    /// The input type this model was built for, read back from the architecture that carries it.
    /// </summary>
    /// <remarks>
    /// Exists so the clone plan can source an <c>inputType</c> constructor argument. Seven models in
    /// this family -- the GANs and the image translators -- pass that argument straight into the
    /// architecture they construct and keep no copy of their own, so every one of them was reported
    /// unrebuildable over a value it still holds: one level down, where the lookup cannot see it,
    /// because a member is sourced from a type's own members and its bases, never from a member OF a
    /// member. Deriving it in the shared base keeps the value in one place instead of adding a second
    /// copy to each model, which could then disagree with the architecture it was built from.
    /// </remarks>
    private InputType _inputType => Architecture.InputType;
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

/// <summary>
/// Base class for vector models whose declared <see cref="NeuralNetworkBase{T}.Layers"/> collection is
/// the complete, literal inference graph in execution order.
/// </summary>
/// <remarks>
/// <para>
/// This is an execution-topology contract, not a shape guess. Models opt in by choosing this base only
/// after their forward has been verified to apply every declared layer exactly once in list order. The
/// base then supplies framework services that are safe only for a true sequential graph, including the
/// allocation-efficient named-activation fold.
/// </para>
/// <para>
/// The capability is intentionally limited to a model that directly chooses this base. A subclass of
/// such a model falls back to tracing its actual forward because it may override inference and introduce
/// branches, shared stages, input transforms, or generation loops. A still-sequential subclass can opt
/// in explicitly after verifying its own forward contract.
/// </para>
/// </remarks>
public abstract class SequentialVectorModelLayoutBase<T> : VectorModelLayoutBase<T>
{
    protected SequentialVectorModelLayoutBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T> lossFunction,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction, maxGradNorm) { }

    protected SequentialVectorModelLayoutBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : base(lossFunction, maxGradNorm) { }

    /// <inheritdoc/>
    protected override bool SupportsSequentialActivationFold
        => GetType().BaseType == typeof(SequentialVectorModelLayoutBase<T>);
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

    /// <summary>
    /// Promotes an unbatched image to the canonical NCHW training layout.
    /// </summary>
    /// <remarks>
    /// Image classifiers commonly replace <c>PredictCore</c> only to install a fused inference fast
    /// path, while their trainable layer graph still requires the same unit-batch promotion as the
    /// base eager path. State that family contract here so objective probes and public training use
    /// identical shapes without repeating it in every ResNet/MobileNet/EfficientNet implementation.
    /// </remarks>
    protected override Tensor<T> PrepareInputForTraining(Tensor<T> input)
        => NormalizeInputBatchDim(input);
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
