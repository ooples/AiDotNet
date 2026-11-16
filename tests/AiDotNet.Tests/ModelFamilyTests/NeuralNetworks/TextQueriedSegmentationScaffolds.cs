using System;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.OpenVocabulary;
using AiDotNet.ComputerVision.Segmentation.Referring;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Shared configuration for the manual scaffolds of the text-queried segmentation models. Each model
/// now takes a required <c>ITextEncoder&lt;T&gt;</c>, which the generator cannot supply, so these
/// scaffolds construct it with a <see cref="DeterministicTextEncoder{T}"/> standing in for CLIP.
/// </summary>
internal static class TextQueriedScaffold
{
    internal const int Channels = 3;
    internal const int Height = 64;
    internal const int Width = 64;
    internal const int NumClasses = 4;
    internal const int EmbeddingDimension = 16;

    internal static int[] InputShape => [Channels, Height, Width];

    internal static NeuralNetworkArchitecture<float> Architecture() => new(
        inputType: InputType.ThreeDimensional,
        taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputHeight: Height,
        inputWidth: Width,
        inputDepth: Channels,
        outputSize: NumClasses);

    internal static DeterministicTextEncoder<float> Encoder() => new(EmbeddingDimension);

    /// <summary>
    /// The decoders emit their logits at the encoder's last stride, so the output shape is measured
    /// once from a real forward rather than restated per model.
    /// </summary>
    internal static int[] ProbeOutputShape(Func<INeuralNetworkModel<float>> create)
    {
        var network = create();
        var output = network.Predict(new Tensor<float>(InputShape));
        return output.Shape.ToArray();
    }
}

/// <summary>Manual scaffold for CAT-Seg (Cho et al. 2024).</summary>
public class CATSegTests : SegmentationTestBase<float>
{
    private int[]? _outputShape;
    protected override int[] InputShape => TextQueriedScaffold.InputShape;
    protected override int[] OutputShape => _outputShape ??= TextQueriedScaffold.ProbeOutputShape(CreateNetwork);

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new CATSeg<float>(TextQueriedScaffold.Architecture(), TextQueriedScaffold.Encoder(),
            options: new CATSegOptions { NumClasses = TextQueriedScaffold.NumClasses, DropRate = 0.0 });
}

/// <summary>Manual scaffold for SAN (Xu et al. 2023).</summary>
public class SANTests : SegmentationTestBase<float>
{
    private int[]? _outputShape;
    protected override int[] InputShape => TextQueriedScaffold.InputShape;
    protected override int[] OutputShape => _outputShape ??= TextQueriedScaffold.ProbeOutputShape(CreateNetwork);

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SAN<float>(TextQueriedScaffold.Architecture(), TextQueriedScaffold.Encoder(),
            options: new SANOptions { NumClasses = TextQueriedScaffold.NumClasses, DropRate = 0.0 });
}

/// <summary>Manual scaffold for SED (Xie et al. 2024).</summary>
public class SEDTests : SegmentationTestBase<float>
{
    private int[]? _outputShape;
    protected override int[] InputShape => TextQueriedScaffold.InputShape;
    protected override int[] OutputShape => _outputShape ??= TextQueriedScaffold.ProbeOutputShape(CreateNetwork);

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SED<float>(TextQueriedScaffold.Architecture(), TextQueriedScaffold.Encoder(),
            options: new SEDOptions { NumClasses = TextQueriedScaffold.NumClasses, DropRate = 0.0 });
}

/// <summary>Manual scaffold for Mask-Adapter (Li et al. 2025).</summary>
public class MaskAdapterTests : SegmentationTestBase<float>
{
    private int[]? _outputShape;
    protected override int[] InputShape => TextQueriedScaffold.InputShape;
    protected override int[] OutputShape => _outputShape ??= TextQueriedScaffold.ProbeOutputShape(CreateNetwork);

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new MaskAdapter<float>(TextQueriedScaffold.Architecture(), TextQueriedScaffold.Encoder(),
            options: new MaskAdapterOptions { NumClasses = TextQueriedScaffold.NumClasses, DropRate = 0.0 });
}

/// <summary>Manual scaffold for Open-Vocabulary SAM (Yuan et al. 2024).</summary>
public class OpenVocabSAMTests : SegmentationTestBase<float>
{
    private int[]? _outputShape;
    protected override int[] InputShape => TextQueriedScaffold.InputShape;
    protected override int[] OutputShape => _outputShape ??= TextQueriedScaffold.ProbeOutputShape(CreateNetwork);

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new OpenVocabSAM<float>(TextQueriedScaffold.Architecture(), TextQueriedScaffold.Encoder(),
            options: new OpenVocabSAMOptions { NumClasses = TextQueriedScaffold.NumClasses, DropRate = 0.0 });
}

/// <summary>Manual scaffold for LISA (Lai et al. 2024).</summary>
public class LISATests : SegmentationTestBase<float>
{
    private int[]? _outputShape;
    protected override int[] InputShape => TextQueriedScaffold.InputShape;
    protected override int[] OutputShape => _outputShape ??= TextQueriedScaffold.ProbeOutputShape(CreateNetwork);

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new LISA<float>(TextQueriedScaffold.Architecture(), TextQueriedScaffold.Encoder(),
            options: new LISAOptions { NumClasses = TextQueriedScaffold.NumClasses, DropRate = 0.0 });
}

/// <summary>Manual scaffold for OMG-LLaVA (Zhang et al. 2024).</summary>
public class OMGLLaVATests : SegmentationTestBase<float>
{
    private int[]? _outputShape;
    protected override int[] InputShape => TextQueriedScaffold.InputShape;
    protected override int[] OutputShape => _outputShape ??= TextQueriedScaffold.ProbeOutputShape(CreateNetwork);

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new OMGLLaVA<float>(TextQueriedScaffold.Architecture(), TextQueriedScaffold.Encoder(),
            options: new OMGLLaVAOptions { NumClasses = TextQueriedScaffold.NumClasses, DropRate = 0.0 });
}

/// <summary>Manual scaffold for PixelLM (Ren et al. 2024).</summary>
public class PixelLMTests : SegmentationTestBase<float>
{
    private int[]? _outputShape;
    protected override int[] InputShape => TextQueriedScaffold.InputShape;
    protected override int[] OutputShape => _outputShape ??= TextQueriedScaffold.ProbeOutputShape(CreateNetwork);

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new PixelLM<float>(TextQueriedScaffold.Architecture(), TextQueriedScaffold.Encoder(),
            options: new PixelLMOptions { NumClasses = TextQueriedScaffold.NumClasses, DropRate = 0.0 });
}

/// <summary>
/// Manual scaffold for VideoLISA (Bai et al. 2024), at the reduced width and depth its generated
/// fixture used: the paper-default 70.8M-parameter model overruns the per-test CPU budget.
/// </summary>
public class VideoLISATests : SegmentationTestBase<float>
{
    private int[]? _outputShape;
    protected override int[] InputShape => TextQueriedScaffold.InputShape;
    protected override int[] OutputShape => _outputShape ??= TextQueriedScaffold.ProbeOutputShape(CreateNetwork);

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new VideoLISA<float>(TextQueriedScaffold.Architecture(), TextQueriedScaffold.Encoder(),
            options: new VideoLISAOptions
            {
                NumClasses = TextQueriedScaffold.NumClasses,
                DropRate = 0.0,
                ChannelDimensions = new[] { 16, 32, 64, 96 },
                EncoderDepths = new[] { 1, 1, 1, 1 },
                DecoderDimension = 32,
                LearningRate = 1e-4,
                WeightDecay = 1e-4
            });
}
