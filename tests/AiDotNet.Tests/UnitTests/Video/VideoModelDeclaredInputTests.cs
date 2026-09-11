using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Generation;
using AiDotNet.Video.Inpainting;
using AiDotNet.Video.Options;
using AiDotNet.Video.Understanding;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

/// <summary>
/// Video models must run on a tensor of their own declared input shape (<c>GetInputShape()</c>).
/// </summary>
/// <remarks>
/// <para>
/// Each case here failed that probe for a model-specific reason, and each was judged on which side was
/// wrong:
/// </para>
/// <list type="bullet">
/// <item>BasicVSR++, OpenSora and ProPainter declare a single <c>[C, H, W]</c> frame or latent, which is a
/// valid input under their own contracts (a one-frame clip; a single latent; a one-frame clip), but their
/// Predict indexed a fourth axis it never added. Predict was wrong.</item>
/// <item>VideoCLIP's input layout is <c>[Frames, C, H, W]</c> and it owns a clip length (<c>numFrames</c>),
/// but its default architecture declared a single <c>[3, 224, 224]</c> frame. The declaration was
/// wrong.</item>
/// </list>
/// </remarks>
public class VideoModelDeclaredInputTests
{
    private static NeuralNetworkArchitecture<double> FrameArchitecture(int size) => new(
        inputType: InputType.ThreeDimensional,
        taskType: NeuralNetworkTaskType.Regression,
        inputHeight: size, inputWidth: size, inputDepth: 3,
        outputSize: 2);

    private static Tensor<T> Random<T>(int[] shape, int seed, Func<double, T> convert)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<T>(shape);
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = convert(rng.NextDouble());
        }
        return t;
    }

    private static Tensor<double> Random(int[] shape, int seed) => Random(shape, seed, v => v);

    private static Tensor<double> WithLeadingAxis(Tensor<double> tensor)
    {
        var shape = new int[tensor.Rank + 1];
        shape[0] = 1;
        for (int i = 0; i < tensor.Rank; i++) shape[i + 1] = tensor.Shape[i];
        var result = new Tensor<double>(shape);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private static void AssertSameValues(Tensor<double> expected, Tensor<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        var e = expected.Data.Span;
        var a = actual.Data.Span;
        for (int i = 0; i < e.Length; i++)
        {
            Assert.Equal(e[i], a[i], 10);
        }
    }

    [Fact]
    public void BasicVSRPlusPlus_UpscalesItsDeclaredSingleFrame_AsAOneFrameClip()
    {
        const int size = 32;
        var model = new BasicVSRPlusPlus<double>(
            FrameArchitecture(size), scaleFactor: 2, numFeatures: 8, numResidualBlocks: 1, numPropagations: 1);

        int[] declared = model.GetArchitecture().GetInputShape();
        Assert.Equal(new[] { 3, size, size }, declared);

        // It used to read [C, H, W] as a clip of C two-dimensional frames and fail inside the feature
        // extractor / flow estimator.
        var frame = Random(declared, seed: 1);
        var upscaled = model.Predict(frame);

        Assert.Equal(new[] { 3, size * 2, size * 2 }, upscaled.Shape.ToArray());

        // Exactly the one-frame clip [1, C, H, W], without its frame axis.
        var clip = model.Predict(WithLeadingAxis(frame));
        Assert.Equal(new[] { 1, 3, size * 2, size * 2 }, clip.Shape.ToArray());
        AssertSameValues(clip, upscaled);
    }

    [Fact]
    public void OpenSora_DenoisesItsDeclaredSingleLatent()
    {
        const int size = 16;
        var model = new OpenSora<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.Generative,
                inputHeight: size, inputWidth: size, inputDepth: 3,
                outputSize: size * size * 3),
            numFrames: 2, hiddenDim: 32, numLayers: 1, numInferenceSteps: 4);

        int[] declared = model.GetArchitecture().GetInputShape();
        Assert.Equal(new[] { 3, size, size }, declared);

        // The denoiser indexed axis 3 of its features, so a rank-3 latent threw IndexOutOfRange.
        var latent = Random(declared, seed: 2);
        var denoised = model.Predict(latent);

        Assert.Equal(new[] { 3, size, size }, denoised.Shape.ToArray());
        AssertSameValues(model.Predict(WithLeadingAxis(latent)), denoised);
    }

    [Fact]
    public void ProPainter_ReconstructsItsDeclaredSingleFrame_AsAOneFrameClip()
    {
        const int size = 16;
        var model = new ProPainter<double>(FrameArchitecture(size), numFeatures: 8, numTransformerBlocks: 1, numHeads: 2);

        int[] declared = model.GetArchitecture().GetInputShape();
        Assert.Equal(new[] { 3, size, size }, declared);

        // The image path read axis 3, so a rank-3 frame threw IndexOutOfRange.
        var frame = Random(declared, seed: 3);
        var reconstructed = model.Predict(frame);

        Assert.Equal(3, reconstructed.Rank);
        Assert.Equal(new[] { size, size }, reconstructed.Shape.ToArray().Skip(1).ToArray());
        AssertSameValues(model.Predict(WithLeadingAxis(frame)), reconstructed);
    }

    [Fact]
    public void VideoCLIP_DefaultConstructor_DeclaresTheClipItEncodes()
    {
        using var model = new VideoCLIP<float>();

        // Its input layout is [Frames, C, H, W] and its default clip is 32 frames; it used to declare a
        // single [3, 224, 224] frame (InputType.ThreeDimensional), which EncodeVideo cannot take.
        Assert.Equal(InputType.FourDimensional, model.GetArchitecture().InputType);
        Assert.Equal(new[] { 32, 3, 224, 224 }, model.GetArchitecture().GetInputShape());
        Assert.Equal(32, model.NumFrames);
    }

    private static VideoCLIP<float> SmallVideoCLIP(NeuralNetworkArchitecture<float> architecture, int? numFrames = null)
    {
        var options = new VideoCLIPVideoOptions
        {
            HiddenDimension = 32,
            NumSpatialBlocks = 1,
            NumTemporalBlocks = 1,
            NumTextBlocks = 1,
            WarmupSteps = 0,
        };

        return numFrames is int frames
            ? new VideoCLIP<float>(architecture, numFrames: frames, embeddingDim: 4, textMaxLength: 8, vocabSize: 64, options: options)
            : new VideoCLIP<float>(architecture, embeddingDim: 4, textMaxLength: 8, vocabSize: 64, options: options);
    }

    private static NeuralNetworkArchitecture<float> ClipArchitecture(int frames) => new(
        inputType: InputType.FourDimensional,
        taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputFrames: frames, inputDepth: 3, inputHeight: 32, inputWidth: 32,
        outputSize: 4);

    [Fact]
    public void VideoCLIP_TakesTheArchitecturesFrameCount_AndPredictsOnTheDeclaredClip()
    {
        using var model = SmallVideoCLIP(ClipArchitecture(frames: 4));

        Assert.Equal(4, model.NumFrames);
        Assert.Equal(4, model.GetModelMetadata().AdditionalInfo["NumFrames"]);

        int[] declared = model.GetArchitecture().GetInputShape();
        Assert.Equal(new[] { 4, 3, 32, 32 }, declared);

        var embedding = model.Predict(Random(declared, seed: 4, v => (float)v));
        Assert.Equal(4, embedding.Length); // one embeddingDim-wide vector for the one clip
    }

    [Fact]
    public void VideoCLIP_DeclaringTheClip_DoesNotChangeTheResolvedLayerStack()
    {
        // The layer stack runs per frame, and its lazy layers are resolved ahead of the first forward
        // from the architecture's declared input. Declaring the clip [Frames, C, H, W] instead of one frame
        // must not leave the stack unresolved: ParameterCount at construction (which weight-streaming and
        // other memory decisions read) has to match the one-frame declaration's.
        var frameArchitecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputDepth: 3, inputHeight: 32, inputWidth: 32, outputSize: 4);

        using var declaredAsFrame = SmallVideoCLIP(frameArchitecture, numFrames: 4);
        using var declaredAsClip = SmallVideoCLIP(ClipArchitecture(frames: 4));

        Assert.True(declaredAsFrame.ParameterCount > 0);
        Assert.Equal(declaredAsFrame.ParameterCount, declaredAsClip.ParameterCount);
    }

    [Fact]
    public void VideoCLIP_ConflictingExplicitFrameCount_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() => SmallVideoCLIP(ClipArchitecture(frames: 4), numFrames: 8));
        Assert.Equal("numFrames", ex.ParamName);
    }
}
