using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.ActionRecognition;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.ActionRecognition;

/// <summary>
/// Regression tests for VideoMAE's masked-autoencoder pretraining path and its frame-count wiring.
/// </summary>
/// <remarks>
/// VideoMAE's default layer stack is laid out as
/// <c>[patch-embed | 12 encoder blocks | reduce-conv, pool, classifier | 4 decoder blocks | reconstruction head]</c>.
/// These tests pin the three places where the model's own bookkeeping disagreed with that layout or with
/// the paper: the decoder started one slot early (on the classifier), the tube mask sized its masked count
/// from tubelets x spatial patches while sampling only spatial patches (so every patch was masked once a
/// clip had two or more tubelets), and the architecture's declared frame count was ignored.
/// </remarks>
public class VideoMAEPretrainingTests
{
    private const int PatchSize = 16;
    private const int TubeletSize = 2;
    private const int Channels = 3;

    private static NeuralNetworkArchitecture<double> VideoArch(int frames, int size, int classes = 4) => new(
        inputType: InputType.FourDimensional,
        taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputFrames: frames,
        inputDepth: Channels,
        inputHeight: size,
        inputWidth: size,
        outputSize: classes);

    private static Tensor<double> RandomClip(int frames, int size, int seed)
    {
        var rng = new Random(seed);
        var clip = new Tensor<double>([1, frames, Channels, size, size]);
        var span = clip.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = rng.NextDouble();
        }
        return clip;
    }

    [Fact]
    public void DecodeForReconstruction_RunsTheDecoderBlocksAndReconstructionHead_NotTheClassifier()
    {
        const int frames = 4;
        const int size = 32;
        const int features = 16;
        var model = new VideoMAE<double>(VideoArch(frames, size), numClasses: 4, numFrames: frames, numFeatures: features);

        int numTubelets = frames / TubeletSize;
        int patches = size / PatchSize;

        // Encoder output layout: one [numFeatures, patchesH, patchesW] map per (clip, tubelet).
        var encoded = new Tensor<double>([numTubelets, features, patches, patches]);
        var rng = new Random(7);
        var span = encoded.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = rng.NextDouble();
        }

        var reconstruction = model.DecodeForReconstruction(encoded);

        // The reconstruction head predicts every pixel of a tubelet patch: channels * tubeletSize * P * P
        // values per patch position. Starting the decoder on the classifier's Dense layer instead produced
        // a numClasses- or numFeatures-wide map (the head, the last layer, was never reached).
        Assert.Equal(
            new[] { numTubelets, Channels * TubeletSize * PatchSize * PatchSize, patches, patches },
            reconstruction.Shape.ToArray());
    }

    [Fact]
    public void PretrainMAE_OnAMultiTubeletClip_ReturnsAFinitePositiveLoss()
    {
        const int frames = 4;
        const int size = 32;
        var model = new VideoMAE<double>(VideoArch(frames, size), numClasses: 4, numFrames: frames, numFeatures: 16, maskRatio: 0.5);

        double loss = model.PretrainMAE(RandomClip(frames, size, seed: 11));

        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), $"loss was {loss}");
        Assert.True(loss > 0.0, $"reconstruction loss of a random clip against an untrained decoder must be positive, was {loss}");
    }

    [Fact]
    public void ComputeReconstructionLoss_ComparesEachMaskedTubeletPatchWithItsOwnPixels()
    {
        const int frames = 4;
        const int size = 32;
        // Raw-pixel target, so the exact per-pixel correspondence is checkable by hand. The default
        // (per-patch normalised) target is pinned in VideoMAEPretrainingFidelityTests.
        var model = new VideoMAE<double>(VideoArch(frames, size), numClasses: 4, numFrames: frames, numFeatures: 8,
            options: new AiDotNet.Video.Options.VideoMAEOptions { NormalizeTarget = false });
        var clip = RandomClip(frames, size, seed: 13);

        int numTubelets = frames / TubeletSize;
        int patches = size / PatchSize;
        int patchDim = Channels * TubeletSize * PatchSize * PatchSize;

        // Only patch (0, 1) is masked.
        var mask = new bool[1, patches, patches];
        mask[0, 0, 1] = true;

        // The exact per-patch target: channel ((ts * C + c) * P + y) * P + x of tubelet t at patch (ph, pw)
        // is pixel [t * tubeletSize + ts, c, ph * P + y, pw * P + x] -- the same (ts, c) fold PatchEmbed uses.
        var perfect = new Tensor<double>([numTubelets, patchDim, patches, patches]);
        for (int t = 0; t < numTubelets; t++)
            for (int ts = 0; ts < TubeletSize; ts++)
                for (int c = 0; c < Channels; c++)
                    for (int y = 0; y < PatchSize; y++)
                        for (int x = 0; x < PatchSize; x++)
                            for (int ph = 0; ph < patches; ph++)
                                for (int pw = 0; pw < patches; pw++)
                                {
                                    int channel = (((ts * Channels) + c) * PatchSize + y) * PatchSize + x;
                                    perfect[t, channel, ph, pw] = clip[0, t * TubeletSize + ts, c, ph * PatchSize + y, pw * PatchSize + x];
                                }

        Assert.Equal(0.0, model.ComputeReconstructionLoss(perfect, clip, mask), 12);

        // Off by 1 on the masked patch only -> MSE exactly 1; errors on VISIBLE patches must not count.
        var shifted = perfect.Clone();
        for (int t = 0; t < numTubelets; t++)
            for (int ch = 0; ch < patchDim; ch++)
            {
                shifted[t, ch, 0, 1] = shifted[t, ch, 0, 1] + 1.0;
                shifted[t, ch, 1, 0] = shifted[t, ch, 1, 0] + 5.0; // visible: ignored
            }

        Assert.Equal(1.0, model.ComputeReconstructionLoss(shifted, clip, mask), 12);
    }

    [Theory]
    [InlineData(16, 64, 0.75, 12)]  // 8 tubelets, 4x4 = 16 spatial patches -> 12 masked
    [InlineData(4, 64, 0.5, 8)]     // 2 tubelets, 16 patches -> 8 masked
    [InlineData(16, 224, 0.9, 176)] // paper default: 8 tubelets, 14x14 = 196 patches -> int(0.9 * 196) = 176
    public void CreateTubeMask_MasksTheConfiguredFractionOfSpatialPatches_ForMultiTubeletClips(
        int frames, int size, double maskRatio, int expectedMaskedPerClip)
    {
        var model = new VideoMAE<double>(VideoArch(frames, size), numClasses: 4, numFrames: frames, numFeatures: 8, maskRatio: maskRatio);

        const int batch = 3;
        var mask = model.CreateTubeMask(batch);

        int patches = size / PatchSize;
        Assert.Equal(batch, mask.GetLength(0));
        Assert.Equal(patches, mask.GetLength(1));
        Assert.Equal(patches, mask.GetLength(2));

        // Tube masking (Tong et al. 2022, Sec. 3.3): ONE spatial mask per clip, shared by every tubelet,
        // with the ratio applied to the per-frame patch count. The old code computed the masked count from
        // numTubelets * spatialPatches but sampled only spatialPatches indices, so any clip with 2+
        // tubelets had every patch masked (nothing visible to the encoder).
        for (int b = 0; b < batch; b++)
        {
            int masked = 0;
            for (int h = 0; h < patches; h++)
            {
                for (int w = 0; w < patches; w++)
                {
                    if (mask[b, h, w])
                    {
                        masked++;
                    }
                }
            }
            Assert.Equal(expectedMaskedPerClip, masked);
        }
    }

    [Fact]
    public void Constructor_UsesTheFrameCountDeclaredByTheArchitecture()
    {
        var model = new VideoMAE<double>(VideoArch(frames: 8, size: 32), numClasses: 4, numFeatures: 8);

        Assert.Equal(8, model.NumFrames);
        Assert.Equal(8, model.GetModelMetadata().AdditionalInfo["NumFrames"]);
    }

    [Fact]
    public void Constructor_AgreeingExplicitFrameCount_IsAccepted()
    {
        var model = new VideoMAE<double>(VideoArch(frames: 4, size: 32), numClasses: 4, numFrames: 4, numFeatures: 8);

        Assert.Equal(4, model.NumFrames);
    }

    [Fact]
    public void Constructor_ConflictingExplicitFrameCount_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new VideoMAE<double>(VideoArch(frames: 8, size: 32), numClasses: 4, numFrames: 4, numFeatures: 8));

        Assert.Equal("numFrames", ex.ParamName);
    }

    [Fact]
    public void Constructor_WithoutADeclaredFrameCount_KeepsTheNumFramesArgument()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputDepth: Channels, inputHeight: 32, inputWidth: 32, outputSize: 4);

        var model = new VideoMAE<double>(arch, numClasses: 4, numFrames: 6, numFeatures: 8);

        Assert.Equal(6, model.NumFrames);
    }
}
