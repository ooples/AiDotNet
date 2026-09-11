using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.ActionRecognition;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.ActionRecognition;

/// <summary>
/// TimeSformer must build for the clip length its architecture declares, and follow the library's
/// unbatched-in / unbatched-out and per-sample-softmax conventions.
/// </summary>
/// <remarks>
/// The frame count sizes the positional table (<c>frames * patches + 1</c> tokens) and is the group size the
/// divided space-time blocks fall back to when run without an explicit frame count. The constructor used to
/// take it from <c>numFrames</c> (default 8) and ignore <c>Architecture.InputFrames</c>, so a 16-frame
/// architecture built an 8-frame model: it reported 8, grouped by 8, and could not run its own declared
/// clip. The rule is VideoMAE's: the architecture wins, a conflicting explicit value throws.
/// </remarks>
public class TimeSformerFrameCountTests
{
    private const int Size = 32;
    private const int Patch = 16;
    private const int Classes = 5;

    private static NeuralNetworkArchitecture<double> ClipArchitecture(int frames) => new(
        inputType: InputType.FourDimensional,
        taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputFrames: frames, inputDepth: 3, inputHeight: Size, inputWidth: Size,
        outputSize: Classes);

    private static TimeSformer<double> Small(NeuralNetworkArchitecture<double> architecture, int? numFrames = null)
        => numFrames is int frames
            ? new TimeSformer<double>(architecture, numClasses: Classes, embedDim: 16, numHeads: 2, numLayers: 1,
                numFrames: frames, patchSize: Patch)
            : new TimeSformer<double>(architecture, numClasses: Classes, embedDim: 16, numHeads: 2, numLayers: 1,
                patchSize: Patch);

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<double>(shape);
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = rng.NextDouble();
        }
        return t;
    }

    [Fact]
    public void Constructor_UsesTheFrameCountDeclaredByTheArchitecture()
    {
        var model = Small(ClipArchitecture(frames: 16));

        Assert.Equal(16, model.NumFrames);
        Assert.Equal(16, model.GetModelMetadata().AdditionalInfo["NumFrames"]);

        // Every divided space-time block is built for the same clip length.
        var blocks = model.Layers.OfType<TimeSformerBlockLayer<double>>().ToList();
        Assert.NotEmpty(blocks);
        Assert.All(blocks, block => Assert.Equal("16", block.GetMetadata()["NumFrames"]));
    }

    [Fact]
    public void Predict_OnTheDeclaredSixteenFrameClip_Succeeds()
    {
        var model = Small(ClipArchitecture(frames: 16));

        // [16, 3, 32, 32] -> 16 frames x 4 patches + CLS = 65 tokens, which the 16-frame model's positional
        // table is sized for. Unbatched in, so one unbatched probability vector out.
        var probabilities = model.Predict(Random(model.Architecture.GetInputShape(), seed: 1));

        Assert.Equal(new[] { Classes }, probabilities.Shape.ToArray());
    }

    [Fact]
    public void BlockPlainForward_GroupsTokensByTheDeclaredFrameCount()
    {
        var model = Small(ClipArchitecture(frames: 16));
        var block = model.Layers.OfType<TimeSformerBlockLayer<double>>().First();

        // A 16-frame clip's tokens: 1 CLS + 16 frames x 4 patches, 16 wide.
        var tokens = Random([1, 1 + 16 * 4, 16], seed: 2);

        // The frame-count-less Forward (used by anything that runs the layer on its own) must group the
        // patch tokens into the same 16 frames the tokenizer produced, not the default 8.
        var plain = block.Forward(tokens);
        var explicitSixteen = block.Forward(tokens, 16);

        Assert.Equal(explicitSixteen.Length, plain.Length);
        for (int i = 0; i < plain.Length; i++)
        {
            Assert.Equal(explicitSixteen.Data.Span[i], plain.Data.Span[i], 12);
        }
    }

    [Fact]
    public void Constructor_ConflictingExplicitFrameCount_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() => Small(ClipArchitecture(frames: 16), numFrames: 4));
        Assert.Equal("numFrames", ex.ParamName);
    }

    [Fact]
    public void Constructor_WithoutADeclaredFrameCount_KeepsTheNumFramesArgument()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: Classes);

        Assert.Equal(4, Small(arch, numFrames: 4).NumFrames);
    }

    [Fact]
    public void Predict_UnbatchedClip_ReturnsAnUnbatchedProbabilityVector()
    {
        var model = Small(ClipArchitecture(frames: 4));

        // Classify documents [NumClasses] for a [T, C, H, W] clip, the output layout is batch-optional,
        // and the base PredictCore squeezes the batch it adds. It returned [1, NumClasses].
        var probabilities = model.Predict(Random([4, 3, Size, Size], seed: 3));

        Assert.Equal(new[] { Classes }, probabilities.Shape.ToArray());
    }

    [Fact]
    public void Predict_Batch_NormalisesEachClipSeparately()
    {
        var model = Small(ClipArchitecture(frames: 4));

        var probabilities = model.Predict(Random([3, 4, 3, Size, Size], seed: 4));

        // One probability vector per clip. The softmax used to run over the WHOLE [B, NumClasses] tensor,
        // so the three rows summed to 1 together instead of each summing to 1.
        Assert.Equal(new[] { 3, Classes }, probabilities.Shape.ToArray());
        for (int b = 0; b < 3; b++)
        {
            double sum = 0;
            for (int c = 0; c < Classes; c++)
            {
                double p = probabilities[b, c];
                Assert.InRange(p, 0.0, 1.0);
                sum += p;
            }

            Assert.Equal(1.0, sum, 9);
        }
    }
}
