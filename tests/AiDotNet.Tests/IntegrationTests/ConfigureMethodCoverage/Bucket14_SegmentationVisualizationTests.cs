using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 14 — ConfigureSegmentationVisualization, and the renderer that finally consumes it.
/// </summary>
/// <remarks>
/// <para>
/// Before the fix in this PR this surface was the last fully-inert Configure* method on the facade:
/// <c>_segmentationVisualizationConfig</c> was assigned, exposed through an internal accessor nobody
/// called, and there was no segmentation-overlay renderer anywhere in the library for it to feed.
/// The analyzer rule AIDN097 exists to make exactly that shape fail the build.
/// </para>
/// <para>
/// The renderer follows <c>torchvision.utils.draw_segmentation_masks(image, masks, alpha, colors)</c>:
/// a free function returning a new image tensor, because visualization is an evaluation-time concern.
/// The configured defaults still ride through to the result so a caller who set them once on the
/// builder does not have to repeat them at the call site.
/// </para>
/// </remarks>
[Collection("ConfigureMethodCoverage")]
public class Bucket14_SegmentationVisualizationTests : ConfigureMethodTestBase
{
    /// <summary>
    /// The stored-but-not-consumed regression guard: configure a NON-default visualization and assert
    /// the built result still carries it. If the field is ever orphaned again this fails.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureSegmentationVisualization_FlowsThroughToTheBuiltResult()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        // This test verifies configuration transport, not optimization. A parameter-free,
        // self-supervised model keeps that contract isolated from architecture cloning.
        var model = new PassThroughSelfSupervisedModel();

        var config = new SegmentationVisualizationConfig
        {
            Alpha = 0.25,               // deliberately not the 0.5 default
            DrawContours = false,       // deliberately not the true default
            ContourThickness = 7,
            MinDisplayConfidence = 0.9,
        };

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureSegmentationVisualization(config)
            .BuildAsync();

        Assert.NotNull(result.Options);
        var carried = result.Options!.SegmentationVisualization;

        Assert.NotNull(carried);
        Assert.Same(config, result.SegmentationVisualization);
        Assert.Equal(0.25, carried!.Alpha);
        Assert.False(carried.DrawContours);
        Assert.Equal(7, carried.ContourThickness);
        Assert.Equal(0.9, carried.MinDisplayConfidence);

        var withParameters = Assert.IsType<AiDotNet.Models.Results.AiModelResult<float, Tensor<float>, Tensor<float>>>(
            result.WithParameters(result.GetParameters()));
        var deepCopy = Assert.IsType<AiDotNet.Models.Results.AiModelResult<float, Tensor<float>, Tensor<float>>>(
            result.DeepCopy());
        Assert.Same(config, withParameters.SegmentationVisualization);
        Assert.Same(config, deepCopy.SegmentationVisualization);
    }

    /// <summary>
    /// Only pixels inside the mask may change. A renderer that tinted the whole image would still
    /// "look right" on a screenshot, so this asserts the untouched region is bit-identical.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void DrawSegmentationMasks_BlendsInsideTheMaskOnly()
    {
        var image = SolidImage(8, 8, 0.0f);
        var masks = new Tensor<float>([1, 8, 8]);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                masks[0, y, x] = 1.0f;

        var rendered = SegmentationRenderer.DrawSegmentationMasks(image, masks, alpha: 1.0);

        bool anyInsideChanged = false;
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                bool inside = y < 4 && x < 4;
                float sum = rendered[0, y, x] + rendered[1, y, x] + rendered[2, y, x];
                if (inside)
                {
                    if (sum > 0f) anyInsideChanged = true;
                }
                else
                {
                    Assert.True(sum == 0f,
                        $"Pixel ({y},{x}) is outside every mask but was modified (sum={sum}).");
                }
            }
        }

        Assert.True(anyInsideChanged, "No pixel inside the mask was tinted — the overlay did nothing.");
    }

    /// <summary>
    /// alpha = 0 is the identity. Guards the blend direction: a renderer with the weights swapped
    /// would replace the image at alpha 0 instead of leaving it alone.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void DrawSegmentationMasks_AlphaZero_LeavesTheImageUnchanged()
    {
        var image = SolidImage(4, 4, 0.375f);
        var masks = OnesMask(4, 4);

        var rendered = SegmentationRenderer.DrawSegmentationMasks(image, masks, alpha: 0.0);

        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    Assert.Equal(0.375f, rendered[c, y, x], 5);
    }

    /// <summary>
    /// A [0,1] image must come back in [0,1]. Mixing an 8-bit palette into a normalized image without
    /// rescaling would saturate every masked pixel to white — visible only much later, as a washed-out
    /// picture, which is why it is asserted rather than eyeballed.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void DrawSegmentationMasks_PreservesTheInputValueRange()
    {
        var normalized = SolidImage(4, 4, 0.5f);
        var rendered = SegmentationRenderer.DrawSegmentationMasks(normalized, OnesMask(4, 4), alpha: 1.0);

        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    Assert.InRange(rendered[c, y, x], 0.0f, 1.0f);

        var eightBit = SolidImage(4, 4, 128.0f);
        var renderedEightBit = SegmentationRenderer.DrawSegmentationMasks(eightBit, OnesMask(4, 4), alpha: 1.0);

        bool anyAboveOne = false;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                {
                    Assert.InRange(renderedEightBit[c, y, x], 0.0f, 255.0f);
                    if (renderedEightBit[c, y, x] > 1.0f) anyAboveOne = true;
                }

        Assert.True(anyAboveOne,
            "A [0,255] image was rendered back into the [0,1] range — the palette was not scaled to the input range.");
    }

    /// <summary>
    /// The generated palette walks hue by the golden-ratio conjugate rather than drawing random
    /// colours, so the same masks render identically on every run and machine.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void DrawSegmentationMasks_IsDeterministic()
    {
        var masks = new Tensor<float>([3, 6, 6]);
        for (int m = 0; m < 3; m++)
            for (int x = 0; x < 6; x++)
                masks[m, m * 2, x] = 1.0f;

        var first = SegmentationRenderer.DrawSegmentationMasks(SolidImage(6, 6, 0.1f), masks);
        var second = SegmentationRenderer.DrawSegmentationMasks(SolidImage(6, 6, 0.1f), masks);

        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 6; y++)
                for (int x = 0; x < 6; x++)
                    Assert.Equal(first[c, y, x], second[c, y, x], 6);
    }

    /// <summary>
    /// A bare mask stack carries no classes or scores, so the mask-only overload cannot label. It must
    /// throw and point at the overload that can, rather than no-op — silently ignoring the setting is
    /// the defect this PR exists to remove.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void ShowLabels_OnTheMaskOnlyOverload_ThrowsRatherThanSilentlyIgnoringTheSetting()
    {
        var image = SolidImage(4, 4, 0.5f);
        var masks = OnesMask(4, 4);

        var withLabels = new SegmentationVisualizationConfig { ShowLabels = true };
        var ex = Assert.Throws<NotSupportedException>(
            () => SegmentationRenderer.DrawSegmentationMasks(image, masks, withLabels));
        Assert.Contains("Render(image, SegmentationOutput, config)", ex.Message);

        var withScores = new SegmentationVisualizationConfig { ShowLabels = false, ShowScores = true };
        Assert.Throws<NotSupportedException>(
            () => SegmentationRenderer.DrawSegmentationMasks(image, masks, withScores));

        // ...and with both off it renders normally, so the guard is not simply rejecting everything.
        var plain = new SegmentationVisualizationConfig { ShowLabels = false, ShowScores = false };
        var rendered = SegmentationRenderer.DrawSegmentationMasks(image, masks, plain);
        Assert.Equal(3, rendered.Shape[0]);
    }

    /// <summary>
    /// Labels genuinely render through <see cref="SegmentationRenderer.Render{T}"/>, using the embedded
    /// 5x7 bitmap font. Asserted by comparing against the same render with labels off: enabling labels
    /// must change pixels OUTSIDE the mask (the text sits above it), which a no-op cannot do.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Render_WithShowLabels_ActuallyDrawsText()
    {
        var image = SolidImage(32, 48, 0.0f);
        var instanceMasks = new Tensor<float>([1, 32, 48]);
        for (int y = 12; y < 20; y++)
            for (int x = 4; x < 30; x++)
                instanceMasks[0, y, x] = 1.0f;

        var output = new SegmentationOutput<float>
        {
            InstanceMasks = instanceMasks,
            InstanceClasses = new[] { 7 },
            InstanceScores = new[] { 0.87f },
            NumClasses = 10,
            ImageHeight = 32,
            ImageWidth = 48,
        };

        var withoutLabels = SegmentationRenderer.Render(image, output,
            new SegmentationVisualizationConfig { Alpha = 1.0, DrawContours = false, ShowLabels = false });
        var withLabels = SegmentationRenderer.Render(image, output,
            new SegmentationVisualizationConfig { Alpha = 1.0, DrawContours = false, ShowLabels = true, ShowScores = true });

        int changedOutsideMask = 0;
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 48; x++)
                if (instanceMasks[0, y, x] == 0f
                    && withLabels[0, y, x] != withoutLabels[0, y, x])
                    changedOutsideMask++;

        Assert.True(changedOutsideMask > 0,
            "Enabling ShowLabels/ShowScores changed no pixel outside the mask — the label was not drawn.");
    }

    /// <summary>
    /// The default config is renderable. ShowLabels defaults to true, so a renderer that could not draw
    /// text would make the most obvious call — Render(image, output) — throw.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Render_WithDefaultConfig_Succeeds()
    {
        var image = SolidImage(24, 24, 0.2f);
        var instanceMasks = new Tensor<float>([1, 24, 24]);
        for (int y = 8; y < 16; y++)
            for (int x = 8; x < 16; x++)
                instanceMasks[0, y, x] = 1.0f;

        var output = new SegmentationOutput<float>
        {
            InstanceMasks = instanceMasks,
            InstanceClasses = new[] { 1 },
            NumClasses = 2,
            ImageHeight = 24,
            ImageWidth = 24,
        };

        var rendered = SegmentationRenderer.Render(image, output);
        Assert.Equal(3, rendered.Shape[0]);
    }

    /// <summary>
    /// Instances scoring below MinDisplayConfidence must not be drawn — the threshold is a real filter,
    /// not a stored number.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Render_SkipsInstancesBelowMinDisplayConfidence()
    {
        var image = SolidImage(4, 4, 0.0f);
        var instanceMasks = new Tensor<float>([2, 4, 4]);
        for (int x = 0; x < 4; x++)
        {
            instanceMasks[0, 0, x] = 1.0f;   // high-scoring instance, row 0
            instanceMasks[1, 3, x] = 1.0f;   // low-scoring instance, row 3
        }

        var output = new SegmentationOutput<float>
        {
            InstanceMasks = instanceMasks,
            InstanceScores = new[] { 0.95f, 0.10f },
            NumClasses = 3,
            ImageHeight = 4,
            ImageWidth = 4,
        };

        var config = new SegmentationVisualizationConfig
        {
            Alpha = 1.0,
            DrawContours = false,
            MinDisplayConfidence = 0.5,
            ShowLabels = false,   // keep label pixels from polluting the row-only filter assertions
        };

        var rendered = SegmentationRenderer.Render(image, output, config);

        float keptRow = rendered[0, 0, 0] + rendered[1, 0, 0] + rendered[2, 0, 0];
        float droppedRow = rendered[0, 3, 0] + rendered[1, 3, 0] + rendered[2, 3, 0];

        Assert.True(keptRow > 0f, "The instance scoring 0.95 was above the 0.5 threshold but was not drawn.");
        Assert.True(droppedRow == 0f,
            $"The instance scoring 0.10 was below the 0.5 threshold but was still drawn (sum={droppedRow}).");
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task Render_ClassMapUsesTheClassIdAsThePaletteIndex()
    {
        await Task.Yield();
        var classMap = new Tensor<float>([2, 2]);
        classMap[0, 0] = 1.0f;
        var output = new SegmentationOutput<float>
        {
            ClassMap = classMap,
            NumClasses = 2,
            ImageHeight = 2,
            ImageWidth = 2,
        };
        var palette = new byte[,] { { 255, 0, 0 }, { 0, 255, 0 } };

        var rendered = SegmentationRenderer.Render(
            SolidImage(2, 2, 0.0f),
            output,
            new SegmentationVisualizationConfig
            {
                Alpha = 1.0,
                DrawContours = false,
                ShowLabels = false,
                ShowScores = false,
                ColorPalette = palette,
            });

        Assert.Equal(0.0f, rendered[0, 0, 0], 6);
        Assert.Equal(1.0f, rendered[1, 0, 0], 6);
        Assert.Equal(0.0f, rendered[2, 0, 0], 6);
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task Render_ConfidenceFilterAppliesToBoxesAndKeepsMaskColorsAligned()
    {
        await Task.Yield();
        var masks = new Tensor<float>([2, 8, 8]);
        masks[0, 1, 1] = 1.0f;
        masks[1, 5, 5] = 1.0f;
        var boxes = new Tensor<float>([2, 4]);
        boxes[0, 0] = 0; boxes[0, 1] = 0; boxes[0, 2] = 2; boxes[0, 3] = 2;
        boxes[1, 0] = 4; boxes[1, 1] = 4; boxes[1, 2] = 6; boxes[1, 3] = 6;
        var output = new SegmentationOutput<float>
        {
            InstanceMasks = masks,
            InstanceBoxes = boxes,
            InstanceScores = new[] { 0.1f, 0.9f },
            NumClasses = 2,
            ImageHeight = 8,
            ImageWidth = 8,
        };

        var rendered = SegmentationRenderer.Render(
            SolidImage(8, 8, 0.0f),
            output,
            new SegmentationVisualizationConfig
            {
                Alpha = 1.0,
                DrawContours = false,
                ShowBoundingBoxes = true,
                ShowLabels = false,
                ShowScores = false,
                MinDisplayConfidence = 0.5,
                ColorPalette = new byte[,] { { 255, 0, 0 }, { 0, 255, 0 } },
            });

        Assert.Equal(0.0f, rendered[0, 0, 0]);
        Assert.Equal(0.0f, rendered[1, 0, 0]);
        Assert.Equal(0.0f, rendered[2, 0, 0]);
        Assert.Equal(1.0f, rendered[0, 4, 4], 6);
        Assert.Equal(0.0f, rendered[1, 4, 4], 6);
        Assert.Equal(1.0f, rendered[0, 5, 5], 6);
        Assert.Equal(0.0f, rendered[1, 5, 5], 6);
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task Render_InvalidBoxShapesFailWithTheInstanceBoxContract()
    {
        await Task.Yield();
        var masks = OnesMask(4, 4);
        var config = new SegmentationVisualizationConfig
        {
            ShowBoundingBoxes = true,
            ShowLabels = false,
            ShowScores = false,
        };

        ArgumentException rankError = Assert.Throws<ArgumentException>(() => SegmentationRenderer.Render(
            SolidImage(4, 4, 0.0f),
            new SegmentationOutput<float> { InstanceMasks = masks, InstanceBoxes = new Tensor<float>([4]), NumClasses = 1 },
            config));
        ArgumentException widthError = Assert.Throws<ArgumentException>(() => SegmentationRenderer.Render(
            SolidImage(4, 4, 0.0f),
            new SegmentationOutput<float> { InstanceMasks = masks, InstanceBoxes = new Tensor<float>([1, 3]), NumClasses = 1 },
            config));

        Assert.Contains("[N,4]", rankError.Message, StringComparison.Ordinal);
        Assert.Contains("[N,4]", widthError.Message, StringComparison.Ordinal);
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task Render_LabelAnchorComesFromARealTopmostMaskPixel()
    {
        await Task.Yield();
        var masks = new Tensor<float>([1, 32, 64]);
        masks[0, 15, 20] = 1.0f; // topmost pixel
        masks[0, 20, 5] = 1.0f;  // leftmost pixel on a different row
        var output = new SegmentationOutput<float>
        {
            InstanceMasks = masks,
            NumClasses = 1,
            ImageHeight = 32,
            ImageWidth = 64,
        };

        var rendered = SegmentationRenderer.Render(
            SolidImage(32, 64, 0.0f),
            output,
            new SegmentationVisualizationConfig
            {
                Alpha = 0.0,
                DrawContours = false,
                ShowLabels = true,
                ShowScores = false,
            });

        int minimumChangedX = int.MaxValue;
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 64; x++)
                if (rendered[0, y, x] != 0.0f || rendered[1, y, x] != 0.0f || rendered[2, y, x] != 0.0f)
                    minimumChangedX = Math.Min(minimumChangedX, x);

        Assert.NotEqual(int.MaxValue, minimumChangedX);
        Assert.True(minimumChangedX >= 20,
            $"The label started at x={minimumChangedX}; it combined unrelated min-X/min-Y extents instead of anchoring to the topmost mask pixel at x=20.");
    }

    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task BitmapFontRejectsNonRgbDestinationsWithAnActionableContract()
    {
        await Task.Yield();
        var error = Assert.Throws<ArgumentException>(() => BitmapFont5x7.DrawText(
            new Tensor<float>([2, 4, 4]),
            AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>(),
            "label",
            0,
            0,
            1.0,
            1.0,
            1.0));
        Assert.Contains("[3,H,W]", error.Message, StringComparison.Ordinal);
    }

    private static Tensor<float> SolidImage(int height, int width, float value)
    {
        var image = new Tensor<float>([3, height, width]);
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    image[c, y, x] = value;
        return image;
    }

    private static Tensor<float> OnesMask(int height, int width)
    {
        var mask = new Tensor<float>([1, height, width]);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                mask[0, y, x] = 1.0f;
        return mask;
    }

    private sealed class PassThroughSelfSupervisedModel :
        IFullModel<float, Tensor<float>, Tensor<float>>, ISelfSupervisedModel
    {
        public long ParameterCount => 0;
        public bool SupportsParameterInitialization => false;
        public ILossFunction<float> DefaultLossFunction => new MeanSquaredErrorLoss<float>();

        public Vector<float> SanitizeParameters(Vector<float> parameters) => parameters;
        public Vector<float> GetParameters() => new(0);

        public void SetParameters(Vector<float> parameters)
        {
            if (parameters is null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }
            if (parameters.Length != 0)
            {
                throw new ArgumentException("This test model is parameter-free.", nameof(parameters));
            }
        }

        public IFullModel<float, Tensor<float>, Tensor<float>> WithParameters(Vector<float> parameters)
        {
            SetParameters(parameters);
            return this;
        }

        public Tensor<float> Predict(Tensor<float> input) => input;
        public void Train(Tensor<float> input, Tensor<float> expectedOutput) { }

        public ModelMetadata<float> GetModelMetadata() => new()
        {
            Name = nameof(PassThroughSelfSupervisedModel),
            FeatureCount = 1,
            Complexity = 1,
        };

        public byte[] Serialize() => [];
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public Dictionary<string, float> GetFeatureImportance() => [];
        public IFullModel<float, Tensor<float>, Tensor<float>> DeepCopy() => this;
        public IFullModel<float, Tensor<float>, Tensor<float>> Clone() => this;
        public void Dispose() { }
    }
}
