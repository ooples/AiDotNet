using AiDotNet.ComputerVision.Segmentation.Common;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 14 — ConfigureSegmentationVisualization, and the renderer that finally consumes it.
/// </summary>
/// <remarks>
/// <para>
/// Before the fix in this PR this surface was the last fully-inert Configure* method on the facade:
/// <c>_segmentationVisualizationConfig</c> was assigned, exposed through an internal accessor nobody
/// called, and there was no segmentation-overlay renderer anywhere in the library for it to feed.
/// The analyzer rule AIDN091 exists to make exactly that shape fail the build.
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
    private readonly ITestOutputHelper _output;
    public Bucket14_SegmentationVisualizationTests(ITestOutputHelper output) { _output = output; }

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
        var model = MakeCanaryModel();

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
        Assert.Equal(0.25, carried!.Alpha);
        Assert.False(carried.DrawContours);
        Assert.Equal(7, carried.ContourThickness);
        Assert.Equal(0.9, carried.MinDisplayConfidence);
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
    /// ShowLabels/ShowScores need a glyph rasterizer this library does not depend on. They must throw
    /// rather than no-op: silently ignoring them would be the very defect this PR exists to remove.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void ShowLabels_ThrowsRatherThanSilentlyIgnoringTheSetting()
    {
        var image = SolidImage(4, 4, 0.5f);
        var masks = OnesMask(4, 4);

        var withLabels = new SegmentationVisualizationConfig { ShowLabels = true };
        Assert.Throws<NotSupportedException>(
            () => SegmentationRenderer.DrawSegmentationMasks(image, masks, withLabels));

        var withScores = new SegmentationVisualizationConfig { ShowLabels = false, ShowScores = true };
        Assert.Throws<NotSupportedException>(
            () => SegmentationRenderer.DrawSegmentationMasks(image, masks, withScores));

        // ...and with both off it renders normally, so the guard is not simply rejecting everything.
        var plain = new SegmentationVisualizationConfig { ShowLabels = false, ShowScores = false };
        var rendered = SegmentationRenderer.DrawSegmentationMasks(image, masks, plain);
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
            ShowLabels = false,   // defaults to true; no glyph rasterizer, so it must be turned off
        };

        var rendered = SegmentationRenderer.Render(image, output, config);

        float keptRow = rendered[0, 0, 0] + rendered[1, 0, 0] + rendered[2, 0, 0];
        float droppedRow = rendered[0, 3, 0] + rendered[1, 3, 0] + rendered[2, 3, 0];

        Assert.True(keptRow > 0f, "The instance scoring 0.95 was above the 0.5 threshold but was not drawn.");
        Assert.True(droppedRow == 0f,
            $"The instance scoring 0.10 was below the 0.5 threshold but was still drawn (sum={droppedRow}).");
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
}
