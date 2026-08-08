using System;
using System.Linq;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Verifies InstantStyle's two mechanisms (arXiv:2404.02733) rather than just that the model
/// constructs.
/// </summary>
/// <remarks>
/// The paper's whole contribution is those two strategies: subtract content-text features from
/// reference-image features in the shared CLIP space, and inject reference features into style
/// blocks ONLY. A latent-diffusion wrapper that does neither is not InstantStyle, which is exactly
/// the state this model was in before.
/// </remarks>
public class InstantStyleMechanismTests
{
    private static Tensor<double> Features(int n, double start, double step)
    {
        var t = new Tensor<double>([1, n]);
        for (int i = 0; i < n; i++) t[0, i] = start + step * i;
        return t;
    }

    [Fact]
    public void BlockLayoutMatchesTheSdxlUNet()
    {
        // "There are 11 transformer blocks with SDXL, 4 for downsample blocks, 1 for middle block,
        // 6 for upsample blocks."
        Assert.Equal(11, InstantStyleModel<double>.TransformerBlockCount);
        Assert.Equal(4 + 1 + 6, InstantStyleModel<double>.TransformerBlockCount);
    }

    [Fact]
    public void StyleAndLayoutBlocksAreTheOnesThePaperIdentifies()
    {
        // up_blocks.0.attentions.1 == "the 6th" == style; down_blocks.2.attentions.1 == "the 4th"
        // == spatial layout.
        Assert.Equal(6, InstantStyleModel<double>.StyleBlockIndex);
        Assert.Equal(4, InstantStyleModel<double>.LayoutBlockIndex);
    }

    [Fact]
    public void InjectionIsExclusiveToTheStyleBlockByDefault()
    {
        // "Most time, the 6th block is enough to capture style" — so only block 6 by default, and
        // the exclusivity is the mechanism: every other block must receive nothing.
        var model = new InstantStyleModel<double>();
        Assert.False(model.InjectLayoutBlock);
        Assert.Equal(new[] { 6 }, model.InjectionBlocks.ToArray());

        for (int block = 1; block <= InstantStyleModel<double>.TransformerBlockCount; block++)
        {
            Assert.Equal(block == 6, model.IsInjectionBlock(block));
        }
    }

    [Fact]
    public void LayoutBlockJoinsOnlyWhenExplicitlyEnabled()
    {
        // "the 4th matters only when the layout is a part of style in some cases"
        var model = new InstantStyleModel<double> { InjectLayoutBlock = true };
        Assert.Equal(new[] { 4, 6 }, model.InjectionBlocks.ToArray());
        Assert.True(model.IsInjectionBlock(4));
        Assert.True(model.IsInjectionBlock(6));
        Assert.False(model.IsInjectionBlock(5));
    }

    [Fact]
    public void BlockScalesAreZeroEverywhereExceptInjectionBlocks()
    {
        var model = new InstantStyleModel<double>();
        var scales = model.BuildBlockInjectionScales();

        Assert.Equal(11, scales.Length);
        Assert.Equal(1.0, scales[InstantStyleModel<double>.StyleBlockIndex - 1]);
        Assert.Equal(0.0, scales.Where((_, i) => i != InstantStyleModel<double>.StyleBlockIndex - 1).Sum());

        model.InjectLayoutBlock = true;
        var both = model.BuildBlockInjectionScales();
        Assert.Equal(2.0, both.Sum());
    }

    [Fact]
    public void DecouplingSubtractsContentTextFromImageFeatures()
    {
        // "after subtracting the content text features from the image features, the style and
        // content can be explicitly decoupled"
        var model = new InstantStyleModel<double>();
        var image = Features(8, 1.0, 1.0);     // 1..8
        var content = Features(8, 0.25, 0.5);  // 0.25, 0.75, ...

        var style = model.DecoupleStyleFromContent(image, content);

        Assert.Equal(image.Shape.ToArray(), style.Shape.ToArray());
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(image[0, i] - content[0, i], style[0, i], 12);
        }
    }

    [Fact]
    public void DecouplingIsAnIdentityWhenThereIsNoContentToRemove()
    {
        var model = new InstantStyleModel<double>();
        var image = Features(6, 2.0, 0.5);
        var zero = new Tensor<double>([1, 6]);

        var style = model.DecoupleStyleFromContent(image, zero);
        for (int i = 0; i < 6; i++) Assert.Equal(image[0, i], style[0, i], 12);
    }

    [Fact]
    public void DecouplingRejectsFeaturesFromDifferentSpaces()
    {
        // The subtraction is only meaningful because both operands live in the same CLIP space;
        // mismatched shapes mean they do not, so this must fail loudly rather than broadcast.
        var model = new InstantStyleModel<double>();
        Assert.Throws<ArgumentException>(
            () => model.DecoupleStyleFromContent(Features(8, 1, 1), Features(4, 1, 1)));
        Assert.Throws<ArgumentNullException>(
            () => model.DecoupleStyleFromContent(null!, Features(4, 1, 1)));
        Assert.Throws<ArgumentNullException>(
            () => model.DecoupleStyleFromContent(Features(4, 1, 1), null!));
    }
}
