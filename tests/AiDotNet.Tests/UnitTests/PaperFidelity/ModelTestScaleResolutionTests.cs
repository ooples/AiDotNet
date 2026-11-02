using AiDotNet.Testing;
using AiDotNet.VisionLanguage.InstructionTuned;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

/// <summary>
/// The test assembly declares options types of its own (OptionPerturbationTests.DepthOptions), and a
/// ModelTestScale the generator emitted for them shadowed AiDotNet's table for every generated fixture:
/// CreateBoundedOptions returned null for every library options type.
/// </summary>
public class ModelTestScaleResolutionTests
{
    [Fact]
    public void ModelTestScale_ResolvesToTheLibraryTable()
    {
        Assert.Equal(typeof(QwenVLOptions).Assembly, typeof(ModelTestScale).Assembly);
    }

    [Fact]
    public void CreateBoundedOptions_BoundsALibraryOptionsType()
    {
        var defaults = new QwenVLOptions();
        var bounded = Assert.IsType<QwenVLOptions>(ModelTestScale.CreateBoundedOptions(typeof(QwenVLOptions)));

        Assert.True(bounded.ResamplerDim < defaults.ResamplerDim,
            $"ResamplerDim stayed at its paper default {defaults.ResamplerDim}.");
    }
}
