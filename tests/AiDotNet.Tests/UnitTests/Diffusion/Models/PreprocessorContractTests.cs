using AiDotNet.Diffusion.Preprocessing;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests for Phase 2 condition preprocessors.
/// </summary>
public class PreprocessorContractTests
{
    #region Constructor Tests

    [Fact]
    public void CannyEdgePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new CannyEdgePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact]
    public void DepthEstimationPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new DepthEstimationPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact]
    public void NormalMapPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new NormalMapPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(3, preprocessor.OutputChannels);
    }

    [Fact]
    public void OpenPosePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new OpenPosePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact]
    public void DWPosePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new DWPosePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact]
    public void SemanticSegPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new SemanticSegPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact]
    public void LineArtPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new LineArtPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact]
    public void SoftEdgePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new SoftEdgePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact]
    public void MLSDPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new MLSDPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact]
    public void ScribblePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new ScribblePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact]
    public void TilePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new TilePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(3, preprocessor.OutputChannels);
    }

    [Fact]
    public void InpaintingMaskPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new InpaintingMaskPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact]
    public void ShufflePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new ShufflePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact]
    public void QRCodePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new QRCodePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact]
    public void ColorPalettePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new ColorPalettePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact]
    public void ContentShufflePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new ContentShufflePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact]
    public void SAMPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new SAMPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact]
    public void MediaPipeFacePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new MediaPipeFacePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    #endregion
}
