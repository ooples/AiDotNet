using AiDotNet.Diffusion.Preprocessing;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests for Phase 2 condition preprocessors.
/// </summary>
public class PreprocessorContractTests
{
    #region Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task CannyEdgePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new CannyEdgePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task DepthEstimationPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new DepthEstimationPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task NormalMapPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new NormalMapPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(3, preprocessor.OutputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task OpenPosePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new OpenPosePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task DWPosePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new DWPosePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SemanticSegPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new SemanticSegPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task LineArtPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new LineArtPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task SoftEdgePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new SoftEdgePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task MLSDPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new MLSDPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task ScribblePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new ScribblePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task TilePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new TilePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(3, preprocessor.OutputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task InpaintingMaskPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new InpaintingMaskPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.Equal(1, preprocessor.OutputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task ShufflePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new ShufflePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task QRCodePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new QRCodePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ColorPalettePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new ColorPalettePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ContentShufflePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new ContentShufflePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SAMPreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new SAMPreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task MediaPipeFacePreprocessor_DefaultConstructor_CreatesValidPreprocessor()
    {
        var preprocessor = new MediaPipeFacePreprocessor<double>();

        Assert.NotNull(preprocessor);
        Assert.True(preprocessor.OutputChannels > 0);
    }

    #endregion
}
