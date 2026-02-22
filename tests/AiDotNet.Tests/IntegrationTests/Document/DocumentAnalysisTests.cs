using AiDotNet.Document;
using AiDotNet.Document.Analysis.PageSegmentation;
using AiDotNet.Document.Analysis.TableDetection;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for document analysis models and result classes.
/// </summary>
public class DocumentAnalysisTests
{
    private static NeuralNetworkArchitecture<double> CreateArchitecture(int imageSize = 64)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: imageSize,
            inputWidth: imageSize,
            inputDepth: 3,
            outputSize: 13);
    }

    private static Tensor<double> CreateSmallImage(int size = 64)
    {
        int totalSize = 1 * 3 * size * size;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, 3, size, size }, data);
    }

    #region DocBank Tests

    [Fact]
    public void DocBank_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DocBank<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void DocBank_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DocBank<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void DocBank_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DocBank<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocBank", meta.Name);
    }

    #endregion

    #region TableTransformer Tests

    [Fact]
    public void TableTransformer_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new TableTransformer<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void TableTransformer_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new TableTransformer<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void TableTransformer_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new TableTransformer<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("TableTransformer", meta.Name);
    }

    #endregion

    #region Result Class Tests

    [Fact]
    public void DocumentLayoutResult_Construction_EmptyRegions()
    {
        var result = new DocumentLayoutResult<double>();
        Assert.Empty(result.Regions);
        Assert.Equal(0, result.TotalRegions);
    }

    [Fact]
    public void TextDetectionResult_Construction_EmptyRegions()
    {
        var result = new TextDetectionResult<double>();
        Assert.Empty(result.TextRegions);
        Assert.Equal(0, result.RegionCount);
    }

    [Fact]
    public void OCRResult_Construction_EmptyText()
    {
        var result = new OCRResult<double>();
        Assert.Equal(string.Empty, result.FullText);
        Assert.Empty(result.Words);
        Assert.Empty(result.Lines);
        Assert.Empty(result.Blocks);
    }

    [Fact]
    public void TableStructureResult_Construction_EmptyCells()
    {
        var result = new TableStructureResult<double>();
        Assert.Empty(result.Cells);
        Assert.Equal(0, result.NumRows);
        Assert.Equal(0, result.NumColumns);
    }

    [Fact]
    public void TableStructureResult_ToStringGrid_ReturnsEmptyForNoData()
    {
        var result = new TableStructureResult<double>
        {
            NumRows = 0,
            NumColumns = 0
        };
        var grid = result.ToStringGrid();
        Assert.Empty(grid);
    }

    [Fact]
    public void TextRegion_Construction_HasDefaults()
    {
        var region = new TextRegion<double>();
        Assert.NotNull(region.BoundingBox);
        Assert.Equal(0, region.Index);
    }

    [Fact]
    public void LayoutRegion_ConfidenceLevel_CategorizesCorrectly()
    {
        var lowConfidence = new LayoutRegion<double> { ConfidenceValue = 0.2 };
        var highConfidence = new LayoutRegion<double> { ConfidenceValue = 0.95 };
        Assert.Equal(ConfidenceLevel.VeryLow, lowConfidence.ConfidenceLevel);
        Assert.Equal(ConfidenceLevel.VeryHigh, highConfidence.ConfidenceLevel);
    }

    [Fact]
    public void DocumentLayoutResult_GetRegionsByType_FiltersCorrectly()
    {
        var result = new DocumentLayoutResult<double>
        {
            Regions =
            [
                new LayoutRegion<double> { ElementType = LayoutElementType.Text, ConfidenceValue = 0.9 },
                new LayoutRegion<double> { ElementType = LayoutElementType.Table, ConfidenceValue = 0.8 },
                new LayoutRegion<double> { ElementType = LayoutElementType.Text, ConfidenceValue = 0.7 },
            ]
        };

        var textRegions = result.GetRegionsByType(LayoutElementType.Text).ToList();
        Assert.Equal(2, textRegions.Count);
    }

    [Fact]
    public void DocumentLayoutResult_GetHighConfidenceRegions_FiltersCorrectly()
    {
        var result = new DocumentLayoutResult<double>
        {
            Regions =
            [
                new LayoutRegion<double> { ConfidenceValue = 0.9 },
                new LayoutRegion<double> { ConfidenceValue = 0.5 },
                new LayoutRegion<double> { ConfidenceValue = 0.3 },
            ]
        };

        var highConf = result.GetHighConfidenceRegions(0.6).ToList();
        Assert.Single(highConf);
    }

    [Fact]
    public void OCRWord_Construction_HasDefaults()
    {
        var word = new OCRWord<double>();
        Assert.Equal(string.Empty, word.Text);
        Assert.NotNull(word.BoundingBox);
    }

    [Fact]
    public void TableCell_Construction_HasDefaults()
    {
        var cell = new TableCell<double>();
        Assert.Equal(1, cell.RowSpan);
        Assert.Equal(1, cell.ColSpan);
        Assert.Equal(string.Empty, cell.Text);
        Assert.False(cell.IsHeader);
    }

    #endregion

    #region DocumentModelOptions Tests

    [Fact]
    public void DocumentModelOptions_DefaultValues_AreNull()
    {
        var options = new DocumentModelOptions<double>();
        Assert.Null(options.HiddenDimension);
        Assert.Null(options.NumAttentionHeads);
        Assert.Null(options.NumLayers);
        Assert.Null(options.DropoutRate);
        Assert.Null(options.ImageSize);
        Assert.Null(options.MaxSequenceLength);
        Assert.Null(options.LearningRate);
        Assert.Null(options.RandomSeed);
    }

    [Fact]
    public void DocumentModelOptions_CustomValues_SetCorrectly()
    {
        var options = new DocumentModelOptions<double>
        {
            HiddenDimension = 512,
            NumAttentionHeads = 8,
            ImageSize = 384,
            MaxSequenceLength = 1024,
        };

        Assert.Equal(512, options.HiddenDimension);
        Assert.Equal(8, options.NumAttentionHeads);
        Assert.Equal(384, options.ImageSize);
        Assert.Equal(1024, options.MaxSequenceLength);
    }

    #endregion
}
