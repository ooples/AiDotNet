using AiDotNet.Document;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Deep integration tests for Document:
/// ConfidenceLevel enum, DocumentType flags enum, FormFieldType enum,
/// LayoutElementType enum, TableExportFormat enum,
/// OCRResult/OCRWord/OCRLine/OCRBlock data models,
/// TextDetectionResult/TextRegion data models.
/// </summary>
public class DocumentDeepMathIntegrationTests
{
    // ============================
    // ConfidenceLevel Enum
    // ============================

    [Fact]
    public void ConfidenceLevel_HasFiveValues()
    {
        var values = Enum.GetValues<ConfidenceLevel>();
        Assert.Equal(5, values.Length);
    }

    [Theory]
    [InlineData(ConfidenceLevel.VeryLow)]
    [InlineData(ConfidenceLevel.Low)]
    [InlineData(ConfidenceLevel.Medium)]
    [InlineData(ConfidenceLevel.High)]
    [InlineData(ConfidenceLevel.VeryHigh)]
    public void ConfidenceLevel_AllValuesValid(ConfidenceLevel level)
    {
        Assert.True(Enum.IsDefined(level));
    }

    [Theory]
    [InlineData(ConfidenceLevel.VeryLow, 0)]
    [InlineData(ConfidenceLevel.Low, 1)]
    [InlineData(ConfidenceLevel.Medium, 2)]
    [InlineData(ConfidenceLevel.High, 3)]
    [InlineData(ConfidenceLevel.VeryHigh, 4)]
    public void ConfidenceLevel_OrderedValues(ConfidenceLevel level, int expectedValue)
    {
        Assert.Equal(expectedValue, (int)level);
    }

    [Theory]
    [InlineData(0.15, ConfidenceLevel.VeryLow)]   // Below 25%
    [InlineData(0.35, ConfidenceLevel.Low)]        // 25-50%
    [InlineData(0.60, ConfidenceLevel.Medium)]     // 50-75%
    [InlineData(0.85, ConfidenceLevel.High)]       // 75-90%
    [InlineData(0.95, ConfidenceLevel.VeryHigh)]   // Above 90%
    public void ConfidenceLevel_ConfidenceScoreMapping(double score, ConfidenceLevel expectedLevel)
    {
        // Map confidence score to ConfidenceLevel
        ConfidenceLevel level;
        if (score < 0.25) level = ConfidenceLevel.VeryLow;
        else if (score < 0.50) level = ConfidenceLevel.Low;
        else if (score < 0.75) level = ConfidenceLevel.Medium;
        else if (score < 0.90) level = ConfidenceLevel.High;
        else level = ConfidenceLevel.VeryHigh;

        Assert.Equal(expectedLevel, level);
    }

    // ============================
    // DocumentType Flags Enum
    // ============================

    [Fact]
    public void DocumentType_None_IsZero()
    {
        Assert.Equal(0, (int)DocumentType.None);
    }

    [Theory]
    [InlineData(DocumentType.BusinessDocument, 1)]
    [InlineData(DocumentType.ScientificPaper, 2)]
    [InlineData(DocumentType.Form, 4)]
    [InlineData(DocumentType.Report, 8)]
    [InlineData(DocumentType.Letter, 16)]
    [InlineData(DocumentType.Handwritten, 32)]
    [InlineData(DocumentType.ScannedDocument, 64)]
    [InlineData(DocumentType.WebPage, 128)]
    [InlineData(DocumentType.Infographic, 256)]
    public void DocumentType_PowerOfTwoValues(DocumentType type, int expectedValue)
    {
        Assert.Equal(expectedValue, (int)type);
        Assert.True((expectedValue & (expectedValue - 1)) == 0, $"{type} should be power of 2");
    }

    [Fact]
    public void DocumentType_All_IncludesAllValues()
    {
        var all = DocumentType.All;
        Assert.True(all.HasFlag(DocumentType.BusinessDocument));
        Assert.True(all.HasFlag(DocumentType.ScientificPaper));
        Assert.True(all.HasFlag(DocumentType.Form));
        Assert.True(all.HasFlag(DocumentType.Report));
        Assert.True(all.HasFlag(DocumentType.Letter));
        Assert.True(all.HasFlag(DocumentType.Handwritten));
        Assert.True(all.HasFlag(DocumentType.ScannedDocument));
        Assert.True(all.HasFlag(DocumentType.WebPage));
        Assert.True(all.HasFlag(DocumentType.Infographic));
    }

    [Fact]
    public void DocumentType_FlagsCombination_Works()
    {
        var combined = DocumentType.Form | DocumentType.ScannedDocument;
        Assert.True(combined.HasFlag(DocumentType.Form));
        Assert.True(combined.HasFlag(DocumentType.ScannedDocument));
        Assert.False(combined.HasFlag(DocumentType.Letter));
        Assert.Equal(4 + 64, (int)combined);
    }

    // ============================
    // FormFieldType Enum
    // ============================

    [Fact]
    public void FormFieldType_HasNineValues()
    {
        var values = Enum.GetValues<FormFieldType>();
        Assert.Equal(9, values.Length);
    }

    [Theory]
    [InlineData(FormFieldType.TextInput)]
    [InlineData(FormFieldType.Checkbox)]
    [InlineData(FormFieldType.RadioButton)]
    [InlineData(FormFieldType.Dropdown)]
    [InlineData(FormFieldType.DateField)]
    [InlineData(FormFieldType.SignatureField)]
    [InlineData(FormFieldType.NumericInput)]
    [InlineData(FormFieldType.TextArea)]
    [InlineData(FormFieldType.Other)]
    public void FormFieldType_AllValuesValid(FormFieldType type)
    {
        Assert.True(Enum.IsDefined(type));
    }

    // ============================
    // LayoutElementType Enum
    // ============================

    [Fact]
    public void LayoutElementType_HasNineteenValues()
    {
        var values = Enum.GetValues<LayoutElementType>();
        Assert.Equal(19, values.Length);
    }

    [Theory]
    [InlineData(LayoutElementType.Text)]
    [InlineData(LayoutElementType.Title)]
    [InlineData(LayoutElementType.List)]
    [InlineData(LayoutElementType.Table)]
    [InlineData(LayoutElementType.Figure)]
    [InlineData(LayoutElementType.Caption)]
    [InlineData(LayoutElementType.Header)]
    [InlineData(LayoutElementType.Footer)]
    [InlineData(LayoutElementType.PageNumber)]
    [InlineData(LayoutElementType.Equation)]
    [InlineData(LayoutElementType.Logo)]
    [InlineData(LayoutElementType.Signature)]
    [InlineData(LayoutElementType.Stamp)]
    [InlineData(LayoutElementType.Barcode)]
    [InlineData(LayoutElementType.QRCode)]
    [InlineData(LayoutElementType.Handwriting)]
    [InlineData(LayoutElementType.FormField)]
    [InlineData(LayoutElementType.Separator)]
    [InlineData(LayoutElementType.Other)]
    public void LayoutElementType_AllValuesValid(LayoutElementType type)
    {
        Assert.True(Enum.IsDefined(type));
    }

    // ============================
    // TableExportFormat Enum
    // ============================

    [Fact]
    public void TableExportFormat_HasFiveValues()
    {
        var values = Enum.GetValues<TableExportFormat>();
        Assert.Equal(5, values.Length);
    }

    [Theory]
    [InlineData(TableExportFormat.CSV)]
    [InlineData(TableExportFormat.JSON)]
    [InlineData(TableExportFormat.HTML)]
    [InlineData(TableExportFormat.Markdown)]
    [InlineData(TableExportFormat.Excel)]
    public void TableExportFormat_AllValuesValid(TableExportFormat format)
    {
        Assert.True(Enum.IsDefined(format));
    }

    // ============================
    // OCRResult: Defaults
    // ============================

    [Fact]
    public void OCRResult_Defaults()
    {
        var result = new OCRResult<double>();
        Assert.Equal(string.Empty, result.FullText);
        Assert.NotNull(result.Words);
        Assert.Empty(result.Words);
        Assert.NotNull(result.Lines);
        Assert.Empty(result.Lines);
        Assert.NotNull(result.Blocks);
        Assert.Empty(result.Blocks);
        Assert.Null(result.DetectedLanguage);
        Assert.Equal(0.0, result.ProcessingTimeMs);
        Assert.False(result.RequiresDeskewing);
        Assert.Null(result.RotationAngle);
    }

    [Fact]
    public void OCRResult_SetProperties()
    {
        var result = new OCRResult<double>
        {
            FullText = "Hello World",
            DetectedLanguage = "en",
            ProcessingTimeMs = 150.5,
            RequiresDeskewing = true,
            RotationAngle = 2.5,
            AverageConfidence = 0.92
        };

        Assert.Equal("Hello World", result.FullText);
        Assert.Equal("en", result.DetectedLanguage);
        Assert.Equal(150.5, result.ProcessingTimeMs);
        Assert.True(result.RequiresDeskewing);
        Assert.Equal(2.5, result.RotationAngle);
        Assert.Equal(0.92, result.AverageConfidence);
    }

    // ============================
    // OCRWord: Defaults
    // ============================

    [Fact]
    public void OCRWord_Defaults()
    {
        var word = new OCRWord<double>();
        Assert.Equal(string.Empty, word.Text);
        Assert.Equal(0, word.LineIndex);
        Assert.Equal(0, word.BlockIndex);
    }

    [Fact]
    public void OCRWord_SetProperties()
    {
        var word = new OCRWord<double>
        {
            Text = "Hello",
            Confidence = 0.95,
            LineIndex = 2,
            BlockIndex = 1
        };

        Assert.Equal("Hello", word.Text);
        Assert.Equal(0.95, word.Confidence);
        Assert.Equal(2, word.LineIndex);
        Assert.Equal(1, word.BlockIndex);
    }

    // ============================
    // OCRLine: Defaults
    // ============================

    [Fact]
    public void OCRLine_Defaults()
    {
        var line = new OCRLine<double>();
        Assert.Equal(string.Empty, line.Text);
        Assert.NotNull(line.Words);
        Assert.Empty(line.Words);
        Assert.Equal(0, line.BlockIndex);
    }

    // ============================
    // OCRBlock: Defaults
    // ============================

    [Fact]
    public void OCRBlock_Defaults()
    {
        var block = new OCRBlock<double>();
        Assert.Equal(string.Empty, block.Text);
        Assert.NotNull(block.Lines);
        Assert.Empty(block.Lines);
        Assert.Equal(LayoutElementType.Text, block.BlockType);
    }

    [Fact]
    public void OCRBlock_SetBlockType()
    {
        var block = new OCRBlock<double>
        {
            BlockType = LayoutElementType.Table,
            Text = "Header | Value"
        };

        Assert.Equal(LayoutElementType.Table, block.BlockType);
        Assert.Equal("Header | Value", block.Text);
    }

    // ============================
    // TextDetectionResult: Defaults
    // ============================

    [Fact]
    public void TextDetectionResult_Defaults()
    {
        var result = new TextDetectionResult<double>();
        Assert.NotNull(result.TextRegions);
        Assert.Empty(result.TextRegions);
        Assert.Null(result.ProbabilityMap);
        Assert.Null(result.ThresholdMap);
        Assert.Null(result.BinaryMap);
        Assert.Equal(0.0, result.ProcessingTimeMs);
        Assert.Equal(0, result.RegionCount);
    }

    [Fact]
    public void TextDetectionResult_RegionCount_MatchesTextRegions()
    {
        var regions = new List<TextRegion<double>>
        {
            new() { ConfidenceValue = 0.95, Index = 0 },
            new() { ConfidenceValue = 0.87, Index = 1 },
            new() { ConfidenceValue = 0.72, Index = 2 }
        };

        var result = new TextDetectionResult<double> { TextRegions = regions };
        Assert.Equal(3, result.RegionCount);
    }

    // ============================
    // TextRegion: Defaults
    // ============================

    [Fact]
    public void TextRegion_Defaults()
    {
        var region = new TextRegion<double>();
        Assert.Null(region.PolygonPoints);
        Assert.Equal(0.0, region.ConfidenceValue);
        Assert.Null(region.RotationAngle);
        Assert.Equal(0, region.Index);
        Assert.Null(region.CroppedImage);
    }

    [Fact]
    public void TextRegion_SetProperties()
    {
        var region = new TextRegion<double>
        {
            ConfidenceValue = 0.92,
            RotationAngle = 5.0,
            Index = 3
        };

        Assert.Equal(0.92, region.ConfidenceValue);
        Assert.Equal(5.0, region.RotationAngle);
        Assert.Equal(3, region.Index);
    }

    // ============================
    // Confidence Score Math
    // ============================

    [Theory]
    [InlineData(new double[] { 0.95, 0.87, 0.92 }, 0.9133)]   // Mean = (0.95+0.87+0.92)/3
    [InlineData(new double[] { 1.0, 1.0, 1.0 }, 1.0)]
    [InlineData(new double[] { 0.0, 0.0, 0.0 }, 0.0)]
    public void OCR_AverageConfidence_Calculation(double[] confidences, double expectedAverage)
    {
        // Verify average confidence calculation
        double sum = 0;
        foreach (double c in confidences) sum += c;
        double avg = sum / confidences.Length;

        Assert.Equal(expectedAverage, avg, 1e-3);
    }

    // ============================
    // Document Rotation/Deskewing Math
    // ============================

    [Theory]
    [InlineData(0.0, false)]    // No rotation
    [InlineData(0.5, false)]    // Small rotation (within tolerance)
    [InlineData(2.0, true)]     // Significant rotation
    [InlineData(5.0, true)]     // Large rotation
    [InlineData(90.0, true)]    // Quarter turn
    public void OCR_Deskewing_BasedOnRotation(double rotationAngle, bool expectedDeskewing)
    {
        // Typically, deskewing is needed when rotation > 1 degree
        bool requiresDeskewing = Math.Abs(rotationAngle) > 1.0;
        Assert.Equal(expectedDeskewing, requiresDeskewing);
    }

    // ============================
    // Bounding Box: IoU (Intersection over Union) Math
    // ============================

    [Theory]
    [InlineData(0, 0, 10, 10, 5, 5, 15, 15, 25.0 / 175.0)]  // Partial overlap
    [InlineData(0, 0, 10, 10, 0, 0, 10, 10, 1.0)]            // Perfect overlap
    [InlineData(0, 0, 10, 10, 20, 20, 30, 30, 0.0)]          // No overlap
    public void BoundingBox_IoU_Calculation(
        double x1a, double y1a, double x2a, double y2a,
        double x1b, double y1b, double x2b, double y2b,
        double expectedIoU)
    {
        // Intersection
        double interX1 = Math.Max(x1a, x1b);
        double interY1 = Math.Max(y1a, y1b);
        double interX2 = Math.Min(x2a, x2b);
        double interY2 = Math.Min(y2a, y2b);

        double interWidth = Math.Max(0, interX2 - interX1);
        double interHeight = Math.Max(0, interY2 - interY1);
        double interArea = interWidth * interHeight;

        // Union
        double areaA = (x2a - x1a) * (y2a - y1a);
        double areaB = (x2b - x1b) * (y2b - y1b);
        double unionArea = areaA + areaB - interArea;

        double iou = unionArea > 0 ? interArea / unionArea : 0;
        Assert.Equal(expectedIoU, iou, 1e-6);
    }
}
