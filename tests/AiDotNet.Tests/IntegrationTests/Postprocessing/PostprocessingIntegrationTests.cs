using AiDotNet.Postprocessing.Document;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Postprocessing;

/// <summary>
/// Integration tests for postprocessing classes.
/// </summary>
public class PostprocessingIntegrationTests
{
    #region TextPostprocessor Tests

    [Fact]
    public void TextPostprocessor_Construction_WithDefaults()
    {
        using var processor = new TextPostprocessor<double>();
        Assert.NotNull(processor);
    }

    [Fact]
    public void TextPostprocessor_Construction_WithOptions()
    {
        var opts = new TextPostprocessorOptions
        {
            RemoveControlCharacters = true,
            NormalizeWhitespace = true,
            FixCommonOcrErrors = false,
            ApplySpellCorrection = false,
        };
        using var processor = new TextPostprocessor<double>(opts);
        Assert.NotNull(processor);
    }

    [Fact]
    public void TextPostprocessor_Process_ReturnsNonNull()
    {
        using var processor = new TextPostprocessor<double>();
        var result = processor.Process("Hello  World");
        Assert.NotNull(result);
        Assert.True(result.Length > 0);
    }

    [Fact]
    public void TextPostprocessor_Process_NormalizesWhitespace()
    {
        var opts = new TextPostprocessorOptions
        {
            NormalizeWhitespace = true,
            RemoveDuplicateSpaces = true,
            FixCommonOcrErrors = false,
            ApplySpellCorrection = false,
        };
        using var processor = new TextPostprocessor<double>(opts);
        var result = processor.Process("Hello   World");
        Assert.DoesNotContain("   ", result);
    }

    #endregion

    #region StructuredOutputParser Tests

    [Fact]
    public void StructuredOutputParser_Construction()
    {
        using var parser = new StructuredOutputParser<double>();
        Assert.NotNull(parser);
    }

    [Fact]
    public void StructuredOutputParser_ParseKeyValuePairs_ExtractsPairs()
    {
        using var parser = new StructuredOutputParser<double>();
        var text = "Name: John\nAge: 30\nCity: New York";
        var result = parser.ParseKeyValuePairs(text);
        Assert.NotNull(result);
        Assert.True(result.Count > 0);
    }

    #endregion

    #region SpellCorrection Tests

    [Fact]
    public void SpellCorrection_Construction_WithDefaults()
    {
        using var corrector = new SpellCorrection<double>();
        Assert.NotNull(corrector);
    }

    [Fact]
    public void SpellCorrection_Construction_WithMaxEditDistance()
    {
        using var corrector = new SpellCorrection<double>(3);
        Assert.NotNull(corrector);
    }

    [Fact]
    public void SpellCorrection_Process_ReturnsNonNull()
    {
        using var corrector = new SpellCorrection<double>();
        var result = corrector.Process("hello world");
        Assert.NotNull(result);
    }

    #endregion

    #region EntityLinking Tests

    [Fact]
    public void EntityLinking_Construction()
    {
        using var linker = new EntityLinking<double>();
        Assert.NotNull(linker);
    }

    [Fact]
    public void EntityLinking_RegisterEntity_DoesNotThrow()
    {
        using var linker = new EntityLinking<double>();
        var entity = new Entity
        {
            Text = "Microsoft",
            Type = EntityType.Organization,
            Confidence = 1.0,
        };
        linker.RegisterEntity(entity);
    }

    #endregion

    #region Data Class Tests

    [Fact]
    public void Entity_Properties_SetCorrectly()
    {
        var entity = new Entity
        {
            Text = "Apple Inc.",
            Type = EntityType.Organization,
            StartIndex = 0,
            EndIndex = 10,
            Confidence = 0.95,
            NormalizedValue = "apple_inc",
            CanonicalName = "Apple Inc.",
        };

        Assert.Equal("Apple Inc.", entity.Text);
        Assert.Equal(EntityType.Organization, entity.Type);
        Assert.Equal(0, entity.StartIndex);
        Assert.Equal(10, entity.EndIndex);
        Assert.Equal(0.95, entity.Confidence, 1e-6);
        Assert.Equal("apple_inc", entity.NormalizedValue);
        Assert.NotNull(entity.Attributes);
    }

    [Fact]
    public void ValidationResult_Properties_SetCorrectly()
    {
        var result = new ValidationResult
        {
            IsValid = true,
        };

        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void ValidationResult_WithErrors_IsInvalid()
    {
        var result = new ValidationResult
        {
            IsValid = false,
        };
        result.Errors.Add("Missing required field: Name");

        Assert.False(result.IsValid);
        Assert.Single(result.Errors);
    }

    [Fact]
    public void DocumentSchema_Properties_SetCorrectly()
    {
        var schema = new DocumentSchema();
        schema.RequiredFields.Add("Name");
        schema.RequiredFields.Add("Date");
        schema.FieldTypes["Amount"] = DataType.Decimal;
        schema.FieldPatterns["Date"] = @"\d{4}-\d{2}-\d{2}";

        Assert.Equal(2, schema.RequiredFields.Count);
        Assert.Single(schema.FieldTypes);
        Assert.Single(schema.FieldPatterns);
    }

    [Fact]
    public void TextPostprocessorOptions_DefaultValues()
    {
        var opts = new TextPostprocessorOptions();
        Assert.True(opts.RemoveControlCharacters);
        Assert.True(opts.NormalizeCharacters);
        Assert.True(opts.NormalizeWhitespace);
        Assert.True(opts.FixCommonOcrErrors);
        Assert.False(opts.ApplySpellCorrection);
        Assert.True(opts.MergeBrokenLines);
        Assert.True(opts.RemoveDuplicateSpaces);
    }

    [Fact]
    public void InvoiceData_Construction()
    {
        var invoice = new InvoiceData
        {
            InvoiceNumber = "INV-001",
            Vendor = "Acme Corp",
            Customer = "John Doe",
            Total = 100.50m,
            Tax = 10.05m,
        };

        Assert.Equal("INV-001", invoice.InvoiceNumber);
        Assert.Equal("Acme Corp", invoice.Vendor);
        Assert.Equal(100.50m, invoice.Total);
        Assert.NotNull(invoice.LineItems);
    }

    [Fact]
    public void ReceiptData_Construction()
    {
        var receipt = new ReceiptData
        {
            StoreName = "Store ABC",
            StoreAddress = "123 Main St",
            Total = 25.99m,
            PaymentMethod = "Credit Card",
        };

        Assert.Equal("Store ABC", receipt.StoreName);
        Assert.Equal(25.99m, receipt.Total);
        Assert.NotNull(receipt.Items);
    }

    #endregion
}
