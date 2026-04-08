using AiDotNet.Postprocessing.Document;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Postprocessing;

/// <summary>
/// Integration tests for postprocessing classes.
/// </summary>
public class PostprocessingIntegrationTests
{
    #region TextPostprocessor Tests

    [Fact(Timeout = 120000)]
    public async Task TextPostprocessor_Construction_WithDefaults()
    {
        using var processor = new TextPostprocessor<double>();
        Assert.NotNull(processor);
    }

    [Fact(Timeout = 120000)]
    public async Task TextPostprocessor_Construction_WithOptions()
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

    [Fact(Timeout = 120000)]
    public async Task TextPostprocessor_Process_PreservesText()
    {
        using var processor = new TextPostprocessor<double>();
        var result = processor.Process("Hello  World");
        Assert.NotNull(result);
        Assert.Contains("Hello", result);
        Assert.Contains("World", result);
    }

    [Fact(Timeout = 120000)]
    public async Task TextPostprocessor_Process_NormalizesWhitespace()
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
        Assert.Equal("Hello World", result);
    }

    #endregion

    #region StructuredOutputParser Tests

    [Fact(Timeout = 120000)]
    public async Task StructuredOutputParser_Construction()
    {
        using var parser = new StructuredOutputParser<double>();
        Assert.NotNull(parser);
    }

    [Fact(Timeout = 120000)]
    public async Task StructuredOutputParser_ParseKeyValuePairs_ExtractsPairsCorrectly()
    {
        using var parser = new StructuredOutputParser<double>();
        var text = "Name: John\nAge: 30\nCity: New York";
        var result = parser.ParseKeyValuePairs(text);
        Assert.NotNull(result);
        Assert.Equal(3, result.Count);
        Assert.Equal("John", result["Name"]);
        Assert.Equal("30", result["Age"]);
        Assert.Equal("New York", result["City"]);
    }

    #endregion

    #region SpellCorrection Tests

    [Fact(Timeout = 120000)]
    public async Task SpellCorrection_Construction_WithDefaults()
    {
        using var corrector = new SpellCorrection<double>();
        Assert.NotNull(corrector);
    }

    [Fact(Timeout = 120000)]
    public async Task SpellCorrection_Construction_WithMaxEditDistance()
    {
        using var corrector = new SpellCorrection<double>(3);
        Assert.NotNull(corrector);
    }

    [Fact(Timeout = 120000)]
    public async Task SpellCorrection_Process_ReturnsProcessedText()
    {
        using var corrector = new SpellCorrection<double>();
        var result = corrector.Process("hello world");
        Assert.NotNull(result);
        Assert.True(result.Length > 0, "Spell correction should return non-empty text");
        Assert.Contains("hello", result.ToLowerInvariant());
    }

    #endregion

    #region EntityLinking Tests

    [Fact(Timeout = 120000)]
    public async Task EntityLinking_Construction()
    {
        using var linker = new EntityLinking<double>();
        Assert.NotNull(linker);
    }

    [Fact(Timeout = 120000)]
    public async Task EntityLinking_RegisterEntity_PersistsEntity()
    {
        using var linker = new EntityLinking<double>();
        var entity = new Entity
        {
            Text = "Microsoft",
            Type = EntityType.Organization,
            Confidence = 1.0,
        };
        linker.RegisterEntity(entity);

        // Verify the entity was registered by linking it
        var linked = linker.LinkEntity(entity);
        Assert.NotNull(linked);
        Assert.Equal("Microsoft", linked.Text);
    }

    #endregion

    #region Data Class Tests

    [Fact(Timeout = 120000)]
    public async Task Entity_Properties_SetCorrectly()
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
        Assert.Equal("Apple Inc.", entity.CanonicalName);
        Assert.NotNull(entity.Attributes);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidationResult_Properties_SetCorrectly()
    {
        var result = new ValidationResult
        {
            IsValid = true,
        };

        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
        Assert.Empty(result.Warnings);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidationResult_WithErrors_IsInvalid()
    {
        var result = new ValidationResult
        {
            IsValid = false,
        };
        result.Errors.Add("Missing required field: Name");

        Assert.False(result.IsValid);
        Assert.Single(result.Errors);
    }

    [Fact(Timeout = 120000)]
    public async Task DocumentSchema_Properties_SetCorrectly()
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

    [Fact(Timeout = 120000)]
    public async Task TextPostprocessorOptions_DefaultValues()
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

    [Fact(Timeout = 120000)]
    public async Task InvoiceData_Construction()
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
        Assert.Equal("John Doe", invoice.Customer);
        Assert.Equal(100.50m, invoice.Total);
        Assert.Equal(10.05m, invoice.Tax);
        Assert.NotNull(invoice.LineItems);
    }

    [Fact(Timeout = 120000)]
    public async Task ReceiptData_Construction()
    {
        var receipt = new ReceiptData
        {
            StoreName = "Store ABC",
            StoreAddress = "123 Main St",
            Total = 25.99m,
            PaymentMethod = "Credit Card",
        };

        Assert.Equal("Store ABC", receipt.StoreName);
        Assert.Equal("123 Main St", receipt.StoreAddress);
        Assert.Equal(25.99m, receipt.Total);
        Assert.Equal("Credit Card", receipt.PaymentMethod);
        Assert.NotNull(receipt.Items);
    }

    #endregion
}
