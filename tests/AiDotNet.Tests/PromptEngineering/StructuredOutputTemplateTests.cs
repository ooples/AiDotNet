using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

public class StructuredOutputTemplateTests
{
    [Fact]
    public void Constructor_WithJsonFormat_CreatesTemplate()
    {
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Json);

        Assert.NotNull(template);
        Assert.Contains("JSON", template.Template);
    }

    [Fact]
    public void Constructor_WithXmlFormat_CreatesTemplate()
    {
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Xml);

        Assert.NotNull(template);
        Assert.Contains("XML", template.Template);
    }

    [Fact]
    public void Constructor_WithCsvFormat_CreatesTemplate()
    {
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Csv);

        Assert.NotNull(template);
        Assert.Contains("CSV", template.Template);
    }

    [Fact]
    public void Constructor_WithMarkdownFormat_CreatesTemplate()
    {
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Markdown);

        Assert.NotNull(template);
        Assert.Contains("Markdown", template.Template);
    }

    [Fact]
    public void Constructor_WithYamlFormat_CreatesTemplate()
    {
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Yaml);

        Assert.NotNull(template);
        Assert.Contains("YAML", template.Template);
    }

    [Fact]
    public void Constructor_WithSchema_IncludesSchema()
    {
        var schema = @"{ ""name"": ""string"", ""age"": ""number"" }";
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Json, schema);

        Assert.NotNull(template);
        Assert.Contains("name", template.Template);
        Assert.Contains("age", template.Template);
    }

    [Fact]
    public void Constructor_WithExample_IncludesExample()
    {
        var example = @"{ ""name"": ""John"", ""age"": 30 }";
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Json, null, example);

        Assert.NotNull(template);
        Assert.Contains("John", template.Template);
    }

    [Fact]
    public void Constructor_WithCustomTemplate_UsesCustomTemplate()
    {
        var customTemplate = "Custom output format: {task}";
        var template = new StructuredOutputTemplate(customTemplate);

        Assert.Equal(customTemplate, template.Template);
    }

    [Fact]
    public void WithTask_SetsTask()
    {
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Json)
            .WithTask("Extract user information");

        Assert.NotNull(template);
        Assert.Contains("Extract user information", template.Template);
    }

    [Fact]
    public void Format_ReplacesVariables()
    {
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Json);
        var variables = new Dictionary<string, string>
        {
            ["task"] = "Parse this data"
        };

        var result = template.Format(variables);

        Assert.Contains("Parse this data", result);
    }

    [Fact]
    public void Json_StaticMethod_CreatesJsonTemplate()
    {
        var template = StructuredOutputTemplate.Json();

        Assert.NotNull(template);
        Assert.Contains("JSON", template.Template);
    }

    [Fact]
    public void Json_WithSchema_IncludesSchema()
    {
        var schema = @"{ ""id"": ""number"" }";
        var template = StructuredOutputTemplate.Json(schema);

        Assert.NotNull(template);
        Assert.Contains("id", template.Template);
    }

    [Fact]
    public void Xml_StaticMethod_CreatesXmlTemplate()
    {
        var template = StructuredOutputTemplate.Xml("root");

        Assert.NotNull(template);
        Assert.Contains("root", template.Template);
    }

    [Fact]
    public void Csv_StaticMethod_CreatesCsvTemplate()
    {
        var template = StructuredOutputTemplate.Csv("name", "age", "email");

        Assert.NotNull(template);
        Assert.Contains("name", template.Template);
        Assert.Contains("age", template.Template);
        Assert.Contains("email", template.Template);
    }

    [Fact]
    public void Markdown_StaticMethod_CreatesMarkdownTemplate()
    {
        var template = StructuredOutputTemplate.Markdown();

        Assert.NotNull(template);
        Assert.Contains("Markdown", template.Template);
    }

    [Fact]
    public void Yaml_StaticMethod_CreatesYamlTemplate()
    {
        var template = StructuredOutputTemplate.Yaml();

        Assert.NotNull(template);
        Assert.Contains("YAML", template.Template);
    }

    [Fact]
    public void Builder_CreatesTemplateWithFormat()
    {
        var template = StructuredOutputTemplate.Builder()
            .WithFormat(StructuredOutputTemplate.OutputFormat.Json)
            .Build();

        Assert.NotNull(template);
        Assert.Contains("JSON", template.Template);
    }

    [Fact]
    public void Builder_WithSchema_IncludesSchema()
    {
        var template = StructuredOutputTemplate.Builder()
            .WithFormat(StructuredOutputTemplate.OutputFormat.Json)
            .WithSchema(@"{ ""field"": ""type"" }")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("field", template.Template);
    }

    [Fact]
    public void Builder_WithExample_IncludesExample()
    {
        var template = StructuredOutputTemplate.Builder()
            .WithFormat(StructuredOutputTemplate.OutputFormat.Json)
            .WithExample(@"{ ""sample"": ""data"" }")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("sample", template.Template);
    }

    [Fact]
    public void Builder_WithTask_IncludesTask()
    {
        var template = StructuredOutputTemplate.Builder()
            .WithFormat(StructuredOutputTemplate.OutputFormat.Json)
            .WithTask("Convert to JSON")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Convert to JSON", template.Template);
    }

    [Fact]
    public void Builder_AddField_GeneratesSchema()
    {
        var template = StructuredOutputTemplate.Builder()
            .WithFormat(StructuredOutputTemplate.OutputFormat.Json)
            .AddField("username", "string")
            .AddField("score", "number")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("username", template.Template);
        Assert.Contains("score", template.Template);
    }

    [Fact]
    public void Template_ContainsFormatInstructions()
    {
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Json);

        Assert.Contains("valid", template.Template.ToLower());
    }

    [Fact]
    public void Template_ContainsImportantSection()
    {
        var template = new StructuredOutputTemplate(StructuredOutputTemplate.OutputFormat.Json);

        Assert.Contains("Important", template.Template);
    }
}
