using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Analysis;
using AiDotNet.PromptEngineering.Chains;
using AiDotNet.PromptEngineering.Compression;
using AiDotNet.PromptEngineering.FewShot;
using AiDotNet.PromptEngineering.Templates;
using AiDotNet.PromptEngineering.Tools;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.PromptEngineering;

/// <summary>
/// Comprehensive integration tests for the PromptEngineering module.
/// Tests Templates, Analysis, FewShot selectors, Compression, Chains, and Tools.
/// </summary>
public class PromptEngineeringIntegrationTests
{
    #region SimplePromptTemplate Tests

    [Fact]
    public void SimplePromptTemplate_Constructor_SetsTemplate()
    {
        // Arrange & Act
        var template = new SimplePromptTemplate("Hello, {name}!");

        // Assert
        Assert.Equal("Hello, {name}!", template.Template);
    }

    [Fact]
    public void SimplePromptTemplate_Constructor_ThrowsOnNullTemplate()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new SimplePromptTemplate(null!));
    }

    [Fact]
    public void SimplePromptTemplate_Constructor_ThrowsOnEmptyTemplate()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SimplePromptTemplate(""));
        Assert.Throws<ArgumentException>(() => new SimplePromptTemplate("   "));
    }

    [Fact]
    public void SimplePromptTemplate_InputVariables_ExtractsVariables()
    {
        // Arrange
        var template = new SimplePromptTemplate("Hello, {name}! Your age is {age}.");

        // Assert
        Assert.Equal(2, template.InputVariables.Count);
        Assert.Contains("name", template.InputVariables);
        Assert.Contains("age", template.InputVariables);
    }

    [Fact]
    public void SimplePromptTemplate_InputVariables_HandlesDuplicates()
    {
        // Arrange
        var template = new SimplePromptTemplate("Hello, {name}! {name} is great!");

        // Assert: duplicates are not counted twice
        Assert.Single(template.InputVariables);
        Assert.Contains("name", template.InputVariables);
    }

    [Fact]
    public void SimplePromptTemplate_Format_SubstitutesVariables()
    {
        // Arrange
        var template = new SimplePromptTemplate("Hello, {name}!");
        var variables = new Dictionary<string, string> { ["name"] = "World" };

        // Act
        var result = template.Format(variables);

        // Assert
        Assert.Equal("Hello, World!", result);
    }

    [Fact]
    public void SimplePromptTemplate_Format_MultipleVariables()
    {
        // Arrange
        var template = new SimplePromptTemplate("Translate '{text}' from {source} to {target}.");
        var variables = new Dictionary<string, string>
        {
            ["text"] = "Hello",
            ["source"] = "English",
            ["target"] = "Spanish"
        };

        // Act
        var result = template.Format(variables);

        // Assert
        Assert.Equal("Translate 'Hello' from English to Spanish.", result);
    }

    [Fact]
    public void SimplePromptTemplate_Format_ThrowsOnNullVariables()
    {
        // Arrange
        var template = new SimplePromptTemplate("Hello, {name}!");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => template.Format(null!));
    }

    [Fact]
    public void SimplePromptTemplate_Format_ThrowsOnMissingVariable()
    {
        // Arrange
        var template = new SimplePromptTemplate("Hello, {name}! You are {age} years old.");
        var variables = new Dictionary<string, string> { ["name"] = "World" };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => template.Format(variables));
    }

    [Fact]
    public void SimplePromptTemplate_Validate_ReturnsTrueWhenAllVariablesPresent()
    {
        // Arrange
        var template = new SimplePromptTemplate("Hello, {name}!");
        var variables = new Dictionary<string, string> { ["name"] = "World" };

        // Act
        var result = template.Validate(variables);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void SimplePromptTemplate_Validate_ReturnsFalseWhenMissingVariable()
    {
        // Arrange
        var template = new SimplePromptTemplate("Hello, {name}! Your age is {age}.");
        var variables = new Dictionary<string, string> { ["name"] = "World" };

        // Act
        var result = template.Validate(variables);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void SimplePromptTemplate_Validate_ReturnsFalseOnNullVariables()
    {
        // Arrange
        var template = new SimplePromptTemplate("Hello, {name}!");

        // Act
        var result = template.Validate(null!);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void SimplePromptTemplate_FromTemplate_CreatesInstance()
    {
        // Act
        var template = SimplePromptTemplate.FromTemplate("Hello, {name}!");

        // Assert
        Assert.NotNull(template);
        Assert.Equal("Hello, {name}!", template.Template);
    }

    [Fact]
    public void SimplePromptTemplate_NoVariables_FormatReturnsTemplate()
    {
        // Arrange
        var template = new SimplePromptTemplate("Hello, World!");
        var variables = new Dictionary<string, string>();

        // Act
        var result = template.Format(variables);

        // Assert
        Assert.Equal("Hello, World!", result);
    }

    #endregion

    #region PromptMetrics Tests

    [Fact]
    public void PromptMetrics_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var metrics = new PromptMetrics();

        // Assert
        Assert.Equal(0, metrics.TokenCount);
        Assert.Equal(0, metrics.CharacterCount);
        Assert.Equal(0, metrics.WordCount);
        Assert.Equal(0, metrics.VariableCount);
        Assert.Equal(0, metrics.ExampleCount);
        Assert.Equal(0.0, metrics.ComplexityScore);
        Assert.Equal(0.0m, metrics.EstimatedCost);
        Assert.NotNull(metrics.DetectedPatterns);
        Assert.Empty(metrics.DetectedPatterns);
        Assert.NotNull(metrics.ModelName);
    }

    [Fact]
    public void PromptMetrics_SetProperties_ReturnsCorrectValues()
    {
        // Arrange & Act
        var metrics = new PromptMetrics
        {
            TokenCount = 100,
            CharacterCount = 500,
            WordCount = 80,
            VariableCount = 3,
            ExampleCount = 2,
            ComplexityScore = 0.5,
            EstimatedCost = 0.001m,
            ModelName = "gpt-4"
        };

        // Assert
        Assert.Equal(100, metrics.TokenCount);
        Assert.Equal(500, metrics.CharacterCount);
        Assert.Equal(80, metrics.WordCount);
        Assert.Equal(3, metrics.VariableCount);
        Assert.Equal(2, metrics.ExampleCount);
        Assert.Equal(0.5, metrics.ComplexityScore);
        Assert.Equal(0.001m, metrics.EstimatedCost);
        Assert.Equal("gpt-4", metrics.ModelName);
    }

    [Fact]
    public void PromptMetrics_AnalyzedAt_IsSetToCurrentTime()
    {
        // Arrange & Act
        var before = DateTime.UtcNow;
        var metrics = new PromptMetrics();
        var after = DateTime.UtcNow;

        // Assert
        Assert.InRange(metrics.AnalyzedAt, before, after);
    }

    [Fact]
    public void PromptMetrics_DetectedPatterns_CanBeSet()
    {
        // Arrange & Act
        var metrics = new PromptMetrics
        {
            DetectedPatterns = new List<string> { "instruction", "question", "translation" }
        };

        // Assert
        Assert.Equal(3, metrics.DetectedPatterns.Count);
        Assert.Contains("instruction", metrics.DetectedPatterns);
        Assert.Contains("question", metrics.DetectedPatterns);
        Assert.Contains("translation", metrics.DetectedPatterns);
    }

    #endregion

    #region PromptIssue Tests

    [Fact]
    public void PromptIssue_DefaultValues()
    {
        // Arrange & Act
        var issue = new PromptIssue();

        // Assert
        Assert.Equal(IssueSeverity.Info, issue.Severity);
        Assert.Equal(string.Empty, issue.Message);
        Assert.Equal(string.Empty, issue.Code);
        Assert.Null(issue.Position);
        Assert.Null(issue.Length);
    }

    [Fact]
    public void PromptIssue_SetProperties()
    {
        // Arrange & Act
        var issue = new PromptIssue
        {
            Severity = IssueSeverity.Warning,
            Message = "Token limit approaching",
            Code = "PE001",
            Position = 100,
            Length = 10
        };

        // Assert
        Assert.Equal(IssueSeverity.Warning, issue.Severity);
        Assert.Equal("Token limit approaching", issue.Message);
        Assert.Equal("PE001", issue.Code);
        Assert.Equal(100, issue.Position);
        Assert.Equal(10, issue.Length);
    }

    [Fact]
    public void IssueSeverity_HasCorrectValues()
    {
        // Assert
        Assert.Equal(0, (int)IssueSeverity.Info);
        Assert.Equal(1, (int)IssueSeverity.Warning);
        Assert.Equal(2, (int)IssueSeverity.Error);
    }

    #endregion

    #region ValidationOptions Tests

    [Fact]
    public void ValidationOptions_DefaultValues()
    {
        // Arrange & Act
        var options = new ValidationOptions();

        // Assert
        Assert.Equal(8192, options.MaxTokens);
        Assert.True(options.CheckForInjection);
        Assert.True(options.ValidateVariables);
        Assert.Equal(IssueSeverity.Info, options.MinSeverityToReport);
    }

    [Fact]
    public void ValidationOptions_Strict_HasCorrectValues()
    {
        // Act
        var options = ValidationOptions.Strict;

        // Assert
        Assert.Equal(4000, options.MaxTokens);
        Assert.True(options.CheckForInjection);
        Assert.True(options.ValidateVariables);
        Assert.Equal(IssueSeverity.Info, options.MinSeverityToReport);
    }

    [Fact]
    public void ValidationOptions_Lenient_HasCorrectValues()
    {
        // Act
        var options = ValidationOptions.Lenient;

        // Assert
        Assert.Equal(128000, options.MaxTokens);
        Assert.False(options.CheckForInjection);
        Assert.False(options.ValidateVariables);
        Assert.Equal(IssueSeverity.Error, options.MinSeverityToReport);
    }

    #endregion

    #region CompressionResult Tests

    [Fact]
    public void CompressionResult_DefaultValues()
    {
        // Arrange & Act
        var result = new CompressionResult();

        // Assert
        Assert.Equal(string.Empty, result.OriginalPrompt);
        Assert.Equal(string.Empty, result.CompressedPrompt);
        Assert.Equal(0, result.OriginalTokenCount);
        Assert.Equal(0, result.CompressedTokenCount);
        Assert.Equal(0, result.TokensSaved);
        Assert.Equal(0.0, result.CompressionRatio);
        Assert.Equal(0.0m, result.EstimatedCostSavings);
        Assert.Equal(string.Empty, result.CompressionMethod);
        Assert.NotNull(result.Warnings);
        Assert.Empty(result.Warnings);
        Assert.False(result.IsSuccessful);
    }

    [Fact]
    public void CompressionResult_TokensSaved_CalculatesCorrectly()
    {
        // Arrange & Act
        var result = new CompressionResult
        {
            OriginalTokenCount = 100,
            CompressedTokenCount = 70
        };

        // Assert
        Assert.Equal(30, result.TokensSaved);
    }

    [Fact]
    public void CompressionResult_CompressionRatio_CalculatesCorrectly()
    {
        // Arrange & Act
        var result = new CompressionResult
        {
            OriginalTokenCount = 100,
            CompressedTokenCount = 70
        };

        // Assert
        Assert.Equal(0.3, result.CompressionRatio, 6);
    }

    [Fact]
    public void CompressionResult_CompressionRatio_ZeroOriginal_ReturnsZero()
    {
        // Arrange & Act
        var result = new CompressionResult
        {
            OriginalTokenCount = 0,
            CompressedTokenCount = 0
        };

        // Assert
        Assert.Equal(0.0, result.CompressionRatio);
    }

    [Fact]
    public void CompressionResult_IsSuccessful_TrueWhenCompressed()
    {
        // Arrange & Act
        var result = new CompressionResult
        {
            OriginalTokenCount = 100,
            CompressedTokenCount = 80
        };

        // Assert
        Assert.True(result.IsSuccessful);
    }

    [Fact]
    public void CompressionResult_IsSuccessful_FalseWhenNoCompression()
    {
        // Arrange & Act
        var result = new CompressionResult
        {
            OriginalTokenCount = 100,
            CompressedTokenCount = 100
        };

        // Assert
        Assert.False(result.IsSuccessful);
    }

    [Fact]
    public void CompressionResult_CompressedAt_IsSetToCurrentTime()
    {
        // Arrange & Act
        var before = DateTime.UtcNow;
        var result = new CompressionResult();
        var after = DateTime.UtcNow;

        // Assert
        Assert.InRange(result.CompressedAt, before, after);
    }

    #endregion

    #region CompressionOptions Tests

    [Fact]
    public void CompressionOptions_DefaultValues()
    {
        // Arrange & Act
        var options = new CompressionOptions();

        // Assert
        Assert.Equal(0.2, options.TargetReduction);
        Assert.Null(options.MaxTokens);
        Assert.Equal(10, options.MinTokenCount);
        Assert.True(options.PreserveVariables);
        Assert.True(options.PreserveCodeBlocks);
        Assert.Equal("gpt-4", options.ModelName);
    }

    [Fact]
    public void CompressionOptions_Default_ReturnsDefaultOptions()
    {
        // Act
        var options = CompressionOptions.Default;

        // Assert
        Assert.Equal(0.2, options.TargetReduction);
    }

    [Fact]
    public void CompressionOptions_Aggressive_ReturnsAggressiveOptions()
    {
        // Act
        var options = CompressionOptions.Aggressive;

        // Assert
        Assert.Equal(0.5, options.TargetReduction);
        Assert.Equal(20, options.MinTokenCount);
    }

    [Fact]
    public void CompressionOptions_Conservative_ReturnsConservativeOptions()
    {
        // Act
        var options = CompressionOptions.Conservative;

        // Assert
        Assert.Equal(0.1, options.TargetReduction);
        Assert.True(options.PreserveVariables);
        Assert.True(options.PreserveCodeBlocks);
    }

    #endregion

    #region FixedExampleSelector Tests

    [Fact]
    public void FixedExampleSelector_Constructor_CreatesEmptySelector()
    {
        // Act
        var selector = new FixedExampleSelector<double>();

        // Assert
        Assert.Equal(0, selector.ExampleCount);
    }

    [Fact]
    public void FixedExampleSelector_AddExample_IncreasesCount()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();

        // Act
        selector.AddExample(new FewShotExample { Input = "test", Output = "result" });

        // Assert
        Assert.Equal(1, selector.ExampleCount);
    }

    [Fact]
    public void FixedExampleSelector_AddExample_ThrowsOnNullExample()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => selector.AddExample(null!));
    }

    [Fact]
    public void FixedExampleSelector_AddExample_ThrowsOnEmptyInput()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            selector.AddExample(new FewShotExample { Input = "", Output = "result" }));
    }

    [Fact]
    public void FixedExampleSelector_AddExample_ThrowsOnEmptyOutput()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            selector.AddExample(new FewShotExample { Input = "test", Output = "" }));
    }

    [Fact]
    public void FixedExampleSelector_SelectExamples_ReturnsFirstN()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();
        selector.AddExample(new FewShotExample { Input = "first", Output = "1" });
        selector.AddExample(new FewShotExample { Input = "second", Output = "2" });
        selector.AddExample(new FewShotExample { Input = "third", Output = "3" });

        // Act
        var selected = selector.SelectExamples("any query", 2);

        // Assert
        Assert.Equal(2, selected.Count);
        Assert.Equal("first", selected[0].Input);
        Assert.Equal("second", selected[1].Input);
    }

    [Fact]
    public void FixedExampleSelector_SelectExamples_ReturnsAllIfCountExceedsAvailable()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();
        selector.AddExample(new FewShotExample { Input = "only", Output = "1" });

        // Act
        var selected = selector.SelectExamples("any query", 10);

        // Assert
        Assert.Single(selected);
        Assert.Equal("only", selected[0].Input);
    }

    [Fact]
    public void FixedExampleSelector_SelectExamples_ThrowsOnEmptyQuery()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();
        selector.AddExample(new FewShotExample { Input = "test", Output = "result" });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => selector.SelectExamples("", 1));
        Assert.Throws<ArgumentException>(() => selector.SelectExamples("   ", 1));
    }

    [Fact]
    public void FixedExampleSelector_SelectExamples_ThrowsOnNonPositiveCount()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();
        selector.AddExample(new FewShotExample { Input = "test", Output = "result" });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => selector.SelectExamples("query", 0));
        Assert.Throws<ArgumentException>(() => selector.SelectExamples("query", -1));
    }

    [Fact]
    public void FixedExampleSelector_SelectExamples_ReturnsEmptyWhenNoExamples()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();

        // Act
        var selected = selector.SelectExamples("any query", 5);

        // Assert
        Assert.Empty(selected);
    }

    [Fact]
    public void FixedExampleSelector_RemoveExample_DecreasesCount()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();
        var example = new FewShotExample { Input = "test", Output = "result" };
        selector.AddExample(example);

        // Act
        var removed = selector.RemoveExample(example);

        // Assert
        Assert.True(removed);
        Assert.Equal(0, selector.ExampleCount);
    }

    [Fact]
    public void FixedExampleSelector_RemoveExample_ReturnsFalseWhenNotFound()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();
        var example = new FewShotExample { Input = "test", Output = "result" };

        // Act
        var removed = selector.RemoveExample(example);

        // Assert
        Assert.False(removed);
    }

    [Fact]
    public void FixedExampleSelector_RemoveExample_ReturnsFalseOnNull()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();

        // Act
        var removed = selector.RemoveExample(null!);

        // Assert
        Assert.False(removed);
    }

    [Fact]
    public void FixedExampleSelector_GetAllExamples_ReturnsAllExamples()
    {
        // Arrange
        var selector = new FixedExampleSelector<double>();
        selector.AddExample(new FewShotExample { Input = "first", Output = "1" });
        selector.AddExample(new FewShotExample { Input = "second", Output = "2" });

        // Act
        var all = selector.GetAllExamples();

        // Assert
        Assert.Equal(2, all.Count);
    }

    #endregion

    #region ToolRegistry Tests

    [Fact]
    public void ToolRegistry_Constructor_CreatesEmptyRegistry()
    {
        // Act
        var registry = new ToolRegistry();

        // Assert
        Assert.Equal(0, registry.Count);
    }

    [Fact]
    public void ToolRegistry_RegisterTool_IncrementsCount()
    {
        // Arrange
        var registry = new ToolRegistry();
        var tool = new MockFunctionTool("test_tool", "Test description");

        // Act
        registry.RegisterTool(tool);

        // Assert
        Assert.Equal(1, registry.Count);
    }

    [Fact]
    public void ToolRegistry_RegisterTool_ThrowsOnNullTool()
    {
        // Arrange
        var registry = new ToolRegistry();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => registry.RegisterTool(null!));
    }

    [Fact]
    public void ToolRegistry_RegisterTool_ThrowsOnDuplicateName()
    {
        // Arrange
        var registry = new ToolRegistry();
        var tool1 = new MockFunctionTool("test_tool", "First tool");
        var tool2 = new MockFunctionTool("test_tool", "Second tool");
        registry.RegisterTool(tool1);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => registry.RegisterTool(tool2));
    }

    [Fact]
    public void ToolRegistry_GetTool_ReturnsTool()
    {
        // Arrange
        var registry = new ToolRegistry();
        var tool = new MockFunctionTool("test_tool", "Test description");
        registry.RegisterTool(tool);

        // Act
        var retrieved = registry.GetTool("test_tool");

        // Assert
        Assert.NotNull(retrieved);
        Assert.Equal("test_tool", retrieved.Name);
    }

    [Fact]
    public void ToolRegistry_GetTool_ReturnsNullWhenNotFound()
    {
        // Arrange
        var registry = new ToolRegistry();

        // Act
        var retrieved = registry.GetTool("nonexistent");

        // Assert
        Assert.Null(retrieved);
    }

    [Fact]
    public void ToolRegistry_GetTool_IsCaseInsensitive()
    {
        // Arrange
        var registry = new ToolRegistry();
        var tool = new MockFunctionTool("Test_Tool", "Test description");
        registry.RegisterTool(tool);

        // Act
        var retrieved = registry.GetTool("TEST_TOOL");

        // Assert
        Assert.NotNull(retrieved);
    }

    [Fact]
    public void ToolRegistry_GetTool_ReturnsNullOnEmptyName()
    {
        // Arrange
        var registry = new ToolRegistry();

        // Act & Assert
        Assert.Null(registry.GetTool(""));
        Assert.Null(registry.GetTool("   "));
    }

    [Fact]
    public void ToolRegistry_UnregisterTool_RemovesTool()
    {
        // Arrange
        var registry = new ToolRegistry();
        var tool = new MockFunctionTool("test_tool", "Test description");
        registry.RegisterTool(tool);

        // Act
        var result = registry.UnregisterTool("test_tool");

        // Assert
        Assert.True(result);
        Assert.Equal(0, registry.Count);
        Assert.Null(registry.GetTool("test_tool"));
    }

    [Fact]
    public void ToolRegistry_UnregisterTool_ReturnsFalseWhenNotFound()
    {
        // Arrange
        var registry = new ToolRegistry();

        // Act
        var result = registry.UnregisterTool("nonexistent");

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ToolRegistry_UnregisterTool_ReturnsFalseOnEmptyName()
    {
        // Arrange
        var registry = new ToolRegistry();

        // Act & Assert
        Assert.False(registry.UnregisterTool(""));
        Assert.False(registry.UnregisterTool("   "));
    }

    [Fact]
    public void ToolRegistry_HasTool_ReturnsTrueWhenExists()
    {
        // Arrange
        var registry = new ToolRegistry();
        var tool = new MockFunctionTool("test_tool", "Test description");
        registry.RegisterTool(tool);

        // Act
        var result = registry.HasTool("test_tool");

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ToolRegistry_HasTool_ReturnsFalseWhenNotExists()
    {
        // Arrange
        var registry = new ToolRegistry();

        // Act
        var result = registry.HasTool("nonexistent");

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ToolRegistry_HasTool_ReturnsFalseOnEmptyName()
    {
        // Arrange
        var registry = new ToolRegistry();

        // Act & Assert
        Assert.False(registry.HasTool(""));
        Assert.False(registry.HasTool("   "));
    }

    [Fact]
    public void ToolRegistry_GetAllTools_ReturnsAllTools()
    {
        // Arrange
        var registry = new ToolRegistry();
        registry.RegisterTool(new MockFunctionTool("tool1", "First"));
        registry.RegisterTool(new MockFunctionTool("tool2", "Second"));

        // Act
        var tools = registry.GetAllTools();

        // Assert
        Assert.Equal(2, tools.Count);
    }

    [Fact]
    public void ToolRegistry_ExecuteTool_ExecutesTool()
    {
        // Arrange
        var registry = new ToolRegistry();
        var tool = new MockFunctionTool("calculator", "Calculator tool");
        registry.RegisterTool(tool);
        var args = JObject.Parse("{\"a\": 1, \"b\": 2}");

        // Act
        var result = registry.ExecuteTool("calculator", args);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void ToolRegistry_ExecuteTool_ThrowsWhenToolNotFound()
    {
        // Arrange
        var registry = new ToolRegistry();
        var args = JObject.Parse("{}");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => registry.ExecuteTool("nonexistent", args));
    }

    [Fact]
    public void ToolRegistry_Clear_RemovesAllTools()
    {
        // Arrange
        var registry = new ToolRegistry();
        registry.RegisterTool(new MockFunctionTool("tool1", "First"));
        registry.RegisterTool(new MockFunctionTool("tool2", "Second"));

        // Act
        registry.Clear();

        // Assert
        Assert.Equal(0, registry.Count);
    }

    [Fact]
    public void ToolRegistry_GenerateToolsDescription_ReturnsDescription()
    {
        // Arrange
        var registry = new ToolRegistry();
        registry.RegisterTool(new MockFunctionTool("calculator", "Performs calculations"));

        // Act
        var description = registry.GenerateToolsDescription();

        // Assert
        Assert.Contains("calculator", description);
        Assert.Contains("Performs calculations", description);
    }

    #endregion

    #region SequentialChain Tests

    [Fact]
    public void SequentialChain_Constructor_CreatesChain()
    {
        // Act
        var chain = new SequentialChain<string, string>("TestChain", "Test description");

        // Assert
        Assert.Empty(chain.Steps);
    }

    [Fact]
    public void SequentialChain_AddStep_AddsStep()
    {
        // Arrange
        var chain = new SequentialChain<string, string>("TestChain");

        // Act
        chain.AddStep("Step1", input => input.ToString().ToUpper());

        // Assert
        Assert.Single(chain.Steps);
        Assert.Equal("Step1", chain.Steps[0]);
    }

    [Fact]
    public void SequentialChain_AddStep_ThrowsOnEmptyName()
    {
        // Arrange
        var chain = new SequentialChain<string, string>("TestChain");

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            chain.AddStep("", input => input));
        Assert.Throws<ArgumentException>(() =>
            chain.AddStep("   ", input => input));
    }

    [Fact]
    public void SequentialChain_AddStep_ThrowsOnNullFunction()
    {
        // Arrange
        var chain = new SequentialChain<string, string>("TestChain");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            chain.AddStep("Step1", (Func<object, object>)null!));
    }

    [Fact]
    public void SequentialChain_Run_ExecutesStepsInOrder()
    {
        // Arrange
        var chain = new SequentialChain<string, string>("TestChain");
        chain.AddStep("ToUpper", input => input.ToString()!.ToUpper());
        chain.AddStep("AddExclaim", input => input.ToString() + "!");

        // Act
        var result = chain.Run("hello");

        // Assert
        Assert.Equal("HELLO!", result);
    }

    [Fact]
    public void SequentialChain_Run_SingleStep()
    {
        // Arrange
        var chain = new SequentialChain<int, int>("TestChain");
        chain.AddStep("Double", input => (int)input * 2);

        // Act
        var result = chain.Run(5);

        // Assert
        Assert.Equal(10, result);
    }

    [Fact]
    public void SequentialChain_AddStep_ReturnsSelfForChaining()
    {
        // Arrange
        var chain = new SequentialChain<string, string>("TestChain");

        // Act
        var result = chain
            .AddStep("Step1", input => input.ToString()!.ToUpper())
            .AddStep("Step2", input => input.ToString() + "!");

        // Assert
        Assert.Same(chain, result);
        Assert.Equal(2, chain.Steps.Count);
    }

    [Fact]
    public async Task SequentialChain_RunAsync_ExecutesSteps()
    {
        // Arrange
        var chain = new SequentialChain<string, string>("TestChain");
        chain.AddStep("ToUpper", input => input.ToString()!.ToUpper());

        // Act
        var result = await chain.RunAsync("hello", CancellationToken.None);

        // Assert
        Assert.Equal("HELLO", result);
    }

    [Fact]
    public async Task SequentialChain_RunAsync_SupportsAsyncSteps()
    {
        // Arrange
        var chain = new SequentialChain<string, string>("TestChain");
        chain.AddStepAsync("AsyncStep", async (input, ct) =>
        {
            await Task.Delay(10, ct);
            return input.ToString()!.ToUpper();
        });

        // Act
        var result = await chain.RunAsync("hello", CancellationToken.None);

        // Assert
        Assert.Equal("HELLO", result);
    }

    [Fact]
    public async Task SequentialChain_RunAsync_SupportsCancellation()
    {
        // Arrange
        var chain = new SequentialChain<string, string>("TestChain");
        chain.AddStepAsync("SlowStep", async (input, ct) =>
        {
            await Task.Delay(5000, ct);
            return input;
        });

        var cts = new CancellationTokenSource();
        cts.CancelAfter(50);

        // Act & Assert
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            chain.RunAsync("hello", cts.Token));
    }

    [Fact]
    public void SequentialChain_Run_ThrowsWhenAsyncStepUsed()
    {
        // Arrange
        var chain = new SequentialChain<string, string>("TestChain");
        chain.AddStepAsync("AsyncStep", async (input, ct) =>
        {
            await Task.Delay(10, ct);
            return input;
        });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => chain.Run("hello"));
    }

    #endregion

    #region FewShotExample Tests

    [Fact]
    public void FewShotExample_DefaultValues()
    {
        // Arrange & Act
        var example = new FewShotExample();

        // Assert
        Assert.Equal(string.Empty, example.Input);
        Assert.Equal(string.Empty, example.Output);
    }

    [Fact]
    public void FewShotExample_SetProperties()
    {
        // Arrange & Act
        var example = new FewShotExample
        {
            Input = "Hello in English",
            Output = "Hola en Espa\u00f1ol"
        };

        // Assert
        Assert.Equal("Hello in English", example.Input);
        Assert.Equal("Hola en Espa\u00f1ol", example.Output);
    }

    #endregion

    #region Helper Classes

    /// <summary>
    /// Mock implementation of IFunctionTool for testing.
    /// </summary>
    private class MockFunctionTool : IFunctionTool
    {
        public string Name { get; }
        public string Description { get; }
        public JObject ParameterSchema => new JObject();

        public MockFunctionTool(string name, string description)
        {
            Name = name;
            Description = description;
        }

        public string Execute(JObject arguments)
        {
            return $"Executed {Name} with {arguments}";
        }

        public bool ValidateArguments(JObject arguments)
        {
            return arguments != null;
        }
    }

    #endregion
}
