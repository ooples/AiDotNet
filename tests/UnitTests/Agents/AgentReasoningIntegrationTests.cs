using AiDotNet.Enums;
using AiDotNet.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.Agents;

/// <summary>
/// Integration tests for AI Agent assistance with Chain-of-Thought reasoning.
/// Tests Phase 4 implementation of reasoning-enabled agent assistance.
/// </summary>
public class AgentReasoningIntegrationTests
{
    /// <summary>
    /// Tests that AgentAssistanceOptions can be configured with reasoning enabled.
    /// </summary>
    [Fact]
    public void AgentAssistanceOptions_WithReasoningEnabled_ConfiguresCorrectly()
    {
        // Arrange & Act
        var options = AgentAssistanceOptions.Create()
            .EnableReasoning()
            .WithMaxReasoningSteps(7)
            .WithReasoningConfidenceThreshold(0.8)
            .Build();

        // Assert
        Assert.True(options.EnableReasoning);
        Assert.Equal(7, options.MaxReasoningSteps);
        Assert.Equal(0.8, options.ReasoningConfidenceThreshold);
    }

    /// <summary>
    /// Tests that AgentAssistanceOptions defaults have reasoning disabled.
    /// </summary>
    [Fact]
    public void AgentAssistanceOptions_Default_HasReasoningDisabled()
    {
        // Arrange & Act
        var options = AgentAssistanceOptions.Default;

        // Assert
        Assert.False(options.EnableReasoning);
        Assert.Equal(5, options.MaxReasoningSteps);
        Assert.Equal(0.7, options.ReasoningConfidenceThreshold);
    }

    /// <summary>
    /// Tests that Comprehensive preset enables reasoning.
    /// </summary>
    [Fact]
    public void AgentAssistanceOptions_Comprehensive_EnablesReasoning()
    {
        // Arrange & Act
        var options = AgentAssistanceOptions.Comprehensive;

        // Assert
        Assert.True(options.EnableReasoning);
        Assert.Equal(7, options.MaxReasoningSteps);
    }

    /// <summary>
    /// Tests that Minimal preset has reasoning disabled.
    /// </summary>
    [Fact]
    public void AgentAssistanceOptions_Minimal_HasReasoningDisabled()
    {
        // Arrange & Act
        var options = AgentAssistanceOptions.Minimal;

        // Assert
        Assert.False(options.EnableReasoning);
    }

    /// <summary>
    /// Tests that WithMaxReasoningSteps validates range.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(11)]
    [InlineData(-1)]
    public void AgentAssistanceOptions_WithMaxReasoningSteps_ValidatesRange(int invalidSteps)
    {
        // Arrange
        var builder = AgentAssistanceOptions.Create();

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            builder.WithMaxReasoningSteps(invalidSteps));
    }

    /// <summary>
    /// Tests that WithReasoningConfidenceThreshold validates range.
    /// </summary>
    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    [InlineData(2.0)]
    public void AgentAssistanceOptions_WithReasoningConfidenceThreshold_ValidatesRange(double invalidThreshold)
    {
        // Arrange
        var builder = AgentAssistanceOptions.Create();

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            builder.WithReasoningConfidenceThreshold(invalidThreshold));
    }

    /// <summary>
    /// Tests that AgentConfiguration can be created with reasoning-enabled options.
    /// </summary>
    [Fact]
    public void AgentConfiguration_WithReasoningOptions_ConfiguresCorrectly()
    {
        // Arrange
        var assistanceOptions = AgentAssistanceOptions.Create()
            .EnableReasoning()
            .WithMaxReasoningSteps(5)
            .Build();

        // Act
        var agentConfig = new AgentConfiguration<double>
        {
            ApiKey = "test-key",
            Provider = LLMProvider.OpenAI,
            IsEnabled = true,
            AssistanceOptions = assistanceOptions
        };

        // Assert
        Assert.NotNull(agentConfig.AssistanceOptions);
        Assert.True(agentConfig.AssistanceOptions.EnableReasoning);
        Assert.Equal(5, agentConfig.AssistanceOptions.MaxReasoningSteps);
    }

    /// <summary>
    /// Tests that AgentConfiguration uses Default options when not specified.
    /// </summary>
    [Fact]
    public void AgentConfiguration_WithoutOptions_UsesDefault()
    {
        // Arrange & Act
        var agentConfig = new AgentConfiguration<double>
        {
            ApiKey = "test-key",
            Provider = LLMProvider.OpenAI,
            IsEnabled = true
            // AssistanceOptions not set
        };

        // Assert - should use default when null
        Assert.Null(agentConfig.AssistanceOptions);
        // Actual default application happens in PredictionModelResult.AskAsync
    }

    /// <summary>
    /// Tests that Clone preserves reasoning settings.
    /// </summary>
    [Fact]
    public void AgentAssistanceOptions_Clone_PreservesReasoningSettings()
    {
        // Arrange
        var original = AgentAssistanceOptions.Create()
            .EnableReasoning()
            .WithMaxReasoningSteps(8)
            .WithReasoningConfidenceThreshold(0.85)
            .EnableDataAnalysis()
            .Build();

        // Act
        var cloned = original.Clone();

        // Assert
        Assert.Equal(original.EnableReasoning, cloned.EnableReasoning);
        Assert.Equal(original.MaxReasoningSteps, cloned.MaxReasoningSteps);
        Assert.Equal(original.ReasoningConfidenceThreshold, cloned.ReasoningConfidenceThreshold);
        Assert.Equal(original.EnableDataAnalysis, cloned.EnableDataAnalysis);
    }

    /// <summary>
    /// Tests fluent builder pattern with multiple reasoning configurations.
    /// </summary>
    [Fact]
    public void AgentAssistanceOptions_FluentBuilder_ChainsCorrectly()
    {
        // Arrange & Act
        var options = AgentAssistanceOptions.Create()
            .EnableDataAnalysis()
            .EnableModelSelection()
            .EnableReasoning()
            .WithMaxReasoningSteps(6)
            .WithReasoningConfidenceThreshold(0.75)
            .Build();

        // Assert
        Assert.True(options.EnableDataAnalysis);
        Assert.True(options.EnableModelSelection);
        Assert.False(options.EnableHyperparameterTuning);
        Assert.False(options.EnableFeatureAnalysis);
        Assert.False(options.EnableMetaLearningAdvice);
        Assert.True(options.EnableReasoning);
        Assert.Equal(6, options.MaxReasoningSteps);
        Assert.Equal(0.75, options.ReasoningConfidenceThreshold);
    }

    /// <summary>
    /// Tests that DisableReasoning works correctly.
    /// </summary>
    [Fact]
    public void AgentAssistanceOptions_DisableReasoning_WorksCorrectly()
    {
        // Arrange & Act
        var options = AgentAssistanceOptions.Create()
            .EnableReasoning()
            .DisableReasoning()
            .Build();

        // Assert
        Assert.False(options.EnableReasoning);
    }

    /// <summary>
    /// Tests valid reasoning step range boundaries.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public void AgentAssistanceOptions_WithMaxReasoningSteps_AcceptsValidRange(int validSteps)
    {
        // Arrange & Act
        var options = AgentAssistanceOptions.Create()
            .WithMaxReasoningSteps(validSteps)
            .Build();

        // Assert
        Assert.Equal(validSteps, options.MaxReasoningSteps);
    }

    /// <summary>
    /// Tests valid confidence threshold range boundaries.
    /// </summary>
    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(0.7)]
    [InlineData(1.0)]
    public void AgentAssistanceOptions_WithReasoningConfidenceThreshold_AcceptsValidRange(double validThreshold)
    {
        // Arrange & Act
        var options = AgentAssistanceOptions.Create()
            .WithReasoningConfidenceThreshold(validThreshold)
            .Build();

        // Assert
        Assert.Equal(validThreshold, options.ReasoningConfidenceThreshold);
    }
}
