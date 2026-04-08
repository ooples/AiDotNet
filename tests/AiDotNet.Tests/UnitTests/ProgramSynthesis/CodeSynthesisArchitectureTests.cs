using AiDotNet.Enums;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

/// <summary>
/// Unit tests for CodeSynthesisArchitecture class.
/// </summary>
public class CodeSynthesisArchitectureTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesInstance()
    {
        // Arrange & Act
        var architecture = new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: ProgramLanguage.Python,
            codeTaskType: CodeTask.Generation,
            numEncoderLayers: 6,
            numDecoderLayers: 6,
            numHeads: 8,
            modelDimension: 512,
            feedForwardDimension: 2048,
            maxSequenceLength: 512,
            vocabularySize: 50000,
            maxProgramLength: 100);

        // Assert
        Assert.NotNull(architecture);
        Assert.Equal(SynthesisType.Neural, architecture.SynthesisType);
        Assert.Equal(ProgramLanguage.Python, architecture.TargetLanguage);
        Assert.Equal(CodeTask.Generation, architecture.CodeTaskType);
        Assert.Equal(6, architecture.NumEncoderLayers);
        Assert.Equal(6, architecture.NumDecoderLayers);
        Assert.Equal(8, architecture.NumHeads);
        Assert.Equal(512, architecture.ModelDimension);
        Assert.Equal(2048, architecture.FeedForwardDimension);
        Assert.Equal(512, architecture.MaxSequenceLength);
        Assert.Equal(50000, architecture.VocabularySize);
        Assert.Equal(100, architecture.MaxProgramLength);
    }

    [Fact]
    public void Constructor_DefaultValues_CreatesInstanceWithDefaults()
    {
        // Arrange & Act
        var architecture = new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Hybrid,
            targetLanguage: ProgramLanguage.CSharp,
            codeTaskType: CodeTask.Completion);

        // Assert
        Assert.NotNull(architecture);
        Assert.Equal(6, architecture.NumEncoderLayers);
        Assert.Equal(0, architecture.NumDecoderLayers);
        Assert.Equal(8, architecture.NumHeads);
        Assert.Equal(512, architecture.ModelDimension);
        Assert.Equal(0.1, architecture.DropoutRate);
        Assert.True(architecture.UsePositionalEncoding);
        Assert.False(architecture.UseDataFlow);
    }

    [Fact]
    public void Constructor_WithDataFlow_SetsDataFlowCorrectly()
    {
        // Arrange & Act
        var architecture = new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: ProgramLanguage.Java,
            codeTaskType: CodeTask.BugDetection,
            useDataFlow: true);

        // Assert
        Assert.True(architecture.UseDataFlow);
    }

    [Fact]
    public void Constructor_DifferentLanguages_CreatesCorrectInstances()
    {
        // Arrange & Act
        var pythonArch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.Python,
            CodeTask.Generation);

        var javaArch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.Java,
            CodeTask.Translation);

        var csharpArch = new CodeSynthesisArchitecture<double>(
            SynthesisType.Neural,
            ProgramLanguage.CSharp,
            CodeTask.Refactoring);

        // Assert
        Assert.Equal(ProgramLanguage.Python, pythonArch.TargetLanguage);
        Assert.Equal(ProgramLanguage.Java, javaArch.TargetLanguage);
        Assert.Equal(ProgramLanguage.CSharp, csharpArch.TargetLanguage);
    }
}
