using System.Linq;
using System.Reflection;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class ProgramSynthesisDtoCoverageTests
{
    [Theory]
    [InlineData(typeof(CodeBugDetectionRequest), CodeTask.BugDetection)]
    [InlineData(typeof(CodeBugFixingRequest), CodeTask.BugFixing)]
    [InlineData(typeof(CodeCloneDetectionRequest), CodeTask.CloneDetection)]
    [InlineData(typeof(CodeCompletionRequest), CodeTask.Completion)]
    [InlineData(typeof(CodeDocumentationRequest), CodeTask.Documentation)]
    [InlineData(typeof(CodeGenerationRequest), CodeTask.Generation)]
    [InlineData(typeof(CodeRefactoringRequest), CodeTask.Refactoring)]
    [InlineData(typeof(CodeReviewRequest), CodeTask.CodeReview)]
    [InlineData(typeof(CodeSearchRequest), CodeTask.Search)]
    [InlineData(typeof(CodeSummarizationRequest), CodeTask.Summarization)]
    [InlineData(typeof(CodeTestGenerationRequest), CodeTask.TestGeneration)]
    [InlineData(typeof(CodeTranslationRequest), CodeTask.Translation)]
    [InlineData(typeof(CodeUnderstandingRequest), CodeTask.Understanding)]
    public void Requests_ExposeExpectedTask(Type requestType, CodeTask expectedTask)
    {
        var request = Assert.IsAssignableFrom<CodeTaskRequestBase>(Activator.CreateInstance(requestType));
        Assert.Equal(expectedTask, request.Task);
    }

    [Theory]
    [InlineData(typeof(CodeBugDetectionResult), CodeTask.BugDetection)]
    [InlineData(typeof(CodeBugFixingResult), CodeTask.BugFixing)]
    [InlineData(typeof(CodeCloneDetectionResult), CodeTask.CloneDetection)]
    [InlineData(typeof(CodeCompletionResult), CodeTask.Completion)]
    [InlineData(typeof(CodeDocumentationResult), CodeTask.Documentation)]
    [InlineData(typeof(CodeGenerationResult), CodeTask.Generation)]
    [InlineData(typeof(CodeRefactoringResult), CodeTask.Refactoring)]
    [InlineData(typeof(CodeReviewResult), CodeTask.CodeReview)]
    [InlineData(typeof(CodeSearchResult), CodeTask.Search)]
    [InlineData(typeof(CodeSummarizationResult), CodeTask.Summarization)]
    [InlineData(typeof(CodeTestGenerationResult), CodeTask.TestGeneration)]
    [InlineData(typeof(CodeTranslationResult), CodeTask.Translation)]
    [InlineData(typeof(CodeUnderstandingResult), CodeTask.Understanding)]
    public void Results_ExposeExpectedTask(Type resultType, CodeTask expectedTask)
    {
        var result = Assert.IsAssignableFrom<CodeTaskResultBase>(Activator.CreateInstance(resultType));
        Assert.Equal(expectedTask, result.Task);
    }

    [Fact]
    public void ProgramSynthesis_PublicDtos_AreConstructible_AndPropertiesAreAccessible()
    {
        var assembly = typeof(CodeTaskRequestBase).Assembly;

        var dtoTypes = assembly
            .GetExportedTypes()
            .Where(t => t.IsClass)
            .Where(t => !t.IsAbstract)
            .Where(t => !t.ContainsGenericParameters)
            .Where(t => t.Namespace is not null)
            .Where(t =>
                t.Namespace.StartsWith("AiDotNet.ProgramSynthesis.Execution", StringComparison.Ordinal) ||
                t.Namespace.StartsWith("AiDotNet.ProgramSynthesis.Models", StringComparison.Ordinal) ||
                t.Namespace.StartsWith("AiDotNet.ProgramSynthesis.Options", StringComparison.Ordinal) ||
                t.Namespace.StartsWith("AiDotNet.ProgramSynthesis.Requests", StringComparison.Ordinal) ||
                t.Namespace.StartsWith("AiDotNet.ProgramSynthesis.Results", StringComparison.Ordinal))
            .Where(t => t.GetConstructor(Type.EmptyTypes) is not null)
            .ToList();

        Assert.NotEmpty(dtoTypes);

        foreach (var dtoType in dtoTypes)
        {
            var instance = Activator.CreateInstance(dtoType);
            Assert.NotNull(instance);

            foreach (var property in dtoType.GetProperties(BindingFlags.Instance | BindingFlags.Public))
            {
                if (property.GetIndexParameters().Length != 0)
                {
                    continue;
                }

                if (property.CanWrite && property.SetMethod is not null && property.SetMethod.IsPublic)
                {
                    var value = CreateSimpleValue(property.PropertyType);
                    if (value is not null || !property.PropertyType.IsValueType)
                    {
                        property.SetValue(instance, value);
                    }
                }

                if (property.CanRead && property.GetMethod is not null && property.GetMethod.IsPublic)
                {
                    _ = property.GetValue(instance);
                }
            }
        }
    }

    private static object? CreateSimpleValue(Type type)
    {
        if (type == typeof(string))
        {
            return "value";
        }

        if (type == typeof(bool))
        {
            return true;
        }

        if (type == typeof(int))
        {
            return 1;
        }

        if (type == typeof(long))
        {
            return 1L;
        }

        if (type == typeof(double))
        {
            return 1.0;
        }

        if (type == typeof(float))
        {
            return 1.0f;
        }

        if (type == typeof(Guid))
        {
            return Guid.Empty;
        }

        if (type == typeof(DateTime))
        {
            return new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        }

        if (type.IsEnum)
        {
            var values = Enum.GetValues(type);
            return values.Length == 0 ? Activator.CreateInstance(type) : values.GetValue(0);
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
        {
            type = Nullable.GetUnderlyingType(type)!;
            return CreateSimpleValue(type);
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(List<>))
        {
            return Activator.CreateInstance(type);
        }

        if (type.IsArray)
        {
            return Array.CreateInstance(type.GetElementType()!, 0);
        }

        if (!type.IsAbstract && type.GetConstructor(Type.EmptyTypes) is not null)
        {
            return Activator.CreateInstance(type);
        }

        return null;
    }
}

