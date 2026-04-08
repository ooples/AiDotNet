using System.Reflection;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class CodeModelArchitectureSerializationTests
{
    [Fact]
    public void WriteAndReadAndValidate_MatchingArchitecture_DoesNotThrow()
    {
        var architecture = CreateArchitecture(maxSequenceLength: 16, vocabularySize: 32, useDataFlow: true, numEncoderLayers: 1, numDecoderLayers: 2);

        var payload = WriteArchitecture(architecture, includeUseDataFlow: true, includeEncoderDecoderLayerCounts: true);

        ReadAndValidate(
            payload,
            architecture,
            modelName: "TestModel",
            includeUseDataFlow: true,
            includeEncoderDecoderLayerCounts: true);
    }

    [Fact]
    public void ReadAndValidate_MismatchedArchitecture_Throws()
    {
        var written = CreateArchitecture(maxSequenceLength: 16, vocabularySize: 32, useDataFlow: true, numEncoderLayers: 1, numDecoderLayers: 2);
        var expected = CreateArchitecture(maxSequenceLength: 16, vocabularySize: 33, useDataFlow: true, numEncoderLayers: 1, numDecoderLayers: 2);

        var payload = WriteArchitecture(written, includeUseDataFlow: true, includeEncoderDecoderLayerCounts: true);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            ReadAndValidate(payload, expected, modelName: "TestModel", includeUseDataFlow: true, includeEncoderDecoderLayerCounts: true));

        Assert.Contains("does not match", ex.Message);
    }

    private static CodeSynthesisArchitecture<double> CreateArchitecture(
        int maxSequenceLength,
        int vocabularySize,
        bool useDataFlow,
        int numEncoderLayers,
        int numDecoderLayers)
    {
        return new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: ProgramLanguage.Generic,
            codeTaskType: CodeTask.Generation,
            numEncoderLayers: numEncoderLayers,
            numDecoderLayers: numDecoderLayers,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            maxSequenceLength: maxSequenceLength,
            vocabularySize: vocabularySize,
            maxProgramLength: 8,
            dropoutRate: 0.0,
            usePositionalEncoding: false,
            useDataFlow: useDataFlow);
    }

    private static byte[] WriteArchitecture(
        CodeSynthesisArchitecture<double> architecture,
        bool includeUseDataFlow,
        bool includeEncoderDecoderLayerCounts)
    {
        var type = Type.GetType("AiDotNet.ProgramSynthesis.Engines.CodeModelArchitectureSerialization, AiDotNet");
        Assert.NotNull(type);

        var method = type!.GetMethod("Write", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        var generic = method!.MakeGenericMethod(typeof(double));
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        generic.Invoke(null, new object[] { writer, architecture, includeUseDataFlow, includeEncoderDecoderLayerCounts });
        writer.Flush();
        return ms.ToArray();
    }

    private static void ReadAndValidate(
        byte[] payload,
        CodeSynthesisArchitecture<double> architecture,
        string modelName,
        bool includeUseDataFlow,
        bool includeEncoderDecoderLayerCounts)
    {
        var type = Type.GetType("AiDotNet.ProgramSynthesis.Engines.CodeModelArchitectureSerialization, AiDotNet");
        Assert.NotNull(type);

        var method = type!.GetMethod("ReadAndValidate", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);

        var generic = method!.MakeGenericMethod(typeof(double));
        using var ms = new MemoryStream(payload);
        using var reader = new BinaryReader(ms);

        try
        {
            generic.Invoke(null, new object[] { reader, architecture, modelName, includeUseDataFlow, includeEncoderDecoderLayerCounts });
        }
        catch (TargetInvocationException ex) when (ex.InnerException is InvalidOperationException invalid)
        {
            throw invalid;
        }
    }
}

