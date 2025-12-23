using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Engines;

internal static class CodeModelArchitectureSerialization
{
    internal static void Write<T>(
        BinaryWriter writer,
        CodeSynthesisArchitecture<T> architecture,
        bool includeUseDataFlow,
        bool includeEncoderDecoderLayerCounts)
    {
        writer.Write((int)architecture.TargetLanguage);
        writer.Write(architecture.MaxSequenceLength);
        writer.Write(architecture.VocabularySize);

        if (includeUseDataFlow)
        {
            writer.Write(architecture.UseDataFlow);
        }

        if (includeEncoderDecoderLayerCounts)
        {
            writer.Write(architecture.NumEncoderLayers);
            writer.Write(architecture.NumDecoderLayers);
        }
    }

    internal static void ReadAndValidate<T>(
        BinaryReader reader,
        CodeSynthesisArchitecture<T> architecture,
        string modelName,
        bool includeUseDataFlow,
        bool includeEncoderDecoderLayerCounts)
    {
        var targetLanguage = (ProgramLanguage)reader.ReadInt32();
        var maxSeqLength = reader.ReadInt32();
        var vocabSize = reader.ReadInt32();

        bool? useDataFlow = null;
        if (includeUseDataFlow)
        {
            useDataFlow = reader.ReadBoolean();
        }

        int? numEncoderLayers = null;
        int? numDecoderLayers = null;
        if (includeEncoderDecoderLayerCounts)
        {
            numEncoderLayers = reader.ReadInt32();
            numDecoderLayers = reader.ReadInt32();
        }

        var mismatched = targetLanguage != architecture.TargetLanguage;
        mismatched |= maxSeqLength != architecture.MaxSequenceLength;
        mismatched |= vocabSize != architecture.VocabularySize;

        if (!mismatched && useDataFlow is { } dataFlow)
        {
            mismatched = dataFlow != architecture.UseDataFlow;
        }

        if (!mismatched && numEncoderLayers is { } encoderLayers)
        {
            mismatched = encoderLayers != architecture.NumEncoderLayers;
        }

        if (!mismatched && numDecoderLayers is { } decoderLayers)
        {
            mismatched = decoderLayers != architecture.NumDecoderLayers;
        }

        if (!mismatched)
        {
            return;
        }

        throw new InvalidOperationException(
            $"Serialized {modelName} architecture does not match the current instance. " +
            $"Serialized: TargetLanguage={targetLanguage}, MaxSequenceLength={maxSeqLength}, VocabularySize={vocabSize}" +
            (useDataFlow is null ? string.Empty : $", UseDataFlow={useDataFlow.Value}") +
            (numEncoderLayers is null ? string.Empty : $", NumEncoderLayers={numEncoderLayers.Value}") +
            (numDecoderLayers is null ? string.Empty : $", NumDecoderLayers={numDecoderLayers.Value}") +
            $". Expected: TargetLanguage={architecture.TargetLanguage}, MaxSequenceLength={architecture.MaxSequenceLength}, VocabularySize={architecture.VocabularySize}" +
            (useDataFlow is null ? string.Empty : $", UseDataFlow={architecture.UseDataFlow}") +
            (numEncoderLayers is null ? string.Empty : $", NumEncoderLayers={architecture.NumEncoderLayers}") +
            (numDecoderLayers is null ? string.Empty : $", NumDecoderLayers={architecture.NumDecoderLayers}") +
            ".");
    }
}
