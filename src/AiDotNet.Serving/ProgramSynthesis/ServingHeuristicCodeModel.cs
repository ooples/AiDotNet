using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.ProgramSynthesis;

internal sealed class ServingHeuristicCodeModel : CodeModelBase<double>
{
    private readonly CodeSynthesisArchitecture<double> _architecture;

    public ServingHeuristicCodeModel(CodeSynthesisArchitecture<double> architecture)
        : base(architecture, new CrossEntropyLoss<double>())
    {
        _architecture = architecture;
        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        // Intentionally empty: Serving uses this model for structured task dispatch and heuristics.
    }

    public override Tensor<double> Predict(Tensor<double> input) => input;

    public override void UpdateParameters(Vector<double> parameters)
    {
    }

    public override void Train(Tensor<double> input, Tensor<double> expectedOutput)
    {
    }

    public override ModelMetadata<double> GetModelMetadata()
    {
        return new ModelMetadata<double>
        {
            ModelType = ModelType.Transformer,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", nameof(ServingHeuristicCodeModel) },
                { "TargetLanguage", _architecture.TargetLanguage.ToString() },
                { "CodeTaskType", _architecture.CodeTaskType.ToString() }
            }
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_architecture.TargetLanguage);
        writer.Write(_architecture.MaxSequenceLength);
        writer.Write(_architecture.VocabularySize);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        var targetLanguage = (ProgramLanguage)reader.ReadInt32();
        var maxSeqLength = reader.ReadInt32();
        var vocabSize = reader.ReadInt32();

        if (targetLanguage != _architecture.TargetLanguage ||
            maxSeqLength != _architecture.MaxSequenceLength ||
            vocabSize != _architecture.VocabularySize)
        {
            throw new InvalidOperationException(
                $"Serialized model architecture does not match this instance. " +
                $"Serialized: Language={targetLanguage}, MaxSequenceLength={maxSeqLength}, VocabularySize={vocabSize}. " +
                $"Instance: Language={_architecture.TargetLanguage}, MaxSequenceLength={_architecture.MaxSequenceLength}, VocabularySize={_architecture.VocabularySize}.");
        }
    }

    protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance()
    {
        return new ServingHeuristicCodeModel(_architecture);
    }

    public static ServingHeuristicCodeModel CreateDefault(ProgramLanguage targetLanguage = ProgramLanguage.Generic)
    {
        var architecture = new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: targetLanguage,
            codeTaskType: CodeTask.Generation,
            maxSequenceLength: 128,
            vocabularySize: 1024,
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            dropoutRate: 0.0,
            usePositionalEncoding: false);

        return new ServingHeuristicCodeModel(architecture);
    }
}
