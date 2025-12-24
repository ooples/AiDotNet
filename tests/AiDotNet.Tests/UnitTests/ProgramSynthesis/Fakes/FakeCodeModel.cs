using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis.Fakes;

internal sealed class FakeCodeModel : CodeModelBase<double>
{
    private readonly CodeSynthesisArchitecture<double> _architecture;

    public FakeCodeModel(CodeSynthesisArchitecture<double> architecture)
        : base(architecture, new CrossEntropyLoss<double>())
    {
        _architecture = architecture;
        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        // Intentionally empty (tests focus on task wiring and structured result shapes).
    }

    public override Tensor<double> Predict(Tensor<double> input)
    {
        return input;
    }

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
                { "ModelName", nameof(FakeCodeModel) },
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
        _ = (ProgramLanguage)reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance()
    {
        return new FakeCodeModel(_architecture);
    }

    public static FakeCodeModel CreateDefault(ProgramLanguage targetLanguage = ProgramLanguage.CSharp)
    {
        var architecture = new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: targetLanguage,
            codeTaskType: CodeTask.Generation,
            maxSequenceLength: 64,
            vocabularySize: 256,
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            dropoutRate: 0.0,
            usePositionalEncoding: false);

        return new FakeCodeModel(architecture);
    }
}
