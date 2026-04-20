using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NER.TransformerBased;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class TinyBERTNERTests : TransformerNERTestBase
{
    // TinyBERT overrides TransformerNEROptions.HiddenDimension to 312
    // (vs the 768 default used by BERT-base). The generator's default
    // [8, 768] InputShape would fail MultiHeadAttention weight matching,
    // so we pin it to 312 here and construct the model via its
    // architecture-only constructor (which wires up CreateTinyBERTDefaults
    // with the 312-dim attention weights).
    protected override int[] InputShape => [8, 312];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new TinyBERTNER<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 128,
                outputSize: 4));
}
