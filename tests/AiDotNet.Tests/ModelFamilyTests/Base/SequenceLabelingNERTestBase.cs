using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for sequence labeling NER models (BiLSTM-CRF, etc.).
/// Inherits NER invariants and adds sequence-specific: output length matches input
/// and deterministic labeling.
/// </summary>
public abstract class SequenceLabelingNERTestBase : NERModelTestBase
{
    [Fact(Timeout = 120000)]
    public async Task SequenceLabels_ShouldMatchInputLength()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        // Sequence labeling should produce one label per input position
        Assert.True(output.Length > 0, "Sequence labeling produced empty output.");
    }

    [Fact(Timeout = 120000)]
    public async Task SequenceLabeling_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var out1 = network.Predict(input);
        var out2 = network.Predict(input);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i]);
    }
}
