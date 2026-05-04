using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression suite for AiDotNet#1234 —
/// <see cref="SequenceTokenSliceLayer{T}"/> was missing its
/// <see cref="DeserializationHelper.CreateLayerFromType{T}"/> branch, so
/// <c>NeuralNetworkBase{T}.Clone()</c> / <c>DeepCopy()</c> threw
/// <c>NotSupportedException: Layer type SequenceTokenSliceLayer`1 is not
/// supported for deserialization (no known constructor found).</c> on
/// every <see cref="TransformerArchitecture{T}"/> that emitted last-token
/// or CLS-token pooling.
///
/// <para>
/// The reporter's specific impact (HarmonicEngine PR #105) was that
/// <c>OptimizerBase.InitializeRandomSolution</c> internally calls
/// <c>Clone()</c> on the architecture every random restart, so even users
/// who never explicitly serialized the network hit this on the first
/// <c>Optimize()</c> call.
/// </para>
///
/// <para>
/// Fix: dedicated branch in <c>DeserializationHelper.CreateLayerFromType</c>
/// that round-trips the <c>Position</c> enum through
/// <c>additionalParams["Position"]</c> (which
/// <see cref="SequenceTokenSliceLayer{T}.GetMetadata"/> already populates)
/// and reconstructs the layer via its single
/// <c>(Position position)</c> constructor.
/// </para>
/// </summary>
public class SequenceTokenSliceLayerDeserializationIssue1234Tests
{
    [Fact]
    public void DeserializationHelper_RecreatesSequenceTokenSliceLayer_DefaultPosition()
    {
        // No metadata supplied — the helper must default Position to Last so
        // legacy networks serialized before the metadata key existed don't
        // fail or silently switch pooling semantics on round-trip.
        var instance = DeserializationHelper.CreateLayerFromType<float>(
            "SequenceTokenSliceLayer`1",
            inputShape: new[] { 1, 4, 8 },
            outputShape: new[] { 1, 8 });

        var layer = Assert.IsType<SequenceTokenSliceLayer<float>>(instance);
        Assert.Equal(0, layer.ParameterCount);
        Assert.False(layer.SupportsTraining);
    }

    [Fact]
    public void DeserializationHelper_RecreatesSequenceTokenSliceLayer_FirstPosition()
    {
        var instance = DeserializationHelper.CreateLayerFromType<float>(
            "SequenceTokenSliceLayer`1",
            inputShape: new[] { 1, 4, 8 },
            outputShape: new[] { 1, 8 },
            additionalParams: new System.Collections.Generic.Dictionary<string, object>
            {
                ["Position"] = SequenceTokenSliceLayer<float>.Position.First.ToString(),
            });

        var layer = Assert.IsType<SequenceTokenSliceLayer<float>>(instance);

        // Verify behaviorally that Position.First was restored: forward on a
        // sequence whose first position holds a known signal must select that
        // signal. We can't read the private _position field, so we let the
        // layer's actual forward output be the witness.
        var input = new Tensor<float>(new[] { 1, 3, 2 });
        // Position 0 holds [10, 20]; positions 1, 2 hold zeros. If the layer
        // sliced position 0 we expect output [[10, 20]]; if it (incorrectly)
        // sliced position seq-1 we'd see [[0, 0]].
        input[0, 0, 0] = 10f;
        input[0, 0, 1] = 20f;

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(2, output.Shape[1]);
        Assert.Equal(10f, output[0, 0]);
        Assert.Equal(20f, output[0, 1]);
    }

    [Fact]
    public void DeserializationHelper_DefaultPosition_IsLastToken()
    {
        // No metadata — must default to Position.Last (autoregressive-LM
        // convention; matches LayerHelper.CreateDefaultTransformerLayers).
        var instance = DeserializationHelper.CreateLayerFromType<float>(
            "SequenceTokenSliceLayer`1",
            inputShape: new[] { 1, 4, 8 },
            outputShape: new[] { 1, 8 });

        var layer = Assert.IsType<SequenceTokenSliceLayer<float>>(instance);

        var input = new Tensor<float>(new[] { 1, 3, 2 });
        input[0, 2, 0] = 10f;  // last position
        input[0, 2, 1] = 20f;

        var output = layer.Forward(input);

        Assert.Equal(10f, output[0, 0]);
        Assert.Equal(20f, output[0, 1]);
    }

    [Fact]
    public void Transformer_TokenInputArchitecture_SupportsClone()
    {
        // Reporter's exact failing path: any token-input Transformer (vocab > 0)
        // gets a SequenceTokenSliceLayer inserted by LayerHelper to pool the
        // sequence axis down to [batch, dim]. Clone() must round-trip.
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 256,
            inputSize: 16,
            outputSize: 32,
            maxSequenceLength: 16,
            vocabularySize: 32);

        var net = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Pre-fix this throws NotSupportedException on the SequenceTokenSliceLayer
        // entry of the layer chain. Post-fix the clone succeeds and recovers an
        // independent network instance.
        var clone = net.Clone();
        var cloneTransformer = Assert.IsType<Transformer<float>>(clone);
        Assert.NotSame(net, cloneTransformer);

        // Stronger postcondition (matches the deep-copy test's layer-chain
        // assertion): the cloned network MUST contain a
        // SequenceTokenSliceLayer in its layer chain. Bare type / not-same
        // checks pass even if the slice layer was silently dropped during
        // deser, which would regress the original #1234 failure mode
        // (token-input Transformer ends up with no last-token pooling and
        // emits [batch, seq, dim] instead of [batch, dim]).
        Assert.True(
            cloneTransformer.Layers.Any(l => l is SequenceTokenSliceLayer<float>),
            "Cloned Transformer must contain a SequenceTokenSliceLayer (last-token pooling).");
    }

    [Fact]
    public void Transformer_TokenInputArchitecture_SupportsDeepCopy()
    {
        // Same path as Clone(), but exercising DeepCopy() directly so a future
        // refactor that splits the two implementations doesn't hide a regression.
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 256,
            inputSize: 16,
            outputSize: 32,
            maxSequenceLength: 16,
            vocabularySize: 32);

        var net = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var deepCopy = net.DeepCopy();
        Assert.NotNull(deepCopy);
        var deepCopyTransformer = Assert.IsType<Transformer<float>>(deepCopy);
        Assert.NotSame(net, deepCopy);

        // The cloned network must have the same layer chain as the original.
        // Specifically the SequenceTokenSliceLayer must be present in the chain.
        var hasSliceLayer = deepCopyTransformer.Layers.Any(l => l is SequenceTokenSliceLayer<float>);
        Assert.True(hasSliceLayer,
            "Cloned Transformer must contain a SequenceTokenSliceLayer (last-token pooling) — " +
            "if this assertion fails, the clone path lost the layer or DeserializationHelper " +
            "regressed and silently dropped it.");
    }
}
