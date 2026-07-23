using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Regression tests for issue #1264 — Transformer must default to Adam, not
/// vanilla GradientDescent, on every code path that would otherwise pick a
/// fallback optimizer (constructor and deserialization).
/// </summary>
public class TransformerDefaultOptimizerTests
{
    private static TransformerArchitecture<float> MakeArch() =>
        new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 1,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            inputSize: 8,
            outputSize: 8,
            maxSequenceLength: 4,
            vocabularySize: 8);

    private static string GetOptimizerName<T>(Transformer<T> t)
    {
        // GetModelMetadata.AdditionalInfo["Optimizer"] holds the runtime
        // optimizer's GetType().Name — the only public surface to inspect
        // it (the field is private).
        var metadata = t.GetModelMetadata();
        return metadata.AdditionalInfo!["Optimizer"]?.ToString() ?? "";
    }

    [Fact]
    public void Constructor_NullOptimizer_DefaultsToAdam()
    {
        var transformer = new Transformer<float>(MakeArch(), optimizer: null);
        Assert.Contains("Adam", GetOptimizerName(transformer));
    }

    [Fact]
    public void Deserialize_MissingOptimizer_FallsBackToAdam_NotGradientDescent()
    {
        // Directly exercise DeserializeNetworkSpecificData's null-optimizer
        // fallback path. A naive Serialize → Deserialize round-trip never
        // reaches the fallback because Serialize always writes the
        // optimizer's type-name (so DeserializeInterface returns the
        // serialized optimizer, not null). To actually verify the fix
        // for #1264, hand-build a BinaryReader stream where the optimizer
        // field carries an empty type-name (the wire format that means
        // "no optimizer"), and confirm the fallback constructs Adam with
        // the Vaswani 2017 hyperparameters — NOT the legacy
        // GradientDescentOptimizer that the issue reported.
        var arch = MakeArch();

        // The Transformer-specific DeserializeNetworkSpecificData expects
        // (in order): 5 int32s, 1 double for dropout, then two
        // DeserializeInterface payloads (loss function, optimizer) where
        // each payload is a single length-prefixed string. Empty string
        // → null → fallback path.
        using var ms = new System.IO.MemoryStream();
        using (var w = new System.IO.BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            w.Write(2);  // numHeads
            w.Write(1);  // numEncoderLayers
            w.Write(0);  // numDecoderLayers
            w.Write(4);  // maxSequenceLength (matches MakeArch)
            w.Write(8);  // vocabularySize (matches MakeArch)
            w.Write(0.1);  // dropoutRate
            w.Write(string.Empty);  // loss function: null sentinel
            w.Write(string.Empty);  // optimizer:    null sentinel ← exercises fallback
        }
        ms.Position = 0;

        var transformer = new Transformer<float>(arch, optimizer: null);
        using (var r = new System.IO.BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            // Invoke the protected method via reflection. The PR's null-
            // optimizer fallback lives directly inside this method.
            var method = typeof(Transformer<float>).GetMethod(
                "DeserializeNetworkSpecificData",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            Assert.NotNull(method);
            method!.Invoke(transformer, new object[] { r });
        }

        var optimizerName = GetOptimizerName(transformer);
        Assert.Contains("Adam", optimizerName);
        Assert.DoesNotContain("GradientDescent", optimizerName);
    }
}
