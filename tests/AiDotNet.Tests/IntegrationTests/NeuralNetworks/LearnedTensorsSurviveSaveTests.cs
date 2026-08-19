using System;
using System.IO;
using System.Reflection;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Pins that a tensor the layer TRAINS is a tensor the layer SAVES.
/// </summary>
/// <remarks>
/// <para>
/// The generated <c>Serialize_Deserialize_ShouldPreserveBehavior</c> invariant cannot catch this
/// class of defect. It compares a freshly constructed layer against a restored one, so an
/// unregistered tensor whose initialization is DETERMINISTIC holds the same value in both and the
/// round trip looks clean. The value only diverges once training has moved it -- which is exactly
/// when a checkpoint matters.
/// </para>
/// <para>
/// This test perturbs the field the way a training step would, then asks whether the change
/// survives a round trip. Each case first measures SENSITIVITY -- whether perturbing the field
/// changes the output at all -- and fails if it does not, because a field the output ignores would
/// report a clean round trip whether or not it was saved, and that silent pass is what let the
/// original defect through.
/// </para>
/// </remarks>
public class LearnedTensorsSurviveSaveTests
{
    private static Tensor<double> Ramp(int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = 0.01 * (i + 1);
        return t;
    }

    public static TheoryData<string, string> Cases() => new()
    {
        // Confirmed lost before this fix, with the measured output drift:
        { "TransNormerLLM", "_gammas" },          // 3.175e-03
        { "TransNormerLLM", "_keyNormScale" },    // 6.528e-05
        { "TransNormerLLM", "_queryNormScale" },  // 6.528e-05
        { "TransNormerLLM", "_outputNormScale" }, // 5.102e-02
        { "MesaNet", "_lnGamma" },                // 4.240e-02
        { "MesaNet", "_lnBeta" },                 // 1.404e-01
        { "TTT", "_lnGamma" },                    // 7.537e-02
        { "TTT", "_lnBeta" },                     // 2.662e-02
        // Controls that were already correct, by each of the two registration mechanisms:
        { "S4D", "_aReal" },                      // [TrainableParameter] attribute
        { "S4D", "_logDelta" },
        { "Mamba", "_aLog" },                     // explicit RegisterTrainableParameter call
        { "Mamba", "_dParam" },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void PerturbingALearnedTensor_SurvivesARoundTrip(string layerName, string fieldName)
    {
        var (make, shape) = Spec(layerName);
        var input = Ramp(shape);

        var layer = make();
        layer.SetTrainingMode(false);
        layer.ResetState();
        layer.Forward(input);

        var field = layer.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.True(field is not null, $"{layerName} has no field named {fieldName}");

        var tensor = field!.GetValue(layer) as Tensor<double>;
        Assert.True(tensor is not null && tensor.Length > 0,
            $"{layerName}.{fieldName} is not a populated Tensor<double>");

        layer.ResetState();
        var baseline = layer.Forward(input).Clone();

        for (int i = 0; i < tensor!.Length; i++) tensor[i] = tensor[i] + 0.25;

        layer.ResetState();
        var mutated = layer.Forward(input).Clone();

        double sensitivity = 0;
        for (int i = 0; i < baseline.Length; i++)
            sensitivity = Math.Max(sensitivity, Math.Abs(baseline[i] - mutated[i]));

        Assert.True(sensitivity > 1e-12,
            $"{layerName}.{fieldName}: perturbing it did not change the output (sensitivity " +
            $"{sensitivity:E3}), so the round-trip assertion below would pass whether or not the " +
            "field is saved and would prove nothing");

        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            layer.Serialize(writer);

        var restored = make();
        ms.Position = 0;
        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            restored.Deserialize(reader);
        restored.SetTrainingMode(false);
        restored.ResetState();
        var restoredOutput = restored.Forward(input).Clone();

        double drift = 0;
        for (int i = 0; i < mutated.Length; i++)
            drift = Math.Max(drift, Math.Abs(mutated[i] - restoredOutput[i]));

        Assert.True(drift <= 1e-12,
            $"{layerName}.{fieldName} was LOST on save: output drifted {drift:E3} after a round " +
            $"trip (the field's own influence on the output is {sensitivity:E3}), which means the " +
            "layer trains a tensor it does not persist");
    }

    private static (Func<LayerBase<double>> Make, int[] Shape) Spec(string name) => name switch
    {
        // modelDimension 8 / numHeads 4 so the head split is valid.
        "TransNormerLLM" => (() => new TransNormerLLMLayer<double>(4, 8, 4), [1, 4, 8]),
        "MesaNet" => (() => new MesaNetLayer<double>(4, 8, 4), [1, 4, 8]),
        "TTT" => (() => new TTTLayer<double>(4, 8, 4), [1, 4, 8]),
        "S4D" => (() => new S4DLayer<double>(4, 4, 4), [1, 4, 4]),
        "Mamba" => (() => new MambaBlock<double>(4, 4, 4, 2, 2), [1, 4, 4]),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown layer"),
    };
}
