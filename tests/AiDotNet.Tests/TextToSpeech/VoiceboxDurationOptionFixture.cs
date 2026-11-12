using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class VoiceboxDurationOptionFixture
{
    internal static long Observe<T>(Voicebox<T> model)
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        model.SetTrainingMode(false);
        var duration = (VoiceboxDurationModel<T>)typeof(Voicebox<T>).GetField("_durationModel", flags)!.GetValue(model)!;
        var phones = new[] { 1, 2 };
        duration.Forward(phones, new Tensor<T>(new[] { 2, 1 }));
        var output = (DenseLayer<T>)typeof(VoiceboxDurationModel<T>).GetField("_output", flags)!.GetValue(duration)!;
        // Fit the real duration readout to three frames per phone, activating the configured cap.
        var weights = new Vector<T>(output.GetParameters().Length);
        weights[weights.Length - 1] = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().FromDouble(Math.Log(4));
        output.SetParameters(weights);
        var durations = (int[])typeof(Voicebox<T>).GetMethod("InferDurations", flags)!
            .Invoke(model, new object[] { phones, new[] { 0, 0 }, 0 })!;
        return OptionSensitivityHarness.Fingerprint(durations.Select(value => (double)value));
    }
}
