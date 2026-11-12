using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

internal static class MinMoOptionFixture
{
    internal static bool Handles(string option) => option.StartsWith("SmallEncoder", StringComparison.Ordinal)
        || option is "LargeEncoderHeads" or "ProjectorHeads" or "DuplexHeads" or "LoraAlpha" or "LoraDropout";

    internal static void Prepare(MinMoOptions options, string option)
    {
        if (option.StartsWith("SmallEncoder", StringComparison.Ordinal))
            options.Encoder = MinMoVoiceEncoder.SenseVoiceSmall;
    }

    internal static long Observe<T>(MinMo<T> model, string option)
    {
        var core = model.Core;
        var numeric = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        long Fingerprint(Tensor<T> value) => OptionSensitivityHarness.Fingerprint(value.Select(numeric.ToDouble));
        if (option is "LoraAlpha" or "LoraDropout")
        {
            // A freshly initialized adapter has zero B weights. Fit actual adapter weights so its
            // scale and training dropout can act, as they do after instruction tuning.
            foreach (var layer in core.Llm.Layers)
                foreach (var adapter in layer.Loras)
                {
                    var weights = adapter.B.GetParameters();
                    for (int i = 0; i < weights.Length; i++) weights[i] = numeric.FromDouble(.03 * Math.Sin(i * .17 + .4));
                    adapter.B.SetParameters(weights);
                }
            var embeddings = core.TokenEmbeddings(new[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            core.Llm.Training = option == "LoraDropout";
            try
            {
                return Fingerprint(core.Llm.Forward(AiDotNetEngine.Current.Reshape(embeddings,
                    new[] { 1, 8, core.Options.HiddenDim }), null, null, null, lastLogitsOnly: true).Hidden);
            }
            finally { core.Llm.Training = false; }
        }
        var speech = core.SpeechEmbeddings(MinMoFixture.Tones<T>(.4), MinMoFixture.SampleRate);
        if (option != "DuplexHeads") return Fingerprint(speech);
        var hidden = core.Llm.Forward(AiDotNetEngine.Current.Reshape(speech,
            new[] { 1, speech.Shape[0], core.Options.HiddenDim }), null, null, null, lastLogitsOnly: true).Hidden;
        return Fingerprint(core.Duplex.Forward(AiDotNetEngine.Current.Reshape(hidden,
            new[] { speech.Shape[0], core.Options.HiddenDim })));
    }
}
