using System.Reflection;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class UniAudioOptionFixture
{
    internal static void PrepareOptions(UniAudioOptions options, string option)
    {
        // Untrained EOS logits can end every trial before a length cap or temperature acts.
        // Exercise those settings through the real sampler, requiring complete audio patches.
        if (option == "MaximumGeneratedFrames") options.MinimumGeneratedFrames = 4;
        if (option == "MaximumLengthRatio")
        {
            options.MaximumGeneratedFrames = 32;
            options.MinimumGeneratedFrames = 32;
        }
        if (option == "Temperature")
        {
            options.MinimumGeneratedFrames = 8;
            options.MaximumGeneratedFrames = 8;
        }
    }

    internal static long ObserveFrontend<T>(UniAudio<T> model)
    {
        // Predict(Tensor) already receives phone IDs and cannot exercise a raw-text frontend.
        // Call the same tokenizer that Synthesize(string) calls, before codec decoding can hide it.
        var tokenize = typeof(UniAudio<T>).GetMethod("TokenizePhoneIds", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var ids = (int[])tokenize.Invoke(model, new object[] { "hello world" })!;
        return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(ids.Select(id => (double)id));
    }
}
