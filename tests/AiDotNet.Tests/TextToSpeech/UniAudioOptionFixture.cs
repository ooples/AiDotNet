using System.Reflection;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class UniAudioOptionFixture
{
    internal static bool IsCodebookOption(string option)
        => option is "CodecCodebookDecay" or "CodecDeadCodeThreshold";

    internal static void PrepareOptions(UniAudioOptions options, string option)
    {
        // Expiration would replace the EMA averages at every step and hide the decay.
        // The threshold probe starts at the supported disabled value, then enables expiration.
        if (IsCodebookOption(option)) options.CodecDeadCodeThreshold = 0;
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

    internal static void TrainCodebooks<T>(UniAudio<T> model)
    {
        // The ordinary TTS fixture trains the language model, whose codec is frozen.
        // EMA settings act in the universal codec's actual training forward instead.
        var codec = (UniAudioCodec<T>)typeof(UniAudio<T>).GetField("_codec",
            BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var audio = new Tensor<T>(new[] { codec.HopLength * 8 });
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < audio.Length; i++)
            audio[i] = ops.FromDouble(.2 * Math.Sin(i * .13) + .1 * Math.Cos(i * .31));
        bool wasTraining = model.IsTrainingMode;
        model.SetTrainingMode(true);
        try
        {
            using var scope = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            codec.TrainingForward(audio, ((UniAudioOptions)model.GetOptions()).NumCodebooks);
        }
        finally { model.SetTrainingMode(wasTraining); }
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
