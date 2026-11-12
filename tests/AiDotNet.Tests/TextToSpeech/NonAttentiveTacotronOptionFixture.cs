using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.TextToSpeech;

internal static class NonAttentiveTacotronOptionFixture
{
    internal static bool UsesTraining(string option) =>
        option is "LearningRate" or "WeightDecay" or "PrenetDropout" or "LstmCellClip";

    internal static void Prepare(NonAttentiveTacotronOptions options, string option)
    {
        switch (option)
        {
            case "MaxTextLength": options.MaxTextLength = 4; break;
            case "MaxMelLength": options.MaxMelLength = 4; break;
            case "PrenetDropout": options.PrenetDropout = 0.5; break;
            case "LstmCellClip": options.LstmCellClip = 0.02; break;
        }
    }

    private static TtsTrainingSample<T> Sample<T>(NonAttentiveTacotron<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        const int tokens = 4;
        const int framesPerToken = 3;
        int channels = model.MelChannels;
        var mel = new Tensor<T>(new[] { tokens * framesPerToken, channels });
        for (int frame = 0; frame < tokens * framesPerToken; frame++)
            for (int channel = 0; channel < channels; channel++)
                mel[frame, channel] = ops.FromDouble(0.6 * Math.Sin(0.17 * frame + 0.11 * channel));
        return new TtsTrainingSample<T>
        {
            Tokens = new Tensor<T>(new[] { tokens }, new Vector<T>(Enumerable.Range(1, tokens).Select(id => ops.FromDouble(id)).ToArray())),
            Mel = mel,
            Durations = Enumerable.Repeat(framesPerToken, tokens).ToArray(),
        };
    }

    internal static long Observe<T>(NonAttentiveTacotron<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return OptionSensitivityHarness.Fingerprint(new[] { ops.ToDouble(model.EvaluateTrainingObjective(Sample(model))) });
    }

    internal static void Train<T>(NonAttentiveTacotron<T> model, int steps)
    {
        var sample = Sample(model);
        for (int step = 0; step < steps; step++) model.Train(sample);
    }
}
