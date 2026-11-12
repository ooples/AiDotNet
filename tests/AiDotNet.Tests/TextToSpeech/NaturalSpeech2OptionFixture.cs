using AiDotNet.Audio.Generation;
using AiDotNet.Models;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class NaturalSpeech2OptionFixture
{
    internal static bool Handles(string option) => option is "SampleRate" or "HopSize" or "PromptMinFraction"
        or "TimeEpsilon" or "TrainingSteps" or "CeResidual";

    internal static void Prepare(NaturalSpeech2Options options, string option)
    {
        if (option == "TimeEpsilon") options.TimeEpsilon = .2;
        if (option == "TrainingSteps")
        {
            options.CosineSchedule = true;
            options.TrainingSteps = 4;
        }
    }

    internal static void Reconcile(NaturalSpeech2Options options, string option)
    {
        if (option is not ("SampleRate" or "HopSize") || options.Codec is not SoundStreamOptions current)
            return;

        var codec = (SoundStreamOptions)CloneEngine.CopyConfiguration(current);
        int originalHop = codec.Ratios.Aggregate(1, (product, stride) => product * stride);
        if (option == "HopSize")
        {
            int lastIndex = codec.Ratios.Length - 1;
            int prefix = originalHop / codec.Ratios[lastIndex];
            if (options.HopSize < prefix || options.HopSize % prefix != 0) return;
            codec.Ratios = (int[])codec.Ratios.Clone();
            codec.Ratios[lastIndex] = options.HopSize / prefix;
        }
        else
        {
            codec.SampleRate = options.SampleRate;
        }
        // Keep the active quantizer count fixed as the codec frame rate changes.
        codec.TargetBandwidthKbps = current.TargetBandwidthKbps
            * (codec.SampleRate / (double)current.SampleRate)
            * (originalHop / (double)options.HopSize);
        options.Codec = codec;
    }

    private static TtsTrainingSample<T> Sample<T>()
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        const int phones = 20;
        const int samples = 1600;
        return new TtsTrainingSample<T>
        {
            Tokens = new Tensor<T>(new[] { phones }, new Vector<T>(
                Enumerable.Range(1, phones).Select(id => ops.FromDouble(id)).ToArray())),
            Audio = new Tensor<T>(new[] { samples }, new Vector<T>(Enumerable.Range(0, samples)
                .Select(i => ops.FromDouble(.3 * Math.Sin(.21 * i) + .08 * Math.Sin(.47 * i))).ToArray())),
            Durations = Enumerable.Repeat(4, phones).ToArray(),
        };
    }

    internal static long Observe<T>(NaturalSpeech2<T> model, string option)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var sample = Sample<T>();
        // The paper freezes a fitted codec before training the diffusion model.
        // An unfitted codebook can assign every frame to the same residual code.
        if (option == "CeResidual") model.Codec.InitializeCodebooks(sample.Audio!);
        return OptionSensitivityHarness.Fingerprint(new[] { ops.ToDouble(model.EvaluateTrainingObjective(sample)) });
    }

    internal static void Train<T>(NaturalSpeech2<T> model, int steps)
    {
        var sample = Sample<T>();
        for (int step = 0; step < steps; step++) model.Train(sample);
    }
}
