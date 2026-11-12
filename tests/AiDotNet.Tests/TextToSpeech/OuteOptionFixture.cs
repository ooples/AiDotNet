using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Exercises native OuteTTS components which waveform-only minimum-length probes bypass.</summary>
internal static class OuteOptionFixture
{
    internal static bool Handles(string option) => option is "DropoutRate" or "RopeHighFrequencyFactor"
        or "RopeLowFrequencyFactor" or "RopeOriginalMaxPositions" or "Chunked" or "ChunkMinWords"
        or "ChunkMaxWords" or "MaxSpeakerSeconds" or "AlignerTemperatures";

    internal static void Prepare(OuteTTSOptions options, string option)
    {
        if (option.StartsWith("Rope", StringComparison.Ordinal))
        {
            // The default generated fixture is Qwen, which disables Llama-3 scaling. Exercise the released Llama variant.
            options.Backbone = OuteTTSBackbone.Llama1B;
            options.VocabSize = 134400;
            options.QueryKeyNorm = false;
            options.RopeTheta = 500000;
            options.RopeScalingFactor = 32;
            options.RopeOriginalMaxPositions = 8192;
            options.RopeLowFrequencyFactor = 1;
            options.RopeHighFrequencyFactor = 4;
        }
        if (option is "Chunked" or "ChunkMinWords" or "ChunkMaxWords")
        {
            options.Chunked = true;
            options.ChunkMinWords = 10;
            options.ChunkMaxWords = 30;
        }
        if (option == "MaxSpeakerSeconds") options.MaxSpeakerSeconds = .05;
        if (option is "MaxSpeakerSeconds" or "AlignerTemperatures") options.AlignerTemperatures = new[] { 0.0 };
    }

    internal static string ChunkText(string option) => option == "ChunkMinWords"
        ? string.Join(" ", Enumerable.Repeat("one", 10)) + ". " + string.Join(" ", Enumerable.Repeat("two", 25)) + "."
        : string.Join(" ", Enumerable.Repeat("word", 80)) + ".";

    internal static double[] Observe<T>(OuteTTS<T> model, string option)
    {
        if (option.StartsWith("Rope", StringComparison.Ordinal)) return model.LanguageModel.InverseFrequencies();
        if (option is "Chunked" or "ChunkMinWords" or "ChunkMaxWords")
            return model.SynthesisChunks(ChunkText(option)).SelectMany(s => new[] { (double)s.Length }
                .Concat(s.Select(c => (double)c))).ToArray();
        if (option == "DropoutRate")
        {
            bool training = model.IsTrainingMode;
            model.SetTrainingMode(true);
            try
            {
                using var _ = new NoGradScope<T>();
                int[] ids = { 1, 3, 5, 7, 9, 11, 13, 15 };
                return new[] { AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble(model.Loss(ids, ids)[0]) };
            }
            finally { model.SetTrainingMode(training); }
        }
        LoadShortWhisperDecision(model);
        if (option == "AlignerTemperatures")
        {
            var audio = Enumerable.Range(0, 1200).Select(i => .3 * Math.Sin(.17 * i)).ToArray();
            var result = model.TranscribeSpeaker(audio);
            if (result.Segments.Count == 0) throw new InvalidOperationException("The native Whisper probe produced no decoded segment.");
            return result.Segments.SelectMany(s => new[] { s.Temperature, s.AvgLogprob, s.NoSpeechProb, s.Start, s.End }).ToArray();
        }
        if (option == "MaxSpeakerSeconds")
        {
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var audio = new Tensor<T>(new[] { 1800 }); // 0.075 s: between the control and its first doubled bound.
            for (int i = 0; i < audio.Length; i++) audio[i] = ops.FromDouble(.3 * Math.Sin(.17 * i));
            try
            {
                var speaker = model.CreateSpeaker(audio);
                return new[] { 1.0, (double)speaker.GlobalFeatures.Energy, speaker.GlobalFeatures.Pitch };
            }
            catch (ArgumentException e) when (e.ParamName == "audio" && e.Message.StartsWith("Speaker audio is longer", StringComparison.Ordinal))
            { return new[] { 0.0 }; }
        }
        throw new ArgumentOutOfRangeException(nameof(option));
    }

    /// <summary>A deterministic parameter fixture for the real Whisper encoder, decoder, token filters, and transcription
    /// loop. Its tied output embedding makes two adjacent timestamps equally likely, then EOT dominant: greedy and
    /// stochastic decoding choose different actual segment timings in two steps, preserving all network components
    /// and release vocabulary/context dimensions.</summary>
    internal static void LoadShortWhisperDecision<T>(OuteTTS<T> model)
    {
        int eot = model.Transcriber.Eot, timestamp = model.Transcriber.TimestampBegin;
        model.Aligner.LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, b) => a * b);
            var values = new double[count];
            if (name == "decoder.ln.bias") return Enumerable.Repeat(1.0, count).ToArray();
            if (name == "decoder.ln.weight") return values;
            if (name == "decoder.token_embedding.weight")
            {
                int width = shape[1];
                for (int row = 0; row < shape[0]; row++)
                    for (int col = 0; col < width; col++)
                        values[row * width + col] = (row == eot ? 30 : row == timestamp || row == timestamp + 1 ? 20 : -16) / (double)width;
                return values;
            }
            if (name.EndsWith(".weight", StringComparison.Ordinal) && shape.Length == 1)
                return Enumerable.Repeat(1.0, count).ToArray();
            int phase = name.Aggregate(0, (a, c) => (a * 31 + c) & 0xffff);
            for (int i = 0; i < count; i++) values[i] = .01 * Math.Sin(i * .37 + phase * .001);
            return values;
        });
    }
}
