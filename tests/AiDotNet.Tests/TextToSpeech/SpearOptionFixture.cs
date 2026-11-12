using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Native SPEAR tasks: semantic denoising, backtranslation, prompted acoustic training and sampling.</summary>
internal static class SpearOptionFixture
{
    internal static bool Handles(string option) => option is "PretrainingDeletionProbability" or "PretrainingDropoutRate"
        or "BacktranslationTrainableDecoderLayers" or "SpeakingDropoutRate" or "PromptSeconds"
        or "Temperature" or "SamplingSeed" or "MaxSemanticTokens";

    internal static void Prepare(SPEARTTSOptions options, string option)
    {
        // N top plus N bottom needs a middle block to distinguish N=1 from N=2.
        if (option == "BacktranslationTrainableDecoderLayers") options.NumDecoderLayers = 4;
        if (option == "MaxSemanticTokens") options.MaxSemanticTokens = 1;
        if (option is "Temperature" or "SamplingSeed") options.MaxAcousticFrames = 16;
    }

    internal static int[,] Grid(SPEARTTSOptions options, int frames = 32)
    {
        var codes = new int[1 + options.NumCodebooks, frames];
        int factor = options.SampleRate / options.HopSize / options.SemanticFrameRate;
        for (int f = 0; f < frames; f++)
        {
            codes[0, f] = (f / factor * 7) % options.SemanticCodes;
            for (int q = 0; q < options.NumCodebooks; q++) codes[q + 1, f] = (3 * f + 5 * q + 1) % options.CodebookSize;
        }
        return codes;
    }

    internal static TtsTrainingSample<T> Sample<T>(SPEARTTS<T> model)
    {
        var options = (SPEARTTSOptions)model.GetOptions();
        var grid = Grid(options);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var tokens = new Tensor<T>(new[] { grid.GetLength(1), grid.GetLength(0) });
        for (int f = 0; f < grid.GetLength(1); f++)
            for (int q = 0; q < grid.GetLength(0); q++) tokens[f, q] = ops.FromDouble(grid[q, f]);
        return new TtsTrainingSample<T> { Tokens = model.TextToTokens("hello there"), CodecTokens = tokens };
    }

    internal static double[] Observe<T>(SPEARTTS<T> model, string option)
    {
        if (option == "MaxSemanticTokens") return model.ReadText(model.TextToTokens("hello there")).Select(id => (double)id).ToArray();
        if (option is "Temperature" or "SamplingSeed")
        {
            var options = (SPEARTTSOptions)model.GetOptions();
            var codes = Grid(options);
            var observed = new List<double>();
            for (int window = 0; window < 4; window++)
            {
                var (ids, firstTarget) = model.SpeakingSequence(codes, new Random(window));
                var generated = model.SampleAcousticTokens(ids.Take(firstTarget).ToArray());
                observed.Add(generated.Count);
                observed.AddRange(generated.Select(id => (double)id));
            }
            return observed.ToArray();
        }
        model.CurrentStage = option switch
        {
            "PretrainingDeletionProbability" or "PretrainingDropoutRate" => SPEARTTSTrainingStage.Pretraining,
            "BacktranslationTrainableDecoderLayers" => SPEARTTSTrainingStage.Backtranslation,
            _ => SPEARTTSTrainingStage.Speaking,
        };
        return new[] { AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble(model.Train(Sample(model))) };
    }
}
