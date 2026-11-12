using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class BarkOptionFixture
{
    internal static bool Handles(string option) => option is "MaxTextLength" or "SamplingSeed" or "TextPadTokenId"
        or "SemanticHistoryLength" or "SemanticPadTokenId" or "SemanticInferenceTokenId" or "SemanticVocabularySize"
        or "CoarseSlidingWindowLength" or "CoarseSemanticContextLength" or "CoarseHistoryLength"
        or "CoarseInferenceTokenId" or "FineWindowLength" or "CodebookSize";

    internal static void PrepareOptions(BarkOptions options, string option)
    {
        if (option == "CoarseSemanticContextLength") options.Coarse.BlockSize = Math.Max(16, options.Coarse.BlockSize);
    }

    internal static long Observe<T>(Bark<T> model, string option)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option is "MaxTextLength" or "TextPadTokenId" or "SemanticHistoryLength" or "SemanticPadTokenId" or "SemanticInferenceTokenId")
            return TtsPaperNumerics.Fingerprint(model.FirstSemanticLogits(new[] { 3 }).Select(value => ops.ToDouble(value)));

        var generation = new BarkGenerationOptions { AllowEarlyStop = false, MaxSemanticTokens = 8 };
        if (option is "SamplingSeed" or "SemanticVocabularySize")
            return TtsPaperNumerics.Fingerprint(model.GenerateSemanticTokens(new[] { 3 }, generation).Select(value => (double)value));

        var historyCodes = new int[model.NumCodebooks, 8];
        for (int q = 0; q < historyCodes.GetLength(0); q++)
            for (int t = 0; t < historyCodes.GetLength(1); t++) historyCodes[q, t] = (3 * q + t) % 16;
        var history = new BarkHistoryPrompt { SemanticTokens = new[] { 1, 3, 5, 7, 9, 11, 13, 15 }, CodecTokens = historyCodes };
        var coarse = model.GenerateCoarseTokens(new[] { 2, 4, 6, 8, 10, 12, 14, 16 }, generation, history);
        if (option == "FineWindowLength")
            return TtsPaperNumerics.Fingerprint(model.GenerateFineTokens(coarse, generation, history).Cast<int>().Select(value => (double)value));
        return TtsPaperNumerics.Fingerprint(coarse.Cast<int>().Select(value => (double)value));
    }
}
