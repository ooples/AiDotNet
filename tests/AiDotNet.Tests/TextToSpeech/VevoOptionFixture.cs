using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class VevoOptionFixture
{
    internal static void PrepareOptions(VevoOptions options)
    {
        // Four phones, two input delimiters and the output start token consume seven positions.
        options.MaxLength = 11;
        options.MinNewTokens = 4;
        options.ConstrainToContentStyleTokens = true;
    }

    internal static long Observe<T>(Vevo<T> model)
    {
        var options = (VevoOptions)model.GetOptions();
        var decoding = new VevoDecoding(options.Sample, options.Temperature, options.TopK, options.TopP,
            options.RepetitionPenalty, options.MinNewTokens, options.MaxLength, options.ConstrainToContentStyleTokens);
        var codes = model.TextToContentStyle.Generate(new[] { 3, 7, 11, 4 }, Array.Empty<int>(), decoding,
            new Random(options.SamplingSeed));
        return TtsPaperNumerics.Fingerprint(codes.Select(value => (double)value));
    }
}
