namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>The released generator explicitly constructs its periodic 20-point Hann window in float32.
/// Keep those input-independent DSP coefficients even when learned weights use double precision.</summary>
internal static class KokoroSignalBuffers
{
    internal static double[] HannFloat32() => new[]
    {
        0.0, 0.02447172999382019, 0.09549149870872498, 0.20610737800598145,
        0.34549152851104736, 0.5, 0.6545085310935974, 0.7938926219940186,
        0.9045084714889526, 0.9755282998085022, 1.0, 0.9755282402038574,
        0.9045084714889526, 0.793892502784729, 0.6545085310935974, 0.5,
        0.3454914391040802, 0.2061072587966919, 0.09549137949943542, 0.02447172999382019
    };
}
