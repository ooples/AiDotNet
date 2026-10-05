namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Renders words in a 5x7 bitmap font onto a black image, so text-family tests can train and score on images
/// whose ground truth is known exactly.
/// </summary>
internal static class SyntheticTextImages
{
    private const int GlyphWidth = 5;
    private const int GlyphHeight = 7;

    private static readonly Dictionary<char, byte[]> Font = new()
    {
        ['H'] = new byte[] { 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 },
        ['I'] = new byte[] { 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111 },
        ['T'] = new byte[] { 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 },
        ['L'] = new byte[] { 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111 },
        ['E'] = new byte[] { 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111 },
    };

    /// <summary>The characters the font can draw.</summary>
    public static string Alphabet => new(Font.Keys.ToArray());

    /// <summary>
    /// Draws <paramref name="text"/> into image <paramref name="batchIndex"/> of <paramref name="image"/> with
    /// its top-left at (<paramref name="left"/>, <paramref name="top"/>). Each font pixel becomes a
    /// <paramref name="scale"/> x <paramref name="scale"/> block, and glyphs sit on a pitch of six font pixels.
    /// Every channel is set to one.
    /// </summary>
    /// <returns>The word's box, from the first glyph's top-left to the last glyph's bottom-right.</returns>
    public static (double Left, double Top, double Right, double Bottom) Draw<T>(Tensor<T> image, int batchIndex,
        string text, int left, int top, int scale)
    {
        var one = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().One;
        int pitch = (GlyphWidth + 1) * scale;
        for (int glyph = 0; glyph < text.Length; glyph++)
        {
            if (!Font.TryGetValue(text[glyph], out var rows))
                throw new ArgumentException($"The test font cannot draw '{text[glyph]}'.", nameof(text));
            for (int row = 0; row < GlyphHeight; row++)
                for (int column = 0; column < GlyphWidth; column++)
                    if ((rows[row] & (1 << (GlyphWidth - 1 - column))) != 0)
                        for (int dy = 0; dy < scale; dy++)
                            for (int dx = 0; dx < scale; dx++)
                                for (int channel = 0; channel < image.Shape[1]; channel++)
                                    image[batchIndex, channel, top + (scale * row) + dy, left + (pitch * glyph) + (scale * column) + dx] = one;
        }
        return (left, top, left + (pitch * (text.Length - 1)) + (GlyphWidth * scale), top + (GlyphHeight * scale));
    }
}
