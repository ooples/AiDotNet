using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Visualization;

/// <summary>
/// A simple 5x7 bitmap font for rendering text on images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class provides a way to draw text on images
/// without requiring external font libraries. Each character is defined as a
/// 5-wide by 7-tall grid of pixels.</para>
/// </remarks>
public class BitmapFont<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Character width in pixels.
    /// </summary>
    public const int CharWidth = 5;

    /// <summary>
    /// Character height in pixels.
    /// </summary>
    public const int CharHeight = 7;

    /// <summary>
    /// Horizontal spacing between characters.
    /// </summary>
    public const int CharSpacing = 1;

    /// <summary>
    /// Font glyphs stored as bit patterns.
    /// Each byte represents one row of a character (using bottom 5 bits).
    /// </summary>
    private static readonly Dictionary<char, byte[]> _glyphs = InitializeGlyphs();

    /// <summary>
    /// Creates a new bitmap font instance.
    /// </summary>
    public BitmapFont()
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Measures the width of a text string in pixels.
    /// </summary>
    /// <param name="text">The text to measure.</param>
    /// <returns>Width in pixels.</returns>
    public int MeasureWidth(string text)
    {
        if (string.IsNullOrEmpty(text))
            return 0;

        return text.Length * (CharWidth + CharSpacing) - CharSpacing;
    }

    /// <summary>
    /// Draws text on an image at the specified position.
    /// </summary>
    /// <param name="image">The image tensor [batch, channels, height, width].</param>
    /// <param name="text">The text to draw.</param>
    /// <param name="x">X position (left edge).</param>
    /// <param name="y">Y position (top edge).</param>
    /// <param name="color">Text color as RGB values (0-1 range).</param>
    /// <param name="scale">Scale factor (1 = original 5x7 size).</param>
    public void DrawText(Tensor<T> image, string text, int x, int y,
        (double R, double G, double B) color, int scale = 1)
    {
        if (string.IsNullOrEmpty(text))
            return;

        int height = image.Shape[2];
        int width = image.Shape[3];
        int channels = image.Shape[1];

        int cursorX = x;

        foreach (char c in text)
        {
            DrawChar(image, c, cursorX, y, color, scale, height, width, channels);
            cursorX += (CharWidth + CharSpacing) * scale;
        }
    }

    /// <summary>
    /// Draws text with a background box.
    /// </summary>
    /// <param name="image">The image tensor.</param>
    /// <param name="text">The text to draw.</param>
    /// <param name="x">X position.</param>
    /// <param name="y">Y position.</param>
    /// <param name="textColor">Text color (0-1 range).</param>
    /// <param name="bgColor">Background color (0-1 range).</param>
    /// <param name="scale">Scale factor.</param>
    /// <param name="padding">Padding around text in pixels.</param>
    public void DrawTextWithBackground(Tensor<T> image, string text, int x, int y,
        (double R, double G, double B) textColor,
        (double R, double G, double B) bgColor,
        int scale = 1, int padding = 2)
    {
        if (string.IsNullOrEmpty(text))
            return;

        int height = image.Shape[2];
        int width = image.Shape[3];
        int channels = image.Shape[1];

        int textWidth = MeasureWidth(text) * scale;
        int textHeight = CharHeight * scale;

        // Draw background
        int bgX1 = Math.Max(0, x - padding);
        int bgY1 = Math.Max(0, y - padding);
        int bgX2 = Math.Min(width - 1, x + textWidth + padding);
        int bgY2 = Math.Min(height - 1, y + textHeight + padding);

        for (int py = bgY1; py <= bgY2; py++)
        {
            for (int px = bgX1; px <= bgX2; px++)
            {
                SetPixel(image, 0, py, px, bgColor.R, bgColor.G, bgColor.B, channels);
            }
        }

        // Draw text
        DrawText(image, text, x, y, textColor, scale);
    }

    /// <summary>
    /// Draws a single character at the specified position.
    /// </summary>
    private void DrawChar(Tensor<T> image, char c, int x, int y,
        (double R, double G, double B) color, int scale,
        int imgHeight, int imgWidth, int channels)
    {
        if (!_glyphs.TryGetValue(char.ToUpper(c), out var glyph))
        {
            // Use a placeholder for unknown characters
            glyph = _glyphs.GetValueOrDefault('?', new byte[7]);
        }

        for (int row = 0; row < CharHeight; row++)
        {
            byte rowBits = glyph.Length > row ? glyph[row] : (byte)0;

            for (int col = 0; col < CharWidth; col++)
            {
                // Check if pixel is set (bit 4-col to read left-to-right)
                bool isSet = (rowBits & (1 << (4 - col))) != 0;

                if (isSet)
                {
                    // Draw scaled pixel
                    for (int sy = 0; sy < scale; sy++)
                    {
                        for (int sx = 0; sx < scale; sx++)
                        {
                            int px = x + col * scale + sx;
                            int py = y + row * scale + sy;

                            if (px >= 0 && px < imgWidth && py >= 0 && py < imgHeight)
                            {
                                SetPixel(image, 0, py, px, color.R, color.G, color.B, channels);
                            }
                        }
                    }
                }
            }
        }
    }

    private void SetPixel(Tensor<T> image, int batch, int y, int x,
        double r, double g, double b, int channels)
    {
        if (channels >= 1)
            image[batch, 0, y, x] = _numOps.FromDouble(MathHelper.Clamp(r, 0.0, 1.0));
        if (channels >= 2)
            image[batch, 1, y, x] = _numOps.FromDouble(MathHelper.Clamp(g, 0.0, 1.0));
        if (channels >= 3)
            image[batch, 2, y, x] = _numOps.FromDouble(MathHelper.Clamp(b, 0.0, 1.0));
    }

    /// <summary>
    /// Initializes the 5x7 font glyphs.
    /// Each character is stored as 7 bytes (one per row).
    /// Each byte uses bits 4-0 for the 5 columns (bit 4 = leftmost).
    /// </summary>
    private static Dictionary<char, byte[]> InitializeGlyphs()
    {
        return new Dictionary<char, byte[]>
        {
            // Space
            [' '] = new byte[] { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },

            // Numbers 0-9
            ['0'] = new byte[] { 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E },
            ['1'] = new byte[] { 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E },
            ['2'] = new byte[] { 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F },
            ['3'] = new byte[] { 0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E },
            ['4'] = new byte[] { 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 },
            ['5'] = new byte[] { 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E },
            ['6'] = new byte[] { 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E },
            ['7'] = new byte[] { 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 },
            ['8'] = new byte[] { 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E },
            ['9'] = new byte[] { 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C },

            // Uppercase letters A-Z
            ['A'] = new byte[] { 0x0E, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11 },
            ['B'] = new byte[] { 0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E },
            ['C'] = new byte[] { 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E },
            ['D'] = new byte[] { 0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C },
            ['E'] = new byte[] { 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F },
            ['F'] = new byte[] { 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10 },
            ['G'] = new byte[] { 0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F },
            ['H'] = new byte[] { 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 },
            ['I'] = new byte[] { 0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E },
            ['J'] = new byte[] { 0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C },
            ['K'] = new byte[] { 0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11 },
            ['L'] = new byte[] { 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F },
            ['M'] = new byte[] { 0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11 },
            ['N'] = new byte[] { 0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11 },
            ['O'] = new byte[] { 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },
            ['P'] = new byte[] { 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 },
            ['Q'] = new byte[] { 0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D },
            ['R'] = new byte[] { 0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11 },
            ['S'] = new byte[] { 0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E },
            ['T'] = new byte[] { 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 },
            ['U'] = new byte[] { 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },
            ['V'] = new byte[] { 0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04 },
            ['W'] = new byte[] { 0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A },
            ['X'] = new byte[] { 0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11 },
            ['Y'] = new byte[] { 0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04 },
            ['Z'] = new byte[] { 0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F },

            // Punctuation and symbols
            ['.'] = new byte[] { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C },
            [','] = new byte[] { 0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08 },
            ['!'] = new byte[] { 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04 },
            ['?'] = new byte[] { 0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04 },
            [':'] = new byte[] { 0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00 },
            [';'] = new byte[] { 0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x04, 0x08 },
            ['\''] = new byte[] { 0x0C, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00 },
            ['"'] = new byte[] { 0x0A, 0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00 },
            ['-'] = new byte[] { 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00 },
            ['_'] = new byte[] { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F },
            ['+'] = new byte[] { 0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00 },
            ['='] = new byte[] { 0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00 },
            ['/'] = new byte[] { 0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00 },
            ['\\'] = new byte[] { 0x00, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00 },
            ['('] = new byte[] { 0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02 },
            [')'] = new byte[] { 0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08 },
            ['['] = new byte[] { 0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E },
            [']'] = new byte[] { 0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E },
            ['{'] = new byte[] { 0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02 },
            ['}'] = new byte[] { 0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08 },
            ['<'] = new byte[] { 0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02 },
            ['>'] = new byte[] { 0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08 },
            ['*'] = new byte[] { 0x00, 0x04, 0x15, 0x0E, 0x15, 0x04, 0x00 },
            ['#'] = new byte[] { 0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A },
            ['$'] = new byte[] { 0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04 },
            ['%'] = new byte[] { 0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03 },
            ['&'] = new byte[] { 0x0C, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0D },
            ['@'] = new byte[] { 0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0F },
            ['^'] = new byte[] { 0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00 },
            ['~'] = new byte[] { 0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00 },
            ['|'] = new byte[] { 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 },
            ['`'] = new byte[] { 0x08, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00 },
        };
    }
}
