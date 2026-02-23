using AiDotNet.Safety.Text;
using AiDotNet.Safety.Watermarking;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Abstract base class for text watermarking modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for text watermarkers including watermark strength
/// configuration and common token processing. Concrete implementations provide
/// the actual watermarking strategy (sampling, lexical, syntactic).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all text watermarkers.
/// Each watermarker type extends this and adds its own way of embedding invisible
/// signatures in AI-generated text.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TextWatermarkerBase<T> : TextSafetyModuleBase<T>, ITextWatermarker<T>
{
    /// <summary>
    /// The watermark strength factor (0.0 to 1.0).
    /// </summary>
    protected readonly double WatermarkStrength;

    /// <summary>
    /// Initializes the text watermarker base.
    /// </summary>
    /// <param name="watermarkStrength">Watermark embedding strength. Default: 0.5.</param>
    protected TextWatermarkerBase(double watermarkStrength = 0.5)
    {
        WatermarkStrength = watermarkStrength;
    }

    /// <inheritdoc />
    public abstract double DetectWatermark(string text);
}
