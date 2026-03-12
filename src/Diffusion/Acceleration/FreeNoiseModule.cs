using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion.Acceleration;

/// <summary>
/// FreeNoise module for tuning-free longer video generation via noise rescheduling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling" (Qiu et al., 2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> The FreeNoise module rearranges noise patterns to enable longer video generation from models trained on short clips. It is a simple, training-free technique that extends video length without quality loss.</para>
/// <para>
/// FreeNoise enables generating longer videos from short-video diffusion models without any
/// fine-tuning. The key idea is noise rescheduling: instead of using independent random noise
/// for all frames, FreeNoise constructs temporally correlated noise by:
/// 1. Generating a base noise sequence for the window size
/// 2. Shifting and blending noise for extended frames
/// 3. Using window-based attention with shared noise patterns
/// This maintains temporal consistency across windows while extending generation length.
/// </para>
/// </remarks>
public class FreeNoiseModule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _windowSize;
    private readonly int _noiseShiftStride;
    private readonly double _blendRatio;
    private readonly Random _random;
    private Tensor<T>? _baseNoise;

    /// <summary>
    /// Gets the window size for noise generation.
    /// </summary>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Gets the noise shift stride.
    /// </summary>
    public int NoiseShiftStride => _noiseShiftStride;

    /// <summary>
    /// Gets the blend ratio between shifted and fresh noise.
    /// </summary>
    public double BlendRatio => _blendRatio;

    /// <summary>
    /// Initializes a new FreeNoise module.
    /// </summary>
    /// <param name="windowSize">Base window size (model's native frame count).</param>
    /// <param name="noiseShiftStride">Number of frames to shift noise for each window.</param>
    /// <param name="blendRatio">Ratio of base noise vs fresh noise (0.0 = all fresh, 1.0 = all base).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public FreeNoiseModule(
        int windowSize = 16,
        int noiseShiftStride = 4,
        double blendRatio = 0.5,
        int? seed = null)
    {
        if (windowSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(windowSize), "Window size must be positive.");
        if (noiseShiftStride <= 0 || noiseShiftStride > windowSize)
            throw new ArgumentOutOfRangeException(nameof(noiseShiftStride), "Shift stride must be between 1 and window size.");

        if (blendRatio < 0.0 || blendRatio > 1.0)
            throw new ArgumentOutOfRangeException(nameof(blendRatio), "Blend ratio must be between 0.0 and 1.0.");

        _windowSize = windowSize;
        _noiseShiftStride = noiseShiftStride;
        _blendRatio = blendRatio;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Generates temporally correlated noise for a target number of frames.
    /// </summary>
    /// <param name="targetFrames">Total number of frames to generate noise for.</param>
    /// <param name="latentShape">Shape of a single frame's latent (e.g., [channels, height, width]).</param>
    /// <returns>Noise tensor for all target frames.</returns>
    public Tensor<T> GenerateRescheduledNoise(int targetFrames, int[] latentShape)
    {
        Guard.Positive(targetFrames);
        Guard.NotNull(latentShape);
        if (latentShape.Length == 0)
        {
            throw new ArgumentException("Latent shape must have at least one dimension.", nameof(latentShape));
        }

        // Generate base noise for the window size
        int elementsPerFrame = 1;
        foreach (int dim in latentShape) elementsPerFrame *= dim;

        var totalShape = new int[latentShape.Length + 1];
        totalShape[0] = targetFrames;
        Array.Copy(latentShape, 0, totalShape, 1, latentShape.Length);

        var noise = new Tensor<T>(totalShape);

        // Fill base window with random noise
        int baseElements = Math.Min(targetFrames, _windowSize) * elementsPerFrame;
        for (int i = 0; i < baseElements; i++)
        {
            noise[i] = NumOps.FromDouble(SampleGaussian());
        }

        // For extended frames, use stride-shifted noise from the base window.
        // Each stride-length block maps to a shifted region of the base noise,
        // following the FreeNoise rescheduling algorithm.
        for (int frame = _windowSize; frame < targetFrames; frame++)
        {
            int extIdx = frame - _windowSize;
            int windowIdx = extIdx / _noiseShiftStride;
            int frameInWindow = extIdx % _noiseShiftStride;
            int sourceFrame = ((windowIdx + 1) * _noiseShiftStride + frameInWindow) % _windowSize;
            int frameOffset = frame * elementsPerFrame;
            int sourceOffset = sourceFrame * elementsPerFrame;

            for (int j = 0; j < elementsPerFrame; j++)
            {
                double baseVal = NumOps.ToDouble(noise[sourceOffset + j]);
                double freshVal = SampleGaussian();
                double blended = _blendRatio * baseVal + (1.0 - _blendRatio) * freshVal;
                noise[frameOffset + j] = NumOps.FromDouble(blended);
            }
        }

        _baseNoise = noise;
        return noise;
    }

    /// <summary>
    /// Gets a copy of the stored base noise from the last generation.
    /// </summary>
    public Tensor<T>? GetBaseNoise()
    {
        if (_baseNoise is null) return null;
        var copy = new Tensor<T>(_baseNoise.Shape);
        _baseNoise.Data.Span.CopyTo(copy.Data.Span);
        return copy;
    }

    /// <summary>
    /// Resets the module state.
    /// </summary>
    public void Reset()
    {
        _baseNoise = null;
    }

    /// <summary>
    /// Samples a value from the standard normal distribution using Box-Muller transform.
    /// </summary>
    private double SampleGaussian()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
