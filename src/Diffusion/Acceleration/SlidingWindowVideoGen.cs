using AiDotNet.Interfaces;

namespace AiDotNet.Diffusion.Acceleration;

/// <summary>
/// Sliding window video generation for memory-efficient long video synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text" (2024)</item>
/// <item>Paper: "FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling" (2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> Sliding Window Video Generation creates long videos by generating overlapping short segments. Each window overlaps with the previous one for smooth transitions, enabling videos much longer than the model's native capacity.</para>
/// <para>
/// Sliding window generation enables producing videos longer than the model's native frame count
/// by generating overlapping windows and blending them. Key components:
/// - Window size: number of frames generated per chunk (model's native capacity)
/// - Stride: number of new frames per window (window - overlap)
/// - Overlap blending: smooth transition between consecutive windows
/// - Conditional anchor frames: use last frames of previous window as conditioning
/// </para>
/// </remarks>
public class SlidingWindowVideoGen<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _windowSize;
    private readonly int _stride;
    private readonly int _overlapSize;
    private readonly BlendingMode _blendingMode;
    private int _currentWindowIndex;
    private Tensor<T>? _previousWindowLatents;

    /// <summary>
    /// Gets the window size (number of frames per chunk).
    /// </summary>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Gets the stride (new frames per window).
    /// </summary>
    public int Stride => _stride;

    /// <summary>
    /// Gets the overlap size between consecutive windows.
    /// </summary>
    public int OverlapSize => _overlapSize;

    /// <summary>
    /// Gets the blending mode for overlapping regions.
    /// </summary>
    public BlendingMode BlendMode => _blendingMode;

    /// <summary>
    /// Gets the current window index.
    /// </summary>
    public int CurrentWindowIndex => _currentWindowIndex;

    /// <summary>
    /// Initializes a new sliding window video generator.
    /// </summary>
    /// <param name="windowSize">Number of frames per generation window.</param>
    /// <param name="stride">Number of new frames per window step.</param>
    /// <param name="blendingMode">How to blend overlapping frames.</param>
    public SlidingWindowVideoGen(
        int windowSize = 16,
        int stride = 12,
        BlendingMode blendingMode = BlendingMode.LinearBlend)
    {
        if (windowSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(windowSize), "Window size must be positive.");
        if (stride <= 0 || stride > windowSize)
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be between 1 and window size.");

        _windowSize = windowSize;
        _stride = stride;
        _overlapSize = windowSize - stride;
        _blendingMode = blendingMode;
        _currentWindowIndex = 0;
    }

    /// <summary>
    /// Computes the number of windows needed for a target video length.
    /// </summary>
    /// <param name="targetFrames">Total number of frames desired.</param>
    /// <returns>Number of generation windows required.</returns>
    public int ComputeNumWindows(int targetFrames)
    {
        if (targetFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetFrames), "Target frames must be positive.");

        if (targetFrames <= _windowSize)
            return 1;

        return 1 + (int)Math.Ceiling((double)(targetFrames - _windowSize) / _stride);
    }

    /// <summary>
    /// Gets the frame range for the current window.
    /// </summary>
    /// <returns>Tuple of (startFrame, endFrame) for the current window.</returns>
    public (int StartFrame, int EndFrame) GetCurrentWindowRange()
    {
        int start = _currentWindowIndex * _stride;
        int end = start + _windowSize;
        return (start, end);
    }

    /// <summary>
    /// Gets the anchor frames from the previous window for conditioning.
    /// </summary>
    /// <returns>Tensor of anchor frames, or null for the first window.</returns>
    public Tensor<T>? GetAnchorFrames()
    {
        return _previousWindowLatents;
    }

    /// <summary>
    /// Blends the current window with the accumulated output.
    /// </summary>
    /// <param name="currentWindow">Latents from the current generation window.</param>
    /// <param name="accumulated">Previously accumulated latents (may be null for first window).</param>
    /// <returns>Blended result.</returns>
    public Tensor<T> BlendWindow(Tensor<T> currentWindow, Tensor<T>? accumulated)
    {
        Guard.NotNull(currentWindow, nameof(currentWindow));
        if (currentWindow.Shape.Length == 0)
            throw new ArgumentException("Current window must have at least one dimension.", nameof(currentWindow));

        if (accumulated == null || _currentWindowIndex == 0)
        {
            _previousWindowLatents = currentWindow;
            return currentWindow;
        }

        // Blend the overlap region between accumulated and current window
        int overlapElements = _overlapSize;
        int elementsPerFrame = currentWindow.Data.Length / Math.Max(1, currentWindow.Shape[0]);
        int accFrames = accumulated.Shape[0];

        var result = new Tensor<T>(accumulated.Shape);
        accumulated.Data.Span.CopyTo(result.Data.Span);

        for (int f = 0; f < overlapElements && f < currentWindow.Shape[0]; f++)
        {
            double t = (double)(f + 1) / (overlapElements + 1);
            double alpha;
            switch (_blendingMode)
            {
                case BlendingMode.CosineBlend:
                    alpha = 0.5 * (1.0 - Math.Cos(Math.PI * t));
                    break;
                case BlendingMode.HardCut:
                    alpha = t >= 0.5 ? 1.0 : 0.0;
                    break;
                case BlendingMode.SlerpBlend:
                    // Hermite smoothstep: smooth ease-in/ease-out transition (S-curve)
                    alpha = t * t * (3.0 - 2.0 * t);
                    break;
                default: // LinearBlend
                    alpha = t;
                    break;
            }
            int accOffset = (accFrames - overlapElements + f) * elementsPerFrame;
            int curOffset = f * elementsPerFrame;
            for (int j = 0; j < elementsPerFrame && accOffset + j < result.Data.Length; j++)
            {
                double accVal = NumOps.ToDouble(result[accOffset + j]);
                double curVal = NumOps.ToDouble(currentWindow[curOffset + j]);
                result[accOffset + j] = NumOps.FromDouble(accVal * (1.0 - alpha) + curVal * alpha);
            }
        }

        _previousWindowLatents = currentWindow;
        return result;
    }

    /// <summary>
    /// Advances to the next window.
    /// </summary>
    public void AdvanceWindow()
    {
        _currentWindowIndex++;
    }

    /// <summary>
    /// Resets the sliding window state for a new video generation.
    /// </summary>
    public void Reset()
    {
        _currentWindowIndex = 0;
        _previousWindowLatents = null;
    }
}

/// <summary>
/// Blending modes for overlapping window regions.
/// </summary>
public enum BlendingMode
{
    /// <summary>Linear interpolation between overlapping frames.</summary>
    LinearBlend,

    /// <summary>Cosine-weighted blending for smoother transitions.</summary>
    CosineBlend,

    /// <summary>Use the latest window's frames (hard cut).</summary>
    HardCut,

    /// <summary>SLERP (spherical linear interpolation) in latent space.</summary>
    SlerpBlend
}
