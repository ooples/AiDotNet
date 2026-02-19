using AiDotNet.Interfaces;

namespace AiDotNet.Diffusion.Acceleration;

/// <summary>
/// Sparse video generation with selective frame denoising for faster inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Sparse VideoGen: Accelerating Video Diffusion Transformers with Flexible Sparsity" (2025)</item>
/// </list></para>
/// <para>
/// SparseVideoGen accelerates video diffusion inference by identifying and skipping redundant
/// computations. The key insight is that not all frames require equal denoising effort:
/// - Keyframes: get full denoising (all transformer blocks)
/// - Intermediate frames: use sparse computation (skip similar blocks)
/// - Selection criteria: based on temporal motion magnitude
/// This achieves 2-4x speedup with minimal visual quality degradation.
/// </para>
/// </remarks>
public class SparseVideoGen<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _totalFrames;
    private readonly int _keyframeInterval;
    private readonly double _sparsityRatio;
    private readonly SparsityStrategy _strategy;
    private bool[]? _keyframeMask;

    /// <summary>
    /// Gets the total number of frames.
    /// </summary>
    public int TotalFrames => _totalFrames;

    /// <summary>
    /// Gets the keyframe interval.
    /// </summary>
    public int KeyframeInterval => _keyframeInterval;

    /// <summary>
    /// Gets the sparsity ratio (fraction of computation saved).
    /// </summary>
    public double SparsityRatio => _sparsityRatio;

    /// <summary>
    /// Gets the sparsity strategy.
    /// </summary>
    public SparsityStrategy Strategy => _strategy;

    /// <summary>
    /// Initializes a new SparseVideoGen module.
    /// </summary>
    /// <param name="totalFrames">Total number of video frames.</param>
    /// <param name="keyframeInterval">Interval between keyframes (every Nth frame is a keyframe).</param>
    /// <param name="sparsityRatio">Target sparsity ratio for non-keyframes (0.0-0.9).</param>
    /// <param name="strategy">Strategy for selecting sparse computation.</param>
    public SparseVideoGen(
        int totalFrames = 16,
        int keyframeInterval = 4,
        double sparsityRatio = 0.5,
        SparsityStrategy strategy = SparsityStrategy.UniformSkip)
    {
        if (totalFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalFrames), "Total frames must be positive.");
        if (keyframeInterval <= 0 || keyframeInterval > totalFrames)
            throw new ArgumentOutOfRangeException(nameof(keyframeInterval), "Keyframe interval must be between 1 and total frames.");

        _totalFrames = totalFrames;
        _keyframeInterval = keyframeInterval;
        _sparsityRatio = Math.Max(0.0, Math.Min(0.9, sparsityRatio));
        _strategy = strategy;

        BuildKeyframeMask();
    }

    private void BuildKeyframeMask()
    {
        _keyframeMask = new bool[_totalFrames];
        // First and last frames are always keyframes
        _keyframeMask[0] = true;
        if (_totalFrames > 1)
            _keyframeMask[_totalFrames - 1] = true;

        // Mark intermediate keyframes at regular intervals
        for (int i = _keyframeInterval; i < _totalFrames; i += _keyframeInterval)
        {
            _keyframeMask[i] = true;
        }
    }

    /// <summary>
    /// Checks if a given frame is a keyframe requiring full computation.
    /// </summary>
    /// <param name="frameIndex">Index of the frame to check.</param>
    /// <returns>True if the frame is a keyframe.</returns>
    public bool IsKeyframe(int frameIndex)
    {
        if (frameIndex < 0 || frameIndex >= _totalFrames)
            throw new ArgumentOutOfRangeException(nameof(frameIndex));

        return _keyframeMask is not null && _keyframeMask[frameIndex];
    }

    /// <summary>
    /// Gets the set of transformer block indices to skip for a non-keyframe.
    /// </summary>
    /// <param name="totalBlocks">Total number of transformer blocks in the model.</param>
    /// <returns>Set of block indices to skip.</returns>
    public HashSet<int> GetBlocksToSkip(int totalBlocks)
    {
        var skipSet = new HashSet<int>();
        int blocksToSkip = (int)(totalBlocks * _sparsityRatio);

        switch (_strategy)
        {
            case SparsityStrategy.UniformSkip:
                // Skip every Nth block
                if (blocksToSkip > 0)
                {
                    int skipInterval = Math.Max(1, totalBlocks / blocksToSkip);
                    for (int i = 1; i < totalBlocks - 1; i += skipInterval)
                    {
                        if (skipSet.Count < blocksToSkip)
                            skipSet.Add(i);
                    }
                }
                break;

            case SparsityStrategy.MiddleSkip:
                // Skip blocks in the middle (first and last are most important)
                int start = totalBlocks / 4;
                int end = 3 * totalBlocks / 4;
                for (int i = start; i < end && skipSet.Count < blocksToSkip; i++)
                {
                    skipSet.Add(i);
                }
                break;

            case SparsityStrategy.AlternatingSkip:
                // Skip every other block
                for (int i = 1; i < totalBlocks - 1 && skipSet.Count < blocksToSkip; i += 2)
                {
                    skipSet.Add(i);
                }
                break;
        }

        return skipSet;
    }

    /// <summary>
    /// Gets the indices of all keyframes.
    /// </summary>
    /// <returns>List of keyframe indices.</returns>
    public List<int> GetKeyframeIndices()
    {
        var indices = new List<int>();
        if (_keyframeMask is not null)
        {
            for (int i = 0; i < _totalFrames; i++)
            {
                if (_keyframeMask[i])
                    indices.Add(i);
            }
        }
        return indices;
    }

    /// <summary>
    /// Gets the number of keyframes.
    /// </summary>
    public int KeyframeCount
    {
        get
        {
            if (_keyframeMask is null) return 0;
            int count = 0;
            foreach (bool isKey in _keyframeMask)
            {
                if (isKey) count++;
            }
            return count;
        }
    }
}

/// <summary>
/// Strategy for selecting which transformer blocks to skip in sparse computation.
/// </summary>
public enum SparsityStrategy
{
    /// <summary>Skip blocks at uniform intervals.</summary>
    UniformSkip,

    /// <summary>Skip blocks in the middle (preserve first and last).</summary>
    MiddleSkip,

    /// <summary>Skip every other block (alternating).</summary>
    AlternatingSkip
}
