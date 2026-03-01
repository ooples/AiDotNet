using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Transforms;

/// <summary>
/// Applies temporal jitter to video frame sequences by randomly shifting frame indices.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Temporal jitter is a data augmentation technique for video models that randomly
/// perturbs the starting position or sampling stride of frames. This forces the model
/// to be robust to temporal alignment variations and prevents overfitting to specific
/// frame positions within video clips.
/// </para>
/// </remarks>
public class TemporalJitterAugmentation<T> : ITransform<Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _maxJitter;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new temporal jitter augmentation.
    /// </summary>
    /// <param name="maxJitter">Maximum number of frames to shift. Default is 4.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public TemporalJitterAugmentation(int maxJitter = 4, int? seed = null)
    {
        _maxJitter = maxJitter;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Applies temporal jitter by circularly shifting frame data.
    /// </summary>
    /// <param name="input">Video tensor of shape [num_frames, frame_features].</param>
    /// <returns>Jittered video tensor with same shape.</returns>
    public Tensor<T> Apply(Tensor<T> input)
    {
        if (input.Shape.Length < 2)
            return input;

        int numFrames = input.Shape[0];
        int featuresPerFrame = input.Shape[1];

        if (numFrames <= 1)
            return input;

        // Random jitter amount in range [-maxJitter, maxJitter]
        int jitter = _random.Next(-_maxJitter, _maxJitter + 1);

        var result = input.Clone();
        var resultSpan = result.Data.Span;

        for (int f = 0; f < numFrames; f++)
        {
            // Circular shift: source frame index wraps around
            int srcFrame = ((f + jitter) % numFrames + numFrames) % numFrames;
            int srcOffset = srcFrame * featuresPerFrame;
            int dstOffset = f * featuresPerFrame;

            for (int j = 0; j < featuresPerFrame; j++)
            {
                resultSpan[dstOffset + j] = input.Data.Span[srcOffset + j];
            }
        }

        return result;
    }
}
