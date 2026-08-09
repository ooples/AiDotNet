using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Video.Prediction;

/// <summary>
/// MCnet's motion/content decomposition: the asymmetric split that gives the model its name.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Villegas et al., arXiv:1706.08033.
/// </para>
/// <code>
///   motion    [d_t, c_t] = f_dyn(x_t - x_{t-1}, d_{t-1}, c_{t-1})     convolutional LSTM
///   content   s_t        = f_cont(x_t)                                plain CNN
///   combine   f_t        = g_comb([d_t, s_t])
///   residual  r_t^l      = f_res([s_t^l, d_t^l])^l                    at EVERY scale
///   decode    x_hat_t+1  = g_dec(f_t, r_t)                            tanh output
/// </code>
/// <para>
/// <b>The asymmetry IS the decomposition, and it is easy to flatten.</b> The motion pathway consumes
/// IMAGE DIFFERENCES <c>x_t - x_{t-1}</c> recurrently; the content pathway consumes a SINGLE frame,
/// <c>x_t</c>, with no recurrence. Feeding both pathways the same frames — or feeding the motion
/// pathway raw frames rather than differences — produces two encoders that learn overlapping
/// representations, which is precisely the entanglement the paper removes.
/// </para>
/// <para>
/// This type owns the differencing, the combination and the per-scale residual pairing. The learned
/// parts (the convolutional LSTM, the CNNs) live in the owning model's layer collection so their
/// weights are real parameters visible to the optimizer and the gradient tape.
/// </para>
/// <para><b>For Beginners:</b> Two things matter when predicting the next video frame: what the scene
/// looks like, and how it is moving. This separates them — movement is read from the CHANGES between
/// consecutive frames, appearance from a single frame — so the model does not have to relearn what a
/// chair looks like every time the camera pans.</para>
/// </remarks>
public sealed class McnetDecomposition<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly int _scales;

    /// <summary>Gets the number of scales at which residual features are communicated.</summary>
    public int Scales => _scales;

    /// <summary>
    /// Creates the decomposition helper.
    /// </summary>
    /// <param name="scales">
    /// Number of scales for the residual connections <c>r_t^l</c>. Must be at least one.
    /// </param>
    public McnetDecomposition(int scales = 3)
    {
        if (scales <= 0)
            throw new ArgumentOutOfRangeException(nameof(scales), scales,
                "At least one scale is required; residual features are communicated at every scale.");
        _scales = scales;
    }

    /// <summary>
    /// The motion pathway's input: the difference sequence <c>x_t - x_{t-1}</c>.
    /// </summary>
    /// <param name="frames">
    /// Observed frames, <c>[time, height, width, channels]</c> — at least two.
    /// </param>
    /// <returns>
    /// <c>[time - 1, height, width, channels]</c>. One fewer than the input, because a difference needs
    /// a predecessor.
    /// </returns>
    /// <remarks>
    /// Differences, NOT raw frames. This is what makes the pathway encode dynamics rather than
    /// appearance: a static scene differences to zero regardless of what it contains, so there is
    /// nothing about content for the motion encoder to latch onto.
    /// </remarks>
    public Tensor<T> MotionInput(Tensor<T> frames)
    {
        if (frames is null) throw new ArgumentNullException(nameof(frames));
        if (frames.Shape.Length != 4)
            throw new ArgumentException(
                $"Expected [time, height, width, channels]; got rank {frames.Shape.Length}.", nameof(frames));

        int time = frames.Shape[0];
        if (time < 2)
            throw new ArgumentException(
                $"At least two frames are required to form a difference; got {time}.", nameof(frames));

        int h = frames.Shape[1], w = frames.Shape[2], c = frames.Shape[3];
        int perFrame = h * w * c;
        var result = new Tensor<T>(new[] { time - 1, h, w, c });

        for (int t = 1; t < time; t++)
        {
            int dst = (t - 1) * perFrame;
            int cur = t * perFrame;
            int prev = (t - 1) * perFrame;
            for (int i = 0; i < perFrame; i++)
            {
                result[dst + i] = Ops.Subtract(frames[cur + i], frames[prev + i]);
            }
        }
        return result;
    }

    /// <summary>
    /// The content pathway's input: the LAST observed frame only.
    /// </summary>
    /// <param name="frames">Observed frames, <c>[time, height, width, channels]</c>.</param>
    /// <returns><c>[height, width, channels]</c>.</returns>
    /// <remarks>
    /// One frame, not the sequence. <c>s_t = f_cont(x_t)</c> has no recurrence and no history: the
    /// spatial layout the decoder needs is whatever is on screen now.
    /// </remarks>
    public Tensor<T> ContentInput(Tensor<T> frames)
    {
        if (frames is null) throw new ArgumentNullException(nameof(frames));
        if (frames.Shape.Length != 4)
            throw new ArgumentException(
                $"Expected [time, height, width, channels]; got rank {frames.Shape.Length}.", nameof(frames));

        int time = frames.Shape[0], h = frames.Shape[1], w = frames.Shape[2], c = frames.Shape[3];
        int perFrame = h * w * c;
        var result = new Tensor<T>(new[] { h, w, c });

        int offset = (time - 1) * perFrame;
        for (int i = 0; i < perFrame; i++) result[i] = frames[offset + i];
        return result;
    }

    /// <summary>
    /// <c>[d_t, s_t]</c>: concatenates motion and content features along the channel axis, ready for
    /// <c>g_comb</c>.
    /// </summary>
    /// <param name="motion">Motion features, <c>[height, width, motionChannels]</c>.</param>
    /// <param name="content">Content features, <c>[height, width, contentChannels]</c>.</param>
    /// <remarks>
    /// Channel-wise concatenation, so the combination layers can weigh the two streams against each
    /// other per spatial position. Summing them instead would force the two representations into one
    /// space and discard the separation the encoders just established.
    /// </remarks>
    public Tensor<T> Combine(Tensor<T> motion, Tensor<T> content)
    {
        if (motion is null) throw new ArgumentNullException(nameof(motion));
        if (content is null) throw new ArgumentNullException(nameof(content));
        if (motion.Shape.Length != 3 || content.Shape.Length != 3)
            throw new ArgumentException("Both feature maps must be [height, width, channels].", nameof(motion));
        if (motion.Shape[0] != content.Shape[0] || motion.Shape[1] != content.Shape[1])
            throw new ArgumentException(
                $"Spatial dimensions must match: motion is [{motion.Shape[0]}, {motion.Shape[1]}] but " +
                $"content is [{content.Shape[0]}, {content.Shape[1]}].", nameof(content));

        int h = motion.Shape[0], w = motion.Shape[1];
        int mc = motion.Shape[2], cc = content.Shape[2];
        var result = new Tensor<T>(new[] { h, w, mc + cc });

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int dst = ((y * w) + x) * (mc + cc);
                int ms = ((y * w) + x) * mc;
                int cs = ((y * w) + x) * cc;
                for (int k = 0; k < mc; k++) result[dst + k] = motion[ms + k];
                for (int k = 0; k < cc; k++) result[dst + mc + k] = content[cs + k];
            }
        }
        return result;
    }

    /// <summary>
    /// Pairs the two pathways' features at a given scale for <c>r_t^l = f_res([s_t^l, d_t^l])^l</c>.
    /// </summary>
    /// <param name="contentAtScale">Content features at scale l.</param>
    /// <param name="motionAtScale">Motion features at scale l.</param>
    /// <param name="scale">The scale index, for validation.</param>
    /// <remarks>
    /// The residual pairing is per-scale and BOTH streams participate. A skip connection carrying only
    /// content — the usual U-Net arrangement — would hand the decoder appearance detail with no
    /// indication of how it is moving, which is the information the residual path exists to supply.
    /// </remarks>
    public Tensor<T> ResidualPair(Tensor<T> contentAtScale, Tensor<T> motionAtScale, int scale)
    {
        if (scale < 0 || scale >= _scales)
            throw new ArgumentOutOfRangeException(nameof(scale), scale,
                $"Scale must be in [0, {_scales - 1}].");

        // Same channel-wise pairing as the bottleneck combination, applied at this scale.
        return Combine(motionAtScale, contentAtScale);
    }
}
