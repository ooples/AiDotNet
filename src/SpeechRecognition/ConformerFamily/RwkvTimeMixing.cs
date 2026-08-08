using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>
/// RWKV time-mixing: the linear-attention recurrence that replaces self-attention, and the reason an
/// RWKV encoder streams without chunking.
/// </summary>
/// <remarks>
/// <para>
/// Used by <see cref="RWKVTransducer{T}"/> for An and Zhang, "Exploring RWKV for Memory Efficient
/// and Low Latency Streaming ASR" (arXiv:2309.14758). Their motivation: "the full-sequence attention
/// mechanism is non-streamable and computationally expensive, thus requiring modifications, such as
/// chunking and caching, for efficient streaming ASR." RWKV needs neither, because it is genuinely
/// recurrent.
/// </para>
/// <code>
///   token shift:  xk_t = mu_k * x_t + (1 - mu_k) * x_(t-1)      (likewise for v and r)
///   state:        a_t = e^(-w) * a_(t-1) + e^(k_t) * v_t
///                 b_t = e^(-w) * b_(t-1) + e^(k_t)
///   output:       wkv_t = (a_(t-1) + e^(u + k_t) * v_t) / (b_(t-1) + e^(u + k_t))
/// </code>
/// <para>
/// <b>Three properties make this streamable, and all three are testable.</b>
/// </para>
/// <list type="number">
/// <item><description><b>The state is CONSTANT size.</b> Two accumulators per channel, whatever the
/// utterance length. A transformer's KV cache grows with every frame, which is the memory cost the
/// paper is removing.</description></item>
/// <item><description><b>Streaming equals batch, exactly.</b> Feeding frames one at a time must
/// produce bit-identical output to processing the whole sequence. That is not an optimization — it
/// is what lets a streaming decoder be correct, and it is the property chunked attention only
/// approximates.</description></item>
/// <item><description><b>The past decays geometrically</b> through <c>e^(-w)</c>, so influence falls
/// off smoothly rather than being truncated at a chunk boundary.</description></item>
/// </list>
/// <para>
/// <b>For Beginners:</b> Ordinary attention re-reads the entire utterance for every new frame, so
/// it cannot start until the speaker stops and it needs more memory the longer they talk. This keeps
/// a small running summary instead, updated once per frame, and older audio fades from it gradually.
/// The answer you get frame-by-frame is identical to the one you would get from the whole recording.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class RwkvTimeMixing<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly int _channels;
    private readonly double _decay;
    private readonly double _bonus;
    private readonly double _tokenShiftMix;

    /// <summary>The running numerator a, one entry per channel.</summary>
    private readonly double[] _stateA;

    /// <summary>The running denominator b, one entry per channel.</summary>
    private readonly double[] _stateB;

    /// <summary>The previous frame, for the token shift.</summary>
    private readonly double[] _previousFrame;

    private bool _hasPrevious;

    /// <summary>Gets the channel count this recurrence is defined over.</summary>
    public int Channels => _channels;

    /// <summary>
    /// Gets the number of scalars retained between frames. Constant in the utterance length — the
    /// property the paper is buying.
    /// </summary>
    public int StateSize => _stateA.Length + _stateB.Length + _previousFrame.Length;

    /// <summary>
    /// Initializes the recurrence.
    /// </summary>
    /// <param name="channels">Feature width.</param>
    /// <param name="timeDecay">w, the per-step decay exponent. Larger forgets faster.</param>
    /// <param name="currentTokenBonus">u, extra weight on the CURRENT frame relative to history.</param>
    /// <param name="tokenShiftMix">
    /// mu, how much of the current frame the shifted input uses. 1.0 disables the shift entirely.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when channels is not positive, the decay
    /// is negative, or the mix is outside [0, 1].</exception>
    public RwkvTimeMixing(int channels, double timeDecay, double currentTokenBonus, double tokenShiftMix)
    {
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "Channel count must be positive.");
        if (timeDecay < 0.0)
            throw new ArgumentOutOfRangeException(nameof(timeDecay), timeDecay,
                "Time decay must be non-negative: e^(-w) with w < 0 would AMPLIFY the past each step " +
                "and the recurrence would diverge over a long utterance.");
        if (tokenShiftMix < 0.0 || tokenShiftMix > 1.0)
            throw new ArgumentOutOfRangeException(nameof(tokenShiftMix), tokenShiftMix,
                "Token shift mix must lie in [0, 1]; it interpolates between the previous and current frame.");

        _channels = channels;
        _decay = timeDecay;
        _bonus = currentTokenBonus;
        _tokenShiftMix = tokenShiftMix;
        _stateA = new double[channels];
        _stateB = new double[channels];
        _previousFrame = new double[channels];
    }

    /// <summary>Clears the recurrent state, starting a fresh utterance.</summary>
    public void Reset()
    {
        Array.Clear(_stateA, 0, _stateA.Length);
        Array.Clear(_stateB, 0, _stateB.Length);
        Array.Clear(_previousFrame, 0, _previousFrame.Length);
        _hasPrevious = false;
    }

    /// <summary>
    /// Advances one frame and returns its output. This is the streaming entry point.
    /// </summary>
    /// <remarks>
    /// The output uses the state as it stood BEFORE this frame was folded in, with the current
    /// frame's contribution added through the bonus term — that ordering is what makes the current
    /// frame distinguishable from history rather than just another decayed entry.
    /// </remarks>
    public Vector<T> Step(Vector<T> frame)
    {
        // REJECTED, NOT PADDED. The loop below reads `c < frame.Length ? frame[c] : 0.0`, so a frame
        // narrower than the recurrence was zero-filled and one wider was truncated -- either way the
        // model kept running and returned a transcription computed from a feature vector that is not
        // the one the caller passed. A width mismatch is a wiring error upstream, and it is far cheaper
        // to see it here than to debug the recognition output it silently corrupts.
        if (frame.Length != _channels)
        {
            throw new ArgumentException(
                $"The recurrence is defined over {_channels} channels but received a frame of " +
                $"{frame.Length}. Padding or truncating it would silently change the model's input.",
                nameof(frame));
        }

        var output = new Vector<T>(_channels);
        double decayFactor = Math.Exp(-_decay);

        for (int c = 0; c < _channels; c++)
        {
            double x = c < frame.Length ? Ops.ToDouble(frame[c]) : 0.0;

            // Token shift: interpolate with the previous frame. On the first frame there is no
            // previous one, so the shifted value is the frame itself rather than a mix with zero,
            // which would silently attenuate the utterance's opening.
            double shifted = _hasPrevious
                ? _tokenShiftMix * x + (1.0 - _tokenShiftMix) * _previousFrame[c]
                : x;

            double k = Math.Exp(Math.Min(shifted, 30.0));            // clamped: e^k overflows fast
            double bonusK = Math.Exp(Math.Min(shifted + _bonus, 30.0));

            double numerator = _stateA[c] + bonusK * shifted;
            double denominator = _stateB[c] + bonusK;
            output[c] = Ops.FromDouble(denominator > 0 ? numerator / denominator : 0.0);

            _stateA[c] = decayFactor * _stateA[c] + k * shifted;
            _stateB[c] = decayFactor * _stateB[c] + k;
            _previousFrame[c] = x;
        }

        _hasPrevious = true;
        return output;
    }

    /// <summary>
    /// Runs a whole sequence, resetting first. Provided so the batch path and the streaming path are
    /// literally the same recurrence.
    /// </summary>
    /// <remarks>
    /// Deliberately implemented by looping <see cref="Step"/> rather than by a separate parallel
    /// formulation. Two implementations of one recurrence drift, and the drift would land exactly on
    /// the streaming-equals-batch property this model's correctness rests on.
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> sequence)
    {
        Reset();
        int frames = sequence.Rank >= 2 ? sequence.Shape[0] : 1;
        int width = frames > 0 ? sequence.Length / frames : sequence.Length;

        // CHECKED HERE TOO, not only in Step. Step validates the frame it is handed, but Forward
        // builds those frames itself from `width` and pads short rows with Ops.Zero before Step ever
        // sees them -- so every frame arrives exactly _channels wide and Step's guard can never fire
        // on this path. A mis-shaped sequence would sail straight through the batch route while the
        // streaming route rejected it, which is the worse of the two outcomes: the two paths are
        // supposed to compute the same function, and this made them disagree on what is even valid.
        if (width != _channels)
        {
            throw new ArgumentException(
                $"The recurrence is defined over {_channels} channels but the sequence has {width} per " +
                $"frame ({frames} frames over {sequence.Length} values).",
                nameof(sequence));
        }

        var result = new Tensor<T>(sequence._shape);
        var frame = new Vector<T>(width);

        for (int t = 0; t < frames; t++)
        {
            for (int c = 0; c < width; c++)
            {
                int flat = t * width + c;
                frame[c] = flat < sequence.Length ? sequence[flat] : Ops.Zero;
            }

            var stepped = Step(frame);
            for (int c = 0; c < width && c < stepped.Length; c++)
            {
                int flat = t * width + c;
                if (flat < result.Length) result[flat] = stepped[c];
            }
        }
        return result;
    }
}
