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

    /// <summary>Receptance, key, value and output projections; null means identity.</summary>
    /// <remarks>
    /// WITHOUT THESE THIS WAS NOT RWKV. The recurrence used the token-shifted input directly as key,
    /// as value AND as the numerator term, and applied no receptance gate at all -- so there was
    /// nothing to learn in the time mixing and no way for it to weight one channel against another.
    /// Left null they are the identity, which is the projection-free reference form the invariant
    /// tests exercise; supplied, they are the trained projections of an
    /// <c>AiDotNet.NeuralNetworks.Layers.SSM.RWKVLayer&lt;T&gt;</c>, so this reproduces that layer's
    /// time mixing one frame at a time.
    /// </remarks>
    private readonly double[,]? _keyProjection;
    private readonly double[,]? _valueProjection;
    private readonly double[,]? _receptanceProjection;
    private readonly double[,]? _outputProjection;

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

    /// <summary>
    /// Initializes the recurrence with trained projections, so it reproduces an
    /// <c>RWKVLayer&lt;T&gt;</c>'s time mixing frame by frame.
    /// </summary>
    /// <param name="channels">Feature width.</param>
    /// <param name="timeDecay">w, the per-step decay exponent.</param>
    /// <param name="currentTokenBonus">u, extra weight on the current frame.</param>
    /// <param name="tokenShiftMix">mu, how much of the current frame the shifted input uses.</param>
    /// <param name="keyProjection">W_k, shape [channels, channels]; null for identity.</param>
    /// <param name="valueProjection">W_v, shape [channels, channels]; null for identity.</param>
    /// <param name="receptanceProjection">W_r, shape [channels, channels]; null for identity.</param>
    /// <param name="outputProjection">W_o, shape [channels, channels]; null for identity.</param>
    /// <exception cref="ArgumentException">A projection is not square at <paramref name="channels"/>.</exception>
    public RwkvTimeMixing(
        int channels,
        double timeDecay,
        double currentTokenBonus,
        double tokenShiftMix,
        Tensor<T>? keyProjection,
        Tensor<T>? valueProjection,
        Tensor<T>? receptanceProjection,
        Tensor<T>? outputProjection)
        : this(channels, timeDecay, currentTokenBonus, tokenShiftMix)
    {
        _keyProjection = ToMatrix(keyProjection, channels, nameof(keyProjection));
        _valueProjection = ToMatrix(valueProjection, channels, nameof(valueProjection));
        _receptanceProjection = ToMatrix(receptanceProjection, channels, nameof(receptanceProjection));
        _outputProjection = ToMatrix(outputProjection, channels, nameof(outputProjection));
    }

    /// <summary>Converts a [channels, channels] projection to doubles once, at construction.</summary>
    /// <remarks>
    /// Converted up front rather than per frame: the recurrence runs this matrix on every timestep of
    /// every utterance, and re-reading <typeparamref name="T"/> through <c>ToDouble</c> inside that
    /// loop would dominate it.
    /// </remarks>
    private static double[,]? ToMatrix(Tensor<T>? projection, int channels, string name)
    {
        if (projection is null) return null;

        if (projection.Rank != 2 || projection.Shape[0] != channels || projection.Shape[1] != channels)
        {
            throw new ArgumentException(
                $"{name} must be [{channels}, {channels}] to project this recurrence's frames; got " +
                $"[{string.Join(", ", projection.Shape.ToArray())}].",
                name);
        }

        var m = new double[channels, channels];
        for (int i = 0; i < channels; i++)
        {
            for (int j = 0; j < channels; j++) m[i, j] = Ops.ToDouble(projection[i, j]);
        }

        return m;
    }

    /// <summary>Applies a projection, or returns the input unchanged when it is the identity.</summary>
    private static double[] Project(double[,]? projection, double[] input, int channels)
    {
        if (projection is null) return input;

        var result = new double[channels];
        for (int o = 0; o < channels; o++)
        {
            double sum = 0.0;
            for (int i = 0; i < channels; i++) sum += projection[o, i] * input[i];
            result[o] = sum;
        }

        return result;
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

        double decayFactor = Math.Exp(-_decay);

        var x = new double[_channels];
        for (int c = 0; c < _channels; c++) x[c] = Ops.ToDouble(frame[c]);

        // Token shift: interpolate with the previous frame. On the first frame there is no previous
        // one, so the shifted value is the frame itself rather than a mix with zero, which would
        // silently attenuate the utterance's opening.
        var shifted = new double[_channels];
        for (int c = 0; c < _channels; c++)
        {
            shifted[c] = _hasPrevious
                ? _tokenShiftMix * x[c] + (1.0 - _tokenShiftMix) * _previousFrame[c]
                : x[c];
        }

        // k, v and r are SEPARATE PROJECTIONS of the shifted input, not three names for it. The value
        // is what the recurrence accumulates, the key is what weights it, and the receptance gates the
        // result -- collapsing them, as this did, leaves the time mixing with nothing to learn.
        var k = Project(_keyProjection, shifted, _channels);
        var v = Project(_valueProjection, shifted, _channels);
        var r = Project(_receptanceProjection, shifted, _channels);

        var wkv = new double[_channels];
        for (int c = 0; c < _channels; c++)
        {
            double expK = Math.Exp(Math.Min(k[c], 30.0));                // clamped: e^k overflows fast
            double expBonusK = Math.Exp(Math.Min(k[c] + _bonus, 30.0));

            // The output uses the state as it stood BEFORE this frame was folded in, with the current
            // frame entering through the bonus term -- that ordering is what makes the current frame
            // distinguishable from history rather than just another decayed entry.
            double numerator = _stateA[c] + expBonusK * v[c];
            double denominator = _stateB[c] + expBonusK;
            wkv[c] = denominator > 0 ? numerator / denominator : 0.0;

            _stateA[c] = decayFactor * _stateA[c] + expK * v[c];
            _stateB[c] = decayFactor * _stateB[c] + expK;
            _previousFrame[c] = x[c];
        }

        // Receptance gate: sigmoid(r) decides how much of the mixed history this frame actually emits.
        var gated = new double[_channels];
        for (int c = 0; c < _channels; c++)
        {
            double sigmoidR = 1.0 / (1.0 + Math.Exp(-Math.Max(-30.0, Math.Min(30.0, r[c]))));
            gated[c] = sigmoidR * wkv[c];
        }

        var projected = Project(_outputProjection, gated, _channels);

        var output = new Vector<T>(_channels);
        for (int c = 0; c < _channels; c++) output[c] = Ops.FromDouble(projected[c]);

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
