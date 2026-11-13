using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
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

    /// <summary>e^k overflows fast, so the exponents are capped here; the receptance is clamped to +/- this.</summary>
    private static readonly T ExponentCap = Ops.FromDouble(30.0);

    private static readonly T NegativeExponentCap = Ops.FromDouble(-30.0);

    private readonly int _channels;
    private readonly double _decay;
    private readonly double _bonus;
    private readonly double _tokenShiftMix;

    /// <summary>e^(-w), mu, 1 - mu and u in <typeparamref name="T"/>, computed once.</summary>
    private readonly T _decayFactor;
    private readonly T _shiftCurrent;
    private readonly T _shiftPrevious;
    private readonly T _bonusValue;

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
    private readonly Tensor<T>? _keyProjection;
    private readonly Tensor<T>? _valueProjection;
    private readonly Tensor<T>? _receptanceProjection;
    private readonly Tensor<T>? _outputProjection;

    /// <summary>The running numerator a, one entry per channel.</summary>
    private Tensor<T> _stateA;

    /// <summary>The running denominator b, one entry per channel.</summary>
    private Tensor<T> _stateB;

    /// <summary>The previous frame, for the token shift.</summary>
    private Tensor<T> _previousFrame;

    private bool _hasPrevious;

    /// <summary>
    /// For a narrow T (Half, BFloat16) the recurrence runs in float here: e^k reaches e^30 and the decayed
    /// state sums grow past it, far beyond Half's 65504.
    /// </summary>
    private readonly RwkvTimeMixing<float>? _wide;

    private static bool IsNarrow =>
        typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16);

    /// <summary>Gets the channel count this recurrence is defined over.</summary>
    public int Channels => _channels;

    /// <summary>
    /// Gets the number of scalars retained between frames. Constant in the utterance length — the
    /// property the paper is buying.
    /// </summary>
    public int StateSize => _wide?.StateSize ?? _stateA.Length + _stateB.Length + _previousFrame.Length;

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
        _decayFactor = Ops.FromDouble(Math.Exp(-_decay));
        _shiftCurrent = Ops.FromDouble(_tokenShiftMix);
        _shiftPrevious = Ops.FromDouble(1.0 - _tokenShiftMix);
        _bonusValue = Ops.FromDouble(_bonus);
        _stateA = new Tensor<T>(new[] { channels });
        _stateB = new Tensor<T>(new[] { channels });
        _previousFrame = new Tensor<T>(new[] { channels });
        if (IsNarrow)
            _wide = new RwkvTimeMixing<float>(channels, timeDecay, currentTokenBonus, tokenShiftMix);
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
        if (IsNarrow)
        {
            _wide = new RwkvTimeMixing<float>(channels, timeDecay, currentTokenBonus, tokenShiftMix,
                _keyProjection?.Cast<float>(), _valueProjection?.Cast<float>(),
                _receptanceProjection?.Cast<float>(), _outputProjection?.Cast<float>());
        }
    }

    /// <summary>Validates a [channels, channels] projection and snapshots it once, at construction.</summary>
    /// <remarks>
    /// Copied up front, so later writes to the caller's tensor cannot change a recurrence that is
    /// already running.
    /// </remarks>
    private static Tensor<T>? ToMatrix(Tensor<T>? projection, int channels, string name)
    {
        if (projection is null) return null;

        if (projection.Rank != 2 || projection.Shape[0] != channels || projection.Shape[1] != channels)
        {
            throw new ArgumentException(
                $"{name} must be [{channels}, {channels}] to project this recurrence's frames; got " +
                $"[{string.Join(", ", projection.Shape.ToArray())}].",
                name);
        }

        return projection.Clone();
    }

    /// <summary>Applies a projection, or returns the input unchanged when it is the identity.</summary>
    private static Tensor<T> Project(IEngine engine, Tensor<T>? projection, Tensor<T> input, int channels)
    {
        if (projection is null) return input;

        // result[o] = sum_i projection[o, i] * input[i]
        return engine.TensorMatMul(projection, input.Reshape(channels, 1)).Reshape(channels);
    }

    /// <summary>min(x, cap) that, like Math.Min, keeps NaN as NaN.</summary>
    private static Tensor<T> CapAbove(IEngine engine, Tensor<T> x, T cap)
        => engine.TensorWhere(engine.TensorGreaterThan(x, cap), engine.TensorFullLike(x, cap), x);

    /// <summary>max(x, floor) that, like Math.Max, keeps NaN as NaN.</summary>
    private static Tensor<T> CapBelow(IEngine engine, Tensor<T> x, T floor)
        => engine.TensorWhere(engine.TensorLessThan(x, floor), engine.TensorFullLike(x, floor), x);

    /// <summary>Clears the recurrent state, starting a fresh utterance.</summary>
    public void Reset()
    {
        _stateA = new Tensor<T>(new[] { _channels });
        _stateB = new Tensor<T>(new[] { _channels });
        _previousFrame = new Tensor<T>(new[] { _channels });
        _hasPrevious = false;
        _wide?.Reset();
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
        // REJECTED, NOT PADDED. An earlier version read `c < frame.Length ? frame[c] : 0.0`, so a frame
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

        // A copy: the frame is kept as the next step's token-shift partner, and the caller may reuse it.
        if (_wide is not null)
            return Tensor<float>.FromVector(_wide.Step(Tensor<T>.FromVector(frame).Cast<float>().ToVector())).Cast<T>().ToVector();

        return StepCore(Tensor<T>.FromVector(frame).Clone()).ToVector();
    }

    /// <summary>Advances one validated <c>[channels]</c> frame, which the recurrence may retain.</summary>
    private Tensor<T> StepCore(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;

        // The recurrence reads values only; its state is not something a gradient flows through.
        using (new NoGradScope<T>())
        {
            // Token shift: interpolate with the previous frame. On the first frame there is no previous
            // one, so the shifted value is the frame itself rather than a mix with zero, which would
            // silently attenuate the utterance's opening.
            var shifted = _hasPrevious
                ? engine.TensorAdd(
                    engine.TensorMultiplyScalar(x, _shiftCurrent),
                    engine.TensorMultiplyScalar(_previousFrame, _shiftPrevious))
                : x;

            // k, v and r are SEPARATE PROJECTIONS of the shifted input, not three names for it. The value
            // is what the recurrence accumulates, the key is what weights it, and the receptance gates the
            // result -- collapsing them, as this did, leaves the time mixing with nothing to learn.
            var k = Project(engine, _keyProjection, shifted, _channels);
            var v = Project(engine, _valueProjection, shifted, _channels);
            var r = Project(engine, _receptanceProjection, shifted, _channels);

            var expK = engine.TensorExp(CapAbove(engine, k, ExponentCap));   // clamped: e^k overflows fast
            var expBonusK = engine.TensorExp(CapAbove(engine, engine.TensorAddScalar(k, _bonusValue), ExponentCap));

            // The output uses the state as it stood BEFORE this frame was folded in, with the current
            // frame entering through the bonus term -- that ordering is what makes the current frame
            // distinguishable from history rather than just another decayed entry.
            var numerator = engine.TensorAdd(_stateA, engine.TensorMultiply(expBonusK, v));
            var denominator = engine.TensorAdd(_stateB, expBonusK);
            var wkv = engine.TensorWhere(
                engine.TensorGreaterThan(denominator, Ops.Zero),
                engine.TensorDivide(numerator, denominator),
                engine.TensorZerosLike(denominator));

            _stateA = engine.TensorAdd(engine.TensorMultiplyScalar(_stateA, _decayFactor), engine.TensorMultiply(expK, v));
            _stateB = engine.TensorAdd(engine.TensorMultiplyScalar(_stateB, _decayFactor), expK);
            _previousFrame = x;

            // Receptance gate: sigmoid(r) = 1 / (1 + e^(-r)), r clamped to [-30, 30], decides how much of
            // the mixed history this frame actually emits.
            var clampedR = CapBelow(engine, CapAbove(engine, r, ExponentCap), NegativeExponentCap);
            var sigmoidR = engine.TensorReciprocal(
                engine.TensorAddScalar(engine.TensorExp(engine.TensorMultiplyScalar(clampedR, Ops.Negate(Ops.One))), Ops.One));
            var gated = engine.TensorMultiply(sigmoidR, wkv);

            var projected = Project(engine, _outputProjection, gated, _channels);

            _hasPrevious = true;
            return projected;
        }
    }

    /// <summary>
    /// Runs a whole sequence, resetting first. Provided so the batch path and the streaming path are
    /// literally the same recurrence.
    /// </summary>
    /// <remarks>
    /// Deliberately implemented by looping the per-frame step rather than by a separate parallel
    /// formulation. Two implementations of one recurrence drift, and the drift would land exactly on
    /// the streaming-equals-batch property this model's correctness rests on.
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> sequence)
    {
        Reset();
        int frames = sequence.Rank >= 2 ? sequence.Shape[0] : 1;
        int width = frames > 0 ? sequence.Length / frames : sequence.Length;

        // CHECKED HERE TOO, not only in Step. Step validates the frame it is handed, but Forward
        // builds those frames itself from `width`, so every frame arrives exactly _channels wide and
        // Step's guard can never fire on this path. A mis-shaped sequence would sail straight through
        // the batch route while the streaming route rejected it, which is the worse of the two
        // outcomes: the two paths are supposed to compute the same function, and this made them
        // disagree on what is even valid.
        if (width != _channels)
        {
            throw new ArgumentException(
                $"The recurrence is defined over {_channels} channels but the sequence has {width} per " +
                $"frame ({frames} frames over {sequence.Length} values).",
                nameof(sequence));
        }

        // width == _channels > 0 makes frames positive, and frames * width is the element count: the
        // frame count is the leading dimension and the width the product of the rest.
        if (_wide is not null)
            return _wide.Forward(sequence.Cast<float>()).Cast<T>();

        var engine = AiDotNetEngine.Current;
        var rows = sequence.Reshape(frames, width);
        var outputs = new Tensor<T>[frames];
        for (int t = 0; t < frames; t++)
        {
            outputs[t] = StepCore(engine.TensorSliceAxis(rows, 0, t).Clone());
        }

        return engine.TensorConcatenate(outputs, 0).Reshape(sequence._shape);
    }
}
