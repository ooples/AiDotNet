using System;
using AiDotNet.Helpers;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// RT-2's action-as-text tokenizer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements the RT-2 paper's action representation: each continuous action dimension is
/// uniformly discretized into <c>NumBins</c> bins (default 256) over its range, and each
/// bin is associated with one of the <c>NumBins</c> least frequently used tokens in the
/// vision-language model's vocabulary. The model emits these tokens autoregressively just
/// as it would emit ordinary text tokens; downstream decoding maps each token back to a
/// continuous action value in the centre of its bin.
/// </para>
/// <para><b>References:</b></para>
/// <list type="bullet">
///   <item>Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", 2023, §3.2 (Robot-Action Fine-tuning), arXiv:2307.15818.</item>
///   <item>Padalkar et al., "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", 2023, action tokenization scheme (Eq. 1) reused.</item>
/// </list>
/// </remarks>
public sealed class RT2ActionTokenizer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double[] _minRange;
    private readonly double[] _maxRange;

    /// <summary>Number of discrete bins per action dimension (RT-2 paper: 256).</summary>
    public int NumBins { get; }

    /// <summary>Number of continuous action dimensions emitted per timestep.</summary>
    public int ActionDim { get; }

    /// <summary>Vocabulary token-ID where the first action bin lives. Bins occupy <c>[TokenIdOffset, TokenIdEndExclusive)</c>.</summary>
    public int TokenIdOffset { get; }

    /// <summary>One past the last action-bin token ID.</summary>
    public int TokenIdEndExclusive => TokenIdOffset + NumBins;

    /// <summary>Total vocabulary size — equals
    /// <see cref="TokenIdEndExclusive"/> because the action bins live
    /// in the LAST <see cref="NumBins"/> token IDs of the vocab per
    /// paper §3.2. Exposed so callers (e.g. <c>GreedyActionToken</c>)
    /// can validate they're passing single-position logit vectors of
    /// the right size.</summary>
    public int VocabSize => TokenIdEndExclusive;

    /// <summary>
    /// Initialises the tokenizer.
    /// </summary>
    /// <param name="actionDim">Number of continuous action dimensions per timestep. RT-2 uses 8 (translation xyz, rotation rpy, gripper open/close, episode-terminate).</param>
    /// <param name="numBins">Discretisation resolution per dimension. Paper value 256.</param>
    /// <param name="vocabSize">VLM vocabulary size; bin tokens are mapped to the last <paramref name="numBins"/> IDs.</param>
    /// <param name="minRange">Per-dimension lower bound; defaults to -1.0 per dim. If a single-element array is supplied it is broadcast.</param>
    /// <param name="maxRange">Per-dimension upper bound; defaults to +1.0 per dim. If a single-element array is supplied it is broadcast.</param>
    public RT2ActionTokenizer(
        int actionDim = 8,
        int numBins = 256,
        int vocabSize = 32000,
        double[]? minRange = null,
        double[]? maxRange = null)
    {
        if (actionDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(actionDim), actionDim, "actionDim must be positive.");
        if (numBins <= 1)
            throw new ArgumentOutOfRangeException(nameof(numBins), numBins, "numBins must be greater than 1.");
        if (vocabSize <= numBins)
            throw new ArgumentOutOfRangeException(nameof(vocabSize), vocabSize, $"vocabSize ({vocabSize}) must exceed numBins ({numBins}) so action bins can be assigned to the least-used vocabulary tokens.");

        ActionDim = actionDim;
        NumBins = numBins;
        TokenIdOffset = vocabSize - numBins;

        _minRange = NormaliseRange(minRange, actionDim, defaultValue: -1.0, nameof(minRange));
        _maxRange = NormaliseRange(maxRange, actionDim, defaultValue: 1.0, nameof(maxRange));

        for (int d = 0; d < actionDim; d++)
        {
            if (!(_maxRange[d] > _minRange[d]))
                throw new ArgumentException($"maxRange[{d}] ({_maxRange[d]}) must be strictly greater than minRange[{d}] ({_minRange[d]}).");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Encodes a continuous action vector into <see cref="ActionDim"/> vocabulary token IDs.
    /// </summary>
    /// <param name="continuousAction">Tensor of length <see cref="ActionDim"/>. Values outside <c>[min, max]</c> are clamped.</param>
    /// <returns>Array of <see cref="ActionDim"/> token IDs, each in <c>[TokenIdOffset, TokenIdEndExclusive)</c>.</returns>
    public int[] EncodeAction(Tensor<T> continuousAction)
    {
        if (continuousAction is null) throw new ArgumentNullException(nameof(continuousAction));
        // Exact-length check: silently truncating extra dimensions
        // makes a misaligned caller (wrong ActionDim, accidentally
        // flattened multi-step tensor, etc.) look like a valid
        // single-step encode and hides the shape bug. ActionDim is
        // the documented contract; reject anything else.
        if (continuousAction.Length != ActionDim)
            throw new ArgumentException($"continuousAction has length {continuousAction.Length} but tokenizer expects exactly {ActionDim}.", nameof(continuousAction));

        var tokens = new int[ActionDim];
        for (int d = 0; d < ActionDim; d++)
        {
            double value = _numOps.ToDouble(continuousAction[d]);
            tokens[d] = TokenIdOffset + ValueToBin(value, d);
        }
        return tokens;
    }

    /// <summary>
    /// Encodes a per-dimension continuous action vector into vocabulary token IDs.
    /// </summary>
    public int[] EncodeAction(Vector<T> continuousAction)
    {
        if (continuousAction is null) throw new ArgumentNullException(nameof(continuousAction));
        if (continuousAction.Length != ActionDim)
            throw new ArgumentException($"continuousAction has length {continuousAction.Length} but tokenizer expects exactly {ActionDim}.", nameof(continuousAction));

        var tokens = new int[ActionDim];
        for (int d = 0; d < ActionDim; d++)
        {
            double value = _numOps.ToDouble(continuousAction[d]);
            tokens[d] = TokenIdOffset + ValueToBin(value, d);
        }
        return tokens;
    }

    /// <summary>
    /// Encodes a multi-step horizon action (shape <c>[horizon, ActionDim]</c> or <c>[horizon * ActionDim]</c>)
    /// into a flat token stream of length <c>horizon * ActionDim</c>.
    /// </summary>
    public int[] EncodeHorizon(Tensor<T> horizonAction, int horizon)
    {
        if (horizonAction is null) throw new ArgumentNullException(nameof(horizonAction));
        if (horizon <= 0) throw new ArgumentOutOfRangeException(nameof(horizon), horizon, "horizon must be positive.");
        int expected = horizon * ActionDim;
        if (horizonAction.Length != expected)
            throw new ArgumentException($"horizonAction has length {horizonAction.Length} but tokenizer expects exactly {expected} (horizon {horizon} × ActionDim {ActionDim}).", nameof(horizonAction));

        var tokens = new int[expected];
        for (int t = 0; t < horizon; t++)
        {
            for (int d = 0; d < ActionDim; d++)
            {
                double value = _numOps.ToDouble(horizonAction[t * ActionDim + d]);
                tokens[t * ActionDim + d] = TokenIdOffset + ValueToBin(value, d);
            }
        }
        return tokens;
    }

    /// <summary>
    /// Decodes vocabulary token IDs (one per action dimension) into a continuous action tensor of length <see cref="ActionDim"/>.
    /// Tokens outside the action-bin range decode to the per-dimension midpoint (graceful fallback for malformed generations).
    /// </summary>
    public Tensor<T> DecodeAction(int[] tokenIds)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));
        if (tokenIds.Length != ActionDim)
            throw new ArgumentException($"tokenIds has length {tokenIds.Length} but tokenizer expects exactly {ActionDim}.", nameof(tokenIds));

        var action = new Tensor<T>([ActionDim]);
        for (int d = 0; d < ActionDim; d++)
        {
            double value = TokenToValue(tokenIds[d], d);
            action[d] = _numOps.FromDouble(value);
        }
        return action;
    }

    /// <summary>
    /// Decodes a flat token stream of length <c>horizon * ActionDim</c> into a horizon-by-dim action tensor.
    /// </summary>
    public Tensor<T> DecodeHorizon(int[] tokenIds, int horizon)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));
        if (horizon <= 0) throw new ArgumentOutOfRangeException(nameof(horizon), horizon, "horizon must be positive.");
        int expected = horizon * ActionDim;
        if (tokenIds.Length != expected)
            throw new ArgumentException($"tokenIds has length {tokenIds.Length} but tokenizer expects exactly {expected} (horizon {horizon} × ActionDim {ActionDim}).", nameof(tokenIds));

        var action = new Tensor<T>([horizon, ActionDim]);
        for (int t = 0; t < horizon; t++)
        {
            for (int d = 0; d < ActionDim; d++)
            {
                double value = TokenToValue(tokenIds[t * ActionDim + d], d);
                action[t * ActionDim + d] = _numOps.FromDouble(value);
            }
        }
        return action;
    }

    /// <summary>Returns true when the supplied token ID falls in this tokenizer's action-bin range.</summary>
    public bool IsActionToken(int tokenId) => tokenId >= TokenIdOffset && tokenId < TokenIdEndExclusive;

    /// <summary>
    /// Given a position in the emitted action-token stream, returns which continuous action dimension it controls.
    /// Use during training to select per-dimension bin ranges if asymmetric clamping is desired.
    /// </summary>
    public int ActionDimOfPosition(int positionInStream) => positionInStream % ActionDim;

    /// <summary>
    /// Selects the action-bin token with the highest logit (greedy
    /// argmax over the tokenizer's vocabulary slice). Expects
    /// <paramref name="logits"/> to be EXACTLY the vocab logit vector
    /// for a single decode position (length == VocabSize).
    /// </summary>
    /// <param name="logits">Length-<c>VocabSize</c> logit vector for one decode position.</param>
    public int GreedyActionToken(Tensor<T> logits)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        // Exact-length match against VocabSize: previously the check
        // was "logits.Length >= TokenIdEndExclusive", which silently
        // accepted a flattened multi-position decoder output
        // (vocab × seqLen) as if it were a single-position logit
        // vector — argmax would then run over the FIRST position's
        // action-window slice and discard everything else without
        // any indication of the misuse. Demand the exact shape so
        // callers stuck on flatten-vs-position errors get a clear
        // diagnostic.
        if (logits.Length != VocabSize)
            throw new ArgumentException($"logits length ({logits.Length}) must equal VocabSize ({VocabSize}) for a single decode position. " +
                "If you're passing the flattened decoder output for multiple positions, slice it to the last-position logits first.", nameof(logits));

        int bestToken = TokenIdOffset;
        double bestLogit = _numOps.ToDouble(logits[TokenIdOffset]);
        for (int t = TokenIdOffset + 1; t < TokenIdEndExclusive; t++)
        {
            double v = _numOps.ToDouble(logits[t]);
            if (v > bestLogit) { bestLogit = v; bestToken = t; }
        }
        return bestToken;
    }

    private int ValueToBin(double value, int dim)
    {
        double lo = _minRange[dim];
        double hi = _maxRange[dim];
        double clamped = Math.Max(lo, Math.Min(hi, value));
        double normalised = (clamped - lo) / (hi - lo);
        int bin = (int)Math.Floor(normalised * NumBins);
        if (bin >= NumBins) bin = NumBins - 1;
        if (bin < 0) bin = 0;
        return bin;
    }

    private double TokenToValue(int tokenId, int dim)
    {
        double lo = _minRange[dim];
        double hi = _maxRange[dim];
        int bin;
        if (tokenId < TokenIdOffset || tokenId >= TokenIdEndExclusive)
        {
            bin = NumBins / 2;
        }
        else
        {
            bin = tokenId - TokenIdOffset;
        }
        double centre = (bin + 0.5) / NumBins;
        return lo + centre * (hi - lo);
    }

    private static double[] NormaliseRange(double[]? input, int actionDim, double defaultValue, string paramName)
    {
        if (input is null)
        {
            var arr = new double[actionDim];
            for (int i = 0; i < actionDim; i++) arr[i] = defaultValue;
            return arr;
        }
        if (input.Length == 1)
        {
            var arr = new double[actionDim];
            for (int i = 0; i < actionDim; i++) arr[i] = input[0];
            return arr;
        }
        if (input.Length != actionDim)
            throw new ArgumentException($"{paramName} length ({input.Length}) must equal actionDim ({actionDim}) or be 1 to broadcast.", paramName);
        return (double[])input.Clone();
    }
}
