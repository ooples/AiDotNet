namespace AiDotNet.Inference;

/// <summary>
/// Tracks whether one model/shape compiled plan has been shown to respond to its input.
/// </summary>
/// <remarks>
/// Held per concrete input shape within one compiled function rather than globally:
/// value-stability is a property of the traced graph, so two shapes or models built by the same
/// process can legitimately differ.
/// </remarks>
internal sealed class JitValueStability<TValue>
{
    private enum PlanState : byte
    {
        Unverified,
        Verified,
        PermanentlyDisabled
    }

    private readonly object _gate = new object();
    private TValue[]? _tracedValues;
    private volatile PlanState _state;

    public bool IsVerified => _state == PlanState.Verified;

    public bool IsPermanentlyDisabled => _state == PlanState.PermanentlyDisabled;

    /// <summary>
    /// Remembers the first input seen, and reports whether this one differs from it.
    /// </summary>
    public bool RecordAndCheckDiffers(Tensor<TValue> input)
    {
        lock (_gate)
        {
            // A caller can observe Unverified just before another request reaches a terminal
            // state. Do not let that stale observation recreate the traced copy after the
            // terminal transition released it.
            if (_state != PlanState.Unverified)
            {
                return false;
            }

            if (_tracedValues is null)
            {
                _tracedValues = new TValue[input.Length];
                for (int i = 0; i < input.Length; i++) _tracedValues[i] = input[i];

                return false;
            }

            if (_tracedValues.Length != input.Length) return true;

            var comparer = EqualityComparer<TValue>.Default;
            for (int i = 0; i < input.Length; i++)
            {
                if (!comparer.Equals(_tracedValues[i], input[i])) return true;
            }

            return false;
        }
    }

    public void MarkVerified()
    {
        lock (_gate)
        {
            // Permanent disablement is terminal. A concurrent prediction may finish an
            // earlier stability check after another prediction has already rejected the
            // plan; that late success must not re-enable the compiled path.
            if (_state != PlanState.PermanentlyDisabled)
            {
                _state = PlanState.Verified;
                _tracedValues = null;
            }
        }
    }

    public bool TryMarkPermanentlyDisabled()
    {
        lock (_gate)
        {
            if (_state == PlanState.PermanentlyDisabled)
            {
                return false;
            }

            _state = PlanState.PermanentlyDisabled;
            _tracedValues = null;
            return true;
        }
    }
}
