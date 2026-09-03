namespace AiDotNet.Configuration;

/// <summary>Settings for the chat client that asks a person instead of a model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These control how often the client checks for your answer, how long it is willing to
/// wait, and what name it reports for itself. The defaults are usually right: check twice a second, wait forever,
/// and call yourself <c>manual</c>.</para>
/// </remarks>
public sealed class ManualChatClientOptions
{
    /// <summary>The longest permitted gap between checks for an answer.</summary>
    public static readonly TimeSpan MaxPollInterval = TimeSpan.FromMinutes(5);

    /// <summary>Gets or sets how often the queue directory is checked for an answer. Defaults to half a second.</summary>
    public TimeSpan PollInterval { get; set; } = TimeSpan.FromMilliseconds(500);

    /// <summary>Gets or sets how long one request waits for an answer; <c>null</c>, the default, waits indefinitely.</summary>
    /// <remarks>
    /// Waiting indefinitely is the sensible default for a mode whose whole point is a person in the loop: a run that
    /// gave up after ten minutes because somebody went to lunch would be worse than useless. Cancelling the run still
    /// ends the wait, so nothing is stuck. Set a timeout when a script rather than a person is answering.
    /// </remarks>
    public TimeSpan? Timeout { get; set; }

    /// <summary>Gets or sets whether leftover task and answer files are removed at construction. Defaults to <c>true</c>.</summary>
    /// <remarks>
    /// A stale answer file from an earlier run would be served instantly as the reply to a prompt it never saw, and
    /// the run would continue with results that are quietly meaningless. Clear it unless you are deliberately
    /// resuming into a queue somebody has already answered.
    /// </remarks>
    public bool ClearStaleTasks { get; set; } = true;

    /// <summary>Gets or sets the identifier the client reports. Defaults to <c>manual</c>.</summary>
    public string ModelId { get; set; } = "manual";

    /// <summary>Creates an independent copy so a running client is unaffected by later mutation.</summary>
    /// <returns>A new instance carrying the same settings.</returns>
    public ManualChatClientOptions Clone() => new()
    {
        PollInterval = PollInterval,
        Timeout = Timeout,
        ClearStaleTasks = ClearStaleTasks,
        ModelId = ModelId
    };

    /// <summary>Rejects intervals and timeouts that cannot be honoured.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The poll interval or the timeout is out of range.</exception>
    /// <exception cref="ArgumentException"><see cref="ModelId"/> is blank.</exception>
    public void Validate()
    {
        if (PollInterval <= TimeSpan.Zero || PollInterval > MaxPollInterval)
        {
            throw new ArgumentOutOfRangeException(nameof(PollInterval), PollInterval,
                "Value must be positive and at most " + MaxPollInterval + ".");
        }

        if (Timeout.HasValue && Timeout.Value <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(Timeout), Timeout.Value,
                "Value must be positive; leave it null to wait indefinitely.");
        }

        if (string.IsNullOrWhiteSpace(ModelId))
            throw new ArgumentException("The reported model identifier cannot be blank.", nameof(ModelId));
    }
}
