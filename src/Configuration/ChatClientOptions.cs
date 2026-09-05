using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>Everything a chat client needs around it: defaults, retry policy, timeouts, filters, and recording.</summary>
/// <remarks>
/// <para>
/// A raw <see cref="IChatClient{T}"/> is a transport. Making it usable in a long unattended run means wrapping it
/// in a retry policy, a time limit, whatever filters the application needs, and — for anything that has to be
/// reproducible — a recorder. These options describe that wrapping once, in one place, and
/// <see cref="ChatClientPipelineFactory"/> assembles it, so the same behaviour applies whether the underlying
/// client is a cloud connector, a local engine, or a bridge to another library.
/// </para>
/// <para>
/// Retry settings live here rather than on a connector base class, which is what makes them apply to every
/// backend rather than only the HTTP ones, and settable by the caller rather than only by a subclass. The
/// reference OpenEvolve implementation hard-codes three retries with a constant five-second delay for its OpenAI
/// path and additionally hands the same retry count to the provider SDK, so a configured three can become sixteen
/// HTTP attempts; exactly one retry layer is applied here.
/// </para>
/// <para>
/// <see cref="RecordingMode"/> is the setting that decides whether an experiment can be repeated.
/// <see cref="ChatClientRecordingMode.Record"/> once against a live provider, commit the file named by
/// <see cref="RecordingPath"/>, and every later run under <see cref="ChatClientRecordingMode.Replay"/> reproduces
/// the same answers offline, at no cost, forever.
/// </para>
/// <para><b>For Beginners:</b> These are the settings that surround your AI model: how many times to retry when
/// the network hiccups, how long to wait before giving up on one call, any extra filters you want on every call,
/// and whether to save the answers so you can replay them later. Sensible defaults are already set — three
/// retries with growing waits — so you usually only touch this when you want recording or a hard timeout.</para>
/// </remarks>
public sealed class ChatClientOptions
{
    private IChatInteractionStore? _resolvedStore;

    /// <summary>Gets or sets settings applied to every call beneath the caller's own per-call settings.</summary>
    public ChatOptions? DefaultChatOptions { get; set; }

    /// <summary>Gets or sets the number of retries after the first attempt; <c>0</c> disables retrying.</summary>
    public int MaxRetries { get; set; } = RetryChatMiddleware.DefaultMaxRetries;

    /// <summary>Gets or sets the wait before the first retry, which doubles for each subsequent retry.</summary>
    public TimeSpan RetryBaseDelay { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>Gets or sets the ceiling on any single retry wait.</summary>
    public TimeSpan RetryMaxDelay { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>Gets or sets the fraction of each retry wait that is randomized, from 0 to 1.</summary>
    public double RetryJitterFactor { get; set; } = 0.25;

    /// <summary>Gets or sets the seed for the retry jitter stream, so a run's waits are reproducible.</summary>
    public ulong RetryJitterSeed { get; set; } = 0x5EED_1234_5678_9ABCUL;

    /// <summary>Gets or sets the time limit for one attempt, or <c>null</c> for no per-attempt limit.</summary>
    public TimeSpan? CallTimeout { get; set; }

    /// <summary>Gets or sets whether an OpenTelemetry span and counters are emitted for each call.</summary>
    public bool EnableTelemetry { get; set; }

    /// <summary>Gets or sets the filters applied to every call, outermost first.</summary>
    /// <remarks>They run inside telemetry and retry, so a filter sees one attempt at a time.</remarks>
    public IList<IChatMiddleware> Middleware { get; set; } = new List<IChatMiddleware>();

    /// <summary>Gets or sets the store recorded answers are written to and replayed from.</summary>
    /// <remarks><c>null</c> with a <see cref="RecordingPath"/> creates a <see cref="JsonFileChatInteractionStore"/>.</remarks>
    public IChatInteractionStore? InteractionStore { get; set; }

    /// <summary>Gets or sets the file a default interaction store reads and writes.</summary>
    public string? RecordingPath { get; set; }

    /// <summary>Gets or sets whether calls are live, recorded, or replayed.</summary>
    public ChatClientRecordingMode RecordingMode { get; set; } = ChatClientRecordingMode.None;

    /// <summary>Gets or sets the model id recordings are looked up under, or <c>null</c> to use the wrapped client's.</summary>
    /// <remarks>
    /// Recordings are keyed per model. Set this when replaying without the original client present, to the model
    /// id the recording was made under.
    /// </remarks>
    public string? ReplayModelId { get; set; }

    /// <summary>Creates an independent copy so a running pipeline is unaffected by later mutation.</summary>
    /// <returns>
    /// A new options instance. The middleware list is copied, but the middleware instances, the interaction
    /// store, and any resolved store are shared, because those are live objects rather than settings.
    /// </returns>
    public ChatClientOptions Clone()
    {
        var middleware = new List<IChatMiddleware>();
        if (Middleware is not null)
        {
            foreach (IChatMiddleware item in Middleware) middleware.Add(item);
        }

        var clone = new ChatClientOptions
        {
            DefaultChatOptions = ChatOptionsMerge.Copy(DefaultChatOptions),
            MaxRetries = MaxRetries,
            RetryBaseDelay = RetryBaseDelay,
            RetryMaxDelay = RetryMaxDelay,
            RetryJitterFactor = RetryJitterFactor,
            RetryJitterSeed = RetryJitterSeed,
            CallTimeout = CallTimeout,
            EnableTelemetry = EnableTelemetry,
            Middleware = middleware,
            InteractionStore = InteractionStore,
            RecordingPath = RecordingPath,
            RecordingMode = RecordingMode,
            ReplayModelId = ReplayModelId
        };

        // A store already opened for a file must not be opened a second time by the
        // copy: two instances over one path would each hold their own view and the
        // last flush would win.
        clone._resolvedStore = _resolvedStore;
        return clone;
    }

    /// <summary>Validates the retry policy, the timeout, the filter list, and the recording configuration.</summary>
    /// <exception cref="ArgumentException">
    /// The filter list is <c>null</c> or holds a <c>null</c> element, or recording is enabled without a store or a
    /// path.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">A retry or timeout value is outside its permitted range.</exception>
    public void Validate()
    {
        if (MaxRetries < 0 || MaxRetries > RetryChatMiddleware.MaxSupportedRetries)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxRetries), MaxRetries,
                $"Value must be between 0 and {RetryChatMiddleware.MaxSupportedRetries}.");
        }

        if (RetryBaseDelay < TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(RetryBaseDelay), RetryBaseDelay, "Value cannot be negative.");
        }

        if (RetryMaxDelay < RetryBaseDelay)
        {
            throw new ArgumentOutOfRangeException(nameof(RetryMaxDelay), RetryMaxDelay,
                "Value cannot be smaller than RetryBaseDelay.");
        }

        if (double.IsNaN(RetryJitterFactor) || double.IsInfinity(RetryJitterFactor)
            || RetryJitterFactor < 0 || RetryJitterFactor > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(RetryJitterFactor), RetryJitterFactor,
                "Value must be a finite number between 0 and 1.");
        }

        if (CallTimeout is { } timeout && timeout <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(CallTimeout), timeout, "Value must be positive.");
        }

        if (!Enum.IsDefined(typeof(ChatClientRecordingMode), RecordingMode))
        {
            throw new ArgumentOutOfRangeException(nameof(RecordingMode), RecordingMode, "Value must be a defined mode.");
        }

        if (Middleware is null)
        {
            throw new ArgumentException("Middleware cannot be null; use an empty list.", nameof(Middleware));
        }

        foreach (IChatMiddleware item in Middleware)
        {
            if (item is null) throw new ArgumentException("Middleware cannot contain a null element.", nameof(Middleware));
        }

        if (RecordingMode == ChatClientRecordingMode.None) return;
        if (InteractionStore is not null) return;
        if (!string.IsNullOrWhiteSpace(RecordingPath)) return;

        throw new ArgumentException(
            $"RecordingMode is {RecordingMode}, which needs either an InteractionStore or a RecordingPath.",
            nameof(RecordingMode));
    }

    /// <summary>Gets the interaction store to use, creating a file-backed one from the path when needed.</summary>
    /// <returns>
    /// <see cref="InteractionStore"/> when set, a <see cref="JsonFileChatInteractionStore"/> over
    /// <see cref="RecordingPath"/> when only that is set, or <c>null</c> when neither is configured. The created
    /// store is remembered, so repeated calls return the same instance.
    /// </returns>
    /// <exception cref="InvalidDataException">The recording file exists but is not a valid interaction store.</exception>
    public IChatInteractionStore? ResolveInteractionStore()
    {
        if (InteractionStore is not null) return InteractionStore;
        if (_resolvedStore is not null) return _resolvedStore;
        if (RecordingPath is not { } path || path.Trim().Length == 0) return null;

        // Replay must not create or rewrite the file it reads, so the file-backed
        // store only flushes on its own when a recording mode is active.
        bool autoFlush = RecordingMode == ChatClientRecordingMode.Record
            || RecordingMode == ChatClientRecordingMode.ReplayWithFallback;
        _resolvedStore = new JsonFileChatInteractionStore(path, autoFlush);
        return _resolvedStore;
    }
}
