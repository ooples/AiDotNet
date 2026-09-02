using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>Assembles a chat client and its surrounding policy — defaults, retry, telemetry, filters, recording.</summary>
/// <remarks>
/// <para>
/// One call turns a bare transport into the client a long unattended run needs. The layers are applied in a fixed
/// order, outermost first: telemetry, so one logical call produces one span and one count no matter how many
/// attempts it took; retry, so the caller's filters see one attempt at a time; then the caller's own filters; and
/// innermost, the record or replay decorator wrapping the real client. That ordering is deliberate — a replayed
/// answer is served before retry can ever be involved, and a recorded run captures exactly the request the
/// provider saw, after every filter has had its say.
/// </para>
/// <para>
/// Because retry lives in the pipeline, it applies to every backend: an HTTP connector, an in-process local
/// engine, a bridge to another library. The reference OpenEvolve implementation puts retry inside its OpenAI
/// client alone, and additionally hands the retry count to the provider SDK, so a configured three retries can
/// become sixteen HTTP attempts. Exactly one retry layer is applied here.
/// </para>
/// <para><b>For Beginners:</b> You have a model client; this hands you back the same client with the safety
/// features switched on — automatic retries, a time limit, optional monitoring, and optional recording so your
/// run can be replayed later without calling the model again. Configure once with
/// <see cref="ChatClientOptions"/> and use the returned client exactly as you used the original.</para>
/// </remarks>
public static class ChatClientPipelineFactory
{
    /// <summary>Wraps a client in the policy described by the options.</summary>
    /// <typeparam name="T">The numeric type of the client.</typeparam>
    /// <param name="inner">The client the pipeline ultimately calls.</param>
    /// <param name="options">The policy to apply; <c>null</c> or a policy that adds nothing returns <paramref name="inner"/>.</param>
    /// <returns>The wrapped client, or <paramref name="inner"/> when the options add no behaviour.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="inner"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">The options are invalid, or recording is enabled without a store.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A retry or timeout value is outside its permitted range.</exception>
    /// <exception cref="InvalidDataException">A configured recording file exists but is not a valid interaction store.</exception>
    public static IChatClient<T> Create<T>(IChatClient<T> inner, ChatClientOptions? options)
    {
        Guard.NotNull(inner);
        if (options is null) return inner;

        ChatClientOptions settings = options.Clone();
        settings.Validate();

        IChatClient<T> client = ApplyRecording(inner, settings);

        var middlewares = new List<IChatMiddleware>();
        if (settings.EnableTelemetry) middlewares.Add(new TelemetryChatMiddleware());
        if (settings.MaxRetries > 0 || settings.CallTimeout.HasValue)
        {
            middlewares.Add(new RetryChatMiddleware(
                settings.MaxRetries,
                settings.RetryBaseDelay,
                settings.RetryMaxDelay,
                settings.CallTimeout,
                settings.RetryJitterFactor,
                settings.RetryJitterSeed));
        }

        foreach (IChatMiddleware middleware in settings.Middleware) middlewares.Add(middleware);
        if (settings.DefaultChatOptions is { } defaults) middlewares.Add(new DefaultsMiddleware(defaults));

        return middlewares.Count == 0 ? client : new MiddlewareChatClient<T>(client, middlewares);
    }

    private static IChatClient<T> ApplyRecording<T>(IChatClient<T> inner, ChatClientOptions settings)
    {
        if (settings.RecordingMode == ChatClientRecordingMode.None) return inner;

        IChatInteractionStore? store = settings.ResolveInteractionStore();
        if (store is null)
        {
            throw new ArgumentException(
                $"RecordingMode is {settings.RecordingMode}, which needs either an InteractionStore or a RecordingPath.",
                nameof(settings));
        }

        switch (settings.RecordingMode)
        {
            case ChatClientRecordingMode.Record:
                return new RecordingChatClient<T>(inner, store);
            case ChatClientRecordingMode.Replay:
                // No fallback: a missing recording must fail loudly rather than
                // quietly reaching the network and making the run non-reproducible.
                return new ReplayingChatClient<T>(store, fallback: null, modelId: settings.ReplayModelId ?? inner.ModelId);
            case ChatClientRecordingMode.ReplayWithFallback:
                return new ReplayingChatClient<T>(store, inner, settings.ReplayModelId ?? inner.ModelId);
            default:
                return inner;
        }
    }

    /// <summary>Fills in client-wide default settings beneath whatever the caller supplied for one call.</summary>
    private sealed class DefaultsMiddleware : IChatMiddleware
    {
        private readonly ChatOptions _defaults;

        public DefaultsMiddleware(ChatOptions defaults) => _defaults = defaults;

        public Task<ChatResponse> InvokeAsync(
            ChatRequestContext context,
            ChatPipelineDelegate next,
            CancellationToken cancellationToken)
        {
            Guard.NotNull(context);
            Guard.NotNull(next);

            // Innermost, so the request the model sees is the fully resolved one and
            // a recording captures the settings that were actually in force.
            context.Options = ChatOptionsMerge.Merge(context.Options, _defaults);
            return next(context, cancellationToken);
        }
    }
}
