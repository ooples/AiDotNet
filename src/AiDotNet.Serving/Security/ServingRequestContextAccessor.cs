namespace AiDotNet.Serving.Security;

/// <summary>
/// Default request context accessor backed by <see cref="AsyncLocal{T}"/>.
/// </summary>
public sealed class ServingRequestContextAccessor : IServingRequestContextAccessor
{
    private static readonly AsyncLocal<ServingRequestContext?> Context = new();

    public ServingRequestContext? Current
    {
        get => Context.Value;
        set => Context.Value = value;
    }
}

