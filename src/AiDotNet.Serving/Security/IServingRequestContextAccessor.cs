namespace AiDotNet.Serving.Security;

/// <summary>
/// Provides access to the current <see cref="ServingRequestContext"/> for the active HTTP request.
/// </summary>
public interface IServingRequestContextAccessor
{
    ServingRequestContext? Current { get; set; }
}

