namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration for protecting admin endpoints.
/// </summary>
/// <remarks>
/// The admin endpoints accept an API key with the <c>Admin</c> scope, or any authenticated caller (typically a
/// JWT) carrying the claim <see cref="ClaimType"/> with the value <see cref="ClaimValue"/>.
/// </remarks>
public sealed class AdminAuthorizationOptions
{
    /// <summary>
    /// Gets or sets the claim type used to identify admin users (for example, <c>"role"</c>).
    /// </summary>
    /// <remarks>
    /// This value must be a non-empty string. It is used by the admin authorization policy at runtime.
    /// </remarks>
    public string ClaimType { get; set; } = "role";

    /// <summary>
    /// Gets or sets the claim value required for admin access (for example, <c>"admin"</c>).
    /// </summary>
    /// <remarks>
    /// This value must be a non-empty string. It is used by the admin authorization policy at runtime.
    /// </remarks>
    public string ClaimValue { get; set; } = "admin";

    /// <summary>
    /// Validates these settings and throws <see cref="InvalidOperationException"/> naming the first invalid one.
    /// </summary>
    /// <param name="configurationPath">The configuration path used in error messages.</param>
    public void Validate(string configurationPath)
    {
        if (string.IsNullOrWhiteSpace(configurationPath))
        {
            configurationPath = "ServingSecurity:Admin";
        }

        if (string.IsNullOrWhiteSpace(ClaimType))
        {
            throw new InvalidOperationException($"{configurationPath}: ClaimType must be a non-empty string.");
        }

        if (string.IsNullOrWhiteSpace(ClaimValue))
        {
            throw new InvalidOperationException($"{configurationPath}: ClaimValue must be a non-empty string.");
        }
    }
}

