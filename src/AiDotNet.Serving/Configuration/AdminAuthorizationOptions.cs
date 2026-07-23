namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration for protecting admin endpoints.
/// </summary>
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

