namespace AiDotNet.Enums;

/// <summary>
/// Represents the current status of a license key after validation.
/// </summary>
public enum LicenseKeyStatus
{
    /// <summary>
    /// The license key is valid and active.
    /// </summary>
    Active,

    /// <summary>
    /// The license key has expired.
    /// </summary>
    Expired,

    /// <summary>
    /// The license key has been revoked by an administrator.
    /// </summary>
    Revoked,

    /// <summary>
    /// The license key has reached its maximum number of allowed seats.
    /// </summary>
    SeatLimitReached,

    /// <summary>
    /// The license key is not valid (wrong format, unknown key, or incorrect).
    /// </summary>
    Invalid,

    /// <summary>
    /// The license key could not be validated online, but the client is allowed
    /// to continue operating until the offline grace period expires.
    /// </summary>
    ValidationPending
}
