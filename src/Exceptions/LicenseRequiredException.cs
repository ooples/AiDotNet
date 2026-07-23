namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when a model persistence operation (save or load) is attempted
/// after the free trial period has expired and no valid license key is configured.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> AiDotNet offers a free trial for model save/load operations.
/// During the trial you can save and load models without restriction. Once the trial expires
/// (after 30 days or 10 save/load operations, whichever comes first), you need to register
/// for a free community license or purchase a commercial license at https://aidotnet.dev.</para>
///
/// <para>Training and inference are never restricted — this exception only applies to
/// <c>SaveModel()</c> and <c>LoadModel()</c> operations.</para>
///
/// <para><b>How to resolve:</b></para>
/// <list type="bullet">
///   <item><description>Register for a free community license at https://aidotnet.dev (for open-source, education, personal use, or companies under $1M revenue with fewer than 5 developers).</description></item>
///   <item><description>Set the license key via environment variable: <c>AIDOTNET_LICENSE_KEY=your-key</c></description></item>
///   <item><description>Or save it to: <c>~/.aidotnet/license.key</c></description></item>
///   <item><description>Or pass it directly: <c>new AiModelBuilder&lt;T, TInput, TOutput&gt;(new AiDotNetLicenseKey("your-key"))</c></description></item>
/// </list>
/// </remarks>
public sealed class LicenseRequiredException : AiDotNetException
{
    /// <summary>
    /// Gets the number of days the trial was active before expiring.
    /// Null if the trial expired due to operation count rather than time.
    /// </summary>
    public int? TrialDaysElapsed { get; }

    /// <summary>
    /// Gets the number of save/load operations performed during the trial.
    /// Null if the trial expired due to time rather than operation count.
    /// </summary>
    public int? OperationsPerformed { get; }

    /// <summary>
    /// Gets the reason the trial expired.
    /// </summary>
    public TrialExpirationReason ExpirationReason { get; }

    /// <summary>
    /// Creates a new <see cref="LicenseRequiredException"/> with a default message.
    /// </summary>
    public LicenseRequiredException()
        : base(DefaultMessage)
    {
        ExpirationReason = TrialExpirationReason.Unknown;
    }

    /// <summary>
    /// Creates a new <see cref="LicenseRequiredException"/> with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public LicenseRequiredException(string message)
        : base(message)
    {
        ExpirationReason = TrialExpirationReason.Unknown;
    }

    /// <summary>
    /// Creates a new <see cref="LicenseRequiredException"/> with a specified error message
    /// and a reference to the inner exception that is the cause of this exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public LicenseRequiredException(string message, Exception innerException)
        : base(message, innerException)
    {
        ExpirationReason = TrialExpirationReason.Unknown;
    }

    /// <summary>
    /// Creates a new <see cref="LicenseRequiredException"/> with trial expiration details.
    /// </summary>
    /// <param name="reason">The reason the trial expired.</param>
    /// <param name="trialDaysElapsed">Number of days the trial was active.</param>
    /// <param name="operationsPerformed">Number of save/load operations performed.</param>
    public LicenseRequiredException(
        TrialExpirationReason reason,
        int? trialDaysElapsed = null,
        int? operationsPerformed = null)
        : base(FormatMessage(reason, trialDaysElapsed, operationsPerformed))
    {
        ExpirationReason = reason;
        TrialDaysElapsed = trialDaysElapsed;
        OperationsPerformed = operationsPerformed;
    }

    private static string FormatMessage(
        TrialExpirationReason reason,
        int? trialDaysElapsed,
        int? operationsPerformed)
    {
        var detail = reason switch
        {
            TrialExpirationReason.TimeExpired =>
                $"Your 30-day free trial has expired (started {trialDaysElapsed} days ago).",
            TrialExpirationReason.OperationLimitReached =>
                $"Your free trial has reached the limit of 10 save/load operations ({operationsPerformed} performed).",
            TrialExpirationReason.LicenseInvalid =>
                "The configured license key is invalid or has been revoked.",
            TrialExpirationReason.LicenseExpired =>
                "The configured license key has expired.",
            TrialExpirationReason.SeatLimitReached =>
                "The configured license key has reached its maximum number of developer seats.",
            _ =>
                "A valid license is required for model save/load operations."
        };

        return $"{detail} " +
               "Register for a free community license at https://aidotnet.dev — " +
               "training and inference remain fully available without a license.";
    }

    private const string DefaultMessage =
        "AiDotNet free trial has expired. Register for a free community license at " +
        "https://aidotnet.dev to continue saving and loading models. " +
        "Training and inference remain fully available without a license.";
}

/// <summary>
/// Specifies the reason a trial period expired or a license check failed.
/// </summary>
public enum TrialExpirationReason
{
    /// <summary>
    /// The reason is unknown or not specified.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// The 30-day free trial period has elapsed.
    /// </summary>
    TimeExpired,

    /// <summary>
    /// The maximum number of free trial operations (10) has been reached.
    /// </summary>
    OperationLimitReached,

    /// <summary>
    /// The provided license key is invalid or unrecognized.
    /// </summary>
    LicenseInvalid,

    /// <summary>
    /// The provided license key has expired.
    /// </summary>
    LicenseExpired,

    /// <summary>
    /// The license has reached its maximum allowed developer seats.
    /// </summary>
    SeatLimitReached
}
