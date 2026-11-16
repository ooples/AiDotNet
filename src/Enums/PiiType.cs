namespace AiDotNet.Enums;

/// <summary>
/// A kind of personally identifiable information the text safety pipeline can detect.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Pick the kinds of private data you want found and redacted. Pattern-shaped data
/// (emails, card numbers, keys) is found by regular expressions; names, addresses and organizations by
/// named-entity heuristics.</para>
/// </remarks>
public enum PiiType
{
    /// <summary>An email address.</summary>
    Email,

    /// <summary>A phone number.</summary>
    PhoneNumber,

    /// <summary>A US Social Security Number.</summary>
    SocialSecurityNumber,

    /// <summary>A payment card number.</summary>
    CreditCardNumber,

    /// <summary>An IPv4 address.</summary>
    IPAddress,

    /// <summary>An API key or bearer token.</summary>
    ApiKey,

    /// <summary>An AWS access key id.</summary>
    AwsAccessKey,

    /// <summary>A plaintext password assignment.</summary>
    Password,

    /// <summary>A passport number.</summary>
    PassportNumber,

    /// <summary>A driver's license number.</summary>
    DriversLicense,

    /// <summary>A person's name.</summary>
    PersonName,

    /// <summary>A postal address or location.</summary>
    Address,

    /// <summary>An organization name.</summary>
    Organization
}
