namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Types of entities that can be extracted from documents.
/// </summary>
public enum EntityType
{
    /// <summary>
    /// Person name.
    /// </summary>
    Person,

    /// <summary>
    /// Organization name.
    /// </summary>
    Organization,

    /// <summary>
    /// Location or place.
    /// </summary>
    Location,

    /// <summary>
    /// Date or time.
    /// </summary>
    Date,

    /// <summary>
    /// Monetary amount.
    /// </summary>
    Money,

    /// <summary>
    /// Percentage.
    /// </summary>
    Percentage,

    /// <summary>
    /// Email address.
    /// </summary>
    Email,

    /// <summary>
    /// Phone number.
    /// </summary>
    Phone,

    /// <summary>
    /// URL or web address.
    /// </summary>
    Url,

    /// <summary>
    /// Product or item.
    /// </summary>
    Product,

    /// <summary>
    /// Event name.
    /// </summary>
    Event,

    /// <summary>
    /// Other or custom entity type.
    /// </summary>
    Other
}
