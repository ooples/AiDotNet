namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Structured invoice data extracted from documents.
/// </summary>
public class InvoiceData
{
    /// <summary>
    /// Invoice number.
    /// </summary>
    public string? InvoiceNumber { get; set; }

    /// <summary>
    /// Invoice date.
    /// </summary>
    public DateTime? Date { get; set; }

    /// <summary>
    /// Vendor/seller name.
    /// </summary>
    public string? Vendor { get; set; }

    /// <summary>
    /// Customer/buyer name.
    /// </summary>
    public string? Customer { get; set; }

    /// <summary>
    /// Total amount.
    /// </summary>
    public decimal? Total { get; set; }

    /// <summary>
    /// Tax amount.
    /// </summary>
    public decimal? Tax { get; set; }

    /// <summary>
    /// Line items.
    /// </summary>
    public List<InvoiceLineItem> LineItems { get; set; } = new();
}
