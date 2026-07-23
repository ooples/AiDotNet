namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Structured receipt data extracted from documents.
/// </summary>
public class ReceiptData
{
    /// <summary>
    /// Store name.
    /// </summary>
    public string? StoreName { get; set; }

    /// <summary>
    /// Store address.
    /// </summary>
    public string? StoreAddress { get; set; }

    /// <summary>
    /// Receipt date.
    /// </summary>
    public DateTime? Date { get; set; }

    /// <summary>
    /// Total amount.
    /// </summary>
    public decimal? Total { get; set; }

    /// <summary>
    /// Tax amount.
    /// </summary>
    public decimal? Tax { get; set; }

    /// <summary>
    /// Payment method.
    /// </summary>
    public string? PaymentMethod { get; set; }

    /// <summary>
    /// Receipt items.
    /// </summary>
    public List<ReceiptItem> Items { get; set; } = new();
}
