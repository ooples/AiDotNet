namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Invoice line item.
/// </summary>
public class InvoiceLineItem
{
    /// <summary>
    /// Quantity.
    /// </summary>
    public int Quantity { get; set; }

    /// <summary>
    /// Item description.
    /// </summary>
    public string Description { get; set; } = "";

    /// <summary>
    /// Unit price.
    /// </summary>
    public decimal UnitPrice { get; set; }

    /// <summary>
    /// Total amount for this line.
    /// </summary>
    public decimal Amount { get; set; }
}
