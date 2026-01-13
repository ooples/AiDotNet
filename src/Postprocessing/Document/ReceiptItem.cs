namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Receipt line item.
/// </summary>
public class ReceiptItem
{
    /// <summary>
    /// Item description.
    /// </summary>
    public string Description { get; set; } = "";

    /// <summary>
    /// Quantity.
    /// </summary>
    public int Quantity { get; set; } = 1;

    /// <summary>
    /// Item price.
    /// </summary>
    public decimal Price { get; set; }
}
