namespace AiDotNet.Document;

/// <summary>
/// Types of layout elements that can be detected in documents.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Documents are made up of different visual elements like text blocks,
/// tables, and figures. Layout detection identifies these elements and their locations.
/// </para>
/// </remarks>
public enum LayoutElementType
{
    /// <summary>
    /// A block of regular text content.
    /// </summary>
    Text,

    /// <summary>
    /// A title or heading.
    /// </summary>
    Title,

    /// <summary>
    /// A bulleted or numbered list.
    /// </summary>
    List,

    /// <summary>
    /// A table with rows and columns.
    /// </summary>
    Table,

    /// <summary>
    /// A figure, image, or diagram.
    /// </summary>
    Figure,

    /// <summary>
    /// A caption for a figure or table.
    /// </summary>
    Caption,

    /// <summary>
    /// A page header.
    /// </summary>
    Header,

    /// <summary>
    /// A page footer.
    /// </summary>
    Footer,

    /// <summary>
    /// A page number.
    /// </summary>
    PageNumber,

    /// <summary>
    /// A mathematical equation or formula.
    /// </summary>
    Equation,

    /// <summary>
    /// A company logo or brand mark.
    /// </summary>
    Logo,

    /// <summary>
    /// A signature area.
    /// </summary>
    Signature,

    /// <summary>
    /// A stamp or seal.
    /// </summary>
    Stamp,

    /// <summary>
    /// A barcode.
    /// </summary>
    Barcode,

    /// <summary>
    /// A QR code.
    /// </summary>
    QRCode,

    /// <summary>
    /// Handwritten text.
    /// </summary>
    Handwriting,

    /// <summary>
    /// A form field (text input, checkbox, etc.).
    /// </summary>
    FormField,

    /// <summary>
    /// A section divider or separator.
    /// </summary>
    Separator,

    /// <summary>
    /// Other or unclassified element type.
    /// </summary>
    Other
}
