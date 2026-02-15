namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for form understanding models that extract structured fields from documents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Form understanding models extract key-value pairs, checkboxes, signatures, and
/// other structured information from forms, invoices, receipts, and similar documents.
/// </para>
/// <para>
/// <b>For Beginners:</b> Form understanding is like having someone read a form and
/// fill out a digital version. The model finds:
/// - Field labels and their values (e.g., "Name: John Smith")
/// - Checkboxes and whether they're checked
/// - Signatures and their locations
///
/// Example usage:
/// <code>
/// var model = new PICK&lt;float&gt;(architecture);
/// var result = model.ExtractFormFields(formImage);
/// foreach (var field in result.Fields)
///     Console.WriteLine($"{field.FieldName}: {field.FieldValue}");
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("FormUnderstanding")]
public interface IFormUnderstanding<T> : IDocumentModel<T>
{
    /// <summary>
    /// Extracts form fields from a document image.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <returns>Form field extraction result.</returns>
    FormFieldResult<T> ExtractFormFields(Tensor<T> documentImage);

    /// <summary>
    /// Extracts form fields with a custom confidence threshold.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="confidenceThreshold">Minimum confidence for field extraction (0-1).</param>
    /// <returns>Form field extraction result.</returns>
    FormFieldResult<T> ExtractFormFields(Tensor<T> documentImage, double confidenceThreshold);

    /// <summary>
    /// Extracts key-value pairs from a document.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <returns>Dictionary of field names to values.</returns>
    Dictionary<string, string> ExtractKeyValuePairs(Tensor<T> documentImage);

    /// <summary>
    /// Detects checkboxes and their states in a document.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <returns>Collection of detected checkboxes.</returns>
    IEnumerable<CheckboxResult<T>> DetectCheckboxes(Tensor<T> documentImage);

    /// <summary>
    /// Detects signatures in a document.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <returns>Collection of detected signatures.</returns>
    IEnumerable<SignatureResult<T>> DetectSignatures(Tensor<T> documentImage);
}

/// <summary>
/// Result of form field extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FormFieldResult<T>
{
    /// <summary>
    /// Gets or sets the extracted form fields.
    /// </summary>
    public IList<FormField<T>> Fields { get; set; } = [];

    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; set; }
}

/// <summary>
/// Represents an extracted form field.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FormField<T>
{
    /// <summary>
    /// Gets or sets the field name/label.
    /// </summary>
    public string FieldName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the field value.
    /// </summary>
    public string FieldValue { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the field type (text, number, date, etc.).
    /// </summary>
    public string FieldType { get; set; } = "text";

    /// <summary>
    /// Gets or sets the confidence score as generic type.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence score as double.
    /// </summary>
    public double ConfidenceValue { get; set; }

    /// <summary>
    /// Gets or sets the bounding box [x1, y1, x2, y2].
    /// </summary>
    public Vector<T> BoundingBox { get; set; } = Vector<T>.Empty();
}

/// <summary>
/// Result of checkbox detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CheckboxResult<T>
{
    /// <summary>
    /// Gets or sets whether the checkbox is checked.
    /// </summary>
    public bool IsChecked { get; set; }

    /// <summary>
    /// Gets or sets the label associated with the checkbox.
    /// </summary>
    public string Label { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score as generic type.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence score as double.
    /// </summary>
    public double ConfidenceValue { get; set; }

    /// <summary>
    /// Gets or sets the bounding box [x1, y1, x2, y2].
    /// </summary>
    public Vector<T> BoundingBox { get; set; } = Vector<T>.Empty();
}

/// <summary>
/// Result of signature detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SignatureResult<T>
{
    /// <summary>
    /// Gets or sets whether a signature is present.
    /// </summary>
    public bool IsPresent { get; set; }

    /// <summary>
    /// Gets or sets the confidence score as generic type.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence score as double.
    /// </summary>
    public double ConfidenceValue { get; set; }

    /// <summary>
    /// Gets or sets the bounding box [x1, y1, x2, y2].
    /// </summary>
    public Vector<T> BoundingBox { get; set; } = Vector<T>.Empty();
}
