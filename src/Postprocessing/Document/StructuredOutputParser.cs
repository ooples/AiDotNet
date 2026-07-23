using System.Text.RegularExpressions;

namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// StructuredOutputParser - Parses document AI outputs into structured data formats.
/// </summary>
/// <remarks>
/// <para>
/// StructuredOutputParser converts raw model outputs and OCR results into
/// structured formats like JSON, tables, key-value pairs, and custom schemas.
/// </para>
/// <para>
/// <b>For Beginners:</b> AI models output text, but applications need structured data.
/// This tool bridges that gap:
///
/// - Convert OCR text to JSON
/// - Extract key-value pairs
/// - Build tabular data
/// - Validate against schemas
///
/// Key features:
/// - Multiple output formats
/// - Schema validation
/// - Custom parsing rules
/// - Error handling
///
/// Example usage:
/// <code>
/// var parser = new StructuredOutputParser&lt;float&gt;();
/// var kvPairs = parser.Process(documentText);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StructuredOutputParser<T> : PostprocessorBase<T, string, Dictionary<string, string>>, IDisposable
{
    #region Fields

    private readonly System.Text.Json.JsonSerializerOptions _jsonOptions;
    private bool _disposed;

    #endregion

    #region Properties

    /// <summary>
    /// Structured output parser does not support inverse transformation.
    /// </summary>
    public override bool SupportsInverse => false;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new StructuredOutputParser with default settings.
    /// </summary>
    public StructuredOutputParser()
    {
        _jsonOptions = new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
        };
    }

    #endregion

    #region Core Implementation

    /// <summary>
    /// Parses text into key-value pairs.
    /// </summary>
    /// <param name="input">The text to parse.</param>
    /// <returns>Dictionary of extracted key-value pairs.</returns>
    protected override Dictionary<string, string> ProcessCore(string input)
    {
        return ParseKeyValuePairs(input);
    }

    /// <summary>
    /// Validates the input text.
    /// </summary>
    protected override void ValidateInput(string input)
    {
        // Allow null/empty strings - they will return empty dictionary
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Parses text into a JSON object based on extraction rules.
    /// </summary>
    /// <param name="text">The text to parse.</param>
    /// <param name="fieldRules">Rules for extracting fields.</param>
    /// <returns>JSON string representation of extracted data.</returns>
    public string ParseToJson(string text, IEnumerable<FieldExtractionRule> fieldRules)
    {
        var result = new Dictionary<string, object?>();

        foreach (var rule in fieldRules)
        {
            var value = ExtractFieldValue(text, rule);
            result[rule.FieldName] = value;
        }

        return System.Text.Json.JsonSerializer.Serialize(result, _jsonOptions);
    }

    /// <summary>
    /// Parses text into key-value pairs.
    /// </summary>
    /// <param name="text">The text to parse.</param>
    /// <returns>Dictionary of extracted key-value pairs.</returns>
    public Dictionary<string, string> ParseKeyValuePairs(string text)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (string.IsNullOrEmpty(text))
            return result;

        // Pattern: "Key: Value" or "Key = Value" or "Key - Value"
        var patterns = new[]
        {
            @"^([^:\n]+):\s*(.+)$",
            @"^([^=\n]+)=\s*(.+)$",
            @"^([^-\n]+)\s+-\s+(.+)$"
        };

        var lines = text.Split('\n');

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed))
                continue;

            foreach (var pattern in patterns)
            {
                var match = RegexHelper.Match(trimmed, pattern);
                if (match.Success)
                {
                    var key = match.Groups[1].Value.Trim();
                    var value = match.Groups[2].Value.Trim();

                    if (!string.IsNullOrEmpty(key) && !string.IsNullOrEmpty(value))
                    {
                        result[key] = value;
                        break;
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Parses tabular text into a list of rows.
    /// </summary>
    /// <param name="text">The text containing tabular data.</param>
    /// <param name="delimiter">The column delimiter.</param>
    /// <returns>List of rows, each row being a list of cell values.</returns>
    public IList<IList<string>> ParseTable(string text, string delimiter = "\t")
    {
        var rows = new List<IList<string>>();
        var lines = text.Split('\n');

        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed))
                continue;

            var cells = trimmed.Split(new[] { delimiter }, StringSplitOptions.None)
                .Select(c => c.Trim())
                .ToList();

            if (cells.Any(c => !string.IsNullOrEmpty(c)))
                rows.Add(cells);
        }

        return rows;
    }

    /// <summary>
    /// Parses tabular text into a list of dictionaries using the first row as headers.
    /// </summary>
    public IList<Dictionary<string, string>> ParseTableWithHeaders(string text, string delimiter = "\t")
    {
        var rows = ParseTable(text, delimiter);
        var result = new List<Dictionary<string, string>>();

        if (rows.Count < 2)
            return result;

        var headers = rows[0];

        for (int i = 1; i < rows.Count; i++)
        {
            var row = rows[i];
            var dict = new Dictionary<string, string>();

            for (int j = 0; j < Math.Min(headers.Count, row.Count); j++)
            {
                if (!string.IsNullOrEmpty(headers[j]))
                    dict[headers[j]] = row[j];
            }

            result.Add(dict);
        }

        return result;
    }

    /// <summary>
    /// Parses a form into a structured object.
    /// </summary>
    /// <param name="text">The form text.</param>
    /// <param name="fieldLabels">Labels to look for.</param>
    /// <returns>Dictionary of field labels to values.</returns>
    public Dictionary<string, string> ParseForm(string text, IEnumerable<string> fieldLabels)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (var label in fieldLabels)
        {
            var value = ExtractFormField(text, label);
            if (value != null)
                result[label] = value;
        }

        return result;
    }

    /// <summary>
    /// Parses an invoice into a structured format.
    /// </summary>
    public InvoiceData ParseInvoice(string text)
    {
        var invoice = new InvoiceData();

        // Extract invoice number
        var invoiceNumMatch = RegexHelper.Match(text, @"(?:Invoice\s*(?:#|No\.?|Number)?:?\s*)(\S+)", RegexOptions.IgnoreCase);
        if (invoiceNumMatch.Success)
            invoice.InvoiceNumber = invoiceNumMatch.Groups[1].Value;

        // Extract date
        var dateMatch = RegexHelper.Match(text, @"(?:Date:?\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})", RegexOptions.IgnoreCase);
        if (dateMatch.Success)
        {
            if (DateTime.TryParse(dateMatch.Groups[1].Value, out var date))
                invoice.Date = date;
        }

        // Extract total
        var totalMatch = RegexHelper.Match(text, @"(?:Total:?\s*|Grand\s+Total:?\s*)[$€£]?\s*([\d,]+\.?\d*)", RegexOptions.IgnoreCase);
        if (totalMatch.Success)
        {
            if (decimal.TryParse(totalMatch.Groups[1].Value.Replace(",", ""), out var total))
                invoice.Total = total;
        }

        // Extract vendor
        var vendorMatch = RegexHelper.Match(text, @"(?:From:?\s*|Vendor:?\s*|Company:?\s*)(.+?)(?:\n|$)", RegexOptions.IgnoreCase);
        if (vendorMatch.Success)
            invoice.Vendor = vendorMatch.Groups[1].Value.Trim();

        // Extract line items (simplified pattern)
        var itemPattern = RegexHelper.Create(@"^(\d+)\s+(.+?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)$", RegexOptions.Multiline);
        var itemMatches = itemPattern.Matches(text);

        foreach (Match match in itemMatches)
        {
            invoice.LineItems.Add(new InvoiceLineItem
            {
                Quantity = int.TryParse(match.Groups[1].Value, out var qty) ? qty : 1,
                Description = match.Groups[2].Value.Trim(),
                UnitPrice = decimal.TryParse(match.Groups[3].Value.Replace(",", ""), out var price) ? price : 0,
                Amount = decimal.TryParse(match.Groups[4].Value.Replace(",", ""), out var amount) ? amount : 0
            });
        }

        return invoice;
    }

    /// <summary>
    /// Parses a receipt into a structured format.
    /// </summary>
    public ReceiptData ParseReceipt(string text)
    {
        var receipt = new ReceiptData();

        // Extract store name (usually at the top)
        var lines = text.Split('\n').Where(l => !string.IsNullOrWhiteSpace(l)).ToList();
        if (lines.Count > 0)
            receipt.StoreName = lines[0].Trim();

        // Extract date
        var dateMatch = RegexHelper.Match(text, @"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", RegexOptions.IgnoreCase);
        if (dateMatch.Success && DateTime.TryParse(dateMatch.Groups[1].Value, out var date))
            receipt.Date = date;

        // Extract total
        var totalMatch = RegexHelper.Match(text, @"(?:Total|Amount|Due):?\s*[$€£]?\s*([\d,]+\.?\d*)", RegexOptions.IgnoreCase);
        if (totalMatch.Success && decimal.TryParse(totalMatch.Groups[1].Value.Replace(",", ""), out var total))
            receipt.Total = total;

        // Extract tax
        var taxMatch = RegexHelper.Match(text, @"(?:Tax|VAT):?\s*[$€£]?\s*([\d,]+\.?\d*)", RegexOptions.IgnoreCase);
        if (taxMatch.Success && decimal.TryParse(taxMatch.Groups[1].Value.Replace(",", ""), out var tax))
            receipt.Tax = tax;

        // Extract items (pattern: item name followed by price)
        var itemPattern = RegexHelper.Create(@"^(.+?)\s+([\d,]+\.?\d*)$", RegexOptions.Multiline);
        var itemMatches = itemPattern.Matches(text);

        foreach (Match match in itemMatches)
        {
            var description = match.Groups[1].Value.Trim();
            // Skip if it looks like a total/tax line
            if (RegexHelper.IsMatch(description, @"(?:total|tax|subtotal|discount)", RegexOptions.IgnoreCase))
                continue;

            if (decimal.TryParse(match.Groups[2].Value.Replace(",", ""), out var price))
            {
                receipt.Items.Add(new ReceiptItem
                {
                    Description = description,
                    Price = price
                });
            }
        }

        return receipt;
    }

    /// <summary>
    /// Validates parsed data against a schema.
    /// </summary>
    public ValidationResult ValidateAgainstSchema(Dictionary<string, object?> data, DocumentSchema schema)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        foreach (var field in schema.RequiredFields)
        {
            if (!data.ContainsKey(field) || data[field] == null)
                errors.Add($"Required field '{field}' is missing");
        }

        foreach (var (fieldName, expectedType) in schema.FieldTypes)
        {
            if (data.TryGetValue(fieldName, out var value) && value != null)
            {
                if (!IsValidType(value, expectedType))
                    errors.Add($"Field '{fieldName}' has invalid type. Expected {expectedType}");
            }
        }

        foreach (var (fieldName, pattern) in schema.FieldPatterns)
        {
            if (data.TryGetValue(fieldName, out var value) && value != null)
            {
                if (!RegexHelper.IsMatch(value.ToString() ?? "", pattern))
                    warnings.Add($"Field '{fieldName}' does not match expected pattern");
            }
        }

        return new ValidationResult
        {
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings
        };
    }

    #endregion

    #region Private Methods

    private object? ExtractFieldValue(string text, FieldExtractionRule rule)
    {
        if (rule.Pattern != null)
        {
            var match = RegexHelper.Match(text, rule.Pattern, RegexOptions.IgnoreCase | RegexOptions.Multiline);
            if (match.Success)
            {
                var value = match.Groups.Count > 1 ? match.Groups[1].Value : match.Value;
                return ConvertValue(value.Trim(), rule.DataType);
            }
        }

        if (rule.Labels != null && rule.Labels.Count > 0)
        {
            foreach (var label in rule.Labels)
            {
                var value = ExtractFormField(text, label);
                if (value != null)
                    return ConvertValue(value, rule.DataType);
            }
        }

        return rule.DefaultValue;
    }

    private string? ExtractFormField(string text, string label)
    {
        // Try various patterns
        var patterns = new[]
        {
            $@"(?:{RegexHelper.Escape(label)})\s*:\s*(.+?)(?:\n|$)",
            $@"(?:{RegexHelper.Escape(label)})\s*=\s*(.+?)(?:\n|$)",
            $@"(?:{RegexHelper.Escape(label)})\s+(.+?)(?:\n|$)"
        };

        foreach (var pattern in patterns)
        {
            var match = RegexHelper.Match(text, pattern, RegexOptions.IgnoreCase);
            if (match.Success)
                return match.Groups[1].Value.Trim();
        }

        return null;
    }

    private object? ConvertValue(string value, DataType dataType)
    {
        return dataType switch
        {
            DataType.String => value,
            DataType.Integer => int.TryParse(value, out var i) ? i : null,
            DataType.Decimal => decimal.TryParse(value.Replace(",", ""), out var d) ? d : null,
            DataType.Boolean => bool.TryParse(value, out var b) ? b : null,
            DataType.DateTime => DateTime.TryParse(value, out var dt) ? dt : null,
            _ => value
        };
    }

    private bool IsValidType(object value, DataType expectedType)
    {
        return expectedType switch
        {
            DataType.String => value is string,
            DataType.Integer => value is int or long,
            DataType.Decimal => value is decimal or double or float,
            DataType.Boolean => value is bool,
            DataType.DateTime => value is DateTime,
            _ => true
        };
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the parser.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            // No managed resources to dispose
        }
        _disposed = true;
    }

    #endregion
}



