namespace AiDotNet.Document;

/// <summary>
/// Types of form fields that can be detected.
/// </summary>
public enum FormFieldType
{
    /// <summary>
    /// A text input field.
    /// </summary>
    TextInput,

    /// <summary>
    /// A checkbox field.
    /// </summary>
    Checkbox,

    /// <summary>
    /// A radio button field.
    /// </summary>
    RadioButton,

    /// <summary>
    /// A dropdown/select field.
    /// </summary>
    Dropdown,

    /// <summary>
    /// A date field.
    /// </summary>
    DateField,

    /// <summary>
    /// A signature field.
    /// </summary>
    SignatureField,

    /// <summary>
    /// A numeric input field.
    /// </summary>
    NumericInput,

    /// <summary>
    /// A multi-line text area.
    /// </summary>
    TextArea,

    /// <summary>
    /// Other or unrecognized field type.
    /// </summary>
    Other
}
