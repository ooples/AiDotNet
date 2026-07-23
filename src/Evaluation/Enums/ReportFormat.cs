namespace AiDotNet.Evaluation.Enums;

/// <summary>
/// Specifies the output format for evaluation reports.
/// </summary>
/// <remarks>
/// <para>
/// Evaluation results can be exported in various formats depending on the use case:
/// documentation, integration with other tools, or human readability.
/// </para>
/// <para>
/// <b>For Beginners:</b> After evaluating your model, you'll want to share or save the results.
/// Different formats serve different purposes:
/// <list type="bullet">
/// <item><b>Console:</b> Quick viewing during development</item>
/// <item><b>Markdown:</b> Documentation and reports</item>
/// <item><b>JSON:</b> Integration with other tools and programmatic access</item>
/// <item><b>CSV:</b> Spreadsheet analysis and comparison</item>
/// </list>
/// </para>
/// </remarks>
public enum ReportFormat
{
    /// <summary>
    /// Plain text format for console output with ASCII formatting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Simple text output that looks good in a terminal window.
    /// Uses ASCII characters for tables and separators.</para>
    /// <para><b>When to use:</b> Quick inspection during development, logging.</para>
    /// </remarks>
    Console = 0,

    /// <summary>
    /// Markdown format for documentation and GitHub/GitLab rendering.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Formatted text that renders nicely on GitHub, GitLab,
    /// Jupyter notebooks, and documentation sites. Tables use pipe characters.</para>
    /// <para><b>When to use:</b> READMEs, documentation, Jupyter notebooks, wikis.</para>
    /// </remarks>
    Markdown = 1,

    /// <summary>
    /// JSON format for programmatic access and tool integration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Structured data format that other programs can easily read.
    /// Perfect for storing results, sending to APIs, or processing with scripts.</para>
    /// <para><b>When to use:</b> APIs, data pipelines, automated testing, storage.</para>
    /// </remarks>
    Json = 2,

    /// <summary>
    /// LaTeX format for academic papers and formal reports.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Format used in academic writing. Tables are formatted
    /// for inclusion in research papers and technical documents.</para>
    /// <para><b>When to use:</b> Research papers, theses, formal technical reports.</para>
    /// </remarks>
    Latex = 3,

    /// <summary>
    /// HTML format for web display and interactive reports.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Web page format with styled tables and potential
    /// interactive elements. Can be viewed in any web browser.</para>
    /// <para><b>When to use:</b> Dashboards, web reports, email reports.</para>
    /// </remarks>
    Html = 4,

    /// <summary>
    /// CSV format for spreadsheet import and data analysis.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Comma-separated values that can be opened in Excel,
    /// Google Sheets, or any data analysis tool. Good for comparing multiple experiments.</para>
    /// <para><b>When to use:</b> Spreadsheet analysis, comparison across experiments.</para>
    /// </remarks>
    Csv = 5,

    /// <summary>
    /// XML format for enterprise integration and configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Structured format commonly used in enterprise systems.
    /// More verbose than JSON but with stricter schema support.</para>
    /// <para><b>When to use:</b> Enterprise integration, systems requiring XML.</para>
    /// </remarks>
    Xml = 6,

    /// <summary>
    /// YAML format for human-readable configuration and MLOps tools.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Human-readable format popular in MLOps tools like
    /// MLflow and Kubeflow. Easy to read and edit manually.</para>
    /// <para><b>When to use:</b> Configuration files, MLOps pipelines, human editing.</para>
    /// </remarks>
    Yaml = 7,

    /// <summary>
    /// Jupyter notebook cell output format with rich display.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Optimized for Jupyter notebooks with rich display
    /// capabilities including styled tables and embedded visualizations.</para>
    /// <para><b>When to use:</b> Jupyter/IPython notebooks, data science workflows.</para>
    /// </remarks>
    JupyterNotebook = 8,

    /// <summary>
    /// PDF format for formal distribution and archival.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Portable document format for sharing reports that
    /// should look the same on any device. Good for formal distribution.</para>
    /// <para><b>When to use:</b> Formal reports, stakeholder communication, archival.</para>
    /// </remarks>
    Pdf = 9
}
