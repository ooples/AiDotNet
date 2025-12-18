using System.Text;
using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Template that creates persona-based prompts for role-playing scenarios.
/// </summary>
/// <remarks>
/// <para>
/// This template helps models adopt specific personas, expertise levels,
/// and communication styles for more targeted and consistent responses.
/// </para>
/// <para><b>For Beginners:</b> Makes the AI adopt a specific character or expert role.
///
/// Example:
/// <code>
/// var template = new RolePlayingTemplate(
///     role: "Senior Software Architect",
///     expertise: new[] { "System Design", "Cloud Architecture", "Microservices" },
///     personality: "Professional but approachable, explains complex concepts clearly"
/// );
///
/// var prompt = template.WithTask("Review this system design")
///     .Format(new Dictionary&lt;string, string&gt;
///     {
///         { "context", "Building a scalable e-commerce platform" }
///     });
/// </code>
///
/// Benefits:
/// - Consistent expertise level in responses
/// - Domain-specific language and knowledge
/// - Appropriate communication style
/// - Better task-relevant responses
/// </para>
/// </remarks>
public class RolePlayingTemplate : PromptTemplateBase
{
    private readonly string _role;
    private readonly IReadOnlyList<string> _expertise;
    private readonly string? _personality;
    private readonly string? _constraints;

    /// <summary>
    /// Initializes a new instance of the RolePlayingTemplate class.
    /// </summary>
    /// <param name="role">The role or persona to adopt.</param>
    /// <param name="expertise">Areas of expertise for the role.</param>
    /// <param name="personality">Optional personality traits and communication style.</param>
    /// <param name="constraints">Optional constraints or limitations for the role.</param>
    public RolePlayingTemplate(
        string role,
        IEnumerable<string>? expertise = null,
        string? personality = null,
        string? constraints = null)
        : base(BuildTemplate(role, expertise, personality, constraints, null))
    {
        _role = role ?? "Assistant";
        _expertise = expertise?.ToList().AsReadOnly() ?? new List<string>().AsReadOnly();
        _personality = personality;
        _constraints = constraints;
    }

    /// <summary>
    /// Initializes a new instance with a custom template string.
    /// </summary>
    /// <param name="template">The custom template string.</param>
    public RolePlayingTemplate(string template)
        : base(template)
    {
        _role = "Custom";
        _expertise = new List<string>().AsReadOnly();
    }

    private static string BuildTemplate(
        string role,
        IEnumerable<string>? expertise,
        string? personality,
        string? constraints,
        string? task)
    {
        var sb = new StringBuilder();

        // Role definition
        sb.AppendLine($"You are a {role}.");
        sb.AppendLine();

        // Expertise areas
        var expertiseList = expertise?.ToList() ?? new List<string>();
        if (expertiseList.Count > 0)
        {
            sb.AppendLine("Your areas of expertise include:");
            foreach (var area in expertiseList)
            {
                sb.AppendLine($"- {area}");
            }
            sb.AppendLine();
        }

        // Personality and style
        if (!string.IsNullOrWhiteSpace(personality))
        {
            sb.AppendLine($"Communication style: {personality}");
            sb.AppendLine();
        }

        // Constraints
        if (!string.IsNullOrWhiteSpace(constraints))
        {
            sb.AppendLine("Important constraints:");
            sb.AppendLine(constraints);
            sb.AppendLine();
        }

        // Task section
        if (!string.IsNullOrWhiteSpace(task))
        {
            sb.AppendLine($"Your task: {task}");
        }
        else
        {
            sb.AppendLine("Your task: {task}");
        }
        sb.AppendLine();

        // Context placeholder
        sb.AppendLine("{context}");

        return sb.ToString();
    }

    /// <summary>
    /// Sets the task for the persona to perform.
    /// </summary>
    /// <param name="task">The task description.</param>
    /// <returns>A new template with the task set.</returns>
    public RolePlayingTemplate WithTask(string task)
    {
        return new RolePlayingTemplate(
            BuildTemplate(_role, _expertise, _personality, _constraints, task));
    }

    /// <summary>
    /// Formats the role-playing template.
    /// </summary>
    protected override string FormatCore(Dictionary<string, string> variables)
    {
        var result = Template;

        foreach (var kvp in variables)
        {
            var placeholder = $"{{{kvp.Key}}}";
            result = result.Replace(placeholder, kvp.Value ?? string.Empty);
        }

        // Clean up empty placeholders
        result = result.Replace("{context}", "");
        result = result.Replace("{task}", "");

        return result.Trim();
    }

    /// <summary>
    /// Creates a technical expert persona.
    /// </summary>
    /// <param name="domain">The technical domain.</param>
    /// <param name="seniorityLevel">The seniority level (Junior, Mid, Senior, Principal).</param>
    public static RolePlayingTemplate TechnicalExpert(string domain, string seniorityLevel = "Senior")
    {
        return new RolePlayingTemplate(
            role: $"{seniorityLevel} {domain} Expert",
            expertise: new[]
            {
                $"{domain} best practices",
                "Problem solving and debugging",
                "Code review and architecture",
                "Performance optimization"
            },
            personality: "Technical, precise, explains concepts clearly with examples",
            constraints: "Provide accurate, production-ready advice. Acknowledge when uncertain."
        );
    }

    /// <summary>
    /// Creates a business analyst persona.
    /// </summary>
    /// <param name="industry">Optional specific industry focus.</param>
    public static RolePlayingTemplate BusinessAnalyst(string? industry = null)
    {
        var role = string.IsNullOrWhiteSpace(industry)
            ? "Business Analyst"
            : $"Business Analyst specializing in {industry}";

        return new RolePlayingTemplate(
            role: role,
            expertise: new[]
            {
                "Requirements gathering and analysis",
                "Process optimization",
                "Stakeholder communication",
                "Data-driven decision making"
            },
            personality: "Professional, analytical, communicates complex ideas simply",
            constraints: "Focus on business value and practical outcomes."
        );
    }

    /// <summary>
    /// Creates a creative writer persona.
    /// </summary>
    /// <param name="style">The writing style or genre.</param>
    public static RolePlayingTemplate CreativeWriter(string style = "General")
    {
        return new RolePlayingTemplate(
            role: $"Creative Writer with expertise in {style}",
            expertise: new[]
            {
                "Storytelling and narrative structure",
                "Character development",
                "Dialogue writing",
                "Genre conventions"
            },
            personality: "Creative, engaging, adapts tone to the content",
            constraints: "Create original, engaging content appropriate for the audience."
        );
    }

    /// <summary>
    /// Creates a teacher/educator persona.
    /// </summary>
    /// <param name="subject">The subject to teach.</param>
    /// <param name="studentLevel">The student level (Beginner, Intermediate, Advanced).</param>
    public static RolePlayingTemplate Teacher(string subject, string studentLevel = "Beginner")
    {
        return new RolePlayingTemplate(
            role: $"{subject} Teacher for {studentLevel} students",
            expertise: new[]
            {
                $"{subject} fundamentals and advanced concepts",
                "Pedagogy and learning techniques",
                "Creating examples and exercises",
                "Assessing understanding"
            },
            personality: "Patient, encouraging, breaks down complex topics into digestible parts",
            constraints: $"Adapt explanations for {studentLevel} level. Use analogies and examples."
        );
    }

    /// <summary>
    /// Creates a code reviewer persona.
    /// </summary>
    /// <param name="languages">Programming languages to focus on.</param>
    public static RolePlayingTemplate CodeReviewer(params string[] languages)
    {
        var langList = languages.Length > 0 ? string.Join(", ", languages) : "multiple languages";

        return new RolePlayingTemplate(
            role: "Senior Code Reviewer",
            expertise: new[]
            {
                $"Code quality in {langList}",
                "Design patterns and best practices",
                "Security vulnerabilities",
                "Performance optimization",
                "Clean code principles"
            },
            personality: "Constructive, thorough, provides actionable feedback",
            constraints: "Be specific about issues. Suggest improvements with examples. Acknowledge good code."
        );
    }

    /// <summary>
    /// Creates a builder for constructing role-playing templates.
    /// </summary>
    public static RolePlayingBuilder Builder() => new();
}

/// <summary>
/// Builder for constructing role-playing templates fluently.
/// </summary>
public class RolePlayingBuilder
{
    private string _role = "Assistant";
    private readonly List<string> _expertise = new();
    private string? _personality;
    private string? _constraints;
    private string? _task;
    private readonly Dictionary<string, string> _customAttributes = new();

    /// <summary>
    /// Sets the role or persona.
    /// </summary>
    public RolePlayingBuilder AsRole(string role)
    {
        _role = role;
        return this;
    }

    /// <summary>
    /// Adds an area of expertise.
    /// </summary>
    public RolePlayingBuilder WithExpertise(string expertise)
    {
        _expertise.Add(expertise);
        return this;
    }

    /// <summary>
    /// Adds multiple areas of expertise.
    /// </summary>
    public RolePlayingBuilder WithExpertise(params string[] expertise)
    {
        _expertise.AddRange(expertise);
        return this;
    }

    /// <summary>
    /// Sets the personality and communication style.
    /// </summary>
    public RolePlayingBuilder WithPersonality(string personality)
    {
        _personality = personality;
        return this;
    }

    /// <summary>
    /// Sets constraints for the role.
    /// </summary>
    public RolePlayingBuilder WithConstraints(string constraints)
    {
        _constraints = constraints;
        return this;
    }

    /// <summary>
    /// Sets the task to perform.
    /// </summary>
    public RolePlayingBuilder WithTask(string task)
    {
        _task = task;
        return this;
    }

    /// <summary>
    /// Adds a custom attribute to the persona.
    /// </summary>
    public RolePlayingBuilder WithAttribute(string name, string value)
    {
        _customAttributes[name] = value;
        return this;
    }

    /// <summary>
    /// Builds the role-playing template.
    /// </summary>
    public RolePlayingTemplate Build()
    {
        var template = new RolePlayingTemplate(_role, _expertise, _personality, _constraints);
        if (_task is not null && !string.IsNullOrWhiteSpace(_task))
        {
            string task = _task;
            return template.WithTask(task);
        }
        return template;
    }
}
