using System.Text;
using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Template optimized for clear, structured instruction-following tasks.
/// </summary>
/// <remarks>
/// <para>
/// This template provides a clear structure for instructions, context, and constraints,
/// helping models understand and follow complex multi-step instructions accurately.
/// </para>
/// <para><b>For Beginners:</b> Structures instructions for clear AI execution.
///
/// Example:
/// <code>
/// var template = new InstructionFollowingTemplate()
///     .WithObjective("Summarize the meeting notes")
///     .AddInstruction("Read through all the notes")
///     .AddInstruction("Identify the main topics discussed")
///     .AddInstruction("Note any action items and their owners")
///     .AddConstraint("Keep summary under 200 words")
///     .AddConstraint("Use bullet points for action items");
///
/// var prompt = template.Format(new Dictionary&lt;string, string&gt;
/// {
///     { "input", meetingNotes }
/// });
/// </code>
///
/// Benefits:
/// - Clear separation of objective, steps, and constraints
/// - Numbered instructions for sequential execution
/// - Explicit constraints prevent unwanted behavior
/// - Input/output sections clearly marked
/// </para>
/// </remarks>
public class InstructionFollowingTemplate : PromptTemplateBase
{
    private readonly string? _objective;
    private readonly List<string> _instructions;
    private readonly List<string> _constraints;
    private readonly string? _inputDescription;
    private readonly string? _outputDescription;

    /// <summary>
    /// Initializes a new instance of the InstructionFollowingTemplate class.
    /// </summary>
    public InstructionFollowingTemplate()
        : base("{objective_section}{instructions_section}{constraints_section}{input_section}{output_section}")
    {
        _instructions = new List<string>();
        _constraints = new List<string>();
    }

    /// <summary>
    /// Initializes a new instance with specified components.
    /// </summary>
    private InstructionFollowingTemplate(
        string? objective,
        List<string> instructions,
        List<string> constraints,
        string? inputDescription,
        string? outputDescription)
        : base(BuildTemplate(objective, instructions, constraints, inputDescription, outputDescription))
    {
        _objective = objective;
        _instructions = instructions;
        _constraints = constraints;
        _inputDescription = inputDescription;
        _outputDescription = outputDescription;
    }

    /// <summary>
    /// Initializes a new instance with a custom template string.
    /// </summary>
    /// <param name="template">The custom template string.</param>
    public InstructionFollowingTemplate(string template)
        : base(template)
    {
        _instructions = new List<string>();
        _constraints = new List<string>();
    }

    private static string BuildTemplate(
        string? objective,
        List<string> instructions,
        List<string> constraints,
        string? inputDescription,
        string? outputDescription)
    {
        var sb = new StringBuilder();

        // Objective section
        if (!string.IsNullOrWhiteSpace(objective))
        {
            sb.AppendLine("## Objective");
            sb.AppendLine(objective);
            sb.AppendLine();
        }

        // Instructions section
        if (instructions.Count > 0)
        {
            sb.AppendLine("## Instructions");
            sb.AppendLine("Follow these steps in order:");
            sb.AppendLine();
            for (int i = 0; i < instructions.Count; i++)
            {
                sb.AppendLine($"{i + 1}. {instructions[i]}");
            }
            sb.AppendLine();
        }

        // Constraints section
        if (constraints.Count > 0)
        {
            sb.AppendLine("## Constraints");
            sb.AppendLine("Adhere to these requirements:");
            sb.AppendLine();
            foreach (var constraint in constraints)
            {
                sb.AppendLine($"- {constraint}");
            }
            sb.AppendLine();
        }

        // Input section
        sb.AppendLine("## Input");
        if (!string.IsNullOrWhiteSpace(inputDescription))
        {
            sb.AppendLine(inputDescription);
        }
        sb.AppendLine("{input}");
        sb.AppendLine();

        // Output section
        sb.AppendLine("## Output");
        if (!string.IsNullOrWhiteSpace(outputDescription))
        {
            sb.AppendLine(outputDescription);
        }

        return sb.ToString();
    }

    /// <summary>
    /// Sets the objective for the task.
    /// </summary>
    /// <param name="objective">The main objective or goal.</param>
    /// <returns>A new template with the objective set.</returns>
    public InstructionFollowingTemplate WithObjective(string objective)
    {
        return new InstructionFollowingTemplate(
            objective,
            new List<string>(_instructions),
            new List<string>(_constraints),
            _inputDescription,
            _outputDescription);
    }

    /// <summary>
    /// Adds an instruction step.
    /// </summary>
    /// <param name="instruction">The instruction to add.</param>
    /// <returns>A new template with the instruction added.</returns>
    public InstructionFollowingTemplate AddInstruction(string instruction)
    {
        var newInstructions = new List<string>(_instructions) { instruction };
        return new InstructionFollowingTemplate(
            _objective,
            newInstructions,
            new List<string>(_constraints),
            _inputDescription,
            _outputDescription);
    }

    /// <summary>
    /// Adds multiple instruction steps.
    /// </summary>
    /// <param name="instructions">The instructions to add.</param>
    /// <returns>A new template with the instructions added.</returns>
    public InstructionFollowingTemplate AddInstructions(params string[] instructions)
    {
        var newInstructions = new List<string>(_instructions);
        newInstructions.AddRange(instructions);
        return new InstructionFollowingTemplate(
            _objective,
            newInstructions,
            new List<string>(_constraints),
            _inputDescription,
            _outputDescription);
    }

    /// <summary>
    /// Adds a constraint.
    /// </summary>
    /// <param name="constraint">The constraint to add.</param>
    /// <returns>A new template with the constraint added.</returns>
    public InstructionFollowingTemplate AddConstraint(string constraint)
    {
        var newConstraints = new List<string>(_constraints) { constraint };
        return new InstructionFollowingTemplate(
            _objective,
            new List<string>(_instructions),
            newConstraints,
            _inputDescription,
            _outputDescription);
    }

    /// <summary>
    /// Adds multiple constraints.
    /// </summary>
    /// <param name="constraints">The constraints to add.</param>
    /// <returns>A new template with the constraints added.</returns>
    public InstructionFollowingTemplate AddConstraints(params string[] constraints)
    {
        var newConstraints = new List<string>(_constraints);
        newConstraints.AddRange(constraints);
        return new InstructionFollowingTemplate(
            _objective,
            new List<string>(_instructions),
            newConstraints,
            _inputDescription,
            _outputDescription);
    }

    /// <summary>
    /// Sets the input description.
    /// </summary>
    /// <param name="description">Description of expected input.</param>
    /// <returns>A new template with the input description set.</returns>
    public InstructionFollowingTemplate WithInputDescription(string description)
    {
        return new InstructionFollowingTemplate(
            _objective,
            new List<string>(_instructions),
            new List<string>(_constraints),
            description,
            _outputDescription);
    }

    /// <summary>
    /// Sets the output description.
    /// </summary>
    /// <param name="description">Description of expected output.</param>
    /// <returns>A new template with the output description set.</returns>
    public InstructionFollowingTemplate WithOutputDescription(string description)
    {
        return new InstructionFollowingTemplate(
            _objective,
            new List<string>(_instructions),
            new List<string>(_constraints),
            _inputDescription,
            description);
    }

    /// <summary>
    /// Formats the instruction-following template.
    /// </summary>
    protected override string FormatCore(Dictionary<string, string> variables)
    {
        var result = Template;

        foreach (var kvp in variables)
        {
            var placeholder = $"{{{kvp.Key}}}";
            result = result.Replace(placeholder, kvp.Value ?? string.Empty);
        }

        return result.Trim();
    }

    /// <summary>
    /// Creates a summarization template.
    /// </summary>
    /// <param name="maxLength">Maximum length of the summary.</param>
    public static InstructionFollowingTemplate Summarization(int maxLength = 200)
    {
        return new InstructionFollowingTemplate()
            .WithObjective("Create a concise summary of the provided content")
            .AddInstruction("Read and understand the full content")
            .AddInstruction("Identify the main points and key information")
            .AddInstruction("Write a clear, coherent summary")
            .AddConstraint($"Keep the summary under {maxLength} words")
            .AddConstraint("Maintain the original meaning and tone")
            .AddConstraint("Do not add information not present in the original");
    }

    /// <summary>
    /// Creates a translation template.
    /// </summary>
    /// <param name="targetLanguage">The target language for translation.</param>
    public static InstructionFollowingTemplate Translation(string targetLanguage)
    {
        return new InstructionFollowingTemplate()
            .WithObjective($"Translate the provided text to {targetLanguage}")
            .AddInstruction("Read and understand the source text completely")
            .AddInstruction("Translate the meaning, not just words")
            .AddInstruction("Preserve formatting and structure where possible")
            .AddConstraint("Maintain the original tone and style")
            .AddConstraint("Use natural, fluent language in the target language")
            .AddConstraint("Keep proper nouns unchanged unless they have standard translations");
    }

    /// <summary>
    /// Creates a classification template.
    /// </summary>
    /// <param name="categories">The available categories.</param>
    public static InstructionFollowingTemplate Classification(params string[] categories)
    {
        var template = new InstructionFollowingTemplate()
            .WithObjective("Classify the provided content into one of the specified categories")
            .AddInstruction("Analyze the content carefully")
            .AddInstruction("Consider which category best matches the content")
            .AddInstruction("Provide the classification with brief reasoning")
            .AddConstraint("Choose only from the provided categories")
            .AddConstraint("If uncertain, choose the most likely category");

        if (categories.Length > 0)
        {
            template = template.AddConstraint($"Available categories: {string.Join(", ", categories)}");
        }

        return template;
    }

    /// <summary>
    /// Creates a question-answering template.
    /// </summary>
    public static InstructionFollowingTemplate QuestionAnswering()
    {
        return new InstructionFollowingTemplate()
            .WithObjective("Answer the question based on the provided context")
            .AddInstruction("Read the context carefully")
            .AddInstruction("Understand what the question is asking")
            .AddInstruction("Find relevant information in the context")
            .AddInstruction("Formulate a clear, direct answer")
            .AddConstraint("Base your answer only on the provided context")
            .AddConstraint("If the answer is not in the context, say so")
            .AddConstraint("Be concise and direct");
    }

    /// <summary>
    /// Creates a builder for constructing instruction-following templates.
    /// </summary>
    public static InstructionFollowingBuilder Builder() => new();
}

/// <summary>
/// Builder for constructing instruction-following templates fluently.
/// </summary>
public class InstructionFollowingBuilder
{
    private string? _objective;
    private readonly List<string> _instructions = new();
    private readonly List<string> _constraints = new();
    private string? _inputDescription;
    private string? _outputDescription;

    /// <summary>
    /// Sets the objective.
    /// </summary>
    public InstructionFollowingBuilder WithObjective(string objective)
    {
        _objective = objective;
        return this;
    }

    /// <summary>
    /// Adds an instruction.
    /// </summary>
    public InstructionFollowingBuilder AddInstruction(string instruction)
    {
        _instructions.Add(instruction);
        return this;
    }

    /// <summary>
    /// Adds a constraint.
    /// </summary>
    public InstructionFollowingBuilder AddConstraint(string constraint)
    {
        _constraints.Add(constraint);
        return this;
    }

    /// <summary>
    /// Sets the input description.
    /// </summary>
    public InstructionFollowingBuilder WithInputDescription(string description)
    {
        _inputDescription = description;
        return this;
    }

    /// <summary>
    /// Sets the output description.
    /// </summary>
    public InstructionFollowingBuilder WithOutputDescription(string description)
    {
        _outputDescription = description;
        return this;
    }

    /// <summary>
    /// Builds the instruction-following template.
    /// </summary>
    public InstructionFollowingTemplate Build()
    {
        var template = new InstructionFollowingTemplate();

        if (_objective is not null && !string.IsNullOrWhiteSpace(_objective))
        {
            string objective = _objective;
            template = template.WithObjective(objective);
        }

        foreach (var instruction in _instructions)
        {
            template = template.AddInstruction(instruction);
        }

        foreach (var constraint in _constraints)
        {
            template = template.AddConstraint(constraint);
        }

        if (_inputDescription is not null && !string.IsNullOrWhiteSpace(_inputDescription))
        {
            string inputDescription = _inputDescription;
            template = template.WithInputDescription(inputDescription);
        }

        if (_outputDescription is not null && !string.IsNullOrWhiteSpace(_outputDescription))
        {
            string outputDescription = _outputDescription;
            template = template.WithOutputDescription(outputDescription);
        }

        return template;
    }
}
