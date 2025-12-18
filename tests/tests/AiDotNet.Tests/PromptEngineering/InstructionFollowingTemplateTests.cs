using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

public class InstructionFollowingTemplateTests
{
    [Fact]
    public void Constructor_Default_CreatesTemplate()
    {
        var template = new InstructionFollowingTemplate();

        Assert.NotNull(template);
    }

    [Fact]
    public void Constructor_WithCustomTemplate_UsesCustomTemplate()
    {
        var customTemplate = "Custom instruction: {input}";
        var template = new InstructionFollowingTemplate(customTemplate);

        Assert.Equal(customTemplate, template.Template);
    }

    [Fact]
    public void WithObjective_SetsObjective()
    {
        var template = new InstructionFollowingTemplate()
            .WithObjective("Summarize the document");

        Assert.NotNull(template);
        Assert.Contains("Summarize the document", template.Template);
    }

    [Fact]
    public void AddInstruction_AddsInstruction()
    {
        var template = new InstructionFollowingTemplate()
            .AddInstruction("Read the document carefully");

        Assert.NotNull(template);
        Assert.Contains("Read the document carefully", template.Template);
    }

    [Fact]
    public void AddInstructions_AddsMultipleInstructions()
    {
        var template = new InstructionFollowingTemplate()
            .AddInstructions("Step 1", "Step 2", "Step 3");

        Assert.NotNull(template);
        Assert.Contains("Step 1", template.Template);
        Assert.Contains("Step 2", template.Template);
        Assert.Contains("Step 3", template.Template);
    }

    [Fact]
    public void AddConstraint_AddsConstraint()
    {
        var template = new InstructionFollowingTemplate()
            .AddConstraint("Keep under 200 words");

        Assert.NotNull(template);
        Assert.Contains("Keep under 200 words", template.Template);
    }

    [Fact]
    public void AddConstraints_AddsMultipleConstraints()
    {
        var template = new InstructionFollowingTemplate()
            .AddConstraints("Be concise", "Use bullet points", "No jargon");

        Assert.NotNull(template);
        Assert.Contains("Be concise", template.Template);
        Assert.Contains("Use bullet points", template.Template);
        Assert.Contains("No jargon", template.Template);
    }

    [Fact]
    public void WithInputDescription_SetsInputDescription()
    {
        var template = new InstructionFollowingTemplate()
            .WithInputDescription("The meeting notes to summarize");

        Assert.NotNull(template);
        Assert.Contains("meeting notes", template.Template);
    }

    [Fact]
    public void WithOutputDescription_SetsOutputDescription()
    {
        var template = new InstructionFollowingTemplate()
            .WithOutputDescription("A bullet-point summary");

        Assert.NotNull(template);
        Assert.Contains("bullet-point summary", template.Template);
    }

    [Fact]
    public void Format_ReplacesInputVariable()
    {
        var template = new InstructionFollowingTemplate()
            .WithObjective("Summarize");
        var variables = new Dictionary<string, string>
        {
            ["input"] = "This is the document content."
        };

        var result = template.Format(variables);

        Assert.Contains("This is the document content.", result);
    }

    [Fact]
    public void Summarization_CreatesSummarizationTemplate()
    {
        var template = InstructionFollowingTemplate.Summarization();

        Assert.NotNull(template);
        Assert.Contains("summary", template.Template.ToLower());
    }

    [Fact]
    public void Summarization_WithMaxLength_IncludesMaxLength()
    {
        var template = InstructionFollowingTemplate.Summarization(100);

        Assert.NotNull(template);
        Assert.Contains("100", template.Template);
    }

    [Fact]
    public void Translation_CreatesTranslationTemplate()
    {
        var template = InstructionFollowingTemplate.Translation("Spanish");

        Assert.NotNull(template);
        Assert.Contains("Spanish", template.Template);
        Assert.Contains("Translate", template.Template);
    }

    [Fact]
    public void Classification_CreatesClassificationTemplate()
    {
        var template = InstructionFollowingTemplate.Classification("Positive", "Negative", "Neutral");

        Assert.NotNull(template);
        Assert.Contains("Positive", template.Template);
        Assert.Contains("Negative", template.Template);
        Assert.Contains("Neutral", template.Template);
    }

    [Fact]
    public void Classification_WithNoCategories_CreatesTemplate()
    {
        var template = InstructionFollowingTemplate.Classification();

        Assert.NotNull(template);
        Assert.Contains("Classify", template.Template);
    }

    [Fact]
    public void QuestionAnswering_CreatesQATemplate()
    {
        var template = InstructionFollowingTemplate.QuestionAnswering();

        Assert.NotNull(template);
        Assert.Contains("Answer", template.Template);
        Assert.Contains("question", template.Template.ToLower());
    }

    [Fact]
    public void Builder_CreatesTemplateWithObjective()
    {
        var template = InstructionFollowingTemplate.Builder()
            .WithObjective("Extract key points")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Extract key points", template.Template);
    }

    [Fact]
    public void Builder_AddInstruction_AddsInstruction()
    {
        var template = InstructionFollowingTemplate.Builder()
            .AddInstruction("First, read carefully")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("First, read carefully", template.Template);
    }

    [Fact]
    public void Builder_AddConstraint_AddsConstraint()
    {
        var template = InstructionFollowingTemplate.Builder()
            .AddConstraint("Be accurate")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Be accurate", template.Template);
    }

    [Fact]
    public void Builder_WithInputDescription_SetsDescription()
    {
        var template = InstructionFollowingTemplate.Builder()
            .WithInputDescription("Raw data file")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Raw data file", template.Template);
    }

    [Fact]
    public void Builder_WithOutputDescription_SetsDescription()
    {
        var template = InstructionFollowingTemplate.Builder()
            .WithOutputDescription("Formatted report")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Formatted report", template.Template);
    }

    [Fact]
    public void Builder_CompleteExample_WorksCorrectly()
    {
        var template = InstructionFollowingTemplate.Builder()
            .WithObjective("Summarize meeting notes")
            .AddInstruction("Read through all notes")
            .AddInstruction("Identify main topics")
            .AddInstruction("Note action items")
            .AddConstraint("Keep under 200 words")
            .AddConstraint("Use bullet points")
            .WithInputDescription("Meeting notes from project kickoff")
            .WithOutputDescription("Executive summary")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Summarize meeting notes", template.Template);
        Assert.Contains("Read through all notes", template.Template);
        Assert.Contains("Identify main topics", template.Template);
        Assert.Contains("action items", template.Template);
        Assert.Contains("200 words", template.Template);
        Assert.Contains("bullet points", template.Template);
    }

    [Fact]
    public void Template_ContainsInstructionsSection()
    {
        var template = new InstructionFollowingTemplate()
            .AddInstruction("Test instruction");

        Assert.Contains("Instructions", template.Template);
    }

    [Fact]
    public void Template_ContainsConstraintsSection()
    {
        var template = new InstructionFollowingTemplate()
            .AddConstraint("Test constraint");

        Assert.Contains("Constraints", template.Template);
    }

    [Fact]
    public void Template_ContainsInputSection()
    {
        var template = new InstructionFollowingTemplate()
            .WithObjective("Test");

        Assert.Contains("Input", template.Template);
    }

    [Fact]
    public void Template_ContainsOutputSection()
    {
        var template = new InstructionFollowingTemplate()
            .WithObjective("Test");

        Assert.Contains("Output", template.Template);
    }

    [Fact]
    public void FluentChaining_WorksCorrectly()
    {
        var template = new InstructionFollowingTemplate()
            .WithObjective("Objective")
            .AddInstruction("Instruction 1")
            .AddInstruction("Instruction 2")
            .AddConstraint("Constraint 1")
            .AddConstraint("Constraint 2")
            .WithInputDescription("Input desc")
            .WithOutputDescription("Output desc");

        Assert.NotNull(template);
        Assert.Contains("Objective", template.Template);
        Assert.Contains("Instruction 1", template.Template);
        Assert.Contains("Instruction 2", template.Template);
        Assert.Contains("Constraint 1", template.Template);
        Assert.Contains("Constraint 2", template.Template);
    }

    [Fact]
    public void Builder_WithEmptyObjective_HandlesGracefully()
    {
        var template = InstructionFollowingTemplate.Builder()
            .WithObjective("")
            .Build();

        Assert.NotNull(template);
    }

    [Fact]
    public void Builder_WithWhitespaceObjective_HandlesGracefully()
    {
        var template = InstructionFollowingTemplate.Builder()
            .WithObjective("   ")
            .Build();

        Assert.NotNull(template);
    }
}
