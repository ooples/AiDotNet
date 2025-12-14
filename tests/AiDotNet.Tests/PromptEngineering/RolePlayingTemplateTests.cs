using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

public class RolePlayingTemplateTests
{
    [Fact]
    public void Constructor_WithRole_CreatesTemplate()
    {
        var template = new RolePlayingTemplate("Software Engineer");

        Assert.NotNull(template);
        Assert.Contains("Software Engineer", template.Template);
    }

    [Fact]
    public void Constructor_WithExpertise_IncludesExpertise()
    {
        var template = new RolePlayingTemplate(
            "Developer",
            new[] { "C#", "Python", "JavaScript" });

        Assert.NotNull(template);
        Assert.Contains("C#", template.Template);
        Assert.Contains("Python", template.Template);
        Assert.Contains("JavaScript", template.Template);
    }

    [Fact]
    public void Constructor_WithPersonality_IncludesPersonality()
    {
        var template = new RolePlayingTemplate(
            "Teacher",
            personality: "Patient and encouraging");

        Assert.NotNull(template);
        Assert.Contains("Patient and encouraging", template.Template);
    }

    [Fact]
    public void Constructor_WithConstraints_IncludesConstraints()
    {
        var template = new RolePlayingTemplate(
            "Analyst",
            constraints: "Only use verified data");

        Assert.NotNull(template);
        Assert.Contains("Only use verified data", template.Template);
    }

    [Fact]
    public void Constructor_WithCustomTemplate_UsesCustomTemplate()
    {
        var customTemplate = "You are a {role}";
        var template = new RolePlayingTemplate(customTemplate);

        Assert.Equal(customTemplate, template.Template);
    }

    [Fact]
    public void WithTask_SetsTask()
    {
        var template = new RolePlayingTemplate("Engineer")
            .WithTask("Review this code");

        Assert.NotNull(template);
        Assert.Contains("Review this code", template.Template);
    }

    [Fact]
    public void Format_ReplacesVariables()
    {
        var template = new RolePlayingTemplate("Expert");
        var variables = new Dictionary<string, string>
        {
            ["context"] = "We are building a web app",
            ["task"] = "Help with architecture"
        };

        var result = template.Format(variables);

        Assert.Contains("We are building a web app", result);
    }

    [Fact]
    public void TechnicalExpert_CreatesExpertTemplate()
    {
        var template = RolePlayingTemplate.TechnicalExpert("C#");

        Assert.NotNull(template);
        Assert.Contains("C#", template.Template);
        Assert.Contains("Expert", template.Template);
    }

    [Fact]
    public void TechnicalExpert_WithSeniorityLevel_IncludesLevel()
    {
        var template = RolePlayingTemplate.TechnicalExpert("Python", "Principal");

        Assert.NotNull(template);
        Assert.Contains("Principal", template.Template);
    }

    [Fact]
    public void BusinessAnalyst_CreatesAnalystTemplate()
    {
        var template = RolePlayingTemplate.BusinessAnalyst();

        Assert.NotNull(template);
        Assert.Contains("Business Analyst", template.Template);
    }

    [Fact]
    public void BusinessAnalyst_WithIndustry_IncludesIndustry()
    {
        var template = RolePlayingTemplate.BusinessAnalyst("Healthcare");

        Assert.NotNull(template);
        Assert.Contains("Healthcare", template.Template);
    }

    [Fact]
    public void CreativeWriter_CreatesWriterTemplate()
    {
        var template = RolePlayingTemplate.CreativeWriter();

        Assert.NotNull(template);
        Assert.Contains("Creative Writer", template.Template);
    }

    [Fact]
    public void CreativeWriter_WithStyle_IncludesStyle()
    {
        var template = RolePlayingTemplate.CreativeWriter("Science Fiction");

        Assert.NotNull(template);
        Assert.Contains("Science Fiction", template.Template);
    }

    [Fact]
    public void Teacher_CreatesTeacherTemplate()
    {
        var template = RolePlayingTemplate.Teacher("Mathematics");

        Assert.NotNull(template);
        Assert.Contains("Mathematics", template.Template);
        Assert.Contains("Teacher", template.Template);
    }

    [Fact]
    public void Teacher_WithStudentLevel_IncludesLevel()
    {
        var template = RolePlayingTemplate.Teacher("Physics", "Advanced");

        Assert.NotNull(template);
        Assert.Contains("Advanced", template.Template);
    }

    [Fact]
    public void CodeReviewer_CreatesReviewerTemplate()
    {
        var template = RolePlayingTemplate.CodeReviewer("C#", "Python");

        Assert.NotNull(template);
        Assert.Contains("Code Reviewer", template.Template);
        Assert.Contains("C#", template.Template);
        Assert.Contains("Python", template.Template);
    }

    [Fact]
    public void CodeReviewer_WithoutLanguages_UsesDefault()
    {
        var template = RolePlayingTemplate.CodeReviewer();

        Assert.NotNull(template);
        Assert.Contains("multiple languages", template.Template);
    }

    [Fact]
    public void Builder_CreatesTemplateWithRole()
    {
        var template = RolePlayingTemplate.Builder()
            .AsRole("Data Scientist")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Data Scientist", template.Template);
    }

    [Fact]
    public void Builder_WithExpertise_IncludesExpertise()
    {
        var template = RolePlayingTemplate.Builder()
            .AsRole("Analyst")
            .WithExpertise("Data Analysis")
            .WithExpertise("Statistics")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Data Analysis", template.Template);
        Assert.Contains("Statistics", template.Template);
    }

    [Fact]
    public void Builder_WithExpertiseArray_IncludesAllExpertise()
    {
        var template = RolePlayingTemplate.Builder()
            .AsRole("Engineer")
            .WithExpertise("Frontend", "Backend", "DevOps")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Frontend", template.Template);
        Assert.Contains("Backend", template.Template);
        Assert.Contains("DevOps", template.Template);
    }

    [Fact]
    public void Builder_WithPersonality_IncludesPersonality()
    {
        var template = RolePlayingTemplate.Builder()
            .AsRole("Coach")
            .WithPersonality("Motivating and supportive")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Motivating and supportive", template.Template);
    }

    [Fact]
    public void Builder_WithConstraints_IncludesConstraints()
    {
        var template = RolePlayingTemplate.Builder()
            .AsRole("Advisor")
            .WithConstraints("Provide factual information only")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Provide factual information only", template.Template);
    }

    [Fact]
    public void Builder_WithTask_IncludesTask()
    {
        var template = RolePlayingTemplate.Builder()
            .AsRole("Assistant")
            .WithTask("Help with scheduling")
            .Build();

        Assert.NotNull(template);
        Assert.Contains("Help with scheduling", template.Template);
    }

    [Fact]
    public void Builder_WithAttribute_DoesNotThrow()
    {
        var template = RolePlayingTemplate.Builder()
            .AsRole("Helper")
            .WithAttribute("specialty", "debugging")
            .Build();

        Assert.NotNull(template);
    }

    [Fact]
    public void Template_ContainsYouAre()
    {
        var template = new RolePlayingTemplate("Expert");

        Assert.Contains("You are", template.Template);
    }

    [Fact]
    public void Template_ContainsTaskPlaceholder()
    {
        var template = new RolePlayingTemplate("Helper");

        Assert.Contains("task", template.Template.ToLower());
    }
}
