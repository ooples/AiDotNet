using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering;
using AiDotNet.PromptEngineering.FewShot;
using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

/// <summary>
/// Integration tests for prompt engineering components working together.
/// Tests end-to-end workflows and component interactions.
/// </summary>
public class IntegrationTests
{
    #region Template Composition Integration

    [Fact]
    public void CompositeTemplate_WithConditional_WorksTogether()
    {
        var composite = CompositePromptTemplate.Builder()
            .Add("System: You are a helpful assistant.")
            .Add(new ConditionalPromptTemplate("{{#if context}}Context: {context}{{/if}}"))
            .Add("User: {question}")
            .Build();

        // With context
        var withContext = composite.Format(new Dictionary<string, string>
        {
            { "context", "We're discussing AI" },
            { "question", "What is machine learning?" }
        });

        Assert.Contains("System: You are a helpful assistant.", withContext);
        Assert.Contains("Context: We're discussing AI", withContext);
        Assert.Contains("User: What is machine learning?", withContext);

        // Without context
        var withoutContext = composite.Format(new Dictionary<string, string>
        {
            { "question", "What is machine learning?" }
        });

        Assert.Contains("System: You are a helpful assistant.", withoutContext);
        Assert.DoesNotContain("Context:", withoutContext);
        Assert.Contains("User: What is machine learning?", withoutContext);
    }

    [Fact]
    public void ChainOfThoughtTemplate_WithContextWindow_TruncatesCorrectly()
    {
        var task = "Solve this very complex mathematical problem";
        var cot = new ChainOfThoughtTemplate(task);

        var prompt = cot.Format(new Dictionary<string, string>
        {
            { "context", "Some background information" }
        });

        // Use context window manager to check if it fits
        var manager = new ContextWindowManager(1000);
        Assert.True(manager.FitsInWindow(prompt));
    }

    [Fact]
    public void RolePlayingTemplate_WithStructuredOutput_CombinesCorrectly()
    {
        // Use the role constructor (not template constructor) by providing expertise param
        var role = new RolePlayingTemplate("data analyst", expertise: null)
            .WithTask("Analyze the following dataset");
        var structured = new StructuredOutputTemplate(
            StructuredOutputTemplate.OutputFormat.Json,
            "{ \"insights\": [], \"recommendations\": [] }");

        var rolePrompt = role.Format(new Dictionary<string, string>
        {
            { "context", "Sales data for Q1 2024" }
        });

        var structuredPrompt = structured.Format(new Dictionary<string, string>
        {
            { "task", rolePrompt }
        });

        Assert.Contains("data analyst", structuredPrompt);
        Assert.Contains("json", structuredPrompt.ToLower());
    }

    #endregion

    #region Few-Shot with Templates Integration

    [Fact]
    public void FewShotPromptTemplate_WithExamples_FormatsCorrectly()
    {
        var selector = new FixedExampleSelector<double>();
        selector.AddExample(new FewShotExample { Input = "Hello", Output = "Hola" });
        selector.AddExample(new FewShotExample { Input = "Goodbye", Output = "Adios" });

        var template = new FewShotPromptTemplate<double>(
            "You are a translator. Translate English to Spanish.\n\n{examples}\n\nNow translate: {query}",
            selector,
            exampleCount: 2,
            exampleFormat: "Input: {input}\nOutput: {output}");

        var prompt = template.Format(new Dictionary<string, string>
        {
            { "query", "Good morning" }
        });

        Assert.Contains("Hello", prompt);
        Assert.Contains("Hola", prompt);
        Assert.Contains("Goodbye", prompt);
        Assert.Contains("Adios", prompt);
        Assert.Contains("Good morning", prompt);
    }

    [Fact]
    public void RandomExampleSelector_SelectsCorrectCount()
    {
        var selector = new RandomExampleSelector<double>(seed: 42);

        for (int i = 0; i < 10; i++)
        {
            selector.AddExample(new FewShotExample
            {
                Input = $"Input {i}",
                Output = $"Output {i}"
            });
        }

        var selected = selector.SelectExamples("test query", 3);

        Assert.Equal(3, selected.Count);
    }

    [Fact]
    public void FixedExampleSelector_ReturnsAllExamples()
    {
        var selector = new FixedExampleSelector<double>();

        selector.AddExample(new FewShotExample { Input = "A", Output = "1" });
        selector.AddExample(new FewShotExample { Input = "B", Output = "2" });
        selector.AddExample(new FewShotExample { Input = "C", Output = "3" });

        var selected = selector.SelectExamples("query", 5);

        // Should return all 3 since we only have 3 examples
        Assert.Equal(3, selected.Count);
    }

    #endregion

    #region Context Window Management Integration

    [Fact]
    public void ContextWindowManager_WithLongPrompt_SplitsCorrectly()
    {
        var manager = new ContextWindowManager(100, text => text.Length);

        var longPrompt = new string('x', 350);
        var chunks = manager.SplitIntoChunks(longPrompt);

        Assert.True(chunks.Count >= 4);
        foreach (var chunk in chunks)
        {
            Assert.True(manager.FitsInWindow(chunk));
        }
    }

    [Fact]
    public void ContextWindowManager_EstimatesTokensReasonably()
    {
        var manager = new ContextWindowManager(4096);

        var shortText = "Hello world";
        var longText = new string('x', 10000);

        var shortEstimate = manager.EstimateTokens(shortText);
        var longEstimate = manager.EstimateTokens(longText);

        Assert.True(shortEstimate < longEstimate);
        Assert.True(shortEstimate > 0);
    }

    [Fact]
    public void ContextWindowManager_ReservedTokens_AccountedFor()
    {
        var manager = new ContextWindowManager(100, text => text.Length);

        var text = new string('x', 80);

        Assert.True(manager.FitsInWindow(text, reservedTokens: 10)); // 80 + 10 = 90, fits
        Assert.False(manager.FitsInWindow(text, reservedTokens: 30)); // 80 + 30 = 110, doesn't fit
    }

    #endregion

    #region Template Chaining Integration

    [Fact]
    public void MultipleTemplates_ChainedTogether_ProducesValidOutput()
    {
        // Step 1: Create a role-playing prompt using role constructor
        var roleTemplate = new RolePlayingTemplate("expert software engineer", expertise: null)
            .WithTask("Review the following code");

        var rolePrompt = roleTemplate.Format(new Dictionary<string, string>
        {
            { "context", "Python code for data processing" }
        });

        // Step 2: Add chain of thought using question constructor
        var cotTemplate = new ChainOfThoughtTemplate("Analyze code quality", context: null);
        var cotPrompt = cotTemplate.Format(new Dictionary<string, string>());

        // Step 3: Request structured output
        var structuredTemplate = new StructuredOutputTemplate(
            StructuredOutputTemplate.OutputFormat.Json,
            "{ \"issues\": [], \"score\": 0, \"suggestions\": [] }");

        var finalPrompt = structuredTemplate.Format(new Dictionary<string, string>
        {
            { "task", cotPrompt }
        });

        // Verify all components are present
        Assert.Contains("Analyze code quality", finalPrompt);
        Assert.Contains("step", finalPrompt.ToLower());
        Assert.Contains("json", finalPrompt.ToLower());
    }

    [Fact]
    public void InstructionTemplate_WithConditionals_ProducesCorrectOutput()
    {
        var instructionTemplate = new InstructionFollowingTemplate()
            .AddInstruction("Read the input carefully")
            .AddInstruction("Identify key points")
            .AddInstruction("Provide a summary");

        var conditionalWrapper = new ConditionalPromptTemplate(
            "{{#if verbose}}Detailed analysis: {detailed}{{/if}}\n{instructions}");

        var instructionPrompt = instructionTemplate.Format(new Dictionary<string, string>
        {
            { "input", "Sample text to analyze" }
        });

        // Without verbose
        var simpleResult = conditionalWrapper.Format(new Dictionary<string, string>
        {
            { "instructions", instructionPrompt }
        });

        Assert.DoesNotContain("Detailed analysis:", simpleResult);
        Assert.Contains("Read the input carefully", simpleResult);

        // With verbose
        var verboseResult = conditionalWrapper.Format(new Dictionary<string, string>
        {
            { "verbose", "true" },
            { "detailed", "Extra information here" },
            { "instructions", instructionPrompt }
        });

        Assert.Contains("Detailed analysis:", verboseResult);
        Assert.Contains("Extra information here", verboseResult);
    }

    #endregion

    #region End-to-End Workflow Integration

    [Fact]
    public void CompleteWorkflow_QASystem_WorksEndToEnd()
    {
        // Simulate a QA system workflow

        // 1. Create few-shot examples using a selector
        var selector = new FixedExampleSelector<double>();
        selector.AddExample(new FewShotExample
        {
            Input = "What is the capital of France?",
            Output = "The capital of France is Paris."
        });
        selector.AddExample(new FewShotExample
        {
            Input = "What is 2 + 2?",
            Output = "2 + 2 equals 4."
        });

        // 2. Create template with example selector
        var fewShotTemplate = new FewShotPromptTemplate<double>(
            "You are a helpful assistant that answers questions accurately.\n\n{examples}\n\n{query}",
            selector,
            exampleCount: 2,
            exampleFormat: "Q: {input}\nA: {output}");

        // 3. Format with new query
        var prompt = fewShotTemplate.Format(new Dictionary<string, string>
        {
            { "query", "What is the largest planet?" }
        });

        // 4. Check with context window
        var contextManager = new ContextWindowManager(4096);
        Assert.True(contextManager.FitsInWindow(prompt));

        // 5. Verify structure
        Assert.Contains("helpful assistant", prompt);
        Assert.Contains("What is the capital of France?", prompt);
        Assert.Contains("What is the largest planet?", prompt);
    }

    [Fact]
    public void CompleteWorkflow_DataAnalysis_WorksEndToEnd()
    {
        // Simulate a data analysis workflow

        // 1. Role-playing setup using role constructor
        var roleTemplate = new RolePlayingTemplate("senior data scientist", expertise: null)
            .WithTask("Analyze the following dataset and provide insights");

        // 2. Add structured output requirement
        var outputSchema = @"{
            ""summary"": ""string"",
            ""trends"": [""string""],
            ""recommendations"": [""string""],
            ""confidence"": ""number""
        }";

        var structuredTemplate = new StructuredOutputTemplate(
            StructuredOutputTemplate.OutputFormat.Json,
            outputSchema);

        // 3. Create composite workflow
        var rolePrompt = roleTemplate.Format(new Dictionary<string, string>
        {
            { "context", "Sales data: Q1=$100K, Q2=$120K, Q3=$95K, Q4=$150K" }
        });

        var finalPrompt = structuredTemplate.Format(new Dictionary<string, string>
        {
            { "task", rolePrompt }
        });

        // 4. Verify completeness
        Assert.Contains("senior data scientist", finalPrompt);
        Assert.Contains("Sales data", finalPrompt);
        Assert.Contains("json", finalPrompt.ToLower());
        Assert.Contains("summary", finalPrompt);
    }

    [Fact]
    public void CompleteWorkflow_CodeReview_WorksEndToEnd()
    {
        // Simulate a code review workflow

        // 1. Instructions for review
        var instructionTemplate = new InstructionFollowingTemplate()
            .AddInstruction("Check for security vulnerabilities")
            .AddInstruction("Verify code follows best practices")
            .AddInstruction("Identify performance issues")
            .AddInstruction("Suggest improvements");

        // 2. Chain of thought for reasoning using question constructor
        var cotTemplate = new ChainOfThoughtTemplate("Review code quality step by step", context: null);

        // 3. Conditional for verbose mode
        var conditionalTemplate = new ConditionalPromptTemplate(
            "{{#if showExamples}}Example issues to look for: SQL injection, XSS, etc.{{/if}}\n{instructions}\n{cot}");

        // 4. Execute workflow
        var instructionPrompt = instructionTemplate.Format(new Dictionary<string, string>
        {
            { "input", "function getUserData(id) { return db.query('SELECT * FROM users WHERE id=' + id); }" }
        });

        var cotPrompt = cotTemplate.Format(new Dictionary<string, string>());

        var finalPrompt = conditionalTemplate.Format(new Dictionary<string, string>
        {
            { "showExamples", "true" },
            { "instructions", instructionPrompt },
            { "cot", cotPrompt }
        });

        // 5. Verify workflow
        Assert.Contains("security vulnerabilities", finalPrompt);
        Assert.Contains("step", finalPrompt.ToLower());
        Assert.Contains("SQL injection", finalPrompt);
    }

    #endregion

    #region Error Handling Integration

    [Fact]
    public void Integration_MissingRequiredVariable_ThrowsAppropriately()
    {
        var template = new SimplePromptTemplate("Hello {name}, your order {orderId} is ready");

        Assert.Throws<ArgumentException>(() => template.Format(new Dictionary<string, string>
        {
            { "name", "John" }
            // orderId is missing
        }));
    }

    [Fact]
    public void Integration_InvalidTemplateChain_HandlesGracefully()
    {
        var template1 = new SimplePromptTemplate("Step 1: {step1}");
        var template2 = new ConditionalPromptTemplate("{{#if result}}{result}{{/if}}");

        var step1Result = template1.Format(new Dictionary<string, string>
        {
            { "step1", "Completed" }
        });

        // Step 2 with empty result should not fail
        var finalResult = template2.Format(new Dictionary<string, string>
        {
            { "result", "" }
        });

        Assert.NotNull(finalResult);
    }

    #endregion
}
