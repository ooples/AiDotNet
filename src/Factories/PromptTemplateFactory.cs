using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Templates;

namespace AiDotNet.Factories;

/// <summary>
/// Factory for creating prompt template instances based on template type.
/// </summary>
/// <remarks>
/// <para>
/// This factory creates prompt templates based on the specified type, following the
/// factory pattern used throughout the AiDotNet library.
/// </para>
/// <para><b>For Beginners:</b> A factory that creates the right kind of template for you.
///
/// Example:
/// ```csharp
/// // Create a simple template
/// var simpleTemplate = PromptTemplateFactory.Create(
///     PromptTemplateType.Simple,
///     "Translate {text} to {language}"
/// );
///
/// // Create a chat template
/// var chatTemplate = PromptTemplateFactory.Create(
///     PromptTemplateType.Chat
/// ) as ChatPromptTemplate;
/// chatTemplate.AddSystemMessage("You are a helpful assistant");
/// ```
/// </para>
/// </remarks>
public static class PromptTemplateFactory
{
    /// <summary>
    /// Creates a prompt template of the specified type.
    /// </summary>
    /// <param name="templateType">The type of template to create.</param>
    /// <param name="template">The template string (not used for Chat type).</param>
    /// <param name="exampleCount">Number of examples for FewShot templates (default: 3).</param>
    /// <returns>A new prompt template instance.</returns>
    /// <exception cref="ArgumentException">Thrown when template type is not supported.</exception>
    public static IPromptTemplate Create(
        PromptTemplateType templateType,
        string? template = null,
        int exampleCount = 3)
    {
        return templateType switch
        {
            PromptTemplateType.Simple => CreateSimpleTemplate(template),
            PromptTemplateType.FewShot => throw new ArgumentException(
                "FewShot templates require an IFewShotExampleSelector<T>. " +
                "Use PromptTemplateFactory.Create<T>(PromptTemplateType.FewShot, template, exampleSelector, exampleCount).",
                nameof(templateType)),
            PromptTemplateType.Chat => CreateChatTemplate(),
            PromptTemplateType.ChainOfThought => CreateChainOfThoughtTemplate(template),
            PromptTemplateType.Tool => CreateSimpleTemplate(template), // Tools use simple for now
            PromptTemplateType.ReAct => CreateReActTemplate(template),
            PromptTemplateType.Optimized => CreateSimpleTemplate(template), // Start with simple
            _ => throw new ArgumentException($"Unsupported template type: {templateType}", nameof(templateType))
        };
    }

    /// <summary>
    /// Creates a prompt template of the specified type with a strongly-typed few-shot example selector.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the selector for scoring and similarity.</typeparam>
    /// <param name="templateType">The type of template to create.</param>
    /// <param name="template">The template string (not used for Chat type).</param>
    /// <param name="exampleSelector">Example selector for FewShot templates.</param>
    /// <param name="exampleCount">Number of examples for FewShot templates (default: 3).</param>
    /// <returns>A new prompt template instance.</returns>
    /// <exception cref="ArgumentException">Thrown when template type is not supported.</exception>
    public static IPromptTemplate Create<T>(
        PromptTemplateType templateType,
        string? template,
        IFewShotExampleSelector<T> exampleSelector,
        int exampleCount = 3)
    {
        if (templateType != PromptTemplateType.FewShot)
        {
            return Create(templateType, template, exampleCount);
        }

        return CreateFewShotTemplate(template, exampleSelector, exampleCount);
    }

    /// <summary>
    /// Creates a simple prompt template.
    /// </summary>
    private static SimplePromptTemplate CreateSimpleTemplate(string? template)
    {
        var resolvedTemplate = template ?? throw new ArgumentException("Template string is required for Simple template type.", nameof(template));

        if (string.IsNullOrWhiteSpace(resolvedTemplate))
        {
            throw new ArgumentException("Template string is required for Simple template type.", nameof(template));
        }

        return new SimplePromptTemplate(resolvedTemplate);
    }

    /// <summary>
    /// Creates a few-shot prompt template.
    /// </summary>
    private static FewShotPromptTemplate<T> CreateFewShotTemplate<T>(
        string? template,
        IFewShotExampleSelector<T> exampleSelector,
        int exampleCount)
    {
        var resolvedTemplate = template ?? throw new ArgumentException("Template string is required for FewShot template type.", nameof(template));

        if (string.IsNullOrWhiteSpace(resolvedTemplate))
        {
            throw new ArgumentException("Template string is required for FewShot template type.", nameof(template));
        }

        if (exampleSelector == null)
            throw new ArgumentNullException(nameof(exampleSelector), "Example selector is required for FewShot template type.");

        return new FewShotPromptTemplate<T>(resolvedTemplate, exampleSelector, exampleCount);
    }

    /// <summary>
    /// Creates a chat prompt template.
    /// </summary>
    private static ChatPromptTemplate CreateChatTemplate()
    {
        return new ChatPromptTemplate();
    }

    /// <summary>
    /// Creates a chain-of-thought prompt template.
    /// </summary>
    private static SimplePromptTemplate CreateChainOfThoughtTemplate(string? template)
    {
        var baseTemplate = template ?? "{input}";
        var cotPrompt = baseTemplate + "\n\nLet's think step by step:";
        return new SimplePromptTemplate(cotPrompt);
    }

    /// <summary>
    /// Creates a ReAct (Reasoning + Acting) prompt template.
    /// </summary>
    private static SimplePromptTemplate CreateReActTemplate(string? template)
    {
        var baseTemplate = template ?? "Answer the following question by alternating between Thought and Action steps.";
        var reactPrompt = baseTemplate + "\n\nQuestion: {question}\n\nThought:";
        return new SimplePromptTemplate(reactPrompt);
    }
}
