namespace AiDotNet.Enums;

/// <summary>
/// Defines the large language model (LLM) providers available for AI agent assistance during model building and inference.
/// </summary>
/// <remarks>
/// <para>
/// This enum specifies which LLM provider to use when enabling agent assistance in the AiModelBuilder.
/// Different providers offer different models with varying capabilities, pricing, and performance characteristics.
/// The selected provider determines which API will be called for agent operations such as model selection,
/// hyperparameter tuning, and conversational assistance.
/// </para>
/// <para><b>For Beginners:</b> This enum lets you choose which AI company's language model to use for helping
/// build your machine learning models.
///
/// Think of these as different AI assistants you can hire:
/// - **OpenAI**: Created GPT-3.5 and GPT-4, known for strong general reasoning
/// - **Anthropic**: Created Claude, designed to be helpful, harmless, and honest
/// - **Azure OpenAI**: Microsoft's enterprise version of OpenAI models with added security
///
/// Each provider requires an API key (like a password) to use their services. You would choose based on:
/// - Which service you already have an account with
/// - Pricing and rate limits
/// - Data privacy requirements (Azure OpenAI keeps data in your region)
/// - Specific model capabilities you need
///
/// For example, if your company uses Microsoft Azure, you might choose AzureOpenAI to keep
/// everything within your existing cloud infrastructure and compliance policies.
/// </para>
/// </remarks>
public enum LLMProvider
{
    /// <summary>
    /// OpenAI's GPT family of models including GPT-3.5, GPT-4, GPT-4-turbo, and GPT-4o.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This option uses OpenAI's language models, which are among the most
    /// powerful and widely-used AI models available. OpenAI created ChatGPT and provides API access
    /// to their models for developers.
    ///
    /// Key characteristics:
    /// - Strong performance on reasoning and analysis tasks
    /// - Well-suited for model selection and hyperparameter recommendations
    /// - Requires an OpenAI API key (obtain from platform.openai.com)
    /// - Pay-per-use pricing based on tokens processed
    ///
    /// Use this when: You want state-of-the-art AI assistance and have an OpenAI account.
    /// </para>
    /// </remarks>
    OpenAI,

    /// <summary>
    /// Anthropic's Claude family of models including Claude 2, Claude 3 Haiku, Claude 3 Sonnet, and Claude 3 Opus.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This option uses Anthropic's Claude models, designed to be particularly
    /// helpful and safe. Anthropic is a newer AI safety company founded by former OpenAI researchers.
    ///
    /// Key characteristics:
    /// - Strong analytical capabilities with detailed explanations
    /// - Large context windows (can process more information at once)
    /// - Requires an Anthropic API key (obtain from console.anthropic.com)
    /// - Known for providing thorough, well-reasoned responses
    ///
    /// Use this when: You want detailed explanations for agent recommendations or need to process
    /// larger amounts of context (like extensive feature descriptions).
    /// </para>
    /// </remarks>
    Anthropic,

    /// <summary>
    /// Microsoft Azure-hosted OpenAI models with enterprise features, compliance, and regional data residency.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This option uses the same OpenAI models as the OpenAI option, but hosted
    /// by Microsoft Azure instead of directly by OpenAI. This provides additional enterprise features.
    ///
    /// Key characteristics:
    /// - Same GPT models as OpenAI but hosted in Microsoft's cloud
    /// - Enterprise-grade security and compliance (SOC 2, HIPAA, etc.)
    /// - Data stays in your specified Azure region
    /// - Integrated with Azure security and billing
    /// - Requires both an Azure subscription and Azure OpenAI access approval
    ///
    /// Use this when: You're working in an enterprise environment that requires data to stay in specific
    /// regions, need enhanced security/compliance, or already use Microsoft Azure infrastructure.
    ///
    /// Note: You'll need to provide additional configuration (endpoint URL and deployment name) when
    /// using this option, as Azure OpenAI requires you to deploy specific models in your Azure account.
    /// </para>
    /// </remarks>
    AzureOpenAI
}
