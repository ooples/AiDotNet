using AiDotNet.Enums;
using Newtonsoft.Json;

namespace AiDotNet.Models;

/// <summary>
/// Stores the configuration settings for AI agent assistance during model building and inference.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates all the settings needed to enable and configure AI agent assistance in the
/// AiModelBuilder. It stores the API credentials, provider selection, and Azure-specific configuration
/// required to connect to large language model (LLM) services. The configuration is used both during model
/// building (for recommendations and analysis) and during inference (for conversational assistance with trained models).
/// </para>
/// <para><b>For Beginners:</b> This class holds the settings that tell the AI agent how to connect to language model services.
///
/// Think of it as a configuration card that contains:
/// - **API Key**: Your password/credentials for the AI service
/// - **Provider**: Which AI company to use (OpenAI, Anthropic, or Azure)
/// - **IsEnabled**: Whether agent assistance is turned on or off
/// - **Azure Settings**: Extra information needed if using Microsoft Azure
///
/// Why you need this:
/// - The AI agent needs credentials to call external AI services
/// - Different providers need different configuration
/// - This keeps all settings in one organized place
/// - The same configuration can be reused across multiple model builds
///
/// For example, if you're using OpenAI's GPT-4, this class would store your OpenAI API key and
/// indicate that the provider is "OpenAI". If you're using Azure, it would also store your Azure
/// endpoint URL and deployment name.
///
/// Security note: API keys are sensitive! This class is marked with [JsonIgnore] attributes to
/// prevent accidentally saving keys to disk when saving your trained models.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters and calculations (typically float or double).</typeparam>
public class AgentConfiguration<T>
{
    /// <summary>
    /// Gets or sets the API key for authenticating with the LLM provider.
    /// </summary>
    /// <value>The API key string, or null if using alternative authentication methods like environment variables.</value>
    /// <remarks>
    /// <para>
    /// The API key is used to authenticate requests to the LLM provider's API. If not provided explicitly,
    /// the system will attempt to resolve the key from global configuration or environment variables.
    /// For security, this value should not be hard-coded but loaded from secure configuration stores.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a password that proves you're authorized to use the AI service.
    ///
    /// Where to get API keys:
    /// - **OpenAI**: Sign up at platform.openai.com and generate a key in your account settings
    /// - **Anthropic**: Get a key from console.anthropic.com after creating an account
    /// - **Azure OpenAI**: Create a key in the Azure Portal after getting approval for Azure OpenAI Service
    ///
    /// You can provide the key in three ways:
    /// 1. Set it here directly (not recommended for production)
    /// 2. Set it globally at application startup using AgentGlobalConfiguration
    /// 3. Store it in an environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    ///
    /// Security tip: Never commit API keys to source control! Use environment variables or secret management
    /// services in production applications.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    public string? ApiKey { get; set; }

    /// <summary>
    /// Gets or sets the LLM provider to use for agent operations.
    /// </summary>
    /// <value>An LLMProvider enum value indicating which service to use. Defaults to OpenAI.</value>
    /// <remarks>
    /// The provider determines which API endpoints are called and what authentication method is used.
    /// Different providers offer different model capabilities, pricing structures, and compliance features.
    /// </remarks>
    public LLMProvider Provider { get; set; } = LLMProvider.OpenAI;

    /// <summary>
    /// Gets or sets a value indicating whether agent assistance is currently enabled.
    /// </summary>
    /// <value>True if the agent should provide assistance; false to disable agent features.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an on/off switch for AI agent assistance.
    ///
    /// When enabled (true):
    /// - The agent can recommend model types
    /// - The agent can suggest hyperparameters
    /// - The agent can analyze your data characteristics
    /// - You can ask questions during model building and inference
    ///
    /// When disabled (false):
    /// - All agent features are turned off
    /// - No API calls are made to LLM services
    /// - Model building proceeds without AI assistance
    ///
    /// This is useful for temporarily disabling agent features without removing all configuration,
    /// or for testing your code with and without agent assistance.
    /// </para>
    /// </remarks>
    public bool IsEnabled { get; set; }

    /// <summary>
    /// Gets or sets the Azure OpenAI endpoint URL (required only when Provider is AzureOpenAI).
    /// </summary>
    /// <value>The full Azure OpenAI resource endpoint URL, or null if not using Azure OpenAI.</value>
    /// <remarks>
    /// <para>
    /// This property is only used when the Provider is set to AzureOpenAI. The endpoint URL is specific to
    /// your Azure OpenAI resource and typically follows the format: https://YOUR-RESOURCE-NAME.openai.azure.com/
    /// </para>
    /// <para><b>For Beginners:</b> If you're using Azure OpenAI instead of regular OpenAI, you need to provide
    /// your Azure endpoint URL here.
    ///
    /// How to find your Azure endpoint:
    /// 1. Go to the Azure Portal (portal.azure.com)
    /// 2. Navigate to your Azure OpenAI resource
    /// 3. Look for "Keys and Endpoint" in the left menu
    /// 4. Copy the endpoint URL (looks like https://your-name.openai.azure.com/)
    ///
    /// This tells the agent where to send requests in Azure's cloud infrastructure instead of OpenAI's servers.
    /// </para>
    /// </remarks>
    public string? AzureEndpoint { get; set; }

    /// <summary>
    /// Gets or sets the Azure OpenAI deployment name (required only when Provider is AzureOpenAI).
    /// </summary>
    /// <value>The name of the deployed model in your Azure OpenAI resource, or null if not using Azure OpenAI.</value>
    /// <remarks>
    /// <para>
    /// Azure OpenAI requires you to deploy specific models with custom names. This property specifies which
    /// deployed model to use for agent operations. Common deployment names include variations of "gpt-4",
    /// "gpt-35-turbo", etc., but the exact name depends on what you named it when deploying.
    /// </para>
    /// <para><b>For Beginners:</b> When using Azure OpenAI, you must first "deploy" a model in the Azure Portal,
    /// giving it a name of your choice. That name goes here.
    ///
    /// How to find your deployment name:
    /// 1. Go to Azure OpenAI Studio (oai.azure.com)
    /// 2. Navigate to "Deployments" in the left menu
    /// 3. You'll see a list of your deployed models with their names
    /// 4. Copy the name of the deployment you want to use
    ///
    /// For example, if you deployed GPT-4 and named it "my-gpt4-model", you would set this property to
    /// "my-gpt4-model". This is different from regular OpenAI where you just specify the model directly.
    /// </para>
    /// </remarks>
    public string? AzureDeployment { get; set; }

    /// <summary>
    /// Gets or sets the agent assistance options that control which types of help the agent provides.
    /// </summary>
    /// <value>Agent assistance options, or null to use AgentAssistanceOptions.Default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls what the agent helps with during model building.
    ///
    /// You can customize exactly which assistance features you want:
    /// - Data analysis (examines your dataset characteristics)
    /// - Model selection (recommends which model type to use)
    /// - Hyperparameter tuning (suggests optimal parameter values)
    /// - Feature analysis (identifies important features)
    /// - Meta-learning advice (guidance on validation and regularization)
    ///
    /// If you don't set this, it defaults to AgentAssistanceOptions.Default which enables
    /// data analysis and model selection (the most commonly useful features).
    ///
    /// For full assistance, use AgentAssistanceOptions.Comprehensive.
    /// For minimal assistance, use AgentAssistanceOptions.Minimal.
    /// </para>
    /// </remarks>
    public AgentAssistanceOptions? AssistanceOptions { get; set; }
}
