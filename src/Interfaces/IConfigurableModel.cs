using AiDotNet.Models.Options;

namespace AiDotNet.Interfaces;

/// <summary>
/// Provides access to a model's options for post-construction hyperparameter configuration.
/// </summary>
/// <remarks>
/// <para>
/// This interface enables the agent hyperparameter auto-apply feature. Models that implement
/// this interface allow the AI agent to set hyperparameter values on their options object
/// before training begins. The GetOptions() method returns the live options instance,
/// allowing mutation of hyperparameter values.
/// </para>
/// <para><b>For Beginners:</b> This interface is like giving the AI agent a key to adjust
/// the model's settings. When the agent recommends hyperparameters like "set learning_rate
/// to 0.01", it needs a way to actually change that setting on the model. This interface
/// provides that access through the GetOptions() method.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters and calculations.</typeparam>
internal interface IConfigurableModel<T>
{
    /// <summary>
    /// Returns the live options instance for this model, allowing hyperparameter mutation before training.
    /// </summary>
    /// <returns>The model's options object (inherits from ModelOptions).</returns>
    ModelOptions GetOptions();
}
