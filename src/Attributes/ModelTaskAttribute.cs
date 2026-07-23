using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Specifies the task(s) that a model is designed to perform.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Apply this attribute to your model class to indicate what the model
/// actually does. You can apply it multiple times if the model performs several tasks.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ModelTask(ModelTask.Classification)]
/// [ModelTask(ModelTask.FeatureExtraction)]
/// public class ResNet&lt;T&gt; : NeuralNetworkBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class ModelTaskAttribute : Attribute
{
    /// <summary>
    /// Gets the task this model performs.
    /// </summary>
    public ModelTask Task { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelTaskAttribute"/> class.
    /// </summary>
    /// <param name="task">The task this model performs.</param>
    public ModelTaskAttribute(ModelTask task)
    {
        Task = task;
    }
}
