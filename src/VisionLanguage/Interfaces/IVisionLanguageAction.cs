namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for Vision-Language-Action (VLA) models that connect visual understanding
/// and language reasoning to physical robotic actions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VLA models bridge the gap between perception (vision-language understanding) and action
/// (robotic control). They take visual observations and optional language instructions to
/// predict action sequences for robot manipulation, navigation, and planning.
/// </para>
/// </remarks>
public interface IVisionLanguageAction<T> : IGenerativeVisionLanguageModel<T>
{
    /// <summary>
    /// Predicts an action sequence from a visual observation and a language instruction.
    /// </summary>
    /// <param name="observation">Visual observation tensor in [channels, height, width] format.</param>
    /// <param name="instruction">Natural language task instruction (e.g., "pick up the red cup").</param>
    /// <returns>Action tensor representing predicted joint/end-effector commands.</returns>
    Tensor<T> PredictAction(Tensor<T> observation, string instruction);

    /// <summary>
    /// Gets the dimensionality of the action space (e.g., number of joint DOFs).
    /// </summary>
    int ActionDimension { get; }

    /// <summary>
    /// Gets the name of the language model backbone.
    /// </summary>
    string LanguageModelName { get; }
}
