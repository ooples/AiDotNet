using AiDotNet.Interpretability;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for model-agnostic explainers that can explain any predictive model's decisions.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Model explainers help you understand WHY a model makes certain predictions.
/// Unlike models that implement IInterpretableModel directly, these explainers work with ANY model -
/// they treat the model as a "black box" and analyze its behavior by observing inputs and outputs.
///
/// Think of it like understanding how a vending machine works: you don't need to see inside it,
/// you just try different button combinations and observe what comes out.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ModelExplainer")]
public interface IModelExplainer<T>
{
    /// <summary>
    /// Gets the name of this explanation method.
    /// </summary>
    string MethodName { get; }

    /// <summary>
    /// Gets whether this explainer provides local (per-instance) explanations.
    /// </summary>
    bool SupportsLocalExplanations { get; }

    /// <summary>
    /// Gets whether this explainer provides global (model-wide) explanations.
    /// </summary>
    bool SupportsGlobalExplanations { get; }
}

/// <summary>
/// Interface for explainers that provide local (per-instance) explanations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TExplanation">The type of explanation produced.</typeparam>
public interface ILocalExplainer<T, TExplanation> : IModelExplainer<T>
{
    /// <summary>
    /// Explains a single prediction.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>An explanation of why the model made this prediction.</returns>
    TExplanation Explain(Vector<T> instance);

    /// <summary>
    /// Explains multiple predictions.
    /// </summary>
    /// <param name="instances">The input instances to explain.</param>
    /// <returns>Explanations for each instance.</returns>
    TExplanation[] ExplainBatch(Matrix<T> instances);
}

/// <summary>
/// Interface for explainers that provide global (model-wide) explanations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TExplanation">The type of explanation produced.</typeparam>
public interface IGlobalExplainer<T, TExplanation> : IModelExplainer<T>
{
    /// <summary>
    /// Generates a global explanation of the model's behavior.
    /// </summary>
    /// <param name="data">Representative data to analyze model behavior.</param>
    /// <returns>A global explanation of the model.</returns>
    TExplanation ExplainGlobal(Matrix<T> data);
}
