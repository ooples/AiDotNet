using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Results;

/// <summary>
/// Partial class providing text-in prediction for models trained through a text vectorizer.
/// </summary>
public partial class AiModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Predicts directly from raw text using the fitted text vectorizer captured during training.
    /// </summary>
    /// <param name="documents">The raw text documents to classify or score.</param>
    /// <returns>The model's predictions for the documents.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="documents"/> is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the model was not trained on text (no vectorizer was configured), or the model's input type is
    /// not a feature <see cref="Matrix{T}"/>.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> When you train on text with <c>ConfigureTextVectorizer(...)</c>, the result
    /// remembers how to turn words into numbers. This method applies that same conversion to new sentences and runs
    /// the model — so you can go straight from a <see cref="string"/> to a prediction without vectorizing by hand.
    /// </para>
    /// <example>
    /// <code>
    /// var vectorizer = new CountVectorizer&lt;double&gt;();
    /// var result = await new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new LogisticRegression&lt;double&gt;())
    ///     .ConfigureTextVectorizer(vectorizer)
    ///     .ConfigureDataLoader(DataLoaders.FromTextDocuments(texts, labels, vectorizer))
    ///     .BuildAsync();
    ///
    /// var prediction = result.PredictText(new[] { "I really enjoyed this!" });
    /// </code>
    /// </example>
    /// </remarks>
    public TOutput PredictText(IEnumerable<string> documents)
    {
        if (documents is null)
        {
            throw new ArgumentNullException(nameof(documents));
        }

        if (TextVectorizer is null)
        {
            throw new InvalidOperationException(
                "This model was not trained on text. Call ConfigureTextVectorizer(...) before BuildAsync(), " +
                "or use Predict(...) with already-vectorized features.");
        }

        Matrix<T> features = TextVectorizer.Transform(documents);

        // The model's input type is the generic TInput; for text models it is a feature Matrix<T>.
        // Box through object so the type test is valid on the unconstrained TInput.
        if ((object)features is not TInput typedFeatures)
        {
            throw new InvalidOperationException(
                "PredictText requires a model whose input type is Matrix<T> (text features).");
        }

        return Predict(typedFeatures);
    }
}
