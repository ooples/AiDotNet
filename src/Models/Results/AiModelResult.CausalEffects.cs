using AiDotNet.CausalInference;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Results;

/// <summary>
/// Causal-effect estimates, surfaced on the result so a causal model is used the same way as any other:
/// configure it on the builder, build, then ask the result what it found.
/// </summary>
/// <remarks>
/// <para>
/// Every causal estimator answers a question that <see cref="Predict(TInput)"/> cannot: not "what is the
/// outcome for this row" but "how much did the treatment change the outcome". Those answers live on
/// <see cref="CausalModelBase{T}"/>, which the result wraps, so before these existed the only way to
/// reach them was to hold the model yourself and step around the builder entirely.
/// </para>
/// <para>
/// <b>For Beginners:</b> Suppose you gave a drug to some patients and not others. Prediction tells you
/// what happens to one patient. These methods tell you what the drug *did* — the average difference it
/// made, and how confident you can be in that number.
/// </para>
/// </remarks>
public partial class AiModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Estimates the Average Treatment Effect: how much the treatment changed the outcome, averaged over
    /// everyone in the sample.
    /// </summary>
    /// <param name="x">The covariates, one row per subject.</param>
    /// <param name="treatment">Who was treated, as a binary indicator per subject.</param>
    /// <param name="outcome">The outcome observed for each subject.</param>
    /// <returns>The estimate and its standard error.</returns>
    /// <exception cref="NotSupportedException">The built model is not a causal estimator.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> "Average treatment effect" is the answer to "if everyone had been treated
    /// instead of nobody, how much would the outcome have moved on average?" The standard error tells
    /// you how much that number would wobble on a different sample of the same size.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Column 0 is the treatment indicator; columns 1.. are the covariates.
    /// var design = new Matrix&lt;double&gt;(new double[,] { { 0, 45 }, { 1, 52 }, { 0, 38 }, { 1, 61 } });
    /// var outcome = new Vector&lt;double&gt;(new double[] { 3.1, 7.2, 2.8, 8.0 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new InverseProbabilityWeighting&lt;double&gt;())
    ///     .Build(design, outcome);
    ///
    /// var covariates = new Matrix&lt;double&gt;(new double[,] { { 45 }, { 52 }, { 38 }, { 61 } });
    /// var treatment = new Vector&lt;int&gt;(new int[] { 0, 1, 0, 1 });
    /// var (ate, se) = result.EstimateATE(covariates, treatment, outcome);
    /// </code>
    /// </example>
    public (T estimate, T standardError) EstimateATE(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (EnsureModel is CausalModelBase<T> causal)
        {
            return causal.EstimateATE(x, treatment, outcome);
        }

        throw new NotSupportedException(
            $"EstimateATE requires a causal model (CausalModelBase<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Estimates the Average Treatment effect on the Treated: the effect among the subjects who actually
    /// received the treatment, rather than over everyone.
    /// </summary>
    /// <param name="x">The covariates, one row per subject.</param>
    /// <param name="treatment">Who was treated, as a binary indicator per subject.</param>
    /// <param name="outcome">The outcome observed for each subject.</param>
    /// <returns>The estimate and its standard error.</returns>
    /// <exception cref="NotSupportedException">The built model is not a causal estimator.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the more useful number when treatment was not assigned at random —
    /// it asks what the treatment did for the people who got it, which is often a different group from
    /// the population as a whole.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var design = new Matrix&lt;double&gt;(new double[,] { { 0, 45 }, { 1, 52 }, { 0, 38 }, { 1, 61 } });
    /// var outcome = new Vector&lt;double&gt;(new double[] { 3.1, 7.2, 2.8, 8.0 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new PropensityScoreMatching&lt;double&gt;())
    ///     .Build(design, outcome);
    ///
    /// var covariates = new Matrix&lt;double&gt;(new double[,] { { 45 }, { 52 }, { 38 }, { 61 } });
    /// var treatment = new Vector&lt;int&gt;(new int[] { 0, 1, 0, 1 });
    /// var (att, se) = result.EstimateATT(covariates, treatment, outcome);
    /// </code>
    /// </example>
    public (T estimate, T standardError) EstimateATT(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (EnsureModel is CausalModelBase<T> causal)
        {
            return causal.EstimateATT(x, treatment, outcome);
        }

        throw new NotSupportedException(
            $"EstimateATT requires a causal model (CausalModelBase<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Estimates the treatment effect for each individual row, rather than one number for the sample.
    /// </summary>
    /// <param name="features">The covariates, one row per subject.</param>
    /// <returns>One estimated effect per row.</returns>
    /// <exception cref="NotSupportedException">The built model is not a causal estimator.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The same treatment can help one person and do nothing for another. This
    /// gives you a per-person estimate, which is what you want if you are deciding who to treat rather
    /// than whether to treat at all.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var design = new Matrix&lt;double&gt;(new double[,] { { 0, 45 }, { 1, 52 }, { 0, 38 }, { 1, 61 } });
    /// var outcome = new Vector&lt;double&gt;(new double[] { 3.1, 7.2, 2.8, 8.0 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new TLearner&lt;double&gt;(maxIterations: 100, learningRate: 0.1))
    ///     .Build(design, outcome);
    ///
    /// var covariates = new Matrix&lt;double&gt;(new double[,] { { 45 }, { 52 }, { 38 }, { 61 } });
    /// var perPersonEffect = result.EstimateTreatmentEffect(covariates);
    /// </code>
    /// </example>
    public Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        if (EnsureModel is CausalModelBase<T> causal)
        {
            return causal.EstimateTreatmentEffect(features);
        }

        throw new NotSupportedException(
            $"EstimateTreatmentEffect requires a causal model (CausalModelBase<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }
}
