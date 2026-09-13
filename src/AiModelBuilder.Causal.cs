using AiDotNet.CausalInference;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Builds a causal model from covariates, who was treated, and what happened to them.
    /// </summary>
    /// <param name="covariates">The covariates, one row per subject. No treatment column.</param>
    /// <param name="treatment">1 for the subjects who were treated, 0 for the controls.</param>
    /// <param name="outcome">The outcome observed for each subject.</param>
    /// <returns>The trained result, the same one <see cref="Build(TInput, TOutput)"/> returns.</returns>
    /// <remarks>
    /// <para>
    /// Causal data is three things — who was treated, what happened, and everything else you know about
    /// them — and <see cref="Build(TInput, TOutput)"/> takes two.
    /// <see cref="CausalModelBase{T}"/> bridges that by reading column 0 of its design matrix as the
    /// treatment indicator, which works but leaves the caller assembling that matrix and one silent
    /// mistake away from a covariate being read as treatment. This overload assembles nothing: the three
    /// signals stay three named arguments.
    /// </para>
    /// <para>
    /// The survival overload alongside this one takes its three arguments in a different order —
    /// <c>Build(features, times, events)</c> — and the two are told apart by which argument is the
    /// <c>Vector&lt;int&gt;</c> indicator. Passing them in the wrong order is a compile error rather
    /// than a silently different call.
    /// </para>
    /// <para>
    /// One limitation, recorded because it is invisible until you hit it: when <typeparamref name="T"/>
    /// is <c>int</c> the two three-argument overloads have the same signature, and a call cannot pick
    /// between them. That is a compile error at the call site rather than a silently different call, and
    /// neither domain has a use for integer times or outcomes, so it is a corner rather than a trap.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A causal model does not ask "what will happen to this person" but "what did
    /// the treatment change". To answer that it has to know who actually received it, which is what the
    /// treatment argument is; the covariates are everything else you measured about them.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Any argument is null.</exception>
    /// <exception cref="ArgumentException">The three inputs disagree on how many subjects there are.</exception>
    /// <exception cref="InvalidOperationException">
    /// No model has been configured, the configured model is not a causal model, or this builder was
    /// declared with input and output types a causal model cannot take.
    /// </exception>
    /// <example>
    /// <code>
    /// var covariates = new Matrix&lt;double&gt;(new double[,]
    /// {
    ///     { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 }
    /// });
    /// var treatment = new Vector&lt;int&gt;(new int[] { 0, 1, 0, 1, 0, 1 });   // who was treated
    /// var outcome = new Vector&lt;double&gt;(new double[] { 3.1, 7.2, 2.8, 8.0, 3.4, 7.6 });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new TLearner&lt;double&gt;(maxIterations: 100, learningRate: 0.1))
    ///     .Build(covariates, treatment, outcome);
    ///
    /// var (ate, se) = result.EstimateATE(covariates, treatment, outcome);
    /// </code>
    /// </example>
    public AiModelResult<T, TInput, TOutput> Build(
        Matrix<T> covariates,
        Vector<int> treatment,
        Vector<T> outcome)
    {
        if (covariates is null) throw new ArgumentNullException(nameof(covariates));
        if (treatment is null) throw new ArgumentNullException(nameof(treatment));
        if (outcome is null) throw new ArgumentNullException(nameof(outcome));

        if (_model is null)
        {
            throw new InvalidOperationException(
                "Build(covariates, treatment, outcome) needs a model. Call ConfigureModel with a causal " +
                "model — CausalForest, SLearner, TLearner, XLearner, DoublyRobustEstimator, " +
                "InverseProbabilityWeighting or PropensityScoreMatching — before building.");
        }

        if (_model is not CausalModelBase<T> causalModel)
        {
            throw new InvalidOperationException(
                $"Build(covariates, treatment, outcome) is for causal models; the configured model is " +
                $"{_model.GetType().Name}. Treatment assignment means nothing to it, so use " +
                $"Build(features, labels).");
        }

        if (covariates.Rows != treatment.Length || covariates.Rows != outcome.Length)
        {
            throw new ArgumentException(
                $"The three inputs disagree on how many subjects there are: covariates has " +
                $"{covariates.Rows} rows, treatment has {treatment.Length} entries and outcome has " +
                $"{outcome.Length}.",
                nameof(treatment));
        }

        for (int i = 0; i < treatment.Length; i++)
        {
            if (treatment[i] != 0 && treatment[i] != 1)
            {
                throw new ArgumentException(
                    $"Treatment assignment is 1 for the treated and 0 for the controls; found " +
                    $"{treatment[i]} at index {i}.",
                    nameof(treatment));
            }
        }

        if (covariates is not TInput covariateInput || outcome is not TOutput outcomeOutput)
        {
            throw new InvalidOperationException(
                $"Build(covariates, treatment, outcome) needs a builder over Matrix<{typeof(T).Name}> " +
                $"and Vector<{typeof(T).Name}>; this one is over {typeof(TInput).Name} and " +
                $"{typeof(TOutput).Name}. Causal models train on a covariate matrix and an outcome vector.");
        }

        // Hand the assignment to the model rather than folding it into the covariate matrix. Packing it
        // in would work for training and then leave the result remembering a column the effect
        // estimators do not take, since they are given the covariates and the treatment separately.
        causalModel.SupplyTreatment(treatment);
        try
        {
            return Build(covariateInput, outcomeOutput);
        }
        finally
        {
            // Clear on the way out too. Train consumes it, but a build that throws first must not leave
            // the model holding an assignment for data it never saw.
            causalModel.SupplyTreatment(null);
        }
    }
}
