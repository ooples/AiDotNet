using AiDotNet.Helpers;
using AiDotNet.Models.Results;
using AiDotNet.SurvivalAnalysis;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Builds a survival model from covariates, observed times, and who was censored.
    /// </summary>
    /// <param name="features">The covariates, one row per subject. No indicator column.</param>
    /// <param name="times">
    /// The observed time for each subject: time to the event, or time to censoring for those who did not
    /// have it.
    /// </param>
    /// <param name="events">
    /// 1 where the event was observed, 0 where the subject was censored.
    /// </param>
    /// <returns>The trained result, the same one <see cref="Build(TInput, TOutput)"/> returns.</returns>
    /// <remarks>
    /// <para>
    /// Survival data is three things — a time, whether the event actually happened, and the covariates —
    /// and <see cref="Build(TInput, TOutput)"/> takes two. <see cref="SurvivalModelBase{T}"/> bridges
    /// that by reading column 0 of its design matrix as the event indicator, which works but leaves the
    /// caller assembling that matrix and one silent mistake away from a covariate being read as
    /// censoring. This overload assembles it for you, so the three signals stay three named arguments
    /// and the convention is nobody's problem but this method's.
    /// </para>
    /// <para>
    /// One limitation, recorded because it is invisible until you hit it: when <typeparamref name="T"/>
    /// is <c>int</c> the two three-argument overloads have the same signature, and a call cannot pick
    /// between them. That is a compile error at the call site rather than a silently different call, and
    /// neither domain has a use for integer times or outcomes, so it is a corner rather than a trap.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> "Censored" means the study ended, or you lost track of someone, before the
    /// thing you were measuring happened to them. You do not know when it will happen — only that it had
    /// not yet. Telling the model which subjects those are is the whole point of survival analysis; a
    /// model told everyone had the event will report survival that is too pessimistic.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Any argument is null.</exception>
    /// <exception cref="ArgumentException">The three inputs disagree on how many subjects there are.</exception>
    /// <exception cref="InvalidOperationException">
    /// No model has been configured, the configured model is not a survival model, or this builder was
    /// declared with input and output types a survival model cannot take.
    /// </exception>
    /// <example>
    /// <code>
    /// var features = new Matrix&lt;double&gt;(new double[,]
    /// {
    ///     { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 }
    /// });
    /// var times = new Vector&lt;double&gt;(new double[] { 5.0, 12.0, 3.0, 18.0, 9.0, 21.0 });
    /// var events = new Vector&lt;int&gt;(new int[] { 1, 0, 1, 0, 1, 1 });   // 0 = censored
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new KaplanMeierEstimator&lt;double&gt;())
    ///     .Build(features, times, events);
    ///
    /// var risk = result.Predict(features);
    /// </code>
    /// </example>
    public AiModelResult<T, TInput, TOutput> Build(
        Matrix<T> features,
        Vector<T> times,
        Vector<int> events)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (times is null) throw new ArgumentNullException(nameof(times));
        if (events is null) throw new ArgumentNullException(nameof(events));

        if (_model is null)
        {
            throw new InvalidOperationException(
                "Build(features, times, events) needs a model. Call ConfigureModel with a survival " +
                "model — KaplanMeierEstimator, NelsonAalenEstimator, CoxProportionalHazards, WeibullAFT, " +
                "LogNormalAFT or RandomSurvivalForest — before building.");
        }

        if (_model is not SurvivalModelBase<T> survivalModel)
        {
            throw new InvalidOperationException(
                $"Build(features, times, events) is for survival models; the configured model is " +
                $"{_model.GetType().Name}. Censoring means nothing to it, so use Build(features, labels).");
        }

        if (features.Rows != times.Length || features.Rows != events.Length)
        {
            throw new ArgumentException(
                $"The three inputs disagree on how many subjects there are: features has " +
                $"{features.Rows} rows, times has {times.Length} entries and events has {events.Length}.",
                nameof(events));
        }

        for (int i = 0; i < events.Length; i++)
        {
            if (events[i] != 0 && events[i] != 1)
            {
                throw new ArgumentException(
                    $"Event indicators are 1 where the event was observed and 0 where the subject was " +
                    $"censored; found {events[i]} at index {i}.",
                    nameof(events));
            }
        }

        if (features is not TInput featureInput || times is not TOutput timesOutput)
        {
            throw new InvalidOperationException(
                $"Build(features, times, events) needs a builder over Matrix<{typeof(T).Name}> and " +
                $"Vector<{typeof(T).Name}>; this one is over {typeof(TInput).Name} and " +
                $"{typeof(TOutput).Name}. Survival models train on a covariate matrix and a time vector.");
        }

        // Hand the indicators to the model rather than folding them into the covariate matrix. Packing
        // them in would work for training and then break prediction: the result would remember an extra
        // column that a caller cannot supply at predict time, because whether the event occurred is
        // exactly what is being predicted.
        survivalModel.SupplyEvents(events);
        try
        {
            return Build(featureInput, timesOutput);
        }
        finally
        {
            // Clear on the way out too. Train consumes them, but a build that throws first must not
            // leave the model holding indicators for data it never saw.
            survivalModel.SupplyEvents(null);
        }
    }
}
