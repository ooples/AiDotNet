using System;

namespace AiDotNet.MetaLearning.Components;

/// <summary>
/// SImPa's PAC-Bayes generalization bounds: the single-task bound (Theorem 1) and the two-level
/// meta-learning bound (Theorem 2) from Nguyen, Do and Carneiro (arXiv:2003.02455).
/// </summary>
/// <remarks>
/// <para>
/// Kept as plain arithmetic over doubles, separate from the algorithm, because these are the paper's
/// theorem statements and are checkable in isolation — the scaling in the denominators and the shape of
/// the log terms are the substance of the result, and burying them inside a training loop is how they
/// drift.
/// </para>
/// <para><b>For Beginners:</b> A PAC-Bayes bound is a promise of the form "the error on data you have not
/// seen is at most the error you measured, plus a penalty". The penalty grows when the learned
/// distribution moves far from the prior, and shrinks as you get more data or more tasks.</para>
/// </remarks>
public static class PacBayesMetaBound
{
    /// <summary>The paper's confidence parameter, <c>epsilon = 0.1</c> for every task.</summary>
    public const double PaperEpsilon = 0.1;

    /// <summary>
    /// Theorem 1, the single-task bound:
    /// <c>empiricalLoss + sqrt((KL[q||p] + ln(m / epsilon)) / (2 * (m - 1)))</c>.
    /// </summary>
    /// <param name="empiricalLoss">Empirical loss on the task's samples.</param>
    /// <param name="klDivergence">KL[q || p] for this task, or a lower-bound estimate of it.</param>
    /// <param name="sampleCount">m, the number of samples for the task. Must exceed 1.</param>
    /// <param name="epsilon">Confidence parameter in (0, 1]. The paper uses 0.1.</param>
    /// <remarks>
    /// The <c>m - 1</c> in the denominator rather than <c>m</c> is why a single sample gives no bound at
    /// all: with one observation there is nothing to generalize from, and the expression correctly
    /// diverges instead of quietly returning the empirical loss.
    /// </remarks>
    public static double SingleTask(
        double empiricalLoss, double klDivergence, int sampleCount, double epsilon = PaperEpsilon)
    {
        ValidateEpsilon(epsilon);
        if (sampleCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleCount), sampleCount,
                "The bound needs at least 2 samples; with one there is no generalization to bound.");
        }
        if (klDivergence < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(klDivergence), klDivergence,
                "A KL divergence cannot be negative.");
        }

        double numerator = klDivergence + Math.Log(sampleCount / epsilon);
        return empiricalLoss + Math.Sqrt(numerator / (2.0 * (sampleCount - 1)));
    }

    /// <summary>
    /// Theorem 2, the meta-learning bound: the empirical validation loss plus a TASK-level and a
    /// META-level complexity term.
    /// </summary>
    /// <param name="empiricalValidationLoss">Mean empirical validation loss across the task batch.</param>
    /// <param name="expectedTaskKL">
    /// <c>E[KL[q(w_i; lambda_i) || p(w_i)]]</c> — the task-posterior KL averaged over tasks.
    /// </param>
    /// <param name="metaKL"><c>KL[q(theta; psi) || p(theta)]</c> for the meta-parameters.</param>
    /// <param name="validationSampleCount">m_i^v, validation samples per task. Must exceed 1.</param>
    /// <param name="taskCount">T, tasks in the batch. Must exceed 1.</param>
    /// <param name="epsilon">Confidence parameter in (0, 1]. The paper uses 0.1.</param>
    /// <returns>The upper bound on the true loss over unseen tasks and samples.</returns>
    /// <remarks>
    /// <para>
    /// The two terms are:
    /// <c>sqrt((E[KL_task] + T^2 * ln(m_v) / (epsilon * (T - 1))) / (2 * (m_v - 1)))</c> and
    /// <c>sqrt((KL_meta + T * ln(T) / epsilon) / (2 * (T - 1)))</c>.
    /// </para>
    /// <para>
    /// TWO LEVELS, not one, and that is the extension the paper makes over single-task PAC-Bayes: the
    /// bound has to cover both an unseen SAMPLE within a known task and an entirely unseen TASK. Dropping
    /// the meta term leaves a bound that says nothing about the case meta-learning exists for, and it is
    /// the term that shrinks with the number of TASKS rather than the number of samples.
    /// </para>
    /// <para>
    /// The <c>T^2</c> in the task-level log term is not a typo for <c>T</c>: the task-level guarantee has
    /// to hold simultaneously across all T tasks, and paying for that union costs a factor that grows
    /// faster than linearly before the <c>(T - 1)</c> division brings it back.
    /// </para>
    /// <para>
    /// COUNTERINTUITIVE CONSEQUENCE, worth stating because it looks like a bug: with both KL terms held
    /// FIXED, increasing T makes this bound LOOSER. The meta term tends to
    /// <c>sqrt(ln(T) / (2 * epsilon))</c> and the task term grows like <c>sqrt(T)</c>. That is correct —
    /// the log terms are union-bound costs and more tasks means more to cover simultaneously. The benefit
    /// of more tasks arrives through the KL terms shrinking as the meta-parameters become better
    /// determined, not through the explicit T dependence.
    /// </para>
    /// <para>
    /// SOURCE CAVEAT: the bracket structure of the task-level term was transcribed from a text rendering
    /// of the paper, in which the grouping of the <c>(T - 1)</c> division was not unambiguous. The form
    /// implemented here is the one consistent with Theorem 1's shape (complexity over
    /// <c>2 * (samples - 1)</c>) and with the meta term's own structure, and it is pinned by
    /// SImPaComponentTests so any correction is a single deliberate edit.
    /// </para>
    /// </remarks>
    public static double MetaLearning(
        double empiricalValidationLoss,
        double expectedTaskKL,
        double metaKL,
        int validationSampleCount,
        int taskCount,
        double epsilon = PaperEpsilon)
    {
        ValidateEpsilon(epsilon);
        if (validationSampleCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(validationSampleCount), validationSampleCount,
                "The task-level term needs at least 2 validation samples per task.");
        }
        if (taskCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(taskCount), taskCount,
                "The meta-level term needs at least 2 tasks; with one task there is no task distribution "
                + "to generalize over.");
        }
        if (expectedTaskKL < 0.0)
            throw new ArgumentOutOfRangeException(nameof(expectedTaskKL), expectedTaskKL, "A KL divergence cannot be negative.");
        if (metaKL < 0.0)
            throw new ArgumentOutOfRangeException(nameof(metaKL), metaKL, "A KL divergence cannot be negative.");

        double taskLog = taskCount * (double)taskCount * Math.Log(validationSampleCount)
                         / (epsilon * (taskCount - 1));
        double taskTerm = Math.Sqrt((expectedTaskKL + taskLog) / (2.0 * (validationSampleCount - 1)));

        double metaLog = taskCount * Math.Log(taskCount) / epsilon;
        double metaTerm = Math.Sqrt((metaKL + metaLog) / (2.0 * (taskCount - 1)));

        return empiricalValidationLoss + taskTerm + metaTerm;
    }

    private static void ValidateEpsilon(double epsilon)
    {
        if (epsilon is <= 0.0 or > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon,
                "epsilon is a confidence parameter and must lie in (0, 1].");
        }
    }
}
