using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// Stockformer's multi-task objective: a masked regression loss on returns plus a classification loss
/// on direction, summed with the paper's task weight (default 1.0, the reference's 1:1 sum).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Transcribed from the reference implementation (github.com/Eric991005/Multitask-Stockformer,
/// <c>lib/Multitask_Stockformer_utils.py</c> and <c>MultiTask_Stockformer_train.py</c> L292-308) for
/// Ma et al., arXiv:2401.06139. Three details here are not what a reading of the paper would suggest:
/// </para>
/// <list type="number">
/// <item><description><b>The regression loss is masked MAE, not MSE.</b>
/// <c>_compute_regression_loss</c> is <c>masked_mae(preds, labels, 0.0)</c>. Entries whose label
/// equals the sentinel are excluded, and the mask is renormalized by its own mean
/// (<c>mask /= mean(mask)</c>) so the result is a mean over VALID entries rather than a mean over all
/// entries with zeros mixed in.</description></item>
/// <item><description><b>The reference sums the two tasks 1:1.</b> The train script has
/// <c>loss = loss_regress + loss_class</c>, with <c>loss = w1*loss_regress + w2*loss_class</c> present
/// but COMMENTED OUT immediately above it. The paper's Eq. 12 does define a weight, so
/// <c>taskLossWeight</c> exposes it with the reference's value as the DEFAULT: calling Compute without
/// it reproduces the reference exactly, and a caller who wants Eq. 12's lambda can set it. What would
/// be an invented deviation is choosing a different default, not offering the knob.</description></item>
/// <item><description><b>Both heads are supervised on BOTH representations.</b> The model emits four
/// tensors — class and regression outputs for the main representation and for the low-frequency one —
/// and each loss term is the SUM of two calls. The classification target is shared by both class
/// terms; the regression targets differ.</description></item>
/// </list>
/// <para><b>For Beginners:</b> The model is graded on two things at once — how close its predicted
/// return is, and whether it got the up/down direction right — and both count equally. Missing data
/// points are skipped rather than treated as zero.</para>
/// </remarks>
public static class StockformerMultiTaskLoss<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Mean absolute error over entries whose label differs from <paramref name="missingSentinel"/>.
    /// </summary>
    /// <remarks>
    /// Mirrors <c>masked_mae</c>: build a 0/1 mask from the labels, divide it by its own mean, apply
    /// it to the absolute errors, and take the mean. The renormalization is what turns "mean over all
    /// entries, missing ones contributing zero" into "mean over valid entries" — dropping it silently
    /// scales the loss down by the valid fraction and quietly weakens this task against the
    /// classification term.
    /// </remarks>
    /// <returns>The masked MAE, or zero when no entry is valid.</returns>
    public static double MaskedMae(Vector<T> predictions, Vector<T> labels, double missingSentinel = 0.0)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (predictions.Length != labels.Length)
            throw new ArgumentException(
                $"Predictions ({predictions.Length}) and labels ({labels.Length}) must be the same length.",
                nameof(predictions));

        int valid = 0;
        double sumAbsolute = 0.0;

        for (int i = 0; i < labels.Length; i++)
        {
            double label = Ops.ToDouble(labels[i]);
            if (double.IsNaN(missingSentinel) ? double.IsNaN(label) : label == missingSentinel) continue;

            double prediction = Ops.ToDouble(predictions[i]);
            double error = Math.Abs(prediction - label);

            // A NaN error is NOT skipped. The mask exists to drop missing LABELS, and this entry has
            // a label. NaN here means the PREDICTION diverged, and skipping it removed the only
            // evidence of that: with every prediction NaN, valid stayed 0 and the method returned
            // 0.0 -- a perfect regression term for a completely broken model. Letting the NaN through
            // makes the loss NaN, which is what a diverged model should report.

            sumAbsolute += error;
            valid++;
        }

        // Every entry masked out. The reference's mask/mean(mask) would be 0/0 here; returning zero
        // keeps the objective finite rather than propagating NaN into the summed loss.
        return valid == 0 ? 0.0 : sumAbsolute / valid;
    }

    /// <summary>
    /// Softmax cross-entropy over direction classes.
    /// </summary>
    /// <param name="logits">Row-major <c>[samples, classes]</c> logits.</param>
    /// <param name="targets">Class indices, one per sample.</param>
    /// <param name="numClasses">Number of direction classes.</param>
    /// <remarks>
    /// Matches <c>torch.nn.CrossEntropyLoss</c>: applied to RAW logits (softmax is internal, so the
    /// caller must not pre-normalize) and averaged over samples. Computed via the log-sum-exp shift so
    /// a large logit cannot overflow.
    /// </remarks>
    public static double CrossEntropy(Vector<T> logits, Vector<T> targets, int numClasses)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (numClasses <= 1)
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses,
                "Cross-entropy needs at least two classes.");
        if (logits.Length != targets.Length * numClasses)
            throw new ArgumentException(
                $"Expected {targets.Length * numClasses} logits for {targets.Length} samples x " +
                $"{numClasses} classes, got {logits.Length}.", nameof(logits));

        double total = 0.0;
        for (int s = 0; s < targets.Length; s++)
        {
            int offset = s * numClasses;

            double max = double.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                double v = Ops.ToDouble(logits[offset + c]);
                if (v > max) max = v;
            }

            double sumExp = 0.0;
            for (int c = 0; c < numClasses; c++) sumExp += Math.Exp(Ops.ToDouble(logits[offset + c]) - max);

            int target = (int)Math.Round(Ops.ToDouble(targets[s]));
            if (target < 0 || target >= numClasses)
                throw new ArgumentException(
                    $"Target class {target} at sample {s} is outside [0, {numClasses - 1}].", nameof(targets));

            // -log softmax(target) = (max + log sum exp) - logit(target)
            total += (max + Math.Log(sumExp)) - Ops.ToDouble(logits[offset + target]);
        }

        return total / targets.Length;
    }

    /// <summary>
    /// The paper's total objective over the model's four outputs.
    /// </summary>
    /// <param name="mainRegression">Regression output for the main representation.</param>
    /// <param name="lowRegression">Regression output for the low-frequency representation.</param>
    /// <param name="mainReturnTarget">Return target for the main representation.</param>
    /// <param name="lowReturnTarget">Return target for the low-frequency representation.</param>
    /// <param name="mainClassLogits">Direction logits for the main representation.</param>
    /// <param name="lowClassLogits">Direction logits for the low-frequency representation.</param>
    /// <param name="directionTarget">Direction target, SHARED by both classification terms.</param>
    /// <param name="numClasses">Number of direction classes.</param>
    /// <param name="missingSentinel">Label value treated as missing by the regression mask.</param>
    /// <param name="taskLossWeight">
    /// Lambda in the paper's <c>L = L_reg + lambda * L_cla</c> (Eq. 12). Defaults to 1.0, which is the
    /// value the reference implementation uses.
    /// </param>
    /// <returns>The regression term, the classification term, and their weighted sum.</returns>
    public static (double Regression, double Classification, double Total) Compute(
        Vector<T> mainRegression, Vector<T> lowRegression,
        Vector<T> mainReturnTarget, Vector<T> lowReturnTarget,
        Vector<T> mainClassLogits, Vector<T> lowClassLogits,
        Vector<T> directionTarget,
        int numClasses,
        double missingSentinel = 0.0,
        double taskLossWeight = 1.0)
    {
        double regression =
            MaskedMae(mainRegression, mainReturnTarget, missingSentinel)
            + MaskedMae(lowRegression, lowReturnTarget, missingSentinel);

        double classification =
            CrossEntropy(mainClassLogits, directionTarget, numClasses)
            + CrossEntropy(lowClassLogits, directionTarget, numClasses);

        // L = L_reg + lambda*L_cla (paper Eq. 12). The reference uses lambda = 1, which is the
        // default here -- but the weighting exists in the paper, so it is a parameter rather than a
        // hardcoded 1:1 sum.
        return (regression, classification, regression + (taskLossWeight * classification));
    }
}
