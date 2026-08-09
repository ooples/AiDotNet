using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Tensor-shaped conveniences over <see cref="ILossFunction{T}"/>, including the library's single
/// implementation of "gradient of a loss with respect to a prediction".
/// </summary>
/// <remarks>
/// <para>
/// Every loss gradient in the library is produced by differentiating
/// <see cref="ILossFunction{T}.ComputeTapeLoss"/> on a <see cref="GradientTape{T}"/>. Loss functions
/// no longer carry a hand-written derivative, so a backward can no longer disagree with the forward
/// it belongs to — the two are the same expression.
/// </para>
/// <para>
/// <b>Prefer your own tape when you have one.</b> <see cref="ComputeGradient{T}"/> opens a tape,
/// records one loss, and discards it. If you are already running a forward pass under a tape, record
/// <c>ComputeTapeLoss</c> on <i>that</i> tape instead: one tape then spans forward and loss, you get
/// parameter gradients directly, and the intermediate prediction-space gradient never has to be
/// materialized. These helpers are for callers holding nothing but two finished tensors.
/// </para>
/// </remarks>
public static class LossFunctionExtensions
{
    /// <summary>
    /// Evaluates the loss for tensor-shaped predictions and targets.
    /// </summary>
    /// <param name="lossFunction">The loss function to evaluate.</param>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The scalar loss value.</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is null.</exception>
    public static T ComputeLoss<T>(
        this ILossFunction<T> lossFunction,
        Tensor<T> predicted,
        Tensor<T> actual)
    {
        if (lossFunction == null)
        {
            throw new ArgumentNullException(nameof(lossFunction));
        }
        if (predicted == null)
        {
            throw new ArgumentNullException(nameof(predicted));
        }
        if (actual == null)
        {
            throw new ArgumentNullException(nameof(actual));
        }

        return lossFunction.CalculateLoss(predicted.ToVector(), actual.ToVector());
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to <paramref name="predicted"/> by
    /// differentiating <see cref="ILossFunction{T}.ComputeTapeLoss"/> on a gradient tape.
    /// </summary>
    /// <param name="lossFunction">The loss function to differentiate.</param>
    /// <param name="predicted">The predicted values; the gradient is taken with respect to these.</param>
    /// <param name="actual">The actual (target) values, treated as a constant.</param>
    /// <returns>A tensor shaped like <paramref name="predicted"/> holding d(loss)/d(predicted).</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the loss does not actually depend on <paramref name="predicted"/> through
    /// tape-recorded operations, which would otherwise return a silently-zero gradient.
    /// </exception>
    public static Tensor<T> ComputeGradient<T>(
        this ILossFunction<T> lossFunction,
        Tensor<T> predicted,
        Tensor<T> actual)
    {
        return ComputeLossAndGradient(lossFunction, predicted, actual).Gradient;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to <paramref name="predicted"/> for callers
    /// working in vector shape.
    /// </summary>
    /// <param name="lossFunction">The loss function to differentiate.</param>
    /// <param name="predicted">The predicted values; the gradient is taken with respect to these.</param>
    /// <param name="actual">The actual (target) values, treated as a constant.</param>
    /// <returns>A vector holding d(loss)/d(predicted), the same length as <paramref name="predicted"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is null.</exception>
    /// <remarks>
    /// A shape adapter over <see cref="ComputeGradient{T}(ILossFunction{T}, Tensor{T}, Tensor{T})"/>,
    /// not a second implementation — the gradient still comes from differentiating
    /// <see cref="ILossFunction{T}.ComputeTapeLoss"/>. This exists for the hand-rolled backward chains
    /// that consume the loss gradient as a vector and propagate it through layers themselves.
    /// </remarks>
    public static Vector<T> ComputeGradient<T>(
        this ILossFunction<T> lossFunction,
        Vector<T> predicted,
        Vector<T> actual)
    {
        if (predicted == null)
        {
            throw new ArgumentNullException(nameof(predicted));
        }
        if (actual == null)
        {
            throw new ArgumentNullException(nameof(actual));
        }

        var gradient = ComputeGradient(
            lossFunction,
            Tensor<T>.FromVector(predicted),
            Tensor<T>.FromVector(actual));

        return gradient.ToVector();
    }

    /// <summary>
    /// Computes the loss value and its gradient with respect to <paramref name="predicted"/> from a
    /// <b>single</b> tape recording.
    /// </summary>
    /// <param name="lossFunction">The loss function to evaluate and differentiate.</param>
    /// <param name="predicted">The predicted values; the gradient is taken with respect to these.</param>
    /// <param name="actual">The actual (target) values, treated as a constant.</param>
    /// <returns>The loss value and d(loss)/d(predicted).</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the loss does not actually depend on <paramref name="predicted"/> through
    /// tape-recorded operations.
    /// </exception>
    /// <remarks>
    /// Reading both from one recording is why this overload exists: computing them separately would
    /// evaluate the forward twice, and — before loss gradients came from the tape — allowed the value
    /// and the gradient to come from two expressions that were not required to agree.
    /// </remarks>
    public static (T Loss, Tensor<T> Gradient) ComputeLossAndGradient<T>(
        this ILossFunction<T> lossFunction,
        Tensor<T> predicted,
        Tensor<T> actual)
    {
        if (lossFunction == null)
        {
            throw new ArgumentNullException(nameof(lossFunction));
        }
        if (predicted == null)
        {
            throw new ArgumentNullException(nameof(predicted));
        }
        if (actual == null)
        {
            throw new ArgumentNullException(nameof(actual));
        }

        using var tape = new GradientTape<T>();

        var lossTensor = lossFunction.ComputeTapeLoss(predicted, actual);
        if (lossTensor is null || lossTensor.Length == 0)
        {
            throw new InvalidOperationException(
                $"{lossFunction.GetType().Name}.ComputeTapeLoss returned no value. It must return a "
                + "rank-0 scalar tensor.");
        }

        var lossValue = lossTensor[0];

        // A loss whose forward never routes the prediction through a recorded operation cannot be
        // differentiated with respect to it. The tape reports that as "no recorded operations";
        // translated here because the bare tape message does not name the loss that caused it.
        if (tape.EntryCount == 0)
        {
            throw new InvalidOperationException(
                $"{lossFunction.GetType().Name}.ComputeTapeLoss recorded no operations on the gradient "
                + "tape, so no gradient with respect to the prediction exists. Build the loss from "
                + "engine operations on the supplied tensors rather than computing it element-wise "
                + "outside the tape.");
        }

        var gradients = tape.ComputeGradients(lossTensor, new List<Tensor<T>> { predicted });

        if (!gradients.TryGetValue(predicted, out var gradient) || gradient is null)
        {
            throw new InvalidOperationException(
                $"{lossFunction.GetType().Name}.ComputeTapeLoss produced a loss that does not depend on "
                + "the prediction, so its gradient is undefined. Check that the forward uses the "
                + "'predicted' tensor it was given rather than a copy of it.");
        }

        return (lossValue, gradient);
    }
}
