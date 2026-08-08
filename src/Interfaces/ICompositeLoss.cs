using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// One declared output of a multi-output model: what it is called, what shape it has, how it is scored,
/// and how heavily it counts.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para>
/// The unit of a DECLARED objective. A model whose paper objective is a sum over several heads cannot
/// express that through a single prediction tensor: whatever the prediction omits receives no gradient,
/// so that branch trains not at all — silently, since the loss still falls and every generic invariant
/// still passes. ABCNet is the case in hand; its recognition head was never supervised.
/// </para>
/// <para>
/// <b>Weight</b> is part of the declaration rather than folded into the loss function because the paper
/// states it separately and readers compare against the paper. ABCNet's objective is
/// <c>L_det + L_bezier + L_rec</c>; Inception's auxiliary classifier is weighted 0.4. A weight buried
/// inside a custom loss cannot be read off, cross-checked, or reported.
/// </para>
/// </remarks>
/// <param name="Name">Human-readable name, used in diagnostics and in the harness's target generation.</param>
/// <param name="Loss">How this output is scored against its target.</param>
/// <param name="Weight">Multiplier applied to this term in the total, per the model's paper.</param>
public readonly record struct OutputSpec<T>(string Name, ILossFunction<T> Loss, double Weight)
{
    /// <summary>A term weighted 1.0 — the common case.</summary>
    public OutputSpec(string name, ILossFunction<T> loss) : this(name, loss, 1.0) { }
}

/// <summary>
/// A model whose training objective is a WEIGHTED SUM over several named outputs, declared rather than
/// hand-assembled at each call site.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para>
/// WHY DECLARED RATHER THAN COMPOSED BY THE CALLER, which is what PyTorch does. In PyTorch a multi-output
/// model returns a tuple and every training loop writes the combination by hand —
/// <c>criterion(out.logits, y) + 0.4 * criterion(out.aux_logits, y)</c> for Inception. That is explicit,
/// which is its virtue, but nothing can INSPECT it: no tool can report which term dominates, no test can
/// assert that a branch is supervised at all, and a loop that forgets a term produces a model that
/// silently never trains part of itself. That is precisely the failure this interface exists to make
/// impossible — ABCNet's recognizer went untrained and no test noticed, because there was nothing to
/// notice with.
/// </para>
/// <para>
/// WHY NOT HIDDEN COLLECTION, which is what Keras does. <c>add_loss</c> lets any layer contribute a term
/// the framework silently sums, and the documented cost is that the objective becomes unreadable: you
/// cannot look at the training path and know what is being optimised. A declaration keeps the
/// explicitness of the PyTorch approach while adding the introspection PyTorch lacks — the terms are
/// named, weighted, enumerable, and therefore assertable.
/// </para>
/// <para>
/// The declaration also lets a test harness stop guessing. Given named outputs with shapes and losses, a
/// harness can GENERATE a correct target per output instead of manufacturing one flat tensor and hoping
/// it fits — which is what neither PyTorch nor Keras can do, because in neither does the model say what
/// it needs.
/// </para>
/// </remarks>
public interface ICompositeLoss<T>
{
    /// <summary>
    /// The objective's terms, in a stable order. One entry per supervised output.
    /// </summary>
    /// <remarks>
    /// Order is stable so that <see cref="ComputeOutputs"/>, the targets a harness generates, and any
    /// diagnostic report all line up index by index without needing to match on name at every use.
    /// </remarks>
    IReadOnlyList<OutputSpec<T>> DeclaredOutputs { get; }

    /// <summary>
    /// Runs the model and returns every declared output, in <see cref="DeclaredOutputs"/> order.
    /// </summary>
    /// <param name="input">The model input.</param>
    /// <returns>One tensor per declared output.</returns>
    /// <remarks>
    /// Must run ON THE TAPE during training, and every returned tensor must be a real product of the
    /// forward rather than a detached copy — the entire purpose is that gradient flows back from each
    /// term into the branch that produced it. A detached output adds a number to the loss and changes no
    /// weights, which is indistinguishable from the bug this replaces.
    /// </remarks>
    IReadOnlyList<Tensor<T>> ComputeOutputs(Tensor<T> input);
}
