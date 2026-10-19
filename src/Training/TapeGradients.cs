using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// The gradient of a scalar objective with respect to a set of leaves, without an update: the episodic heads of the
/// metric meta-learners (Matching, Prototypical and Relation Networks, MetaOptNet, MCL, ATAML, LEO, AWGIM) and their
/// inner adaptation loops.
/// </summary>
/// <remarks>
/// <para>
/// Those learners train a generic <c>IModel</c> body through <c>MetaLearnerBase.ComputeGradients</c> and apply the
/// averaged outer gradient of the body and of their own head weights as one flat SGD step
/// (<c>MetaLearnerBase.ApplyGradients</c>), the update every meta-learner shares. The body is not guaranteed to be a
/// tensor-parameter network, so the fused step cannot own that update; what each of them carried was its own tape
/// around the head's gradient. It lives here once.
/// </para>
/// <para>
/// The tape owns a <c>TensorArena</c> that it rewinds on dispose, and the gradients live in it, so they are handed to
/// <paramref name="consume"/> INSIDE the tape's scope; a consumer copies what it keeps into host vectors there.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class TapeGradients<T>
{
    /// <summary>
    /// Records <paramref name="objective"/> on a fresh tape and hands its gradients with respect to
    /// <paramref name="leaves"/> to <paramref name="consume"/>; returns the objective's value.
    /// </summary>
    /// <param name="objective">Builds the scalar objective from the leaves.</param>
    /// <param name="leaves">The tensors to differentiate with respect to.</param>
    /// <param name="consume">Reads the gradients (a leaf the objective never reached has no entry).</param>
    /// <param name="persistent">Keeps the tape's graph after the backward, for objectives that are read again.</param>
    public static T Of(
        Func<Tensor<T>> objective,
        IReadOnlyList<Tensor<T>> leaves,
        Action<Dictionary<Tensor<T>, Tensor<T>>> consume,
        bool persistent = false)
    {
        Guard.NotNull(objective);
        Guard.NotNull(leaves);
        Guard.NotNull(consume);
        using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = persistent });
        var value = objective();
        if (value is null)
            throw new InvalidOperationException("A gradient objective returned null instead of a scalar tensor.");
        T result = value[0];
        consume(tape.ComputeGradients(value, leaves));
        return result;
    }

    /// <summary>
    /// The gradient of an inner loss with respect to <paramref name="source"/>, on a nested tape inside an enclosing
    /// recording. With <paramref name="createGraph"/> the backward is recorded on the enclosing tape, so outer
    /// differentiation reaches through it (second-order); without, the gradient is a constant.
    /// </summary>
    /// <remarks>
    /// The nested tape is deliberately not disposed: inside a <c>TensorArena</c>, disposing it returns tensors the
    /// enclosing computation still holds, and the next step then reads a recycled buffer. Left to the collector, the
    /// step's graph stays valid for the outer backward. Not disposing it must still not leave it current: a
    /// constructed tape is the thread's recording tape until disposed, so every later operation on the thread would
    /// record onto it and walk its stale graph into recycled arena tensors. The scope restores the enclosing tape on
    /// return, and the nested tape opens no arena of its own.
    /// </remarks>
    public static Tensor<T> Nested(Tensor<T> source, Func<Tensor<T>> innerLoss, bool createGraph)
    {
        Guard.NotNull(source);
        Guard.NotNull(innerLoss);
        using var restoreEnclosingTape = GradientTape<T>.RecordOn(GradientTape<T>.Current);
        var inner = new GradientTape<T>(new GradientTapeOptions { Persistent = true, SuppressArenaScope = true });
        var loss = innerLoss();
        var gradients = inner.ComputeGradients(loss, new[] { source }, createGraph: createGraph);
        var gradient = gradients.TryGetValue(source, out var g) ? g : new Tensor<T>(source._shape);
        return createGraph ? gradient : AiDotNet.Tensors.Engines.AiDotNetEngine.Current.StopGradient(gradient);
    }
}