using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// A fused training step whose forward reads several per-step input slots, run on the shared
/// <see cref="FusedTrainingSession{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Stochastic trainers (CSDI, TabDDPM, TabSyn, FinDiff, AutoDiffTab) feed per-step data that is not one input and
/// one target: a noisy sample, the noise, a timestep encoding, conditioning. The session takes one input and one
/// target, so the slots are flattened into a single <c>[1, N]</c> row and the forward narrows and reshapes them back
/// with engine ops. A compiled plan therefore replays against every step's fresh slot data, and the session supplies
/// the fused lifecycle every base class shares: the optimizer mapping, the committed-failure policy and the
/// parameter-persistence probe.
/// </para>
/// <para>
/// The plan owns the optimizer state from its first step, so a model whose weights are created lazily resolves them
/// with one forward outside the tape before that step; an eager first step would leave its moments in the eager
/// optimizer and the plan would restart them from zero.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public sealed class MultiSlotFusedStep<T> : IDisposable
{
    private readonly FusedTrainingSession<T> _session;
    private bool _disposed;

    /// <summary>Creates a step for <paramref name="owner"/>, the model whose plan and optimizer state it holds.</summary>
    public MultiSlotFusedStep(object owner)
    {
        Guard.NotNull(owner);
        _session = new FusedTrainingSession<T>(owner);
    }

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Trains one step on the fused compiled plan. Returns false when the plan does not apply; the caller then runs
    /// its eager step.
    /// </summary>
    /// <param name="parameterProvider">The tensors to train, read every step.</param>
    /// <param name="zeroGradAction">Optional layer gradient reset before the step.</param>
    /// <param name="freshSlotData">This step's slot data.</param>
    /// <param name="forward">The model forward over the slots.</param>
    /// <param name="computeLoss">The scalar loss from the forward output and the slots.</param>
    /// <param name="optimizer">The optimizer whose configuration the plan runs.</param>
    /// <param name="lossValue">The step's loss when it ran.</param>
    public bool TryStep(
        Func<IReadOnlyList<Tensor<T>>> parameterProvider,
        Action? zeroGradAction,
        IReadOnlyList<Tensor<T>> freshSlotData,
        Func<IReadOnlyList<Tensor<T>>, Tensor<T>> forward,
        Func<Tensor<T>, IReadOnlyList<Tensor<T>>, Tensor<T>> computeLoss,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        out T lossValue)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MultiSlotFusedStep<T>));
        lossValue = MathHelper.GetNumericOperations<T>().Zero;
        Guard.NotNull(parameterProvider);
        Guard.NotNull(forward);
        Guard.NotNull(computeLoss);
        Guard.NotNull(optimizer);
        if (freshSlotData is null || freshSlotData.Count == 0)
            return false;

        var parameters = parameterProvider();
        if (parameters is null || parameters.Count == 0)
        {
            using (new NoGradScope<T>())
                forward(freshSlotData);
            parameters = parameterProvider();
            if (parameters is null || parameters.Count == 0)
                return false;
        }

        var shapes = new int[freshSlotData.Count][];
        var offsets = new int[freshSlotData.Count];
        int total = 0;
        for (int i = 0; i < freshSlotData.Count; i++)
        {
            shapes[i] = freshSlotData[i]._shape;
            offsets[i] = total;
            total += freshSlotData[i].Length;
        }
        if (total == 0)
            return false;

        var packed = new T[total];
        for (int i = 0; i < freshSlotData.Count; i++)
            freshSlotData[i].AsSpan().CopyTo(packed.AsSpan(offsets[i], freshSlotData[i].Length));

        IReadOnlyList<Tensor<T>> Unpack(Tensor<T> row)
        {
            var slots = new Tensor<T>[shapes.Length];
            for (int i = 0; i < shapes.Length; i++)
            {
                int length = freshSlotData[i].Length;
                slots[i] = length == 0
                    ? new Tensor<T>(shapes[i])
                    : Engine.Reshape(Engine.TensorNarrow(row, 1, offsets[i], length), shapes[i]);
            }
            return slots;
        }

        zeroGradAction?.Invoke();
        var outcome = _session.TryStep(new FusedTrainingStepRequest<T>
        {
            Layers = Array.Empty<ITrainableLayer<T>>(),
            Input = new Tensor<T>(new[] { 1, total }, new Vector<T>(packed)),
            Target = new Tensor<T>(new[] { 1, total }, new Vector<T>((T[])packed.Clone())),
            Forward = row => forward(Unpack(row)),
            ComputeLoss = (predicted, row) => computeLoss(predicted, Unpack(row)),
            Optimizer = optimizer,
            ExtraParameters = parameters,
        }, out lossValue);

        if (outcome == FusedStepOutcome.CommittedFailure)
            _session.DropAfterCommittedFailure();
        return outcome == FusedStepOutcome.Stepped;
    }

    /// <summary>Forgets the compiled plan and its optimizer state.</summary>
    public void Invalidate() => _session.Reset(stickyDisable: false);

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _session.Reset(stickyDisable: false);
        _disposed = true;
    }
}