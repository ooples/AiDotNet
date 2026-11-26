using AiDotNet.DistributedTraining;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using OptimizerType = AiDotNet.Tensors.Engines.Compilation.OptimizerType;

namespace AiDotNet.Training;

public static partial class CompiledTapeTrainingStep<T>
{
    // Debug only (AIDOTNET_FUSED_DEBUG): the plan's step, the rate its schedule gives that step, and the norm of the
    // gradients it left (a host read per gradient).
    [AiDotNet.Attributes.ScalarKernel("Debug-only host gradient norm for the fused diagnostics log")]
    internal static string PlanStepDiagnostics(ICompiledTrainingPlan<T> plan)
    {
        if (plan is not ICompiledTrainingPlanIntrospection<T> view) return "no introspection";
        int step = view.OptimizerStep;
        string lr = view.LearningRateSchedules.Count > 0 ? view.LearningRateSchedules[0].GetLr(step).ToString("G9") : "n/a";
        double sum = 0;
        var numOps = MathHelper.GetNumericOperations<T>();
        foreach (var g in plan.Gradients ?? Array.Empty<Tensor<T>>())
            if (g is not null) foreach (var v in g.ToArray()) { double d = numOps.ToDouble(v); sum += d * d; }
        return $"plan step {step}, plan lr(step) {lr}, |grad| {Math.Sqrt(sum):G9}";
    }

    /// <summary>
    /// Compiled data-parallel steps whose optimizer update runs inside the plan (the fused optimizer, captured as a
    /// graph on CUDA), with the gradient all-reduce at the plan's exchange point between the backward and the update.
    /// The eager reference path (the plan computes gradients, the coordinator reduces them and the optimizer instance
    /// steps) stays the default; this one is opted into per coordinator.
    /// </summary>
    private static long s_compiledUpdateSteps, s_lastExchanges, s_lastUpdateLaunches;
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> s_reportedIneligibility = new();
    private static int s_lastSplit;

    /// <summary>
    /// Steps this process ran with the data-parallel update in the plan, and the last such plan's gradient exchanges,
    /// update-graph launches and whether it ran split into a backward graph and an update graph (tests).
    /// </summary>
    internal static (long Steps, long Exchanges, long UpdateGraphLaunches, bool Split) DistributedCompiledUpdateDiagnostics
        => (System.Threading.Interlocked.Read(ref s_compiledUpdateSteps), System.Threading.Interlocked.Read(ref s_lastExchanges),
            System.Threading.Interlocked.Read(ref s_lastUpdateLaunches), System.Threading.Volatile.Read(ref s_lastSplit) != 0);

    private sealed class DistributedCompiledUpdate
    {
        internal DistributedTapeCoordinator<T> Coordinator = null!;
        internal ICompiledTrainingPlan<T>? Plan;
        internal long LocalTokens;
        internal double GlobalLoss;
        internal bool Exchanged;
        internal bool LayoutChecked;
        internal Vector<T>? Flat;
        internal T[]? Host;
        // A replay check still pending once the update runs in the plan: checked at the exchange, before the update.
        internal bool VerifyPending;
        internal double? EagerReplayLoss;
        internal double ReplayTolerance;
    }

    /// <summary>
    /// Whether a distributed step may run its update in the plan: plain DDP (every rank holds every parameter and its
    /// optimizer state), mixed precision only in the plan's own fused form, the update the plan's fused kernel computes exactly (no L2 term from the
    /// optimizer, no reduced-precision moments), and the coordinator opted in.
    /// </summary>
    private static bool DistributedCompiledUpdateEligible(
        NeuralNetworks.NeuralNetworkBase<T> distNet,
        DistributedTapeCoordinator<T> distCoord,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> eagerOptimizer,
        in DistributedFusedConfig config,
        FusedMixedPrecision? mixedPrecision)
    {
        string? why = !distCoord.CompiledOptimizerUpdate ? "not opted in"
            : distCoord.Strategy is not Enums.DistributedStrategy.DDP ? "strategy " + distCoord.Strategy + " shards the optimizer"
            : distNet.ActiveMixedPrecisionContext is not null && mixedPrecision is null ? "mixed precision has no fused form"
            : config.UseBf16Moments || config.Int8MomentBlockSize > 0 ? "reduced-precision moments"
            : distNet.GetDistributedOptimizerRegularizer(eagerOptimizer) is not null ? "the optimizer regularizes its gradients"
            : null;
        if (why is not null)
        {
            Fd("distributed in-plan update ineligible: " + why);
            // Asked for and declined: the step runs the eager data-parallel update instead, which must not go unseen.
            if (distCoord.CompiledOptimizerUpdate && s_reportedIneligibility.TryAdd(why, 0))
                System.Diagnostics.Trace.TraceWarning(
                    "DataParallelUpdateInCompiledPlan is on, but the data-parallel update runs eagerly: " + why + ".");
        }
        return why is null;
    }

    /// <summary>The optimizer the fused kernel runs, as the single-process fused path configures it.</summary>
    internal readonly struct DistributedFusedConfig
    {
        internal DistributedFusedConfig(OptimizerType type, double learningRate, double beta1, double beta2,
            double epsilon, double weightDecay, LrSchedule? schedule, FusedOptimizerExtras? extras,
            double l2Regularization, bool l2AfterClip, bool useBf16Moments, int int8MomentBlockSize)
        {
            Type = type; LearningRate = learningRate; Beta1 = beta1; Beta2 = beta2; Epsilon = epsilon;
            WeightDecay = weightDecay; Schedule = schedule; Extras = extras; L2Regularization = l2Regularization;
            L2AfterClip = l2AfterClip; UseBf16Moments = useBf16Moments; Int8MomentBlockSize = int8MomentBlockSize;
        }

        internal OptimizerType Type { get; }
        internal double LearningRate { get; }
        internal double Beta1 { get; }
        internal double Beta2 { get; }
        internal double Epsilon { get; }
        internal double WeightDecay { get; }
        internal LrSchedule? Schedule { get; }
        internal FusedOptimizerExtras? Extras { get; }
        internal double L2Regularization { get; }
        internal bool L2AfterClip { get; }
        internal bool UseBf16Moments { get; }
        internal int Int8MomentBlockSize { get; }
    }

    /// <summary>
    /// Every rank must take the same path each step (the paths run different collectives), so the switch to the
    /// in-plan update is agreed: it engages when every rank has a plan that is not this step's trace and no replay
    /// check is pending, and stays engaged.
    /// </summary>
    private static bool AgreeOnDistributedCompiledUpdate(DistributedTapeCoordinator<T> distCoord, bool localReady)
    {
        if (distCoord.WorldSize <= 1) return localReady;
        var numOps = MathHelper.GetNumericOperations<T>();
        var vote = new Vector<T>(1);
        vote.AsWritableSpan()[0] = localReady ? numOps.One : numOps.Zero;
        distCoord.Backend.AllReduce(vote, ReductionOperation.Min);
        return numOps.ToDouble(vote.AsSpan()[0]) > 0.5;
    }

    /// <summary>
    /// Points <paramref name="plan"/>'s gradient exchange at the all-reduce: this rank's loss and valid-token count
    /// give its weight (n_r / N, the global mean over tokens), its gradients are scaled by it and summed over ranks,
    /// and the plan's clip and fused update then run on the global gradient on every rank alike.
    /// </summary>
    private static void AttachDistributedExchange(CompiledTrainingPlan<T> plan, DistributedCompiledUpdate state)
    {
        plan.SetGradientExchange(exchange => ExchangeDistributedGradients(exchange, state));
    }

    [AiDotNet.Attributes.ScalarKernel("Host scaling of one step's flat gradient by this rank's token weight before the all-reduce")]
    private static void ExchangeDistributedGradients(GradientExchange<T> exchange, DistributedCompiledUpdate state)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var coordinator = state.Coordinator;
        // Page-locked path (float plans on a backend with it): every gradient and the loss in one batch of async copies
        // and a single wait, written back stream-ordered before the update. Otherwise the per-buffer Read/Write below.
        bool isStaged = exchange.TryReadStaged(out Span<float> staged, out float stagedLoss);
        double localLoss = state.LocalTokens <= 0 ? 0.0
            : isStaged ? stagedLoss : numOps.ToDouble(exchange.ReadLoss());
        if (state.VerifyPending)
        {
            // Before any weight or moment moves. A disagreement here cannot hand the step back to the eager path (the
            // plan already holds the optimizer's moments), so it stops training rather than continue on a wrong plan.
            state.VerifyPending = false;
            if (!VerifyDistributedReplayAgreement(coordinator, numOps, state.LocalTokens > 0, localLoss,
                    state.EagerReplayLoss, state.ReplayTolerance))
                throw new InvalidOperationException(
                    "The compiled plan's replay disagreed with the eager forward after the data-parallel update had moved "
                    + "into the plan: " + (CurrentState.distributedReplayDisagreementReason ?? "no reason recorded"));
        }

        var (rankWeight, globalLoss, _) = coordinator.SynchronizeStepMetadata(state.LocalTokens, localLoss);
        state.GlobalLoss = globalLoss;
        int length = exchange.TotalLength;
        if (!state.LayoutChecked && coordinator.WorldSize > 1)
        {
            // The ranks' gradient layouts must agree before any data moves, or the sum adds unrelated elements.
            if (!coordinator.RanksAgree(length))
                throw new InvalidOperationException(
                    $"Rank {coordinator.Rank}: the ranks' compiled plans exchange different gradient layouts "
                    + $"({length} elements here, a different count on another rank).");
            state.LayoutChecked = true;
        }

        if (isStaged)
        {
            ExchangeStaged(exchange, state, coordinator, staged, rankWeight);
            state.Exchanged = true;
            return;
        }

        var host = state.Host is { } h && h.Length == length ? h : state.Host = new T[length];
        exchange.Read(host);
        if (coordinator.WorldSize > 1)
        {
            var flat = state.Flat is { } f && f.Length == length ? f : state.Flat = new Vector<T>(length);
            var span = flat.AsWritableSpan();
            if (typeof(T) == typeof(float))
            {
                // SIMD (measured 2026-11-25: the per-element generic multiply took ~100 ms of a ~450 ms step for a
                // 16M-parameter model, about a fifth of the exchange).
                ScaleInto((float[])(object)host, (float)rankWeight, ((Vector<float>)(object)flat).AsWritableSpan());
            }
            else
            {
                T weight = numOps.FromDouble(rankWeight);
                for (int i = 0; i < length; i++) span[i] = numOps.Multiply(host[i], weight);
            }
            coordinator.Backend.AllReduce(flat, ReductionOperation.Sum);
            flat.AsSpan().CopyTo(host);
            exchange.Write(host);
        }
        else if (rankWeight != 1.0)
        {
            // One rank: its weight is 1 with tokens, 0 without (then there is nothing to average).
            T weight = numOps.FromDouble(rankWeight);
            for (int i = 0; i < length; i++) host[i] = numOps.Multiply(host[i], weight);
            exchange.Write(host);
        }
        state.Exchanged = true;
    }

    private static void ExchangeStaged(GradientExchange<T> exchange, DistributedCompiledUpdate state,
        DistributedTapeCoordinator<T> coordinator, Span<float> staged, double rankWeight)
    {
        int length = staged.Length;
        if (coordinator.WorldSize > 1)
        {
            var flat = state.Flat is { } f && f.Length == length ? f : state.Flat = new Vector<T>(length);
            var flatSpan = ((Vector<float>)(object)flat).AsWritableSpan();
            ScaleInto(staged, (float)rankWeight, flatSpan);
            coordinator.Backend.AllReduce(flat, ReductionOperation.Sum);
            ((Vector<float>)(object)flat).AsSpan().CopyTo(staged);
        }
        else if (rankWeight != 1.0)
        {
            ScaleInto(staged, (float)rankWeight, staged);
        }
        else
        {
            return;   // nothing changed: the device gradients are already the step's gradients
        }
        exchange.WriteStaged();
    }

    private static void ScaleInto(ReadOnlySpan<float> source, float factor, Span<float> destination)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            var f = new System.Numerics.Vector<float>(factor);
            var src = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(source);
            var dst = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(destination);
            for (int k = 0; k < src.Length; k++) dst[k] = src[k] * f;
            i = src.Length * System.Numerics.Vector<float>.Count;
        }
        for (; i < source.Length; i++) destination[i] = source[i] * factor;
    }

    /// <summary>
    /// One engaged step: configures the plan's fused optimizer on the first (taking over the eager optimizer's state,
    /// so Adam continues the eager steps' moments and step count) or continues the previous plan's (a new batch shape),
    /// then runs the step, whose exchange reduces the gradients across ranks before the plan's update.
    /// </summary>
    private static T RunDistributedCompiledUpdate(
        NeuralNetworks.NeuralNetworkBase<T> distNet,
        DistributedTapeCoordinator<T> distCoord,
        ICompiledTrainingPlan<T> plan,
        Tensor<T>[] parameters,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> eagerOptimizer,
        in DistributedFusedConfig config,
        FusedMixedPrecision? mixedPrecision,
        double maxGradNorm,
        long localTokens,
        bool verifyDistributedReplay,
        double? distributedEagerReplayLoss,
        double distributedReplayTolerance,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients)
    {
        if (plan is not CompiledTrainingPlan<T> concrete)
            throw new InvalidOperationException("The data-parallel in-plan update needs the AiDotNet.Tensors compiled plan.");
        var state = CurrentState.distributedCompiledUpdate as DistributedCompiledUpdate;
        if (state is null)
        {
            state = new DistributedCompiledUpdate { Coordinator = distCoord };
            CurrentState.distributedCompiledUpdate = state;
        }

        if (!ReferenceEquals(state.Plan, plan))
        {
            if (state.Plan is { } previous)
            {
                // A new batch shape's plan takes over the moments, clip and step from the plan that held them.
                plan.ContinueOptimizerFrom(previous);
                if (previous is CompiledTrainingPlan<T> previousConcrete) previousConcrete.SetGradientExchange(null);
            }
            else
            {
                if (config.Schedule is { } schedule)
                    plan.ConfigureOptimizer(config.Type, schedule, config.Beta1, config.Beta2, config.Epsilon,
                        config.WeightDecay, config.Extras);
                else
                    plan.ConfigureOptimizer(config.Type, config.LearningRate, config.Beta1, config.Beta2, config.Epsilon,
                        config.WeightDecay, config.Extras);
                // The reference path clips at maxGradNorm in the coordinator and the optimizer instance then clips at its
                // own threshold too, so the global clip it applies is the smaller of the two.
                maxGradNorm = NeuralNetworks.NeuralNetworkBase<T>.EffectiveGlobalGradNormClip(eagerOptimizer, maxGradNorm);
                if (maxGradNorm > 0.0) plan.SetMaxGradNorm(maxGradNorm);
                // The optimizer's L2 term, in the order the eager step applies it (after the exchange, as the eager
                // reference adds it to the reduced gradient).
                if (config.L2AfterClip)
                {
                    if (plan is not ICompiledTrainingPlanRegularizationOrder ordered)
                        throw new InvalidOperationException("The compiled plan cannot add the L2 term after the clip.");
                    ordered.SetL2Regularization(config.L2Regularization, GradientRegularizationOrder.ClipThenRegularize);
                }
                else if (config.L2Regularization > 0.0)
                    plan.SetL2Regularization(config.L2Regularization);
                // The eager optimizer ran this run's steps until now; its moments and step count move into the plan.
                var linked = LinkFusedOptimizerState(plan, eagerOptimizer);
                linked?.MarkPendingFusedOptimizerStateInstalled();
                Fd($"distributed in-plan update engaged: eager optimizer state {(linked is null ? "NOT installed" : "installed")}, "
                    + $"plan optimizer step {concrete.OptimizerStep}, clip {maxGradNorm}, l2 {config.L2Regularization} after clip {config.L2AfterClip}, "
                    + $"type {config.Type}, wd {config.WeightDecay}, eps {config.Epsilon}, mixed precision {mixedPrecision?.Signature ?? "off"}, "
                    + $"engine {AiDotNet.Tensors.Engines.AiDotNetEngine.Current.GetType().Name}");
                if (eagerOptimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> exported)
                    AttachFusedOptimizerStateLink(plan, exported);
            }
            AttachDistributedExchange(concrete, state);
            state.Plan = plan;
            state.LayoutChecked = false;
        }

        // The plan's loss scaling (and its carry-over to a new shape's plan), as the single-process fused step sets it.
        if (!TryConfigureMixedPrecision(plan, mixedPrecision))
            throw _lastFallbackException ?? new InvalidOperationException("The compiled plan refused the loss scaling.");

        state.LocalTokens = localTokens;
        state.Exchanged = false;
        state.VerifyPending = verifyDistributedReplay;
        state.EagerReplayLoss = distributedEagerReplayLoss;
        state.ReplayTolerance = distributedReplayTolerance;
        distNet.MarkDistributedMutationStarted();
        plan.Step();
        if (!state.Exchanged)
            throw new InvalidOperationException("The compiled plan stepped without running its gradient exchange.");
        System.Threading.Interlocked.Increment(ref s_compiledUpdateSteps);
        System.Threading.Interlocked.Exchange(ref s_lastExchanges, concrete.GradientExchangeCount);
        System.Threading.Interlocked.Exchange(ref s_lastUpdateLaunches, concrete.UpdateGraphReplayCount);
        System.Threading.Volatile.Write(ref s_lastSplit, concrete.HasSplitStepGraph ? 1 : 0);
        // The batch-end bookkeeping every step takes (the single-process fused step's OnBatchEnd): the optimizer's
        // scheduler position, which a per-batch schedule's plan rate follows.
        // Not FinalizeDistributedStepPostUpdate: its host-authoritative cache invalidation would treat the host copies
        // as current, and the plan updated the weights on the device (the caller invalidates for a device update).
        if (eagerOptimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled)
            scheduled.OnBatchEnd();
        if (s_fusedDebug)
            Fd($"distributed in-plan step: {PlanStepDiagnostics(plan)}, eager lr {eagerOptimizer.GetCurrentLearningRate():G9}, "
                + $"global loss {state.GlobalLoss:G9}, tokens {localTokens}, schedule {(config.Schedule is null ? "none" : config.Schedule.GetType().Name)}");
        PublishFusedGradients(onGradients, parameters, plan);
        var numOps = MathHelper.GetNumericOperations<T>();
        T loss = numOps.FromDouble(state.GlobalLoss);
        distNet.SetLastLossFromStepper(loss);
        return loss;
    }
}
