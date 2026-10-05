using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// One training step for any model that trains on the gradient tape: the fused compiled plan when it applies (CPU
/// or GPU), otherwise the eager tape with the model's optimizer. Model base classes own one and call
/// <see cref="Step"/> per batch, so no model writes its own tape loop or its own GPU-resident path.
/// </summary>
/// <remarks>
/// The eager branch is the loop the time-series, diffusion and generative models each carried a copy of: record
/// forward and loss on a tape, take the gradients of the trained tensors, hand them to
/// <see cref="IGradientBasedOptimizer{T, TInput, TOutput}.Step(TapeStepContext{T})"/>, advance the optimizer's
/// schedule. The fused branch is <see cref="FusedTrainingSession{T}"/>.
/// </remarks>
internal sealed class TapeTrainingStepper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <param name="owner">The model being trained; keys its compiled plan.</param>
    /// <param name="onReset">Owner cleanup when a committed fused plan is dropped.</param>
    public TapeTrainingStepper(object owner, Action? onReset = null)
    {
        Guard.NotNull(owner);
        Session = new FusedTrainingSession<T>(owner, onReset);
    }

    /// <summary>The fused lifecycle; exposed for diagnostics and explicit resets.</summary>
    public FusedTrainingSession<T> Session { get; }

    /// <summary>Whether the most recent <see cref="Step"/> ran on the fused compiled plan.</summary>
    public bool LastStepFused { get; private set; }

    /// <summary>Trains one batch and returns its loss.</summary>
    public T Step(FusedTrainingStepRequest<T> request)
    {
        Guard.NotNull(request);
        var outcome = Session.TryStep(request, out T fusedLoss);
        if (outcome == FusedStepOutcome.Stepped)
        {
            LastStepFused = true;
            return fusedLoss;
        }

        if (outcome == FusedStepOutcome.CommittedFailure)
        {
            var cause = Session.LastFallbackException;
            // Device memory pressure or a transient device fault: the plan is unusable for this run, but nothing
            // about the model is wrong, so continue on the eager tape from the current weights.
            if (!IsGpuOutOfMemoryFailure(cause) && !IsGpuTransientFailure(cause))
                throw CommittedPlanCannotContinue(cause);
            Session.Reset(stickyDisable: true);
        }

        LastStepFused = false;
        return EagerStep(request);
    }

    /// <summary>Forward, loss and backward on the tape, then the model's optimizer.</summary>
    public static T EagerStep(FusedTrainingStepRequest<T> request)
    {
        Guard.NotNull(request);
        var parameters = TrainedParameters(request);
        foreach (var layer in request.Layers)
            layer.ZeroGrad();

        Tensor<T> loss;
        Dictionary<Tensor<T>, Tensor<T>> gradients;
        using (var tape = new GradientTape<T>())
        {
            var predicted = request.Forward(request.Input);
            loss = request.ComputeLoss(predicted, request.Target);
            var all = tape.ComputeGradients(loss, parameters);
            gradients = new Dictionary<Tensor<T>, Tensor<T>>(parameters.Length, TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var parameter in parameters)
                if (all.TryGetValue(parameter, out var gradient))
                    gradients[parameter] = gradient;
        }

        request.OnGradients?.Invoke(gradients);
        if (request.ModelGradientClip > 0.0)
            ClipByGlobalNorm(gradients, request.ModelGradientClip);

        T lossValue = loss.Length > 0 ? loss[0] : NumOps.Zero;
        var forward = request.Forward;
        var computeLoss = request.ComputeLoss;
        var context = new TapeStepContext<T>(
            parameters, gradients, lossValue,
            request.Input, request.Target,
            (input, _) => forward(input),
            (predicted, target) => computeLoss(predicted, target),
            parameterBuffer: null);
        request.Optimizer.Step(context);
        if (request.Optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled)
            scheduled.OnBatchEnd();
        return lossValue;
    }

    /// <summary>A device out-of-memory failure, possibly wrapped.</summary>
    internal static bool IsGpuOutOfMemoryFailure(Exception? exception)
    {
        for (var e = exception; e is not null; e = e.InnerException)
        {
            if (e is OutOfMemoryException)
                return true;
            var message = e.Message;
            if (message.Contains("Out of memory", StringComparison.OrdinalIgnoreCase)
                && (message.Contains("cuMem", StringComparison.Ordinal)
                    || message.Contains("CUDA", StringComparison.OrdinalIgnoreCase)
                    || message.Contains("GPU", StringComparison.OrdinalIgnoreCase)
                    || message.Contains("CL_OUT_OF", StringComparison.OrdinalIgnoreCase)))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>A transient device fault (driver, stream, launch, or a buffer released under the step).</summary>
    internal static bool IsGpuTransientFailure(Exception? exception)
    {
        for (var e = exception; e is not null; e = e.InnerException)
        {
            var message = e.Message;
            if (message.Contains("CUDA error", StringComparison.OrdinalIgnoreCase)
                || message.Contains("cuMem", StringComparison.Ordinal)
                || message.Contains("cuStream", StringComparison.Ordinal)
                || message.Contains("cuLaunch", StringComparison.Ordinal)
                || message.Contains("OpenCL", StringComparison.OrdinalIgnoreCase)
                || message.Contains("released before materialization", StringComparison.OrdinalIgnoreCase)
                || message.Contains("buffer was released", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>The error for a committed fused plan that cannot run the next step.</summary>
    internal static InvalidOperationException CommittedPlanCannotContinue(Exception? cause)
    {
        string rootCause = cause is not null
            ? $" Root-cause exception (caught in CompiledTapeTrainingStep): {cause.GetType().FullName}: {cause.Message}"
            : " (No exception was caught: the fused path returned false from one of the explicit refuse paths: plan "
              + "reference changed, optimizer hyperparameters drifted, TensorCodecOptions.EnableCompilation=false, or "
              + "numeric/optimizer type unsupported.)";
        return new InvalidOperationException(
            "Fused compiled training has already run successfully, but the current step cannot engage the fused "
            + "path. The plan-embedded optimizer state cannot be transferred to the eager optimizer, so falling back "
            + "silently would produce a trajectory that diverges from the previous fused steps. Common causes: "
            + "variable input/target shape (new compiled plan), LR scheduler or adaptive-rate changes, attached "
            + "AMSGrad, or a kernel-level exception in plan.Step/ConfigureOptimizer that was caught and swallowed. "
            + "Resolution: keep shapes and optimizer hyperparameters stable across steps, OR reset the model's "
            + "training state, OR disable compilation "
            + "(AiModelBuilder.ConfigureJitCompilation(JitCompilationConfig.Disabled))." + rootCause,
            cause);
    }

    private static Tensor<T>[] TrainedParameters(FusedTrainingStepRequest<T> request)
    {
        if (request.Selection is not null)
            return request.Selection.ToArray();
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        var parameters = new List<Tensor<T>>();
        foreach (var layer in request.Layers)
            foreach (var parameter in layer.GetTrainableParameters())
                if (parameter is not null && seen.Add(parameter))
                    parameters.Add(parameter);
        if (request.ExtraParameters is not null)
            foreach (var parameter in request.ExtraParameters)
                if (seen.Add(parameter))
                    parameters.Add(parameter);
        return parameters.ToArray();
    }

    // PyTorch clip_grad_norm_ on the engine: scale every gradient by max / (norm + 1e-6) when the norm exceeds max.
    private static void ClipByGlobalNorm(Dictionary<Tensor<T>, Tensor<T>> gradients, double maxNorm)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        Tensor<T>? total = null;
        foreach (var gradient in gradients.Values)
        {
            if (gradient.Length == 0) continue;
            var squares = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(gradient, gradient), null), new[] { 1 });
            total = total is null ? squares : engine.TensorAdd(total, squares);
        }
        if (total is null) return;
        double norm = Math.Sqrt(NumOps.ToDouble(total[0]));
        if (!(norm > maxNorm) || double.IsInfinity(norm)) return;
        T scale = NumOps.FromDouble(maxNorm / (norm + 1e-6));
        foreach (var gradient in gradients.Values)
            engine.TensorMultiplyScalarInPlace(gradient, scale);
    }
}