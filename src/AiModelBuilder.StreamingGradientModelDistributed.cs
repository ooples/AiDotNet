using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

/// <content>
/// Distributed streaming training for models that train through <see cref="IGradientComputable{T, TInput, TOutput}"/>
/// instead of the autodiff tape (classical online learners such as linear and logistic regression).
/// </content>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Gets whether the streaming run trains a non-neural gradient model across more than one rank.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Neural networks synchronize on the tape path through
    /// <see cref="DistributedTraining.DistributedTapeCoordinator{T}"/>. Every other model that exposes its gradient
    /// and its parameters is synchronized by <see cref="AccumulateStreamingGradientModelStep"/>.
    /// </para>
    /// <para>
    /// A backend with one rank is not a distributed run: there is no peer to exchange with, so the model takes the
    /// step it takes with no backend configured and ends on the same parameters.
    /// </para>
    /// </remarks>
    private bool IsStreamingGradientModelDistributed(DistributedTraining.ICommunicationBackend<T>? distBackend)
    {
        return distBackend is { WorldSize: > 1 }
            && _model is not null
            && _model is not INeuralNetwork<T>
            && _model is IGradientComputable<T, TInput, TOutput>
            && _model is IParameterizable<T, TInput, TOutput>;
    }

    /// <summary>
    /// Rejects a multi-rank streaming run whose model has no gradient this path can synchronize across ranks.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// More than one rank is configured and the model is neither a <see cref="NeuralNetworks.NeuralNetworkBase{T}"/>
    /// nor a non-neural model implementing both <see cref="IGradientComputable{T, TInput, TOutput}"/> and
    /// <see cref="IParameterizable{T, TInput, TOutput}"/>.
    /// </exception>
    private void EnsureStreamingModelSupportsDistributedTraining(DistributedTraining.ICommunicationBackend<T>? distBackend)
    {
        if (distBackend is not { WorldSize: > 1 }
            || _model is NeuralNetworks.NeuralNetworkBase<T>
            || IsStreamingGradientModelDistributed(distBackend))
        {
            return;
        }

        throw new NotSupportedException(
            $"Distributed streaming training cannot synchronize {_model?.GetType().Name ?? "a null model"} across " +
            $"{distBackend.WorldSize} ranks: the model must derive from NeuralNetworkBase<T> (tape path) or implement " +
            "both IGradientComputable and IParameterizable (gradient path).");
    }

    /// <summary>
    /// Gives every rank rank 0's parameters before the first step, so all ranks start from the same model.
    /// </summary>
    private void SynchronizeStreamingGradientModelInitialParameters(DistributedTraining.ICommunicationBackend<T> distBackend)
    {
        if (_model is null)
        {
            return;
        }

        var parameterizable = InterfaceGuard.Parameterizable(_model);
        Vector<T> localParameters = parameterizable.GetParameters();
        EnsureRanksAgreeOnParameterCount(distBackend, localParameters.Length);
        if (localParameters.Length == 0)
        {
            return;
        }

        Vector<T> rankZeroParameters = distBackend.Broadcast(localParameters, root: 0);
        if (distBackend.Rank != 0)
        {
            parameterizable.SetParameters(rankZeroParameters);
        }
    }

    /// <summary>
    /// Fails on every rank when the ranks hold models with different parameter counts.
    /// </summary>
    /// <remarks>
    /// The count travels as 7-bit chunks so it stays exact for every numeric type. Chunk-wise minimum and maximum
    /// are equal exactly when every rank sent the same count, and every rank sees the same two reductions, so a
    /// mismatch throws everywhere instead of leaving some ranks blocked in the next collective.
    /// </remarks>
    private static void EnsureRanksAgreeOnParameterCount(
        DistributedTraining.ICommunicationBackend<T> distBackend,
        int localParameterCount)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var lowest = new Vector<T>(5);
        var highest = new Vector<T>(5);
        WriteIntAsU7Chunks(numOps, lowest.AsWritableSpan(), 0, localParameterCount);
        WriteIntAsU7Chunks(numOps, highest.AsWritableSpan(), 0, localParameterCount);
        distBackend.AllReduce(lowest, DistributedTraining.ReductionOperation.Min);
        distBackend.AllReduce(highest, DistributedTraining.ReductionOperation.Max);

        int lowestCount = ReadIntFromU7Chunks(numOps, lowest.AsSpan(), 0);
        int highestCount = ReadIntFromU7Chunks(numOps, highest.AsSpan(), 0);
        if (lowestCount != highestCount)
        {
            throw new InvalidOperationException(
                "Distributed streaming training requires the same model on every rank, but the ranks report different " +
                $"parameter counts (rank {distBackend.Rank} has {localParameterCount}). Configure every rank with the " +
                "same model architecture and fitted shape.");
        }
    }

    /// <summary>
    /// What a gradient model has accumulated toward its next update: the gradient and loss sums of the
    /// micro-batches seen since the last update, and how many samples they cover.
    /// </summary>
    private sealed class StreamingGradientStepState
    {
        public StreamingGradientStepState(T zero)
        {
            LossSum = zero;
        }

        public Vector<T>? GradientSum { get; set; }

        public T LossSum { get; set; }

        public long SampleCount { get; set; }

        public int MicroBatches { get; set; }
    }

    /// <summary>
    /// Adds one loader batch to a gradient model's pending update and, once
    /// <paramref name="accumulationSteps"/> batches are in, applies the update.
    /// </summary>
    /// <param name="state">The pending update.</param>
    /// <param name="inputs">This process's samples of the batch; empty on a rank the tail step left idle.</param>
    /// <param name="outputs">The targets matching <paramref name="inputs"/>.</param>
    /// <param name="lossFunction">The loss whose gradient and value are computed.</param>
    /// <param name="distBackend">The backend shared by the ranks, or <c>null</c> for one process.</param>
    /// <param name="globalSampleCount">The number of samples in the batch across all ranks.</param>
    /// <param name="accumulationSteps">How many batches make one update.</param>
    /// <param name="stepLoss">The mean loss over the samples of the update, measured before it.</param>
    /// <returns><c>true</c> when the update was applied; <c>false</c> while batches are still accumulating.</returns>
    /// <remarks>
    /// The update is the mean gradient over every sample of the batch, taken at the parameters the step started
    /// from. That is the same rule for one process and for several ranks: the per-sample gradients are summed in
    /// global sample order and the total is divided by the global sample count, so ranks with different slice
    /// sizes are weighted by their sample counts and an idle rank adds nothing but still applies the same update.
    /// N ranks therefore take, bit for bit, the step one process takes on the same batch.
    /// </remarks>
    private bool AccumulateStreamingGradientModelStep(
        StreamingGradientStepState state,
        TInput[] inputs,
        TOutput[] outputs,
        ILossFunction<T> lossFunction,
        DistributedTraining.ICommunicationBackend<T>? distBackend,
        long globalSampleCount,
        int accumulationSteps,
        out T stepLoss)
    {
        var model = _model ?? throw new InvalidOperationException("Streaming training requires a configured model.");
        var gradientModel = InterfaceGuard.GradientComputable(model);
        var numOps = MathHelper.GetNumericOperations<T>();

        int parameterCount = model is IParameterizable<T, TInput, TOutput> parameterizable
            ? parameterizable.GetParameters().Length
            : -1;

        // Loss and gradient of one local sample, taken at the parameters the step started from (nothing is
        // applied before the whole update is).
        (Vector<T> Gradient, T Loss) Evaluate(int i)
        {
            T loss = SampleLoss(model, lossFunction, inputs[i], outputs[i]);
            Vector<T> gradient = gradientModel.ComputeGradients(inputs[i], outputs[i], lossFunction);
            if (parameterCount < 0)
            {
                parameterCount = gradient.Length;
            }

            if (gradient.Length != parameterCount)
            {
                throw new InvalidOperationException(
                    $"{model.GetType().Name}.ComputeGradients returned {gradient.Length} values where " +
                    $"{parameterCount} were expected; streaming training needs one gradient per parameter.");
            }

            return (gradient, loss);
        }

        // The running sums continue from the batches already accumulated, and every sample is added to them one
        // at a time in global sample order. One process does exactly that, holding one gradient at a time. Several
        // ranks do the same by passing the running sums along in rank order (the loader hands rank r the r-th
        // contiguous slice of the global batch): each rank adds its own samples and broadcasts the result to the
        // others. The additions and their order are then the ones a single process performs, so the result is the
        // same bit for bit, which a sum of per-rank sums is not. The price is one broadcast per rank and step in
        // place of one all-reduce, and each rank keeping its slice's per-sample gradients until its turn (they
        // are computed up front so the ranks work in parallel).
        Vector<T>? gradientSum = state.GradientSum;
        T lossSum = state.LossSum;
        if (distBackend is { WorldSize: > 1 })
        {
            var evaluated = new (Vector<T> Gradient, T Loss)[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                evaluated[i] = Evaluate(i);
            }

            if (parameterCount < 0)
            {
                throw new InvalidOperationException(
                    $"{model.GetType().Name} was given an empty batch and does not expose its parameters, so the " +
                    "size of its gradient is unknown.");
            }

            Vector<T> runningSum = gradientSum ?? new Vector<T>(parameterCount);
            for (int turn = 0; turn < distBackend.WorldSize; turn++)
            {
                if (turn == distBackend.Rank)
                {
                    foreach (var (gradient, loss) in evaluated)
                    {
                        runningSum = runningSum.Add(gradient);
                        lossSum = AddLossScalars(lossSum, loss);
                    }
                }

                (runningSum, lossSum) = BroadcastRunningSums(distBackend, turn, runningSum, lossSum);
            }

            gradientSum = runningSum;
        }
        else
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var (gradient, loss) = Evaluate(i);
                gradientSum = (gradientSum ?? new Vector<T>(parameterCount)).Add(gradient);
                lossSum = AddLossScalars(lossSum, loss);
            }
        }

        if (gradientSum is null)
        {
            // An empty batch with nothing accumulated before it: there is nothing to add and nothing to apply.
            stepLoss = numOps.Zero;
            return false;
        }

        state.GradientSum = gradientSum;
        state.LossSum = lossSum;
        state.SampleCount += globalSampleCount;
        state.MicroBatches++;
        if (state.MicroBatches < accumulationSteps)
        {
            stepLoss = numOps.Zero;
            return false;
        }

        stepLoss = ApplyAccumulatedStreamingGradientModelStep(state);
        return true;
    }

    /// <summary>
    /// Hands the running gradient and loss sums from the rank whose turn it is to every other rank, in one
    /// collective that carries both.
    /// </summary>
    private static (Vector<T> GradientSum, T LossSum) BroadcastRunningSums(
        DistributedTraining.ICommunicationBackend<T> backend,
        int root,
        Vector<T> gradientSum,
        T lossSum)
    {
        int parameterCount = gradientSum.Length;
        Vector<T> running = backend.Broadcast(
            Vector<T>.Concatenate(new List<Vector<T>> { gradientSum, new Vector<T>(new[] { lossSum }) }),
            root);
        return (running.GetSubVector(0, parameterCount), running[parameterCount]);
    }

    /// <summary>
    /// Applies a gradient model's pending update, the mean gradient over every accumulated sample, and clears it.
    /// Also used to flush the micro-batches left over at the end of an epoch.
    /// </summary>
    /// <returns>The mean loss over the accumulated samples.</returns>
    private T ApplyAccumulatedStreamingGradientModelStep(StreamingGradientStepState state)
    {
        var model = _model ?? throw new InvalidOperationException("Streaming training requires a configured model.");
        var numOps = MathHelper.GetNumericOperations<T>();
        Vector<T> gradientSum = state.GradientSum
            ?? throw new InvalidOperationException("There is no accumulated gradient to apply.");
        T lossSum = state.LossSum;
        long accumulatedSamples = state.SampleCount;
        state.GradientSum = null;
        state.LossSum = numOps.Zero;
        state.SampleCount = 0;
        state.MicroBatches = 0;
        if (accumulatedSamples <= 0)
        {
            return numOps.Zero;
        }

        // Every rank holds the same parameters and the same sums, so the penalty and the update are the same on
        // every rank.
        T sampleCount = numOps.FromDouble(accumulatedSamples);
        Vector<T> meanGradient = RegularizeStreamingGradient(gradientSum.Divide(sampleCount));
        InterfaceGuard.GradientComputable(model).ApplyGradients(
            meanGradient, numOps.FromDouble(ResolveStreamingGradientModelLearningRate()));
        EndStreamingGradientModelBatch();
        return numOps.Divide(lossSum, sampleCount);
    }

    /// <summary>
    /// Adds the <see cref="ConfigureRegularization"/> penalty to a gradient-model gradient, taken at the model's
    /// current parameters (for L2, <c>gradient + strength * parameters</c>).
    /// </summary>
    private Vector<T> RegularizeStreamingGradient(Vector<T> gradient)
    {
        if (_regularization is null || _model is null)
        {
            return gradient;
        }

        Vector<T> parameters = InterfaceGuard.Parameterizable(_model).GetParameters();
        if (_regularization is Regularization.RegularizationBase<T, TInput, TOutput> regularizationBase)
        {
            return regularizationBase.Regularize(gradient, parameters);
        }

        if (gradient is TOutput typedGradient && parameters is TOutput typedParameters
            && _regularization.Regularize(typedGradient, typedParameters) is Vector<T> regularized)
        {
            return regularized;
        }

        throw new NotSupportedException(
            $"{_regularization.GetType().Name} cannot regularize the parameter gradient of {_model.GetType().Name}.");
    }

    /// <summary>
    /// Ends an update of a model the streaming loop updates itself: a learning-rate scheduler stepping per batch
    /// advances here, so the next update uses the next scheduled rate.
    /// </summary>
    /// <remarks>
    /// A neural network ends its batches inside its own optimizer step, and distillation trains at a fixed rate.
    /// </remarks>
    private void EndStreamingGradientModelBatch()
    {
        if (_model is not INeuralNetwork<T>
            && _knowledgeDistillationOptions is null
            && _optimizer is Optimizers.GradientBasedOptimizerBase<T, TInput, TOutput> gradientOptimizer)
        {
            gradientOptimizer.OnBatchEnd();
        }
    }

    /// <summary>
    /// Ends an epoch for a model the streaming loop updates itself, so a scheduler stepping per epoch advances.
    /// </summary>
    private void EndStreamingGradientModelEpoch()
    {
        if (_model is not INeuralNetwork<T>
            && _knowledgeDistillationOptions is null
            && _optimizer is Optimizers.GradientBasedOptimizerBase<T, TInput, TOutput> gradientOptimizer)
        {
            gradientOptimizer.OnEpochEnd();
        }
    }

    /// <summary>
    /// The learning rate of a gradient-model step: the optimizer's current rate when it is gradient-based (the
    /// rate its scheduler has reached), its configured initial rate otherwise, and 0.01 when no optimizer is
    /// configured.
    /// </summary>
    private double ResolveStreamingGradientModelLearningRate()
    {
        if (_optimizer is IGradientBasedOptimizer<T, TInput, TOutput> gradientOptimizer)
        {
            return gradientOptimizer.GetCurrentLearningRate();
        }

        return _optimizer is not null ? _optimizer.GetOptions().InitialLearningRate : 0.01;
    }
}
