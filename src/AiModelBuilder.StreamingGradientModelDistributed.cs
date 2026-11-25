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
    /// and its parameters is synchronized by <see cref="ExecuteStreamingDistributedGradientStep"/>.
    /// </para>
    /// <para>
    /// A backend with one rank is not a distributed run: there is no peer to exchange with, so the model takes the
    /// same per-sample updates it takes with no backend configured and ends on the same parameters.
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
    /// Runs one data-parallel step of a gradient model: the update every rank applies is the mean gradient over
    /// the whole global batch, as if one process had trained on all of it.
    /// </summary>
    /// <param name="inputs">This rank's slice of the global batch; empty on a rank the tail step left idle.</param>
    /// <param name="outputs">The targets matching <paramref name="inputs"/>.</param>
    /// <param name="lossFunction">The loss whose gradient and value are computed.</param>
    /// <param name="distBackend">The communication backend shared by the ranks.</param>
    /// <param name="globalSampleCount">The number of samples in the global batch across all ranks.</param>
    /// <returns>The mean loss over the global batch, measured before the update.</returns>
    /// <remarks>
    /// Each rank sums its per-sample gradients, the sums are all-reduced, and the total is divided by the global
    /// sample count. Ranks with different slice sizes are therefore weighted by their sample counts, and an idle
    /// rank contributes zeros but still applies the same update.
    /// </remarks>
    private T ExecuteStreamingDistributedGradientStep(
        TInput[] inputs,
        TOutput[] outputs,
        ILossFunction<T> lossFunction,
        DistributedTraining.ICommunicationBackend<T> distBackend,
        long globalSampleCount)
    {
        var model = _model ?? throw new InvalidOperationException("Streaming training requires a configured model.");
        var gradientModel = InterfaceGuard.GradientComputable(model);
        var numOps = MathHelper.GetNumericOperations<T>();
        int parameterCount = InterfaceGuard.Parameterizable(model).GetParameters().Length;

        var gradientSum = new Vector<T>(parameterCount);
        T lossSum = numOps.Zero;
        for (int i = 0; i < inputs.Length; i++)
        {
            // Loss and gradient are both taken at the parameters the step started from.
            lossSum = AddLossScalars(lossSum, SampleLoss(model, lossFunction, inputs[i], outputs[i]));
            Vector<T> sampleGradient = gradientModel.ComputeGradients(inputs[i], outputs[i], lossFunction);
            if (sampleGradient.Length != parameterCount)
            {
                throw new InvalidOperationException(
                    $"{model.GetType().Name}.ComputeGradients returned {sampleGradient.Length} values for a model with " +
                    $"{parameterCount} parameters; distributed streaming training needs one gradient per parameter.");
            }

            gradientSum = gradientSum.Add(sampleGradient);
        }

        // One collective carries the gradient sum and the loss sum.
        Vector<T> reduced = Vector<T>.Concatenate(
            new List<Vector<T>> { gradientSum, new Vector<T>(new[] { lossSum }) });
        distBackend.AllReduce(reduced, DistributedTraining.ReductionOperation.Sum);

        // Every rank holds the same parameters and the same reduced sum, so the penalty and the update are the
        // same on every rank.
        T sampleCount = numOps.FromDouble(globalSampleCount);
        Vector<T> meanGradient = RegularizeStreamingGradient(reduced.GetSubVector(0, parameterCount).Divide(sampleCount));
        gradientModel.ApplyGradients(meanGradient, numOps.FromDouble(ResolveStreamingGradientModelLearningRate()));
        EndStreamingGradientModelBatch();
        return numOps.Divide(reduced[parameterCount], sampleCount);
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
    /// Refuses gradient accumulation for a model the streaming loop updates sample by sample.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// <c>GradientAccumulationSteps</c> is above 1 and the model is not a neural network.
    /// </exception>
    private void EnsureStreamingGradientAccumulationIsHonoured()
    {
        int accumulationSteps = _streamingTrainingOptions?.GradientAccumulationSteps ?? 1;
        if (accumulationSteps > 1 && _model is not null && _model is not INeuralNetwork<T>)
        {
            throw new NotSupportedException(
                $"GradientAccumulationSteps = {accumulationSteps} is not supported for {_model.GetType().Name} on the " +
                "streaming data-loader path: only neural networks accumulate micro-batches there, and the run " +
                "would stop after fewer samples than the step budget asks for. Set GradientAccumulationSteps to 1.");
        }
    }

    /// <summary>
    /// Ends a loader batch for a model the streaming loop updates itself: a learning-rate scheduler stepping per
    /// batch advances here, so the next update uses the next scheduled rate.
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
