using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Builds the per-batch training step for a streaming run configured with
    /// <see cref="ConfigureKnowledgeDistillation"/>, or returns <c>null</c> when distillation is not configured.
    /// </summary>
    /// <exception cref="NotSupportedException">The student does not expose its gradient.</exception>
    private Func<TInput[], TOutput[], T>? CreateStreamingDistillationStep()
    {
        if (_configuredDistillationStrategy is not null && _knowledgeDistillationOptions is null)
        {
            _knowledgeDistillationOptions = new KnowledgeDistillationOptions<T, TInput, TOutput>
            {
                Strategy = _configuredDistillationStrategy,
            };
        }

        var kd = _knowledgeDistillationOptions;
        if (kd is null)
        {
            return null;
        }

        var model = _model ?? throw new InvalidOperationException("Streaming training requires a configured model.");
        if (model is not IGradientComputable<T, TInput, TOutput> student)
        {
            throw new NotSupportedException(
                $"ConfigureKnowledgeDistillation requires a gradient-computable student, but model " +
                $"'{model.GetType().Name}' does not implement IGradientComputable<T, TInput, TOutput>. " +
                "Use a neural-network student, or drive distillation manually post-build via " +
                "AiModelResult.KnowledgeDistillationOptions.");
        }

        var teacherForward = ResolveTeacherForward(kd);
        KnowledgeDistillation.DistillationLossAdapter<T>? lossAdapterSlot = null;
        KnowledgeDistillation.DistillationLossAdapter<T> AdapterFor(Tensor<T> teacherOutput)
            => lossAdapterSlot ??= new KnowledgeDistillation.DistillationLossAdapter<T>(
                KnowledgeDistillation.DistillationStrategyFactory<T>.ResolveStrategyForOutputWidth(
                    kd.Strategy, kd.Temperature, kd.Alpha,
                    outputClasses: teacherOutput.Shape.Length > 0
                        ? teacherOutput.Shape[teacherOutput.Shape.Length - 1]
                        : teacherOutput.Length,
                    kd.SparseTopK, kd.SparseTopKMinimumClasses));

        var numOps = MathHelper.GetNumericOperations<T>();
        double effectiveLr = _optimizer?.GetOptions()?.InitialLearningRate ?? kd.LearningRate;
        T learningRate = numOps.FromDouble(effectiveLr);
        int maxSubBatchSize = Math.Max(1, kd.BatchSize);

        var nnStudent = student as NeuralNetworks.NeuralNetworkBase<T>;
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? tensorGradOptimizer =
            (object?)_optimizer as IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>;
        if (nnStudent is not null && _optimizer is null)
        {
            tensorGradOptimizer = new Optimizers.AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                nnStudent,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = effectiveLr,
                    UseAMSGrad = false,
                });
        }

        var typedGradOptimizer = _optimizer as IGradientBasedOptimizer<T, TInput, TOutput>;

        return (inputs, targets) =>
        {
            if (inputs.Length == 0)
            {
                return numOps.Zero;
            }

            T lossSum = numOps.Zero;
            int lossBatches = 0;

            for (int start = 0; start < inputs.Length; start += maxSubBatchSize)
            {
                int count = Math.Min(maxSubBatchSize, inputs.Length - start);
                var sampleInputs = new Tensor<T>[count];
                var sampleTargets = new Tensor<T>[count];
                for (int j = 0; j < count; j++)
                {
                    if (inputs[start + j] is not Tensor<T> si || targets[start + j] is not Tensor<T> st)
                    {
                        throw new NotSupportedException(
                            "Knowledge distillation requires tensor-shaped training data (Tensor<T> inputs and " +
                            $"targets); this build supplied {typeof(TInput).Name}/{typeof(TOutput).Name}.");
                    }
                    sampleInputs[j] = si;
                    sampleTargets[j] = st;
                }

                Tensor<T> inputBatch = TryStackTensorBatch(sampleInputs, out var stackedInputs) && stackedInputs is not null
                    ? stackedInputs
                    : sampleInputs[0];
                Tensor<T>? targetBatch = TryStackTensorBatch(sampleTargets, out var stackedTargets) && stackedTargets is not null
                    ? stackedTargets
                    : (count == 1 ? sampleTargets[0] : null);

                Tensor<T> teacherOutput = teacherForward(inputBatch);
                var lossAdapter = AdapterFor(teacherOutput);
                lossAdapter.CurrentTrueLabel = null;
                lossAdapter.CurrentTrueLabelBatch = targetBatch;

                var typedInput = (TInput)(object)inputBatch;
                if (model.Predict(typedInput) is Tensor<T> predictedBatch)
                {
                    lossSum = AddLossScalars(lossSum, lossAdapter.CalculateBatchLoss(predictedBatch, teacherOutput));
                    lossBatches++;
                }

                if (nnStudent is not null && tensorGradOptimizer is not null)
                {
                    nnStudent.TrainDistillationBatch(inputBatch, teacherOutput, lossAdapter, tensorGradOptimizer);
                }
                else
                {
                    var gradients = student.ComputeGradients(typedInput, (TOutput)(object)teacherOutput, lossAdapter);
                    if (typedGradOptimizer is not null)
                    {
                        typedGradOptimizer.ApplyGradients(gradients, model);
                    }
                    else
                    {
                        student.ApplyGradients(gradients, learningRate);
                    }
                }
            }

            return lossBatches > 0 ? numOps.Divide(lossSum, numOps.FromDouble(lossBatches)) : numOps.Zero;
        };
    }
}
