using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

/// <content>
/// Knowledge distillation on the streaming supervised path.
/// </content>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Builds the per-batch distillation step for a streaming run, or returns <c>null</c> when distillation is not
    /// configured.
    /// </summary>
    /// <returns>
    /// A function that trains the student on every sample of a batch against the teacher and returns the mean
    /// distillation loss of the batch, each sample's loss measured before its update.
    /// </returns>
    /// <remarks>
    /// This is the update <see cref="RunKnowledgeDistillationTraining"/> runs over the rows of an in-memory set:
    /// the teacher's output for the sample is the target, the true label feeds the hard-label term, and the
    /// strategy's gradient is applied through the student's own gradient surface at the optimizer's configured
    /// initial learning rate. Nothing in it needs more than the sample at hand, so it runs on a stream unchanged.
    /// </remarks>
    /// <exception cref="NotSupportedException">The student does not expose its gradient.</exception>
    private Func<TInput[], TOutput[], T>? CreateStreamingDistillationStep()
    {
        // A strategy configured without ConfigureKnowledgeDistillation still expresses intent to distill; engage
        // distillation (which fails clearly if no teacher was supplied), as the in-memory path does.
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
        var strategy = kd.Strategy is { } configuredStrategy
            ? configuredStrategy
            : KnowledgeDistillation.DistillationStrategyFactory<T>.CreateResponseBasedStrategy(
                kd.Temperature, kd.Alpha);
        var lossAdapter = new KnowledgeDistillation.DistillationLossAdapter<T>(strategy);
        var numOps = MathHelper.GetNumericOperations<T>();
        T learningRate = numOps.FromDouble(_optimizer?.GetOptions()?.InitialLearningRate ?? 0.01);

        return (inputs, targets) =>
        {
            T lossSum = numOps.Zero;
            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is not Tensor<T> sampleInput || targets[i] is not Tensor<T> sampleTarget)
                {
                    throw new NotSupportedException(
                        "Knowledge distillation requires tensor-shaped training data (Tensor<T> inputs and " +
                        $"targets); this build supplied {typeof(TInput).Name}/{typeof(TOutput).Name}.");
                }

                // One row with a leading batch dimension, the shape the in-memory loop hands the student.
                Tensor<T> inputRow = TryStackTensorBatch(new[] { sampleInput }, out var stacked) && stacked is not null
                    ? stacked
                    : sampleInput;
                Tensor<T> teacherOutput = teacherForward(inputRow);
                lossAdapter.CurrentTrueLabel = sampleTarget.ToVector();

                var typedInput = (TInput)(object)inputRow;
                lossSum = AddLossScalars(
                    lossSum,
                    lossAdapter.CalculateLoss(
                        ConversionsHelper.ConvertToVector<T, TOutput>(model.Predict(typedInput)),
                        teacherOutput.ToVector()));

                var gradients = student.ComputeGradients(typedInput, (TOutput)(object)teacherOutput, lossAdapter);
                student.ApplyGradients(gradients, learningRate);
            }

            return inputs.Length > 0 ? numOps.Divide(lossSum, numOps.FromDouble(inputs.Length)) : numOps.Zero;
        };
    }
}
