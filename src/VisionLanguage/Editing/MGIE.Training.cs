using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.VisionLanguage.Editing;

public partial class MGIE<T>
{
    [AiDotNet.Attributes.Scratch]
    private bool _editTrainingWarmed;

    /// <summary>Runs one MGIE training step on an image, an instruction and the edited target image.</summary>
    /// <param name="sourceImage">RGB in [-1,1], [3,height,width] or [batch,3,height,width].</param>
    /// <param name="instructionTokenIds">The editing instruction, tokenized by the instruction encoder's tokenizer.</param>
    /// <param name="editedImage">The target edit, with the same layout as <paramref name="sourceImage"/>.</param>
    /// <param name="expressiveInstructionTokenIds">
    /// Optional ground-truth expressive instruction the MLLM should derive (tokenized likewise). When supplied, the
    /// instruction loss trains the language model to produce it; otherwise only the edit loss applies.
    /// </param>
    /// <returns>The total objective before the update.</returns>
    /// <remarks>
    /// <para>
    /// The objective is L_all = L_ins + 0.5 L_edit (Fu et al. 2024, Eq. 5). The MLLM reads the image, the instruction
    /// and the expressive instruction, followed by the learned [IMG] tokens. L_ins is the cross-entropy of the
    /// expressive instruction's tokens, each predicted from the position before it. The [IMG] hidden states pass
    /// through the edit head to become the guidance U, and L_edit is the noise-prediction error of the denoiser, which
    /// receives the noisy target latent concatenated with the source latent and cross-attends to U. For
    /// classifier-free guidance, 5% of examples drop the source latent, the guidance, or both (Sec. 3.3).
    /// </para>
    /// <para>
    /// The VAE encodes both images without gradient, and the MLLM is frozen except its word embeddings and LM head by
    /// default; the edit head, the [IMG] embeddings and the denoiser train. Timesteps, noise and dropout decisions are
    /// drawn once, so an optimizer that re-evaluates the objective sees the same example.
    /// </para>
    /// <para><b>For Beginners:</b> show the model a photo, the instruction, and the photo after the edit. It learns to
    /// explain the edit in words and to steer the image generator toward the edited result.</para>
    /// </remarks>
    public T TrainEdit(Tensor<T> sourceImage, IReadOnlyList<int> instructionTokenIds, Tensor<T> editedImage,
        IReadOnlyList<int>? expressiveInstructionTokenIds = null)
    {
        if (instructionTokenIds is null) throw new ArgumentNullException(nameof(instructionTokenIds));
        if (instructionTokenIds.Count == 0)
            throw new ArgumentException("An editing instruction needs at least one token.", nameof(instructionTokenIds));
        var source = NormalizeImageLayout(sourceImage);
        var target = NormalizeImageLayout(editedImage);
        if (!source.Shape.ToArray().SequenceEqual(target.Shape.ToArray()))
            throw new ArgumentException("The source and edited images must have the same layout.", nameof(editedImage));
        int batch = source.Shape[0];
        var expressive = expressiveInstructionTokenIds ?? Array.Empty<int>();
        var tokens = instructionTokenIds.Concat(expressive).ToList();

        var visual = PrepareVisionImage(source);
        Tensor<T> sourceLatent;
        Tensor<T> targetLatent;
        using (new NoGradScope<T>())
        {
            sourceLatent = _vae.Encode(ResizeImage(source, _editOptions.OutputImageSize), sampleMode: false);
            targetLatent = EncodeToLatent(ResizeImage(target, _editOptions.OutputImageSize), sampleMode: true);
            if (!_editTrainingWarmed)
            {
                // Lazily sized MLLM and edit-head weights must exist before the trainable set is collected.
                _ = EncodeEditGuidance(source, tokens);
                _ = _editMapper.Forward(new Tensor<T>(new[] { 1, _editOptions.EditTokenCount, _editOptions.DecoderDim }));
                InvalidateTrainableParametersCache();
                _editTrainingWarmed = true;
            }
        }
        if (sourceLatent.Rank != 4 || sourceLatent.Shape[1] != LATENT_CHANNELS || !sourceLatent.Shape.ToArray().SequenceEqual(targetLatent.Shape.ToArray()))
            throw new InvalidOperationException("The VAE must produce matching [batch,4,height,width] latents for both images.");

        var timesteps = new int[batch];
        var dropSource = new bool[batch];
        var dropGuidance = new bool[batch];
        for (int row = 0; row < batch; row++)
        {
            timesteps[row] = RandomGenerator.Next(Scheduler.Config.TrainTimesteps);
            if (RandomGenerator.NextDouble() < _editOptions.ConditionDropoutProbability)
            {
                // Equally likely: no source image, no instruction guidance, or neither.
                int dropped = RandomGenerator.Next(3);
                dropSource[row] = dropped != 1;
                dropGuidance[row] = dropped != 0;
            }
        }
        var noise = new Tensor<T>(targetLatent.Shape.ToArray());
        var noiseSpan = noise.AsWritableSpan();
        for (int i = 0; i < noiseSpan.Length; i++) noiseSpan[i] = NumOps.FromDouble(RandomGenerator.NextGaussian());
        var noisyTarget = AddTrainingNoise(targetLatent, noise, timesteps);
        var sourceBranches = SelectRows(sourceLatent, new Tensor<T>(sourceLatent.Shape.ToArray()), dropSource);

        int visualCount = _instructionEncoder.JointVisualTokenCount;
        return StepWithCustomLoss(() =>
        {
            var joint = _instructionEncoder.EncodeJointHiddenStates(visual, tokens, _editMapper.EditTokenEmbeddings);
            int length = joint.Shape[1];
            Tensor<T>? loss = null;

            if (expressive.Count > 0 && _editOptions.InstructionLossWeight > 0)
            {
                var positions = new int[batch * expressive.Count];
                var entries = new int[batch * expressive.Count];
                var hidden = Engine.Reshape(joint, new[] { batch * length, _editOptions.DecoderDim });
                for (int row = 0; row < batch; row++)
                {
                    for (int j = 0; j < expressive.Count; j++)
                    {
                        // Text token k is predicted from the hidden state at the position just before it.
                        int position = visualCount + instructionTokenIds.Count + j - 1;
                        positions[row * expressive.Count + j] = row * length + position;
                    }
                }
                var logits = _instructionEncoder.ProjectToVocabulary(
                    Engine.TensorGather(hidden, new Tensor<int>(positions, new[] { positions.Length }), 0));
                int vocabulary = logits.Shape[1];
                for (int i = 0; i < entries.Length; i++)
                    entries[i] = i * vocabulary + expressive[i % expressive.Count];
                var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, 1), new[] { logits.Length });
                var instructionLoss = Engine.TensorMultiplyScalar(
                    Engine.TensorNegate(Engine.ReduceSum(Engine.TensorGather(logProbabilities, new Tensor<int>(entries, new[] { entries.Length }), 0), null)),
                    NumOps.FromDouble(_editOptions.InstructionLossWeight / entries.Length));
                loss = instructionLoss;
            }

            var editStates = Engine.TensorNarrow(joint, 1, length - _editOptions.EditTokenCount, _editOptions.EditTokenCount);
            var guidance = _editMapper.Forward(editStates);
            var nullGuidance = _editMapper.Forward(new Tensor<T>(new[] { batch, _editOptions.EditTokenCount, _editOptions.DecoderDim }));
            var context = SelectRows(guidance, nullGuidance, dropGuidance);
            var input = Engine.TensorConcatenate(new[] { noisyTarget, sourceBranches }, axis: 1);
            // One tape-connected prediction per row: the base batched helper copies through host spans (which
            // severs the gradient) and assumes the prediction has the input's eight channels rather than four.
            var rows = new Tensor<T>[batch];
            for (int row = 0; row < batch; row++)
                rows[row] = _unet.PredictNoise(Engine.TensorNarrow(input, 0, row, 1), timesteps[row],
                    Engine.TensorNarrow(context, 0, row, 1));
            var predicted = rows.Length == 1 ? rows[0] : Engine.TensorConcatenate(rows, axis: 0);
            var difference = Engine.TensorSubtract(predicted, noise);
            var editLoss = Engine.TensorMultiplyScalar(
                Engine.ReduceSum(Engine.TensorMultiply(difference, difference), null),
                NumOps.FromDouble(_editOptions.EditLossWeight / difference.Length));
            return loss is null ? editLoss : Engine.TensorAdd(loss, editLoss);
        });
    }

    private Tensor<T> AddTrainingNoise(Tensor<T> clean, Tensor<T> noise, int[] timesteps)
    {
        if (Scheduler is NoiseSchedulerBase<T> scheduler)
            return scheduler.AddNoiseBatched(clean, noise, timesteps);

        int perRow = clean.Length / timesteps.Length;
        var noisy = new Tensor<T>(clean.Shape.ToArray());
        var cleanSpan = clean.AsSpan();
        var noiseSpan = noise.AsSpan();
        var noisySpan = noisy.AsWritableSpan();
        for (int row = 0; row < timesteps.Length; row++)
        {
            var cleanRow = new Vector<T>(perRow);
            var noiseRow = new Vector<T>(perRow);
            for (int i = 0; i < perRow; i++)
            {
                cleanRow[i] = cleanSpan[row * perRow + i];
                noiseRow[i] = noiseSpan[row * perRow + i];
            }
            var noised = Scheduler.AddNoise(cleanRow, noiseRow, timesteps[row]);
            for (int i = 0; i < perRow; i++) noisySpan[row * perRow + i] = noised[i];
        }
        return noisy;
    }

    /// <summary>Row-wise choice between two equally shaped batches, keeping the tape connection of both.</summary>
    private Tensor<T> SelectRows(Tensor<T> kept, Tensor<T> replacement, bool[] replace)
    {
        if (!replace.Any(value => value)) return kept;
        var rows = new Tensor<T>[replace.Length];
        for (int row = 0; row < rows.Length; row++)
            rows[row] = Engine.TensorNarrow(replace[row] ? replacement : kept, 0, row, 1);
        return rows.Length == 1 ? rows[0] : Engine.TensorConcatenate(rows, axis: 0);
    }
}
