using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tokenization;

namespace AiDotNet.VisionLanguage.Editing;

public partial class MGIE<T>
{
    /// <summary>
    /// Returns visual tokens in the language model's feature space, not VAE latents.
    /// Images use RGB values in [-1,1], [channels,height,width] or [batch,channels,height,width].
    /// The same engine-based resize and normalization is used for instruction conditioning.
    /// </summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        bool unbatched = image.Rank == 3;
        var visual = PrepareVisionImage(NormalizeImageLayout(image));
        var projected = _instructionEncoder.ProjectToLanguageSpace(_instructionEncoder.ExtractVisualFeatures(visual));
        return unbatched
            ? Engine.Reshape(projected, new[] { projected.Shape[1], projected.Shape[2] })
            : projected;
    }

    /// <summary>
    /// Encodes an explicit image/instruction token sequence into trainable diffusion context.
    /// </summary>
    /// <remarks>
    /// This is the differentiable native route: it retains gradients through the image encoder,
    /// text embeddings, joint language layers, appended edit tokens and query mapper. Token IDs
    /// must belong to the injected/default instruction encoder's tokenizer. It does not sample
    /// a textual continuation or detach hidden states; callers can supply an already expressive
    /// instruction during training. The output is [batch,EditQueryCount,768].
    /// </remarks>
    public Tensor<T> EncodeEditGuidance(Tensor<T> image, IReadOnlyList<int> instructionTokenIds)
    {
        if (instructionTokenIds is null) throw new ArgumentNullException(nameof(instructionTokenIds));
        var visual = PrepareVisionImage(NormalizeImageLayout(image));
        var joint = _instructionEncoder.EncodeJointHiddenStates(visual, instructionTokenIds, _editMapper.EditTokenEmbeddings);
        var editStates = Engine.TensorNarrow(joint, 1, joint.Shape[1] - _options.EditTokenCount, _options.EditTokenCount);
        var guidance = _editMapper.Forward(editStates);
        // Both nested stacks may have acquired lazy weights on this first actual joint forward.
        InvalidateTrainableParametersCache();
        return guidance;
    }

    /// <inheritdoc />
    public Tensor<T> EditImage(Tensor<T> image, string instruction) => EditImage(image, instruction, seed: null);

    /// <summary>
    /// Edits an RGB image in [-1,1] using seeded fresh target noise and the source VAE's unscaled mode.
    /// </summary>
    /// <param name="image">[3,height,width] or [batch,3,height,width]. The returned tensor preserves this rank.</param>
    /// <param name="instruction">An editing instruction understood by the model's trained tokenizer/weights.</param>
    /// <param name="seed">Optional sampler seed; it does not reinitialize the model.</param>
    /// <param name="standardNoise">Optional standard Gaussian noise, before the scheduler's initial scaling.</param>
    /// <remarks>
    /// Classifier-free guidance uses text+image, image-only and unconditional branches in one U-Net
    /// batch. Only the evolving noisy target is scheduler-normalized; the source latent channels
    /// are concatenated afterward. This API performs inference without changing the caller's modes.
    /// Construction creates native trainable weights and does not import released MGIE/LLaVA weights
    /// or identify checkpoint-specific [IMG] token IDs. Random-weight outputs are not useful edits.
    /// </remarks>
    public Tensor<T> EditImage(Tensor<T> image, string instruction, int? seed, Vector<T>? standardNoise = null)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (instruction is null) throw new ArgumentNullException(nameof(instruction));
        if (string.IsNullOrWhiteSpace(instruction))
            throw new ArgumentException("An editing instruction cannot be empty.", nameof(instruction));
        var sourceImage = NormalizeImageLayout(image);
        bool unbatched = image.Rank == 3;
        bool encoderMode = _instructionEncoder.IsTrainingMode;
        bool mapperMode = _editMapper.TrainingMode;
        using var noGrad = new NoGradScope<T>();
        using var inference = InferenceMode.Enter();
        try
        {
            _instructionEncoder.SetTrainingMode(false);
            _editMapper.SetTrainingMode(false);
            var guidance = EncodeStringGuidance(sourceImage, instruction);
            var nullStates = new Tensor<T>(new[] { sourceImage.Shape[0], _options.EditTokenCount, _options.DecoderDim });
            var nullGuidance = _editMapper.Forward(nullStates);
            var resizedSource = ResizeImage(sourceImage, _options.OutputImageSize);
            var sourceLatent = _vae.Encode(resizedSource, sampleMode: false);
            if (sourceLatent.Rank != 4 || sourceLatent.Shape[1] != LATENT_CHANNELS)
                throw new InvalidOperationException("The source VAE must produce [batch,4,height,width] latents.");
            var edited = Denoise(sourceLatent, guidance, nullGuidance, seed, standardNoise);
            var decoded = DecodeFromLatent(edited);
            if (decoded.Rank != 4 || decoded.Shape[0] != sourceImage.Shape[0] || decoded.Shape[1] != 3 ||
                decoded.Shape[2] != _options.OutputImageSize || decoded.Shape[3] != _options.OutputImageSize)
                throw new InvalidOperationException("The VAE output geometry does not match OutputImageSize.");
            return unbatched
                ? Engine.Reshape(decoded, new[] { 3, _options.OutputImageSize, _options.OutputImageSize })
                : decoded;
        }
        finally
        {
            _instructionEncoder.SetTrainingMode(encoderMode);
            _editMapper.SetTrainingMode(mapperMode);
        }
    }

    private Tensor<T> Denoise(Tensor<T> sourceLatent, Tensor<T> guidance, Tensor<T> nullGuidance,
        int? seed, Vector<T>? standardNoise)
    {
        int batch = sourceLatent.Shape[0];
        var zeroSource = new Tensor<T>(sourceLatent.Shape.ToArray());
        // These survive every arena reset. Only transient noisy inputs/predictions live in a step.
        var sourceBranches = Engine.TensorConcatenate(new[] { sourceLatent, sourceLatent, zeroSource }, axis: 0);
        var contexts = Engine.TensorConcatenate(new[] { guidance, nullGuidance, nullGuidance }, axis: 0);
        return GenerateConditioned(sourceLatent.Shape.ToArray(), _options.NumDiffusionSteps, seed, standardNoise,
            (normalizedNoisyLatent, timestep) =>
            {
                var targets = Engine.TensorConcatenate(
                    new[] { normalizedNoisyLatent, normalizedNoisyLatent, normalizedNoisyLatent }, axis: 0);
                var input = Engine.TensorConcatenate(new[] { targets, sourceBranches }, axis: 1);
                var predictions = _unet.PredictNoise(input, timestep, contexts);
                if (predictions.Rank != 4 || predictions.Shape[0] != 3 * batch ||
                    predictions.Shape[1] != LATENT_CHANNELS || predictions.Shape[2] != sourceLatent.Shape[2] ||
                    predictions.Shape[3] != sourceLatent.Shape[3])
                    throw new InvalidOperationException("The denoiser must return four-channel noise for all three guidance branches.");
                var textImage = Engine.TensorNarrow(predictions, 0, 0, batch);
                var imageOnly = Engine.TensorNarrow(predictions, 0, batch, batch);
                var unconditional = Engine.TensorNarrow(predictions, 0, 2 * batch, batch);
                var textDelta = Engine.TensorMultiplyScalar(Engine.TensorSubtract(textImage, imageOnly),
                    NumOps.FromDouble(_options.GuidanceScale));
                var imageDelta = Engine.TensorMultiplyScalar(Engine.TensorSubtract(imageOnly, unconditional),
                    NumOps.FromDouble(_options.ImageGuidanceScale));
                return Engine.TensorAdd(unconditional, Engine.TensorAdd(textDelta, imageDelta));
            });
    }

    private Tensor<T> EncodeStringGuidance(Tensor<T> image, string instruction)
    {
        var originalTokens = _instructionEncoder.EncodeInstructionTokens(instruction);
        if (!_options.EnableExpressiveInstructions)
            return EncodeEditGuidance(image, originalTokens);

        // Generation's public contract is one unbatched image. Preserve that contract for each row,
        // then combine the real per-image contexts rather than borrowing row zero's continuation.
        var contexts = new Tensor<T>[image.Shape[0]];
        for (int batch = 0; batch < contexts.Length; batch++)
        {
            var row = Engine.TensorNarrow(image, 0, batch, 1);
            var vision = PrepareVisionImage(row);
            var unbatchedVision = Engine.Reshape(vision, new[] { 3, _options.ImageSize, _options.ImageSize });
            int available = _options.MaxSequenceLength - _instructionEncoder.NumVisualTokens - 1 -
                _options.EditTokenCount - originalTokens.Count;
            if (available < 0)
                throw new ArgumentException("The image and instruction exceed the joint sequence limit.", nameof(instruction));
            int generationLength = Math.Min(_options.MaxGenerationLength, available);
            // A one-token nucleus is greedy and deterministic, while retaining the existing
            // LLaVA generation implementation and configured tokenizer/weights.
            string continuation = generationLength == 0 ? string.Empty :
                _instructionEncoder.Generate(unbatchedVision, instruction, generationLength, temperature: 1, topP: 0);
            var tokens = new List<int>(originalTokens);
            if (!string.IsNullOrWhiteSpace(continuation))
            {
                var extra = _instructionEncoder.EncodeInstructionTokens(continuation);
                for (int i = 0; i < extra.Count && i < available; i++) tokens.Add(extra[i]);
            }
            contexts[batch] = EncodeEditGuidance(row, tokens);
        }
        return contexts.Length == 1 ? contexts[0] : Engine.TensorConcatenate(contexts, axis: 0);
    }

    private Tensor<T> NormalizeImageLayout(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (image.Rank == 3 && image.Shape[0] == 3 && image.Shape[1] > 0 && image.Shape[2] > 0)
            return Engine.Reshape(image, new[] { 1, 3, image.Shape[1], image.Shape[2] });
        if (image.Rank == 4 && image.Shape[0] > 0 && image.Shape[1] == 3 && image.Shape[2] > 0 && image.Shape[3] > 0)
            return image;
        throw new ArgumentException("Expected a nonempty RGB image [3,height,width] or [batch,3,height,width].", nameof(image));
    }

    private Tensor<T> PrepareVisionImage(Tensor<T> image)
    {
        var resized = ResizeImage(image, _options.ImageSize);
        var unitRange = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(resized, NumOps.FromDouble(0.5)), NumOps.FromDouble(0.5));
        var mean = new Tensor<T>(new[] { 1, 3, 1, 1 });
        var inverseStd = new Tensor<T>(new[] { 1, 3, 1, 1 });
        for (int channel = 0; channel < 3; channel++)
        {
            mean[channel] = NumOps.FromDouble(_options.ImageMean[channel]);
            inverseStd[channel] = NumOps.FromDouble(1.0 / _options.ImageStd[channel]);
        }
        return Engine.TensorMultiply(Engine.TensorSubtract(unitRange, mean), inverseStd);
    }

    private Tensor<T> ResizeImage(Tensor<T> image, int size) => image.Shape[2] == size && image.Shape[3] == size
        ? image
        : Engine.Interpolate(image, new[] { size, size }, InterpolateMode.Bilinear, alignCorners: false);

    private LLaVANeuralNetwork<T> CreateInstructionEncoder(int? seed)
    {
        var architecture = new NeuralNetworkArchitecture<T>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputDepth: 3, inputHeight: _options.ImageSize,
            inputWidth: _options.ImageSize, outputSize: _options.DecoderDim)
        { RandomSeed = seed ?? _options.Seed ?? Architecture?.RandomSeed };
        // This is a native training tokenizer, not a claim to load a pretrained LLaVA vocabulary.
        // Checkpoint users inject a native encoder configured with their matching tokenizer.
        var tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.LLaMA,
            vocabSize: _options.VocabSize);
        return new LLaVANeuralNetwork<T>(architecture, imageSize: _options.ImageSize, channels: 3,
            patchSize: _options.VisionPatchSize, vocabularySize: _options.VocabSize,
            maxSequenceLength: _options.MaxSequenceLength, embeddingDimension: _options.DecoderDim,
            visionHiddenDim: _options.VisionDim, numVisionLayers: _options.NumVisionLayers,
            numLmLayers: _options.NumDecoderLayers, numHeads: _options.NumHeads, tokenizer: tokenizer);
    }

    private static void ValidateEditingOptions(MGIEOptions options)
    {
        if (options.VisionPatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(options.VisionPatchSize));
        if (options.ImageSize < options.VisionPatchSize) throw new ArgumentOutOfRangeException(nameof(options.ImageSize));
        if (options.OutputImageSize <= 0) throw new ArgumentOutOfRangeException(nameof(options.OutputImageSize));
        if (options.NumHeads <= 0) throw new ArgumentOutOfRangeException(nameof(options.NumHeads));
        if (options.VisionDim <= 0 || options.VisionDim % options.NumHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(options.VisionDim));
        if (options.DecoderDim <= 0 || options.DecoderDim % options.NumHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(options.DecoderDim));
        if (options.NumVisionLayers <= 0) throw new ArgumentOutOfRangeException(nameof(options.NumVisionLayers));
        if (options.NumDecoderLayers <= 0) throw new ArgumentOutOfRangeException(nameof(options.NumDecoderLayers));
        if (options.VocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(options.VocabSize));
        if (options.EditHiddenDim <= 0 || options.EditHiddenDim > int.MaxValue / 4)
            throw new ArgumentOutOfRangeException(nameof(options.EditHiddenDim));
        if (options.EditNumHeads <= 0 || options.EditHiddenDim % options.EditNumHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(options.EditNumHeads));
        if (options.EditTokenCount <= 0) throw new ArgumentOutOfRangeException(nameof(options.EditTokenCount));
        if (options.EditQueryCount <= 0) throw new ArgumentOutOfRangeException(nameof(options.EditQueryCount));
        if (options.EditHeadLayers <= 0) throw new ArgumentOutOfRangeException(nameof(options.EditHeadLayers));
        if (!FiniteNonnegative(options.DropoutRate) || options.DropoutRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(options.DropoutRate));
        if (options.NumDiffusionSteps <= 0) throw new ArgumentOutOfRangeException(nameof(options.NumDiffusionSteps));
        if (options.MaxGenerationLength < 0) throw new ArgumentOutOfRangeException(nameof(options.MaxGenerationLength));
        long patches = options.ImageSize / options.VisionPatchSize;
        if (patches * patches + 1 + options.EditTokenCount + 1 > options.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(options.MaxSequenceLength), "The sequence must hold image, edit and instruction tokens.");
        if (!FiniteNonnegative(options.GuidanceScale)) throw new ArgumentOutOfRangeException(nameof(options.GuidanceScale));
        if (!FiniteNonnegative(options.ImageGuidanceScale)) throw new ArgumentOutOfRangeException(nameof(options.ImageGuidanceScale));
        if (options.ImageMean.Length != 3) throw new ArgumentException("ImageMean must contain three channels.", nameof(options.ImageMean));
        if (options.ImageStd.Length != 3) throw new ArgumentException("ImageStd must contain three channels.", nameof(options.ImageStd));
        for (int channel = 0; channel < 3; channel++)
        {
            if (double.IsNaN(options.ImageMean[channel]) || double.IsInfinity(options.ImageMean[channel]))
                throw new ArgumentOutOfRangeException(nameof(options.ImageMean));
            if (!FiniteNonnegative(options.ImageStd[channel]) || options.ImageStd[channel] == 0)
                throw new ArgumentOutOfRangeException(nameof(options.ImageStd));
        }
    }

    private static bool FiniteNonnegative(double value) => !double.IsNaN(value) && !double.IsInfinity(value) && value >= 0;
}
