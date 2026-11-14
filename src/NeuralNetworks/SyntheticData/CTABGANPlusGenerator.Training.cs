using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Training;

namespace AiDotNet.NeuralNetworks.SyntheticData;

public partial class CTABGANPlusGenerator<T>
{
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _criticOptimizer;

    // Tensor view of the transformed training matrix, rebuilt when a different matrix is passed in.
    [Scratch]
    private Matrix<T>? _realDataSource;
    [Scratch]
    private Tensor<T>? _realDataTensor;

    private Tensor<T> TrainingGenerator(Tensor<T> input)
    {
        if (_usingCustomLayers)
        {
            var custom = input;
            foreach (var layer in Layers) custom = layer.Forward(custom);
            return custom;
        }
        var current = input;
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            if (i > 0) current = Engine.TensorConcatenate(new[] { current, input }, 1);
            current = Layers[i].Forward(current);
            _genBNLayers[i].SetTrainingMode(true);
            current = Engine.ReLU(_genBNLayers[i].Forward(current));
        }
        return Layers[^1].Forward(Engine.TensorConcatenate(new[] { current, input }, 1));
    }

    private Tensor<T> TrainingActivations(Tensor<T> raw, Tensor<T> gumbelNoise)
        => SyntheticTensorOps<T>.ApplyOutputActivationsBatched(
            raw, _transformer, _columns.Count, gumbelNoise, _options.GumbelTemperature);

    private (Tensor<T> Scores, Tensor<T> Classes, Tensor<T> Features) TrainingCritic(Tensor<T> input, bool dropout)
    {
        var current = input;
        Tensor<T>? hidden = null;
        for (int i = 0; i < _discLayers.Count; i++)
        {
            if (_discLayers[i] is DropoutLayer<T> mask)
            {
                mask.SetTrainingMode(dropout);
                current = mask.Forward(current);
                continue;
            }
            if (i == _discLayers.Count - 1) hidden = current;
            current = _discLayers[i].Forward(current);
            if (i != _discLayers.Count - 1) current = Engine.LeakyReLU(current, NumOps.FromDouble(.2));
        }
        return (current, _classifierHead!.Forward(hidden!), hidden!);
    }

    private Tensor<T> Mean(Tensor<T> tensor) => Engine.ReduceMean(tensor, Enumerable.Range(0, tensor.Rank).ToArray(), false);

    private Tensor<T> ClassLabels(Tensor<T> packed) => Engine.TensorSlice(packed,
        new[] { 0, _targetColTransformOffset }, new[] { packed.Shape[0], _numClasses });

    private Tensor<T> ClassLoss(Tensor<T> logits, Tensor<T> labels) => Engine.TensorNegate(
        Mean(Engine.ReduceSum(Engine.TensorMultiply(labels, Engine.TensorLogSoftmax(logits, 1)), new[] { 1 }, false)));

    private Tensor<T> InformationLoss(Tensor<T> fake, Tensor<T> real)
    {
        Tensor<T> Moments(Tensor<T> values, out Tensor<T> deviation)
        {
            var mean = Engine.ReduceMean(values, new[] { 0 }, true);
            var centered = Engine.TensorSubtract(values, Engine.TensorBroadcastTo(mean, values.Shape.ToArray()));
            var variance = Engine.ReduceMean(Engine.TensorMultiply(centered, centered), new[] { 0 }, true);
            if (values.Shape[0] > 1)
                variance = Engine.TensorMultiplyScalar(variance, NumOps.FromDouble((double)values.Shape[0] / (values.Shape[0] - 1)));
            deviation = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-8)));
            return mean;
        }
        var fakeMean = Moments(fake, out var fakeDeviation);
        var realMean = Moments(real, out var realDeviation);
        var meanDifference = Engine.TensorSubtract(fakeMean, realMean);
        var deviationDifference = Engine.TensorSubtract(fakeDeviation, realDeviation);
        Tensor<T> Norm(Tensor<T> value) => Engine.TensorSqrt(Engine.TensorAddScalar(
            Engine.ReduceSum(Engine.TensorMultiply(value, value), Enumerable.Range(0, value.Rank).ToArray(), false), NumOps.FromDouble(1e-8)));
        return Engine.TensorAdd(Norm(meanDifference), Norm(deviationDifference));
    }

    private void TrainNativeBatch(Matrix<T> data, int packs, bool critic)
    {
        int count = packs * _options.PacSize;
        int embed = _options.EmbeddingDimension;
        var sampler = _sampler ?? throw new InvalidOperationException("CTAB-GAN+ training needs a fitted sampler.");
        var noiseRows = new List<Vector<T>>(count);
        var conditionRows = new List<Vector<T>>(count);
        var maskRows = new List<Vector<T>>(count);
        var rowIndices = new int[count];
        for (int row = 0; row < count; row++)
        {
            // Training-by-sampling (CTAB-GAN+ section 3, after CTGAN): one condition, its column mask and a real row
            // holding that category, then the row's noise.
            var (condition, mask, _, _, index) = sampler.SampleCondVecWithMask();
            conditionRows.Add(condition);
            maskRows.Add(mask);
            rowIndices[row] = index;
            noiseRows.Add(CreateStandardNormalVector(embed));
        }

        var noiseBatch = SyntheticTensorOps<T>.StackRows(noiseRows, embed);
        var conditionBatch = SyntheticTensorOps<T>.StackRows(conditionRows, _condWidth);
        var maskBatch = SyntheticTensorOps<T>.StackRows(maskRows, _condWidth);
        var gumbel = SyntheticTensorOps<T>.GumbelNoise(new[] { count, _dataWidth }, _random);
        if (_realDataTensor is null || !ReferenceEquals(_realDataSource, data))
        {
            _realDataTensor = Tensor<T>.FromMatrix(data);
            _realDataSource = data;
        }
        var real = SyntheticTensorOps<T>.GatherRows(_realDataTensor, rowIndices, _dataWidth);
        var generatorInput = Engine.TensorConcatenate(new[] { noiseBatch, conditionBatch }, 1);
        var realPacked = Engine.TensorConcatenate(new[] { real, conditionBatch }, 1).Reshape(new[] { packs, _packedInputDim });
        bool classification = _hasClassificationTarget && _options.ClassifierWeight > 0;
        if (critic)
        {
            Tensor<T> fakePacked;
            using (new NoGradScope<T>())
                fakePacked = Engine.TensorConcatenate(new[] { TrainingActivations(TrainingGenerator(generatorInput), gumbel), conditionBatch }, 1)
                    .Reshape(new[] { packs, _packedInputDim });
            var stacked = Engine.TensorConcatenate(new[] { realPacked, fakePacked }, 0);
            Tensor<T>? traced = null;
            Tensor<T> Forward(Tensor<T> batch)
            {
                traced = batch;
                var prediction = TrainingCritic(batch, true);
                return Engine.TensorConcatenate(new[] { prediction.Scores, prediction.Classes }, 1);
            }
            Tensor<T> Loss(Tensor<T> prediction, Tensor<T> targets)
            {
                var realScores = Engine.TensorSlice(prediction, new[] { 0, 0 }, new[] { packs, 1 });
                var fakeScores = Engine.TensorSlice(prediction, new[] { packs, 0 }, new[] { packs, 1 });
                var loss = Engine.TensorSubtract(Mean(fakeScores), Mean(realScores));
                if (_options.GradientPenaltyWeight > 0)
                {
                    var tracedBatch = traced ?? throw new InvalidOperationException(
                        "CTAB-GAN+ critic step: the loss ran before the forward pass.");
                    var realInput = Engine.TensorSlice(tracedBatch, new[] { 0, 0 }, new[] { packs, _packedInputDim });
                    var fakeInput = Engine.TensorSlice(tracedBatch, new[] { packs, 0 }, new[] { packs, _packedInputDim });
                    var penalty = WassersteinCriticStep<T>.GradientPenalty(Engine, realInput, fakeInput,
                        batch => TrainingCritic(batch, true).Scores, _random);
                    loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(penalty, NumOps.FromDouble(_options.GradientPenaltyWeight)));
                }
                if (classification)
                    loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(ClassLoss(
                        Engine.TensorSlice(prediction, new[] { 0, 1 }, new[] { packs, _numClasses }), ClassLabels(targets)),
                        NumOps.FromDouble(_options.ClassifierWeight)));
                return loss;
            }
            var classifierHead = _classifierHead ?? throw new InvalidOperationException(
                "CTAB-GAN+ training needs the auxiliary classifier head built by Fit.");
            var layers = _discLayers.Concat(new ILayer<T>[] { classifierHead }).ToList();
            _criticOptimizer ??= new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate });
            FusedTrainingStep<T>.Step(_criticOptimizer, layers.OfType<ITrainableLayer<T>>().ToList(), stacked, realPacked,
                Forward, Loss, _criticOptimizer, maxGradNorm: _options.MaxGradNorm,
                graphBreakReason: "The combined critic/classifier objective requires the nested gradient-penalty tape; compiled replay drops its parameter gradients.",
                extraTensors: TapeTrainingStep<T>.CollectParameters(layers));
            return;
        }

        // The persistent generator input packs [noise | condition | mask | Gumbel noise] so a replayed plan uses fresh
        // draws of each every step.
        var input = Engine.TensorConcatenate(new[] { generatorInput, maskBatch, gumbel }, 1);
        var catBlocks = SyntheticTensorOps<T>.CategoricalOutputBlocks(_transformer, _columns.Count);
        Tensor<T>? tracedInput = null;
        Tensor<T>? tracedHead = null;
        Tensor<T> GeneratorForward(Tensor<T> batch)
        {
            tracedInput = batch;
            var head = TrainingGenerator(Engine.TensorSlice(batch, new[] { 0, 0 }, new[] { count, embed + _condWidth }));
            tracedHead = head;
            var noise = Engine.TensorSlice(batch, new[] { 0, embed + 2 * _condWidth }, new[] { count, _dataWidth });
            return TrainingActivations(head, noise);
        }
        Tensor<T> GeneratorLoss(Tensor<T> fake, Tensor<T> targets)
        {
            var batchInput = tracedInput ?? throw new InvalidOperationException(
                "CTAB-GAN+ generator step: the loss ran before the forward pass.");
            var conditions = Engine.TensorSlice(batchInput, new[] { 0, embed }, new[] { count, _condWidth });
            var packed = Engine.TensorConcatenate(new[] { fake, conditions }, 1).Reshape(new[] { packs, _packedInputDim });
            var prediction = TrainingCritic(packed, false);
            var loss = Engine.TensorNegate(Mean(prediction.Scores));
            // The conditional generator loss CTAB-GAN+ keeps from CTGAN: cross-entropy of the conditioned column's
            // category logits against the requested one-hot.
            if (_condWidth > 0 && catBlocks.Count > 0 && tracedHead is not null)
            {
                var masks = Engine.TensorSlice(batchInput, new[] { 0, embed + _condWidth }, new[] { count, _condWidth });
                loss = Engine.TensorAdd(loss, SyntheticTensorOps<T>.ConditionalCrossEntropy(tracedHead, conditions, masks, catBlocks));
            }
            if (classification)
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(ClassLoss(prediction.Classes,
                    ClassLabels(targets.Reshape(new[] { packs, _options.PacSize * _dataWidth }))), NumOps.FromDouble(_options.ClassifierWeight)));
            if (_options.InformationWeight > 0)
            {
                var realInput = Engine.TensorConcatenate(new[] { targets, conditions }, 1).Reshape(new[] { packs, _packedInputDim });
                // Information loss matches the mean and standard deviation of discriminator features.
                var realFeatures = TrainingCritic(realInput, false).Features;
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(InformationLoss(prediction.Features, realFeatures), NumOps.FromDouble(_options.InformationWeight)));
            }
            return loss;
        }
        var generatorLayers = Layers.Concat(_genBNLayers.Cast<ILayer<T>>()).ToList();
        FusedTrainingStep<T>.Step(_optimizer, generatorLayers.OfType<ITrainableLayer<T>>().ToList(), input, real,
            GeneratorForward, GeneratorLoss, _optimizer, maxGradNorm: _options.MaxGradNorm,
            extraTensors: TapeTrainingStep<T>.CollectParameters(generatorLayers));
    }
}
