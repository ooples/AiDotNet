using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Facade;

public sealed class AiModelBuilderDistillationAndBaseOptimizerGpuTests
{
    private static NeuralNetwork<float> BuildClassifierNetwork(int inputDim, int hiddenDim, int numClasses, int hiddenLayers, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var layers = new List<ILayer<float>>();
        for (int i = 0; i < hiddenLayers; i++)
        {
            layers.Add(new DenseLayer<float>(hiddenDim, activationFunction: new TanhActivation<float>()));
        }
        layers.Add(new DenseLayer<float>(numClasses, activationFunction: new IdentityActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: inputDim,
            outputSize: numClasses,
            layers: layers);

        var net = new NeuralNetwork<float>(arch, lossFunction: new CrossEntropyLoss<float>());
        // Materialize lazy DenseLayer weights once on a sample batch before seeding deterministic parameters.
        _ = net.Predict(new Tensor<float>(new[] { 1, inputDim }));
        var p = net.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            p[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.12);
        }
        net.UpdateParameters(p);
        return net;
    }

    private static (Tensor<float> inputs, Tensor<float> oneHotTargets) CreateClassificationDataset(
        int rows, int inputDim, int numClasses, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var x = new Tensor<float>(new[] { rows, inputDim });
        var y = new Tensor<float>(new[] { rows, numClasses });
        for (int r = 0; r < rows; r++)
        {
            int cls = r % numClasses;
            y[r, cls] = 1.0f;
            for (int c = 0; c < inputDim; c++)
            {
                x[r, c] = (c % numClasses == cls ? 1.25f : -0.15f) + (float)((rng.NextDouble() - 0.5) * 0.1);
            }
        }
        return (x, y);
    }

    [Fact]
    public async Task AiModelBuilder_ConfigureKnowledgeDistillation_BatchesByBatchSizeAndStepsConfiguredAdamW()
    {
        const int rows = 20;
        const int inputDim = 12;
        const int hiddenDim = 16;
        const int numClasses = 32;
        const int batchSize = 4;
        const int epochs = 4;

        var teacher = BuildClassifierNetwork(inputDim, hiddenDim, numClasses, hiddenLayers: 2, seed: 11);
        var student = BuildClassifierNetwork(inputDim, hiddenDim, numClasses, hiddenLayers: 2, seed: 29);
        var (x, y) = CreateClassificationDataset(rows, inputDim, numClasses, seed: 41);

        int teacherBatchCalls = 0;
        int maxObservedBatchRows = 0;
        var epochLosses = new List<float>();

        var adamw = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(
            student,
            new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.02,
                WeightDecay = 0.01,
                MaxIterations = epochs,
            });

        var kdOptions = new KnowledgeDistillationOptions<float, Tensor<float>, Tensor<float>>
        {
            TeacherForward = batch =>
            {
                teacherBatchCalls++;
                if (batch.Shape[0] > maxObservedBatchRows)
                    maxObservedBatchRows = batch.Shape[0];
                Assert.InRange(batch.Shape[0], 1, batchSize);
                return teacher.Predict(batch);
            },
            SparseTopK = 8,
            SparseTopKMinimumClasses = 16,
            Temperature = 2.0,
            Alpha = 0.25,
            BatchSize = batchSize,
            Epochs = epochs,
            ValidateAfterEpoch = false,
            OnEpochComplete = (_, loss) => epochLosses.Add(loss),
        };
        kdOptions.Validate();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(student)
            .ConfigureOptimizer(adamw)
            .ConfigureKnowledgeDistillation(kdOptions)
            .BuildAsync(x, y);

        Assert.NotNull(result);
        Assert.Equal(batchSize, maxObservedBatchRows);
        Assert.Equal(epochs, epochLosses.Count);
        Assert.True(epochLosses[epochLosses.Count - 1] < epochLosses[0],
            $"Expected distillation epoch loss to decrease under AdamW: first={epochLosses[0]:F6}, last={epochLosses[epochLosses.Count - 1]:F6}");
    }

    [Fact]
    public async Task AiModelBuilder_ConfigureKnowledgeDistillation_WorksWithBlockwiseAnchorLocalCreditAndGradientCheckpointing()
    {
        const int rows = 16;
        const int inputDim = 12;
        const int hiddenDim = 12;
        const int numClasses = 24;

        var teacher = BuildClassifierNetwork(inputDim, hiddenDim, numClasses, hiddenLayers: 3, seed: 101);
        var student = BuildClassifierNetwork(inputDim, hiddenDim, numClasses, hiddenLayers: 3, seed: 202);
        var (x, y) = CreateClassificationDataset(rows, inputDim, numClasses, seed: 303);

        var creditRule = new BlockwiseAnchorLocalErrorCreditRule<float>(
            readoutLayerCount: 1,
            anchorStride: 2,
            maxCandidates: 8,
            localWeight: 0.8,
            detachReadoutOnHiddenBlocks: true,
            seed: 404);

        var optimizer = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(
            student,
            new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.015,
                MaxIterations = 3,
            });

        var epochLosses = new List<float>();
        var kdOptions = new KnowledgeDistillationOptions<float, Tensor<float>, Tensor<float>>
        {
            TeacherModel = teacher,
            SparseTopK = 6,
            SparseTopKMinimumClasses = 16,
            Temperature = 2.0,
            Alpha = 0.2,
            BatchSize = 4,
            Epochs = 3,
            ValidateAfterEpoch = false,
            OnEpochComplete = (_, loss) => epochLosses.Add(loss),
        };

        var beforeParams = student.GetParameters().Clone();
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(student)
            .ConfigureOptimizer(optimizer)
            .ConfigureCreditRule(creditRule)
            .ConfigureGradientCheckpointing(new GradientCheckpointingOptions(enabled: true, checkpointEveryNLayers: 2))
            .ConfigureKnowledgeDistillation(kdOptions)
            .BuildAsync(x, y);

        Assert.NotNull(result);
        Assert.Equal(2, student.GradientCheckpointingSegmentSize);
        Assert.Equal(3, epochLosses.Count);

        var afterParams = student.GetParameters();
        double paramDeltaNormSq = 0.0;
        for (int i = 0; i < afterParams.Length; i++)
        {
            Assert.False(float.IsNaN(afterParams[i]) || float.IsInfinity(afterParams[i]));
            double diff = afterParams[i] - beforeParams[i];
            paramDeltaNormSq += diff * diff;
        }
        Assert.True(paramDeltaNormSq > 1e-8, "Student parameters did not update under BlockwiseAnchor credit + KD.");
    }

    [Fact]
    public void GradientBasedOptimizerBase_AllOptimizerFamilies_StepCleanlyAcrossParametersOnActiveEngine()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(777);
        var factories = new Func<NeuralNetwork<float>, IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>>[]
        {
            net => new StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new MomentumOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new NesterovAcceleratedGradientOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new RootMeanSquarePropagationOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new LionOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new NadamOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new LAMBOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new RAdamOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new ASGDOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new RpropOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new AdamOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(net),
            net => new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(net),
        };

        var x = new Tensor<float>(new[] { 4, 8 });
        var y = new Tensor<float>(new[] { 4, 6 });
        for (int r = 0; r < 4; r++)
        {
            y[r, r % 6] = 1.0f;
            for (int c = 0; c < 8; c++)
            {
                x[r, c] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        foreach (var makeOptimizer in factories)
        {
            var net = BuildClassifierNetwork(inputDim: 8, hiddenDim: 12, numClasses: 6, hiddenLayers: 2, seed: 55);
            var opt = makeOptimizer(net);
            var initial = net.GetParameters().Clone();

            for (int step = 0; step < 2; step++)
            {
                using var tape = new GradientTape<float>();
                var pred = net.ForwardForTraining(x);
                var loss = new CrossEntropyLoss<float>().ComputeTapeLoss(pred, y);
                var trainable = new List<Tensor<float>>();
                foreach (var layer in net.Layers)
                {
                    if (layer is LayerBase<float> layerBase)
                    {
                        foreach (var p in layerBase.GetTrainableParameters())
                        {
                            if (p.Length > 0 && !trainable.Contains(p))
                                trainable.Add(p);
                        }
                    }
                }
                var grads = tape.ComputeGradients(loss, trainable);
                var ctx = new TapeStepContext<float>(
                    trainable,
                    grads,
                    loss.Length > 0 ? loss[0] : 0f,
                    x,
                    y,
                    (inp, _) => net.ForwardForTraining(inp),
                    (pr, tg) => new CrossEntropyLoss<float>().ComputeTapeLoss(pr, tg));
                opt.Step(ctx);
            }

            var updated = net.GetParameters();
            double moved = 0.0;
            for (int i = 0; i < updated.Length; i++)
            {
                Assert.False(float.IsNaN(updated[i]) || float.IsInfinity(updated[i]),
                    $"{opt.GetType().Name} produced non-finite parameter at index {i}.");
                double d = updated[i] - initial[i];
                moved += d * d;
            }
            Assert.True(moved > 1e-10, $"{opt.GetType().Name} did not update parameters.");
        }
    }
}
