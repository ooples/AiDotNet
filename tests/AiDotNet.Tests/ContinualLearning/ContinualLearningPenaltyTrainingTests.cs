using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.ContinualLearning.Config;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.ContinualLearning.Trainers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ContinualLearning;

/// <summary>
/// The regularizers of EWC (Kirkpatrick et al. 2017), MAS (Aljundi et al. 2018) and LwF (Li and Hoiem 2017) act on
/// training: learning a second task with the penalty switched on reaches different parameters than without it.
/// </summary>
/// <remarks>
/// Before this was fixed, EWC's and MAS's AdjustGradients returned the task gradient unchanged and the LwF trainer
/// only reported its distillation loss, so all three trainers ran plain fine-tuning and every pair below was
/// identical. Each epoch is one full batch, so the trainer's unseeded shuffle cannot change the averaged gradient.
/// </remarks>
public class ContinualLearningPenaltyTrainingTests
{
    private const int Samples = 12;

    private static NeuralNetwork<double> Network()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(6, (IActivationFunction<double>)new TanhActivation<double>()),
            new DenseLayer<double>(3),
        };
        var network = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 3, layers: layers) { RandomSeed = 5 });
        network.Predict(new Tensor<double>(new[] { 1, 2 }));                 // materialize the lazily built weights
        return network;
    }

    private static (NeuralNetwork<double> A, NeuralNetwork<double> B) Twins()
    {
        var a = Network();
        var b = Network();
        b.SetParameters(a.GetParameters());
        return (a, b);
    }

    private static ContinualLearnerConfig<double> Config() => new()
    {
        LearningRate = 0.05,
        EpochsPerTask = 3,
        BatchSize = Samples,
    };

    private static TaskData Task(double phase) => new(Enumerable.Range(0, Samples).Select(i =>
    {
        double x0 = Math.Sin(0.5 * i + phase), x1 = Math.Cos(0.3 * i - phase);
        return (new[] { x0, x1 }, new[] { x0 + x1, x0 - x1, x0 * x1 });
    }).ToList());

    private static double[] Learn(IContinualLearner<double, Tensor<double>, Tensor<double>> learner, NeuralNetwork<double> network)
    {
        learner.LearnTask(Task(0.0));
        learner.LearnTask(Task(1.3));
        return network.GetParameters().ToArray();
    }

    private static void AssertDiffer(double[] withPenalty, double[] without, string name)
        => Assert.True(withPenalty.Zip(without, (p, q) => Math.Abs(p - q)).Max() > 1e-9,
            $"{name}: the second task trained to the same parameters with and without the penalty, so it never reached the gradient.");

    [Fact(Timeout = 120000)]
    public async Task Ewc_PenaltyChangesTheSecondTask()
    {
        await System.Threading.Tasks.Task.Yield();
        var (a, b) = Twins();
        var loss = new MeanSquaredErrorLoss<double>();
        var on = Learn(new EWCTrainer<double, Tensor<double>, Tensor<double>>(a, loss, Config(),
            new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(loss, new EWCOptions<double> { Lambda = 50.0 })), a);
        var off = Learn(new EWCTrainer<double, Tensor<double>, Tensor<double>>(b, loss, Config(),
            new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(loss, new EWCOptions<double> { Lambda = 0.0 })), b);
        AssertDiffer(on, off, "EWC");
    }

    [Fact(Timeout = 120000)]
    public async Task Mas_PenaltyChangesTheSecondTask()
    {
        await System.Threading.Tasks.Task.Yield();
        var (a, b) = Twins();
        var loss = new MeanSquaredErrorLoss<double>();
        var on = Learn(new MASTrainer<double, Tensor<double>, Tensor<double>>(a, loss, Config(),
            new MemoryAwareSynapses<double, Tensor<double>, Tensor<double>>(loss, new MASOptions<double> { Lambda = 5.0 })), a);
        var off = Learn(new MASTrainer<double, Tensor<double>, Tensor<double>>(b, loss, Config(),
            new MemoryAwareSynapses<double, Tensor<double>, Tensor<double>>(loss, new MASOptions<double> { Lambda = 0.0 })), b);
        AssertDiffer(on, off, "MAS");
    }

    [Fact(Timeout = 120000)]
    public async Task Lwf_DistillationGradientChangesTheSecondTask()
    {
        await System.Threading.Tasks.Task.Yield();
        var (a, b) = Twins();
        var loss = new MeanSquaredErrorLoss<double>();
        var on = Learn(new LwFTrainer<double, Tensor<double>, Tensor<double>>(a, loss, Config(),
            new LearningWithoutForgetting<double, Tensor<double>, Tensor<double>>(loss, new LwFOptions<double>()),
            new LwFTrainerOptions<double> { IncludeDistillationInGradients = true }), a);
        var off = Learn(new LwFTrainer<double, Tensor<double>, Tensor<double>>(b, loss, Config(),
            new LearningWithoutForgetting<double, Tensor<double>, Tensor<double>>(loss, new LwFOptions<double>()),
            new LwFTrainerOptions<double> { IncludeDistillationInGradients = false }), b);
        AssertDiffer(on, off, "LwF");
    }

    [Fact(Timeout = 60000)]
    public async Task Lwf_DistillationObjective_MatchesTheVectorLoss()
    {
        await System.Threading.Tasks.Task.Yield();
        // The paper's soft cross-entropy at T = 2 (Li and Hoiem 2017, Eqs. 2–4), on the tape and as a vector.
        var lwf = new LearningWithoutForgetting<double, Tensor<double>, Tensor<double>>(new MeanSquaredErrorLoss<double>(), new LwFOptions<double>());
        var student = new[] { 0.3, -1.2, 2.0 };
        var teacher = new[] { 1.0, 0.1, -0.4 };
        double expected = 0;
        double[] Soft(double[] z)
        {
            var e = z.Select(v => Math.Exp(v / 2.0)).ToArray();
            return e.Select(v => v / e.Sum()).ToArray();
        }
        var ps = Soft(student);
        var pt = Soft(teacher);
        for (int i = 0; i < 3; i++) expected -= pt[i] * Math.Log(ps[i]);
        var vector = lwf.DistillationLoss.CalculateLoss(new Vector<double>(student), new Vector<double>(teacher));
        var tape = lwf.DistillationLoss.ComputeTapeLoss(new Tensor<double>(new[] { 1, 3 }, new Vector<double>(student)),
            new Tensor<double>(new[] { 1, 3 }, new Vector<double>(teacher)));
        Assert.Equal(expected, vector, 10);
        Assert.Equal(expected, tape[0], 10);
    }

    private sealed class TaskData : IDataset<double, Tensor<double>, Tensor<double>>
    {
        private readonly List<Tensor<double>> _inputs;
        private readonly List<Tensor<double>> _outputs;

        public TaskData(List<(double[] X, double[] Y)> rows)
            : this(rows.Select(r => Row(r.X)).ToList(), rows.Select(r => Row(r.Y)).ToList())
        {
        }

        private TaskData(List<Tensor<double>> inputs, List<Tensor<double>> outputs)
        {
            _inputs = inputs;
            _outputs = outputs;
        }

        private static Tensor<double> Row(double[] values) => new(new[] { 1, values.Length }, new Vector<double>(values));

        public int Count => _inputs.Count;
        public IReadOnlyList<Tensor<double>> Inputs => _inputs;
        public IReadOnlyList<Tensor<double>> Outputs => _outputs;
        public bool HasLabels => true;
        public Tensor<double> GetInput(int index) => _inputs[index];
        public Tensor<double> GetOutput(int index) => _outputs[index];
        public (Tensor<double> Input, Tensor<double> Output) GetSample(int index) => (_inputs[index], _outputs[index]);
        public IDataset<double, Tensor<double>, Tensor<double>> Subset(int[] indices)
            => new TaskData(indices.Select(i => _inputs[i]).ToList(), indices.Select(i => _outputs[i]).ToList());
        public IDataset<double, Tensor<double>, Tensor<double>> Except(int[] indices)
            => Subset(Enumerable.Range(0, Count).Where(i => !indices.Contains(i)).ToArray());
        public IDataset<double, Tensor<double>, Tensor<double>> Merge(IDataset<double, Tensor<double>, Tensor<double>> other)
            => new TaskData(_inputs.Concat(other.Inputs).ToList(), _outputs.Concat(other.Outputs).ToList());
        public IDataset<double, Tensor<double>, Tensor<double>> AddSamples(Tensor<double>[] inputs, Tensor<double>[] outputs)
            => new TaskData(_inputs.Concat(inputs).ToList(), _outputs.Concat(outputs).ToList());
        public IDataset<double, Tensor<double>, Tensor<double>> RemoveSamples(int[] indices) => Except(indices);
        public IDataset<double, Tensor<double>, Tensor<double>> UpdateLabels(int[] indices, Tensor<double>[] labels)
        {
            var outputs = new List<Tensor<double>>(_outputs);
            for (int i = 0; i < indices.Length; i++) outputs[indices[i]] = labels[i];
            return new TaskData(new List<Tensor<double>>(_inputs), outputs);
        }
        public IDataset<double, Tensor<double>, Tensor<double>> Shuffle(Random? random = null) => Clone();
        public (IDataset<double, Tensor<double>, Tensor<double>> Train, IDataset<double, Tensor<double>, Tensor<double>> Test) Split(double trainRatio = 0.8, Random? random = null)
        {
            int train = (int)(Count * trainRatio);
            return (Subset(Enumerable.Range(0, train).ToArray()), Subset(Enumerable.Range(train, Count - train).ToArray()));
        }
        public int[] GetIndices() => Enumerable.Range(0, Count).ToArray();
        public IDataset<double, Tensor<double>, Tensor<double>> Clone() => new TaskData(new List<Tensor<double>>(_inputs), new List<Tensor<double>>(_outputs));
    }
}
