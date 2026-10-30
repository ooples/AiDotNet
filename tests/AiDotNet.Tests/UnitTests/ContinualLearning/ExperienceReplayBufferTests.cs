using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Memory;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ContinualLearning;

/// <summary>
/// The rehearsal memory of experience-replay continual learning (Rolnick et al. 2019): a bounded buffer that
/// keeps an even share of every task seen so far.
/// </summary>
public class ExperienceReplayBufferTests
{
    // Every sample's input carries its task id, so a test can tell which task a stored point came from.
    private sealed class TaskDataset : IDataset<double, double[], double[]>
    {
        private readonly List<double[]> _inputs;
        private readonly List<double[]> _outputs;

        public TaskDataset(int task, int count)
        {
            _inputs = Enumerable.Range(0, count).Select(i => new[] { (double)task, i }).ToList();
            _outputs = Enumerable.Range(0, count).Select(i => new[] { (double)task }).ToList();
        }

        private TaskDataset(List<double[]> inputs, List<double[]> outputs)
        {
            _inputs = inputs;
            _outputs = outputs;
        }

        public int Count => _inputs.Count;
        public IReadOnlyList<double[]> Inputs => _inputs;
        public IReadOnlyList<double[]> Outputs => _outputs;
        public bool HasLabels => true;
        public double[] GetInput(int index) => _inputs[index];
        public double[] GetOutput(int index) => _outputs[index];
        public (double[] Input, double[] Output) GetSample(int index) => (_inputs[index], _outputs[index]);
        public IDataset<double, double[], double[]> Subset(int[] indices) =>
            new TaskDataset(indices.Select(i => _inputs[i]).ToList(), indices.Select(i => _outputs[i]).ToList());
        public IDataset<double, double[], double[]> Except(int[] indices)
        {
            var exclude = new HashSet<int>(indices);
            return Subset(Enumerable.Range(0, Count).Where(i => !exclude.Contains(i)).ToArray());
        }
        public IDataset<double, double[], double[]> Merge(IDataset<double, double[], double[]> other) =>
            throw new NotSupportedException();
        public IDataset<double, double[], double[]> AddSamples(double[][] inputs, double[][] outputs) =>
            throw new NotSupportedException();
        public IDataset<double, double[], double[]> RemoveSamples(int[] indices) => Except(indices);
        public IDataset<double, double[], double[]> UpdateLabels(int[] indices, double[][] labels) =>
            throw new NotSupportedException();
        public IDataset<double, double[], double[]> Shuffle(Random? random = null) => Clone();
        public (IDataset<double, double[], double[]> Train, IDataset<double, double[], double[]> Test) Split(
            double trainRatio = 0.8, Random? random = null) => throw new NotSupportedException();
        public int[] GetIndices() => Enumerable.Range(0, Count).ToArray();
        public IDataset<double, double[], double[]> Clone() =>
            new TaskDataset(new List<double[]>(_inputs), new List<double[]>(_outputs));
    }

    private static ExperienceReplayBuffer<double, double[], double[]> Buffer(int size) =>
        new(size, MemorySamplingStrategy.Reservoir, ReplaySamplingStrategy.Uniform, seed: 7);

    [Fact]
    public void EachNewTask_RebalancesTheBufferToEqualShares()
    {
        var buffer = Buffer(30);

        buffer.AddTaskExamples(new TaskDataset(0, 40), 0);
        buffer.AddTaskExamples(new TaskDataset(1, 40), 1);
        buffer.AddTaskExamples(new TaskDataset(2, 40), 2);

        Assert.Equal(30, buffer.Count);
        var counts = buffer.GetTaskCounts();
        Assert.Equal(10, counts[0]);
        Assert.Equal(10, counts[1]);
        Assert.Equal(10, counts[2]);
    }

    [Fact]
    public void Rebalancing_KeepsEveryStoredPointUnderItsOwnTask()
    {
        var buffer = Buffer(30);
        for (int task = 0; task < 4; task++)
            buffer.AddTaskExamples(new TaskDataset(task, 40), task);

        for (int task = 0; task < 4; task++)
        {
            var points = buffer.GetTaskExamples(task);
            Assert.NotEmpty(points);
            Assert.All(points, p => Assert.Equal(task, (int)p.Input[0]));
        }

        Assert.Equal(buffer.Count, buffer.GetTaskCounts().Values.Sum());
    }

    [Fact]
    public void SampleFromTask_ReturnsOnlyThatTask()
    {
        var buffer = Buffer(20);
        buffer.AddTaskExamples(new TaskDataset(0, 10), 0);
        buffer.AddTaskExamples(new TaskDataset(1, 10), 1);

        var sample = buffer.SampleFromTask(1, 5);

        Assert.Equal(5, sample.Count);
        Assert.All(sample, p => Assert.Equal(1, (int)p.Input[0]));
    }

    [Fact]
    public void RemoveTask_DropsItsPointsAndKeepsTheRestIndexed()
    {
        var buffer = Buffer(30);
        for (int task = 0; task < 3; task++)
            buffer.AddTaskExamples(new TaskDataset(task, 10), task);

        buffer.RemoveTask(1);

        Assert.False(buffer.GetTaskCounts().ContainsKey(1));
        Assert.All(buffer.GetTaskExamples(2), p => Assert.Equal(2, (int)p.Input[0]));
        Assert.Equal(buffer.Count, buffer.GetTaskCounts().Values.Sum());
    }
}
