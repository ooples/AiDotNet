using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;

public partial class DecisionTransformerAgent<T>
{
    [Buffer] private Vector<T> _stateMean = new(0);
    [Buffer] private Vector<T> _stateStd = new(0);
    private T _bestEpisodeReturn = MathHelper.GetNumericOperations<T>().Zero;
    private bool _hasEpisodeReturn;

    [ParameterAlias(nameof(_transformerNetwork))]
    internal INeuralNetwork<T> TransformerNetwork => _transformerNetwork;
    internal IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TransformerOptimizer => _networkOptimizer;
    internal int StoredTransitionCount => _trajectoryBuffer.Count;
    internal int StoredTrajectoryCount => _trajectoryBuffer.Count == 0 ? 0
        : 1 + _trajectoryBuffer.Skip(1).Count(step => step.timestep == 0);
    internal IReadOnlyList<T> CurrentContextReturnsToGo => _currentContext.ReturnsToGo;
    internal T[] GetStateMean() => _stateMean.ToArray();
    internal T[] GetStateStd() => _stateStd.ToArray();

    private static void ValidateOptions(DecisionTransformerOptions<T> options)
    {
        if (options.StateSize <= 0 || options.ActionSize <= 0 || options.EmbeddingDim <= 0
            || options.NumLayers <= 0 || options.NumHeads <= 0 || options.EmbeddingDim % options.NumHeads != 0
            || options.ContextLength <= 0 || options.MaxEpisodeLength <= 0 || options.BufferSize <= 0 || options.BatchSize <= 0
            || !(options.DropoutRate >= 0 && options.DropoutRate < 1)
            || !(options.ReturnScale > 0) || double.IsInfinity(options.ReturnScale)
            || options.WarmupSteps < 0 || !(options.MaxGradientNorm >= 0)
            || !(options.WeightDecay >= 0) || double.IsInfinity(options.WeightDecay)
            || (options.TargetReturn is double target && (double.IsNaN(target) || double.IsInfinity(target))))
            throw new ArgumentOutOfRangeException(nameof(options), "Invalid Decision Transformer configuration.");
    }

    private T InitialReturnToGo() => _options.TargetReturn is double target ? NumOps.FromDouble(target)
        : _hasEpisodeReturn ? _bestEpisodeReturn : NumOps.Zero;

    private void RecomputeStateStatistics()
    {
        _stateMean = new Vector<T>(_options.StateSize);
        _stateStd = new Vector<T>(_options.StateSize);
        for (int j = 0; j < _options.StateSize; j++)
        {
            if (_trajectoryBuffer.Count == 0) { _stateStd[j] = NumOps.One; continue; }
            double mean = _trajectoryBuffer.Average(step => NumOps.ToDouble(step.state[j]));
            double variance = _trajectoryBuffer.Average(step =>
            {
                double gap = NumOps.ToDouble(step.state[j]) - mean;
                return gap * gap;
            });
            _stateMean[j] = NumOps.FromDouble(mean);
            _stateStd[j] = NumOps.FromDouble(Math.Sqrt(variance) + 1e-6);
        }
    }

    private void TrimCompletedTrajectories()
    {
        while (_trajectoryBuffer.Count > _options.BufferSize)
        {
            int next = _trajectoryBuffer.FindIndex(1, step => step.timestep == 0);
            int remove = next > 0 ? next : _trajectoryBuffer.Count - _options.BufferSize;
            _trajectoryBuffer.RemoveRange(0, remove);
        }
        RecomputeStateStatistics();
    }

    internal IReadOnlyList<T> GetStoredReturnsToGo(int trajectoryIndex)
    {
        if (trajectoryIndex < 0 || trajectoryIndex >= StoredTrajectoryCount)
            throw new ArgumentOutOfRangeException(nameof(trajectoryIndex));
        var result = new List<T>();
        int current = 0;
        for (int i = 0; i < _trajectoryBuffer.Count; i++)
        {
            if (i > 0 && _trajectoryBuffer[i].timestep == 0) current++;
            if (current == trajectoryIndex) result.Add(_trajectoryBuffer[i].returnToGo);
        }
        return result;
    }

    internal Vector<T>[] PredictContext(IReadOnlyList<T> returnsToGo, IReadOnlyList<Vector<T>> states,
        IReadOnlyList<Vector<T>?> actions, int firstTimestep)
    {
        int length = states.Count;
        if (length == 0 || length > _options.ContextLength || returnsToGo.Count != length || actions.Count != length)
            throw new ArgumentException("The context must hold 1..ContextLength timesteps with matching lists.");
        var input = new Tensor<T>(new[] { 1, length, _transformerNetwork.StepWidth });
        for (int i = 0; i < length; i++) Pack(input, 0, i, returnsToGo[i], states[i], actions[i], firstTimestep + i);
        _transformerNetwork.SetTrainingMode(false);
        var output = _transformerNetwork.Predict(input);
        var result = new Vector<T>[length];
        for (int i = 0; i < length; i++)
        {
            result[i] = new Vector<T>(_options.ActionSize);
            for (int j = 0; j < _options.ActionSize; j++) result[i][j] = output[i * _options.ActionSize + j];
        }
        return result;
    }
}
