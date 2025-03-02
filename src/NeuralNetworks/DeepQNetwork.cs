namespace AiDotNet.NeuralNetworks;

public class DeepQNetwork<T> : NeuralNetworkBase<T>
{
    private int _actionSpace;
    private readonly List<Experience<T>> _replayBuffer = [];
    private readonly DeepQNetwork<T> _targetNetwork;
    private readonly T _epsilon;
    private readonly int _batchSize = 32;

    public DeepQNetwork(NeuralNetworkArchitecture<T> architecture, double epsilon = 1e16) : base(architecture)
    {
        _epsilon = NumOps.FromDouble(epsilon);
        _targetNetwork = new DeepQNetwork<T>(architecture, epsilon);
    }

    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultDeepQNetworkLayers(Architecture));
        }

        _actionSpace = Architecture.OutputSize;
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    public Vector<T> GetQValues(Vector<T> state)
    {
        return Predict(state);
    }

    public int GetBestAction(Vector<T> state)
    {
        var qValues = GetQValues(state);
        return ArgMax(qValues);
    }

    private int ArgMax(Vector<T> vector)
    {
        T max = vector[0];
        int maxIndex = 0;
        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.GreaterThan(vector[i], max))
            {
                max = vector[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public int GetAction(Vector<T> state)
    {
        if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), _epsilon))
        {
            return Random.Next(_actionSpace);
        }

        return GetBestAction(state);
    }

    public void AddExperience(Vector<T> state, int action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T>(state, action, reward, nextState, done));
        if (_replayBuffer.Count > 10000) // Limit buffer size
        {
            _replayBuffer.RemoveAt(0);
        }
    }

    public void Train(T gamma, T learningRate)
    {
        if (_replayBuffer.Count < _batchSize) return;

        var batch = _replayBuffer.OrderBy(x => Random.Next()).Take(_batchSize).ToList();

        var states = new Matrix<T>(batch.Select(e => e.State).ToArray());
        var actions = batch.Select(e => e.Action).ToArray();
        var rewards = new Vector<T>(batch.Select(e => e.Reward).ToArray());
        var nextStates = new Matrix<T>(batch.Select(e => e.NextState).ToArray());
        var dones = batch.Select(e => e.Done).ToArray();

        // Predict Q-values for current states
        var currentQValues = PredictBatch(states);

        // Predict Q-values for next states
        var nextQValues = _targetNetwork.PredictBatch(nextStates);

        // Compute target Q-values
        var targetQValues = new Matrix<T>(currentQValues.Rows, currentQValues.Columns);
        for (int i = 0; i < _batchSize; i++)
        {
            for (int j = 0; j < _actionSpace; j++)
            {
                if (j == actions[i])
                {
                    T maxNextQ = nextQValues.GetRow(i).Max();
                    T target = dones[i] ? rewards[i] : NumOps.Add(rewards[i], NumOps.Multiply(gamma, maxNextQ));
                    targetQValues[i, j] = target;
                }
                else
                {
                    targetQValues[i, j] = currentQValues[i, j];
                }
            }
        }

        // Compute loss
        Matrix<T> loss = ComputeLoss(currentQValues, targetQValues);

        // Perform backpropagation
        BackPropagate(loss, learningRate);

        // Update target network periodically
        if (Random.Next(100) == 0) // Update every 100 steps on average
        {
            UpdateTargetNetwork();
        }
    }

    private Matrix<T> PredictBatch(Matrix<T> inputs)
    {
        var current = inputs;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromMatrix(current)).ToMatrix();
        }

        return current;
    }

    private Matrix<T> ComputeLoss(Matrix<T> predicted, Matrix<T> target)
    {
        // Mean Squared Error loss
        return predicted.Subtract(target).PointwiseMultiply(predicted.Subtract(target));
    }

    private void BackPropagate(Matrix<T> loss, T learningRate)
    {
        var gradient = Tensor<T>.FromMatrix(loss);
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }
    }

    private void UpdateTargetNetwork()
    {
        // Copy weights from the main network to the target network
        for (int i = 0; i < Layers.Count; i++)
        {
            _targetNetwork.Layers[i].SetParameters(Layers[i].GetParameters());
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Encountered a null layer during serialization.");

            string? fullName = layer.GetType().FullName;
            if (string.IsNullOrEmpty(fullName))
                throw new InvalidOperationException($"Unable to get full name for layer type {layer.GetType()}");

            writer.Write(fullName);
            layer.Serialize(writer);
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            if (string.IsNullOrEmpty(layerTypeName))
                throw new InvalidOperationException("Encountered an empty layer type name during deserialization.");

            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");

            if (!typeof(ILayer<T>).IsAssignableFrom(layerType))
                throw new InvalidOperationException($"Type {layerTypeName} does not implement ILayer<T>");

            object? instance = Activator.CreateInstance(layerType);
            if (instance == null)
                throw new InvalidOperationException($"Failed to create an instance of {layerTypeName}");

            var layer = (ILayer<T>)instance;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }
    }
}