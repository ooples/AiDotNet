namespace AiDotNet.NeuralNetworks;

public class SelfOrganizingMap<T> : NeuralNetworkBase<T>
{
    private Matrix<T> Weights { get; set; }
    private int MapWidth { get; set; }
    private int MapHeight { get; set; }
    private int InputDimension { get; set; }

    public SelfOrganizingMap(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        // Get input dimension from the architecture
        InputDimension = architecture.InputSize;
    
        if (InputDimension <= 0)
        {
            throw new ArgumentException("Input dimension must be greater than zero for SOM.");
        }
    
        // Get map size from the output size
        int mapSize = architecture.OutputSize;
    
        if (mapSize <= 0)
        {
            throw new ArgumentException("Map size (output size) must be greater than zero for SOM.");
        }
    
        // Calculate map dimensions - try to make it as square as possible
        MapWidth = (int)Math.Sqrt(mapSize);
        MapHeight = (int)Math.Ceiling(mapSize / (double)MapWidth);
    
        // Adjust if not a perfect square
        if (MapWidth * MapHeight != mapSize)
        {
            // We'll use the closest perfect square for simplicity
            int perfectSquare = MapWidth * MapWidth;
            if (Math.Abs(perfectSquare - mapSize) < Math.Abs((MapWidth + 1) * (MapWidth + 1) - mapSize))
            {
                MapWidth = MapWidth;
                MapHeight = MapWidth; // Make it a perfect square
            }
            else
            {
                MapWidth = MapWidth + 1;
                MapHeight = MapWidth; // Make it a perfect square
            }
        
            // Log a warning that we're adjusting the map size
            Console.WriteLine($"Warning: Adjusting map size from {mapSize} to {MapWidth * MapHeight} to make it a perfect square.");
        }
    
        Weights = new Matrix<T>(MapWidth * MapHeight, InputDimension);
    
        InitializeWeights();
        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        // SOM doesn't use layers in the same way as feedforward networks
        // Instead, we'll initialize the weights directly
    }

    private void InitializeWeights()
    {
        for (int i = 0; i < MapWidth * MapHeight; i++)
        {
            for (int j = 0; j < InputDimension; j++)
            {
                Weights[i, j] = NumOps.FromDouble(Random.NextDouble());
            }
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        int bmu = FindBestMatchingUnit(input);
        return Weights.GetRow(bmu);
    }

    private int FindBestMatchingUnit(Vector<T> input)
    {
        int bmu = 0;
        T minDistance = NumOps.MaxValue;

        for (int i = 0; i < MapWidth * MapHeight; i++)
        {
            T distance = CalculateDistance(input, Weights.GetRow(i));
            if (NumOps.LessThan(distance, minDistance))
            {
                minDistance = distance;
                bmu = i;
            }
        }

        return bmu;
    }

    private T CalculateDistance(Vector<T> v1, Vector<T> v2)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            T diff = NumOps.Subtract(v1[i], v2[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }

    public void Train(Vector<T> input, int epochs, T initialLearningRate, T initialRadius)
    {
        T timeConstant = NumOps.FromDouble(epochs / Math.Log(Convert.ToDouble(initialRadius)));

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            int bmu = FindBestMatchingUnit(input);
            T learningRate = CalculateLearningRate(initialLearningRate, epoch, epochs);
            T radius = CalculateRadius(initialRadius, epoch, timeConstant);

            UpdateWeights(input, bmu, learningRate, radius);
        }
    }

    private void UpdateWeights(Vector<T> input, int bmu, T learningRate, T radius)
    {
        int bmuX = bmu % MapWidth;
        int bmuY = bmu / MapWidth;

        for (int x = 0; x < MapWidth; x++)
        {
            for (int y = 0; y < MapHeight; y++)
            {
                int index = y * MapWidth + x;
                T distance = NumOps.Sqrt(NumOps.FromDouble((x - bmuX) * (x - bmuX) + (y - bmuY) * (y - bmuY)));

                if (NumOps.LessThan(distance, radius))
                {
                    T influence = CalculateInfluence(distance, radius);
                    Vector<T> weightDelta = CalculateWeightDelta(input, Weights.GetRow(index), learningRate, influence);
                    Vector<T> updatedWeights = new Vector<T>(InputDimension);
                    for (int i = 0; i < InputDimension; i++)
                    {
                        updatedWeights[i] = NumOps.Add(Weights[index, i], weightDelta[i]);
                    }

                    Weights.SetRow(index, updatedWeights);
                }
            }
        }
    }

    private T CalculateLearningRate(T initialLearningRate, int currentEpoch, int totalEpochs)
    {
        return NumOps.Multiply(initialLearningRate, NumOps.Exp(NumOps.Negate(NumOps.FromDouble(currentEpoch / totalEpochs))));
    }

    private T CalculateRadius(T initialRadius, int currentEpoch, T timeConstant)
    {
        return NumOps.Multiply(initialRadius, NumOps.Exp(NumOps.Negate(NumOps.Divide(NumOps.FromDouble(currentEpoch), timeConstant))));
    }

    private T CalculateInfluence(T distance, T radius)
    {
        T distanceSquared = NumOps.Multiply(distance, distance);
        T radiusSquared = NumOps.Multiply(radius, radius);

        return NumOps.Exp(NumOps.Negate(NumOps.Divide(distanceSquared, NumOps.Multiply(NumOps.FromDouble(2), radiusSquared))));
    }

    private Vector<T> CalculateWeightDelta(Vector<T> input, Vector<T> weight, T learningRate, T influence)
    {
        Vector<T> delta = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            delta[i] = NumOps.Multiply(NumOps.Multiply(learningRate, influence), NumOps.Subtract(input[i], weight[i]));
        }

        return delta;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        // This method is not typically used in SOMs
        throw new NotImplementedException("UpdateParameters is not implemented for Self-Organizing Maps.");
    }

    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(MapWidth);
        writer.Write(MapHeight);
        writer.Write(InputDimension);

        for (int i = 0; i < MapWidth * MapHeight; i++)
        {
            for (int j = 0; j < InputDimension; j++)
            {
                writer.Write(Convert.ToDouble(Weights[i, j]));
            }
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        MapWidth = reader.ReadInt32();
        MapHeight = reader.ReadInt32();
        InputDimension = reader.ReadInt32();

        Weights = new Matrix<T>(MapWidth * MapHeight, InputDimension);

        for (int i = 0; i < MapWidth * MapHeight; i++)
        {
            for (int j = 0; j < InputDimension; j++)
            {
                Weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }
}