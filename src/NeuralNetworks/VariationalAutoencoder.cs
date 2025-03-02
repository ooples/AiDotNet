namespace AiDotNet.NeuralNetworks;

public class VariationalAutoencoder<T> : NeuralNetworkBase<T>
{
    public int LatentSize { get; private set; }
    private MeanLayer<T>? MeanLayer { get; set; }
    private LogVarianceLayer<T>? LogVarianceLayer { get; set; }

    public VariationalAutoencoder(NeuralNetworkArchitecture<T> architecture, int latentSize) : base(architecture)
    {
        LatentSize = latentSize;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultVAELayers(Architecture, LatentSize));
        }

        SetSpecificLayers();
    }

    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        bool hasMeanLayer = false;
        bool hasLogVarianceLayer = false;
        bool hasSamplingLayer = false;

        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is MeanLayer<T>)
            {
                hasMeanLayer = true;
            }
            else if (layers[i] is LogVarianceLayer<T>)
            {
                hasLogVarianceLayer = true;
            }
            else if (layers[i] is SamplingLayer<T>)
            {
                hasSamplingLayer = true;
            }
        }

        if (!hasMeanLayer)
        {
            throw new InvalidOperationException("Custom VAE layers must include a MeanLayer.");
        }

        if (!hasLogVarianceLayer)
        {
            throw new InvalidOperationException("Custom VAE layers must include a LogVarianceLayer.");
        }

        if (!hasSamplingLayer)
        {
            throw new InvalidOperationException("Custom VAE layers must include a SamplingLayer for the reparameterization trick.");
        }
    }

    private void SetSpecificLayers()
    {
        MeanLayer = Layers.OfType<MeanLayer<T>>().FirstOrDefault();
        LogVarianceLayer = Layers.OfType<LogVarianceLayer<T>>().FirstOrDefault();

        if (MeanLayer == null || LogVarianceLayer == null)
        {
            throw new InvalidOperationException("MeanLayer and LogVarianceLayer must be present in the network.");
        }
    }

    public (Vector<T> Mean, Vector<T> LogVariance) Encode(Vector<T> input)
    {
        if (MeanLayer == null || LogVarianceLayer == null)
        {
            throw new InvalidOperationException("MeanLayer and LogVarianceLayer have not been properly initialized.");
        }

        var current = input;
        for (int i = 0; i < Layers.Count / 2; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        var mean = MeanLayer.Forward(Tensor<T>.FromVector(current)).ToVector();
        var logVariance = LogVarianceLayer.Forward(Tensor<T>.FromVector(current)).ToVector();

        return (mean, logVariance);
    }

    public Vector<T> Reparameterize(Vector<T> mean, Vector<T> logVariance)
    {
        if (mean.Length != logVariance.Length)
            throw new ArgumentException("Mean and log variance vectors must have the same length.");

        var result = new T[mean.Length];

        for (int i = 0; i < mean.Length; i++)
        {
            // Generate two random numbers from a uniform distribution
            double u1 = 1.0 - Random.NextDouble(); // Uniform(0,1] random number
            double u2 = 1.0 - Random.NextDouble(); // Uniform(0,1] random number

            // Box-Muller transform to generate a sample from a standard normal distribution
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

            // Convert z to T type
            T zT = NumOps.FromDouble(z);

            // Reparameterization trick: sample = mean + exp(0.5 * logVariance) * z
            T halfLogVariance = NumOps.Multiply(NumOps.FromDouble(0.5), logVariance[i]);
            T stdDev = NumOps.Exp(halfLogVariance);
            result[i] = NumOps.Add(mean[i], NumOps.Multiply(stdDev, zT));
        }

        return new Vector<T>(result);
    }

    public Vector<T> Decode(Vector<T> latentVector)
    {
        var current = latentVector;
        for (int i = Layers.Count / 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        var (mean, logVariance) = Encode(input);
        var latentVector = Reparameterize(mean, logVariance);

        return Decode(latentVector);
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

        // Update mean and log variance layers
        if (MeanLayer != null && LogVarianceLayer != null)
        {
            int meanParameterCount = MeanLayer.ParameterCount;
            MeanLayer.UpdateParameters(parameters.SubVector(startIndex, meanParameterCount));
            startIndex += meanParameterCount;

            int logVarianceParameterCount = LogVarianceLayer.ParameterCount;
            LogVarianceLayer.UpdateParameters(parameters.SubVector(startIndex, logVarianceParameterCount));
        }
        else
        {
            throw new InvalidOperationException("MeanLayer and LogVarianceLayer have not been properly initialized.");
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

        // Serialize mean and log variance layers
        if (MeanLayer != null && LogVarianceLayer != null)
        {
            writer.WriteInt32Array(MeanLayer.GetInputShape());
            writer.Write(MeanLayer.Axis);
            MeanLayer.Serialize(writer);

            writer.WriteInt32Array(LogVarianceLayer.GetInputShape());
            writer.Write(LogVarianceLayer.Axis);
            LogVarianceLayer.Serialize(writer);
        }
        else
        {
            throw new InvalidOperationException("MeanLayer and LogVarianceLayer have not been properly initialized.");
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

        // Deserialize mean and log variance layers
        int[] meanInputShape = reader.ReadInt32Array();
        int meanAxis = reader.ReadInt32();
        MeanLayer = new MeanLayer<T>(meanInputShape, meanAxis);
        MeanLayer.Deserialize(reader);

        int[] logVarianceInputShape = reader.ReadInt32Array();
        int logVarianceAxis = reader.ReadInt32();
        LogVarianceLayer = new LogVarianceLayer<T>(logVarianceInputShape, logVarianceAxis);
        LogVarianceLayer.Deserialize(reader);
    }
}