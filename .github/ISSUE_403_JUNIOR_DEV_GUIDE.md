# Issue #403: Junior Developer Implementation Guide
## Federated Learning Infrastructure

---

## Table of Contents
1. [Understanding Federated Learning](#understanding-federated-learning)
2. [Core Concepts and Algorithms](#core-concepts-and-algorithms)
3. [Architecture Design](#architecture-design)
4. [Implementation Strategy](#implementation-strategy)
5. [Privacy-Preserving Techniques](#privacy-preserving-techniques)
6. [Testing Strategy](#testing-strategy)
7. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)

---

## Understanding Federated Learning

### What is Federated Learning?

Federated Learning is a distributed machine learning approach where:
- **Training data remains on client devices** (phones, edge devices, hospitals)
- **Models are trained locally** on each client's private data
- **Only model updates are shared** with a central server
- **Privacy is preserved** by never sending raw data to the server

### Why Federated Learning?

**Traditional Machine Learning**:
```
[Client Data] → [Central Server] → [Train Model] → [Deploy Model]
                  ↑ Problem: Privacy risk, data transmission costs
```

**Federated Learning**:
```
[Client 1: Local Training] ─┐
[Client 2: Local Training] ─┼→ [Central Server: Aggregation] → [Global Model]
[Client 3: Local Training] ─┘
    ↑ Data never leaves clients
```

### Real-World Use Cases

1. **Mobile Keyboard Prediction** (Google Gboard)
   - Each phone trains on user's typing patterns
   - Models improve without sending private messages to server

2. **Healthcare** (Hospital Collaboration)
   - Hospitals train on patient data locally
   - Models aggregate learning without sharing patient records

3. **Financial Services** (Fraud Detection)
   - Banks train on transaction data locally
   - Improve fraud detection without sharing customer data

---

## Core Concepts and Algorithms

### 1. Federated Averaging (FedAvg)

The foundational algorithm for federated learning.

**Algorithm Overview**:
```
Server maintains global model W
For each communication round t = 1, 2, 3, ...:
    1. Server sends W to selected clients
    2. Each client k trains locally:
        - W_k ← W (initialize with global model)
        - For local epochs e = 1..E:
            - W_k ← W_k - η∇L(W_k; D_k)  (gradient descent on local data D_k)
    3. Server aggregates updates:
        - W ← Σ(n_k/n) * W_k  (weighted average by client dataset size)
        where n_k = size of client k's dataset, n = total dataset size
```

**Mathematical Foundation**:

The goal is to minimize the global loss function:
```
F(W) = Σ(n_k/n) * F_k(W)
```
where:
- `F(W)` = Global loss across all clients
- `F_k(W)` = Local loss on client k's data
- `n_k` = Number of samples on client k
- `n = Σn_k` = Total samples across all clients

**Key Parameters**:
- `C`: Fraction of clients selected per round (e.g., 0.1 = 10% of clients)
- `E`: Number of local training epochs on each client
- `B`: Local batch size for client training
- `η`: Learning rate for local updates

**Pseudocode Implementation**:
```csharp
// Server-side FedAvg
public class FederatedAveragingServer<T>
{
    public void Train(int totalRounds, double clientFraction)
    {
        for (int round = 0; round < totalRounds; round++)
        {
            // 1. Select random subset of clients
            var selectedClients = SelectClients(clientFraction);

            // 2. Send global model to each selected client
            var clientUpdates = new List<ModelUpdate<T>>();
            foreach (var client in selectedClients)
            {
                var update = client.LocalTrain(_globalModel);
                clientUpdates.Add(update);
            }

            // 3. Aggregate updates (weighted average)
            _globalModel = AggregateUpdates(clientUpdates);

            // 4. Evaluate global model
            var accuracy = EvaluateGlobalModel();
            Console.WriteLine($"Round {round}: Accuracy = {accuracy:F4}");
        }
    }

    private NeuralNetwork<T> AggregateUpdates(List<ModelUpdate<T>> updates)
    {
        // Weighted average by dataset size
        var totalSamples = updates.Sum(u => u.NumSamples);
        var aggregatedModel = new NeuralNetwork<T>(_architecture);

        foreach (var layer in aggregatedModel.Layers)
        {
            var aggregatedWeights = new Tensor<T>(layer.Weights.Shape);

            foreach (var update in updates)
            {
                var weight = (double)update.NumSamples / totalSamples;
                var clientWeights = update.Model.GetLayer(layer.Name).Weights;
                aggregatedWeights = aggregatedWeights.Add(
                    clientWeights.Multiply(NumOps<T>.FromDouble(weight))
                );
            }

            layer.SetWeights(aggregatedWeights);
        }

        return aggregatedModel;
    }
}
```

### 2. Client Selection Strategies

**Random Selection** (Standard FedAvg):
```csharp
private List<IFederatedClient<T>> SelectClients(double fraction)
{
    int numSelected = (int)(AllClients.Count * fraction);
    return AllClients.OrderBy(c => _random.Next()).Take(numSelected).ToList();
}
```

**Weighted Selection** (Based on data size):
```csharp
private List<IFederatedClient<T>> SelectClientsWeighted(double fraction)
{
    var weights = AllClients.Select(c => c.GetDatasetSize()).ToArray();
    var totalWeight = weights.Sum();

    var selected = new List<IFederatedClient<T>>();
    int numSelected = (int)(AllClients.Count * fraction);

    for (int i = 0; i < numSelected; i++)
    {
        double r = _random.NextDouble() * totalWeight;
        double cumulative = 0;

        for (int j = 0; j < AllClients.Count; j++)
        {
            cumulative += weights[j];
            if (r <= cumulative && !selected.Contains(AllClients[j]))
            {
                selected.Add(AllClients[j]);
                break;
            }
        }
    }

    return selected;
}
```

**Importance-Based Selection** (Based on loss):
```csharp
private List<IFederatedClient<T>> SelectClientsByImportance(double fraction)
{
    // Clients with higher loss get higher selection probability
    var losses = AllClients.Select(c => c.GetLocalLoss(_globalModel)).ToArray();

    // Softmax to convert losses to probabilities
    var maxLoss = losses.Max();
    var expLosses = losses.Select(l => Math.Exp(l - maxLoss)).ToArray();
    var sumExp = expLosses.Sum();
    var probabilities = expLosses.Select(e => e / sumExp).ToArray();

    // Sample without replacement
    return SampleWithoutReplacement(AllClients, probabilities, fraction);
}
```

### 3. Aggregation Algorithms

**Weighted Average** (Standard FedAvg):
```csharp
W_global = Σ(n_k/n) * W_k
```

**Median Aggregation** (Robust to outliers):
```csharp
private NeuralNetwork<T> AggregateMedian(List<ModelUpdate<T>> updates)
{
    var aggregatedModel = new NeuralNetwork<T>(_architecture);

    foreach (var layer in aggregatedModel.Layers)
    {
        var allWeights = updates.Select(u =>
            u.Model.GetLayer(layer.Name).Weights.ToArray()
        ).ToList();

        // Compute median element-wise
        var medianWeights = new T[layer.Weights.Length];
        for (int i = 0; i < medianWeights.Length; i++)
        {
            var values = allWeights.Select(w => w[i]).OrderBy(v => v).ToArray();
            medianWeights[i] = values[values.Length / 2];
        }

        layer.SetWeights(new Tensor<T>(medianWeights, layer.Weights.Shape));
    }

    return aggregatedModel;
}
```

**Krum Aggregation** (Byzantine-robust):
```csharp
// Select the update closest to the majority
private NeuralNetwork<T> AggregateKrum(List<ModelUpdate<T>> updates, int f)
{
    // f = maximum number of Byzantine (malicious) clients
    int n = updates.Count;
    int m = n - f - 2;  // Number of closest updates to consider

    // Compute pairwise distances
    var scores = new double[n];
    for (int i = 0; i < n; i++)
    {
        var distances = new List<double>();
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                distances.Add(ComputeModelDistance(updates[i], updates[j]));
            }
        }

        // Sum of m smallest distances
        scores[i] = distances.OrderBy(d => d).Take(m).Sum();
    }

    // Return update with smallest score
    int bestIndex = scores.Select((s, i) => (s, i)).OrderBy(x => x.s).First().i;
    return updates[bestIndex].Model;
}

private double ComputeModelDistance(ModelUpdate<T> u1, ModelUpdate<T> u2)
{
    double sum = 0;
    foreach (var layer in u1.Model.Layers)
    {
        var w1 = layer.Weights.ToArray();
        var w2 = u2.Model.GetLayer(layer.Name).Weights.ToArray();

        for (int i = 0; i < w1.Length; i++)
        {
            double diff = Convert.ToDouble(w1[i]) - Convert.ToDouble(w2[i]);
            sum += diff * diff;
        }
    }
    return Math.Sqrt(sum);
}
```

### 4. Communication Efficiency

**Model Compression** (Reduce bandwidth):
```csharp
public class ModelCompressor<T>
{
    // Quantization: Convert float32 to int8
    public byte[] QuantizeModel(NeuralNetwork<T> model)
    {
        var allWeights = model.GetAllWeights().ToArray();
        var min = allWeights.Min(w => Convert.ToDouble(w));
        var max = allWeights.Max(w => Convert.ToDouble(w));

        var quantized = new byte[allWeights.Length];
        for (int i = 0; i < allWeights.Length; i++)
        {
            double normalized = (Convert.ToDouble(allWeights[i]) - min) / (max - min);
            quantized[i] = (byte)(normalized * 255);
        }

        return quantized;
    }

    // Sparsification: Send only top-k weights by magnitude
    public SparseUpdate<T> SparsifyUpdate(NeuralNetwork<T> baseModel, NeuralNetwork<T> updatedModel, double topK)
    {
        var differences = ComputeWeightDifferences(baseModel, updatedModel);
        var topIndices = GetTopKIndices(differences, topK);

        return new SparseUpdate<T>
        {
            Indices = topIndices,
            Values = topIndices.Select(i => differences[i]).ToArray()
        };
    }

    // Gradient compression (random sparsification)
    public SparseUpdate<T> CompressGradients(Tensor<T>[] gradients, double compressionRatio)
    {
        var totalSize = gradients.Sum(g => g.Length);
        int numToKeep = (int)(totalSize * compressionRatio);

        // Random selection
        var flatGradients = gradients.SelectMany(g => g.ToArray()).ToArray();
        var indices = Enumerable.Range(0, flatGradients.Length)
            .OrderBy(i => _random.Next())
            .Take(numToKeep)
            .ToArray();

        return new SparseUpdate<T>
        {
            Indices = indices,
            Values = indices.Select(i => flatGradients[i]).ToArray()
        };
    }
}
```

---

## Architecture Design

### Class Hierarchy

```
IFederatedServer<T>
    ├── FederatedAveragingServer<T>
    ├── FederatedProxServer<T> (FedProx variant)
    └── SecureAggregationServer<T> (with differential privacy)

IFederatedClient<T>
    ├── StandardClient<T>
    ├── SimulatedClient<T> (for testing)
    └── EdgeDeviceClient<T> (for IoT devices)

IAggregationStrategy<T>
    ├── WeightedAverageAggregation<T>
    ├── MedianAggregation<T>
    ├── KrumAggregation<T>
    └── TrimmedMeanAggregation<T>

IClientSelector<T>
    ├── RandomSelector<T>
    ├── WeightedSelector<T>
    └── ImportanceBasedSelector<T>

IPrivacyMechanism<T>
    ├── DifferentialPrivacy<T>
    ├── SecureAggregation<T>
    └── HomomorphicEncryption<T>
```

### Interface Definitions

```csharp
namespace AiDotNet.FederatedLearning
{
    /// <summary>
    /// Represents a federated learning server that coordinates training across clients.
    /// </summary>
    public interface IFederatedServer<T> where T : struct
    {
        /// <summary>
        /// Train the global model for a specified number of rounds.
        /// </summary>
        void Train(int numRounds, FederatedTrainingConfig config);

        /// <summary>
        /// Get the current global model.
        /// </summary>
        NeuralNetwork<T> GetGlobalModel();

        /// <summary>
        /// Register a new client with the server.
        /// </summary>
        void RegisterClient(IFederatedClient<T> client);

        /// <summary>
        /// Evaluate the global model on test data.
        /// </summary>
        double EvaluateGlobalModel(Tensor<T> testX, Tensor<T> testY);
    }

    /// <summary>
    /// Represents a client that participates in federated learning.
    /// </summary>
    public interface IFederatedClient<T> where T : struct
    {
        /// <summary>
        /// Train locally on the client's private data.
        /// </summary>
        ModelUpdate<T> LocalTrain(NeuralNetwork<T> globalModel, int epochs, double learningRate);

        /// <summary>
        /// Get the size of the client's local dataset.
        /// </summary>
        int GetDatasetSize();

        /// <summary>
        /// Compute the local loss on the client's data.
        /// </summary>
        double GetLocalLoss(NeuralNetwork<T> model);

        /// <summary>
        /// Unique identifier for the client.
        /// </summary>
        string ClientId { get; }
    }

    /// <summary>
    /// Strategy for aggregating model updates from multiple clients.
    /// </summary>
    public interface IAggregationStrategy<T> where T : struct
    {
        /// <summary>
        /// Aggregate multiple model updates into a single global model.
        /// </summary>
        NeuralNetwork<T> Aggregate(List<ModelUpdate<T>> updates, NeuralNetwork<T> previousGlobalModel);
    }

    /// <summary>
    /// Represents a model update from a client.
    /// </summary>
    public class ModelUpdate<T> where T : struct
    {
        public NeuralNetwork<T> Model { get; set; } = null!;
        public int NumSamples { get; set; }
        public double LocalLoss { get; set; }
        public string ClientId { get; set; } = string.Empty;
        public int Round { get; set; }
    }

    /// <summary>
    /// Configuration for federated training.
    /// </summary>
    public class FederatedTrainingConfig
    {
        public double ClientFraction { get; set; } = 0.1;  // 10% of clients per round
        public int LocalEpochs { get; set; } = 1;
        public int LocalBatchSize { get; set; } = 32;
        public double LocalLearningRate { get; set; } = 0.01;
        public bool UseSecureAggregation { get; set; } = false;
        public bool UseDifferentialPrivacy { get; set; } = false;
        public double DPEpsilon { get; set; } = 1.0;  // Privacy budget
        public double DPDelta { get; set; } = 1e-5;
        public int MinClientsPerRound { get; set; } = 2;
        public int MaxClientsPerRound { get; set; } = 100;
    }
}
```

---

## Privacy-Preserving Techniques

### 1. Differential Privacy

**Definition**: Ensures that individual data points cannot be identified from model updates.

**Mechanism**: Add calibrated noise to gradients before sharing.

```csharp
public class DifferentialPrivacy<T> : IPrivacyMechanism<T> where T : struct
{
    private readonly double _epsilon;  // Privacy budget
    private readonly double _delta;    // Failure probability
    private readonly double _sensitivity;  // L2 sensitivity of gradients

    public DifferentialPrivacy(double epsilon = 1.0, double delta = 1e-5, double sensitivity = 1.0)
    {
        _epsilon = epsilon;
        _delta = delta;
        _sensitivity = sensitivity;
    }

    /// <summary>
    /// Apply differential privacy to model update.
    /// Uses Gaussian mechanism for (ε, δ)-differential privacy.
    /// </summary>
    public ModelUpdate<T> ApplyPrivacy(ModelUpdate<T> update)
    {
        // Compute noise scale: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        double noiseScale = _sensitivity * Math.Sqrt(2 * Math.Log(1.25 / _delta)) / _epsilon;

        var noisyModel = new NeuralNetwork<T>(update.Model.Architecture);

        foreach (var layer in update.Model.Layers)
        {
            var weights = layer.Weights.ToArray();
            var noisyWeights = new T[weights.Length];

            for (int i = 0; i < weights.Length; i++)
            {
                // Add Gaussian noise
                double noise = SampleGaussian(0, noiseScale);
                double noisyValue = Convert.ToDouble(weights[i]) + noise;
                noisyWeights[i] = NumOps<T>.FromDouble(noisyValue);
            }

            noisyModel.GetLayer(layer.Name).SetWeights(
                new Tensor<T>(noisyWeights, layer.Weights.Shape)
            );
        }

        return new ModelUpdate<T>
        {
            Model = noisyModel,
            NumSamples = update.NumSamples,
            ClientId = update.ClientId,
            LocalLoss = update.LocalLoss,
            Round = update.Round
        };
    }

    /// <summary>
    /// Clip gradients to bound sensitivity.
    /// </summary>
    public Tensor<T>[] ClipGradients(Tensor<T>[] gradients, double clipNorm)
    {
        // Compute L2 norm of all gradients
        double totalNorm = 0;
        foreach (var grad in gradients)
        {
            foreach (var value in grad.ToArray())
            {
                double v = Convert.ToDouble(value);
                totalNorm += v * v;
            }
        }
        totalNorm = Math.Sqrt(totalNorm);

        // Clip if exceeds threshold
        if (totalNorm > clipNorm)
        {
            double scaleFactor = clipNorm / totalNorm;
            return gradients.Select(grad =>
                grad.Multiply(NumOps<T>.FromDouble(scaleFactor))
            ).ToArray();
        }

        return gradients;
    }

    private double SampleGaussian(double mean, double stddev)
    {
        // Box-Muller transform
        double u1 = 1.0 - _random.NextDouble();
        double u2 = 1.0 - _random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stddev * randStdNormal;
    }
}
```

**Privacy Budget Management**:
```csharp
public class PrivacyBudgetTracker
{
    private double _remainingEpsilon;
    private readonly double _totalEpsilon;

    public PrivacyBudgetTracker(double totalEpsilon)
    {
        _totalEpsilon = totalEpsilon;
        _remainingEpsilon = totalEpsilon;
    }

    public bool CanAfford(double epsilonCost)
    {
        return _remainingEpsilon >= epsilonCost;
    }

    public void Spend(double epsilonCost)
    {
        if (!CanAfford(epsilonCost))
            throw new InvalidOperationException("Insufficient privacy budget");

        _remainingEpsilon -= epsilonCost;
    }

    public double GetRemainingBudget() => _remainingEpsilon;
}
```

### 2. Secure Aggregation

**Definition**: Server aggregates updates without seeing individual client contributions.

**Mechanism**: Cryptographic protocol where server only sees the sum, not individual values.

```csharp
public class SecureAggregation<T> : IPrivacyMechanism<T> where T : struct
{
    /// <summary>
    /// Simplified secure aggregation using additive secret sharing.
    /// In production, use proper cryptographic protocols (e.g., Boneh-Lynn-Shacham).
    /// </summary>
    public NeuralNetwork<T> AggregateSecurely(List<ModelUpdate<T>> updates)
    {
        int numClients = updates.Count;

        // Step 1: Each client generates pairwise shared secrets
        var secrets = GeneratePairwiseSecrets(numClients);

        // Step 2: Each client masks their update
        var maskedUpdates = new List<MaskedUpdate<T>>();
        for (int i = 0; i < numClients; i++)
        {
            var masked = MaskUpdate(updates[i], secrets[i], i, numClients);
            maskedUpdates.Add(masked);
        }

        // Step 3: Server aggregates masked updates
        var aggregated = AggregateMaskedUpdates(maskedUpdates);

        // Step 4: Masks cancel out in the sum
        // Sum of masks = Σ(s_ij - s_ji) = 0 for all pairs (i,j)

        return aggregated;
    }

    private double[][][] GeneratePairwiseSecrets(int numClients)
    {
        // secrets[i][j] = shared secret between client i and client j
        var secrets = new double[numClients][][];

        for (int i = 0; i < numClients; i++)
        {
            secrets[i] = new double[numClients][];
            for (int j = 0; j < numClients; j++)
            {
                if (i < j)
                {
                    // Generate random secret
                    secrets[i][j] = GenerateRandomVector();
                    secrets[j][i] = secrets[i][j].Select(s => -s).ToArray();  // Negation
                }
            }
        }

        return secrets;
    }

    private MaskedUpdate<T> MaskUpdate(ModelUpdate<T> update, double[][] secrets, int clientId, int numClients)
    {
        var weights = update.Model.GetAllWeights().ToArray();
        var masked = new T[weights.Length];

        for (int i = 0; i < weights.Length; i++)
        {
            double sum = Convert.ToDouble(weights[i]);

            // Add pairwise secrets
            for (int j = 0; j < numClients; j++)
            {
                if (j != clientId && secrets[j] != null)
                {
                    sum += secrets[j][i % secrets[j].Length];
                }
            }

            masked[i] = NumOps<T>.FromDouble(sum);
        }

        return new MaskedUpdate<T> { MaskedWeights = masked };
    }

    private double[] GenerateRandomVector(int length = 1000)
    {
        return Enumerable.Range(0, length)
            .Select(_ => (_random.NextDouble() - 0.5) * 2)
            .ToArray();
    }
}
```

### 3. Homomorphic Encryption

**Definition**: Perform computations on encrypted data without decrypting.

```csharp
public class HomomorphicEncryption<T> where T : struct
{
    // Simplified Paillier-like encryption (conceptual)
    private readonly BigInteger _publicKey;
    private readonly BigInteger _privateKey;

    public BigInteger Encrypt(double value)
    {
        // Simplified: real implementation uses Paillier cryptosystem
        long scaledValue = (long)(value * 1e6);
        return BigInteger.Pow(scaledValue, 2) % _publicKey;
    }

    public double Decrypt(BigInteger ciphertext)
    {
        // Simplified decryption
        var decrypted = BigInteger.ModPow(ciphertext, _privateKey, _publicKey);
        return (double)decrypted / 1e6;
    }

    public BigInteger Add(BigInteger c1, BigInteger c2)
    {
        // Homomorphic addition: E(a) * E(b) = E(a + b)
        return (c1 * c2) % _publicKey;
    }

    public BigInteger ScalarMultiply(BigInteger c, double scalar)
    {
        // Homomorphic scalar multiplication: E(a)^k = E(k*a)
        long scaledScalar = (long)(scalar * 1e6);
        return BigInteger.ModPow(c, scaledScalar, _publicKey);
    }
}
```

---

## Testing Strategy

### Unit Tests

```csharp
[TestClass]
public class FederatedAveragingTests
{
    [TestMethod]
    public void FedAvg_ConvergesOnIID_Data()
    {
        // Arrange: Create server with 10 clients, IID data
        var server = new FederatedAveragingServer<double>(
            new SimpleNNArchitecture(inputSize: 10, outputSize: 2)
        );

        var clients = CreateIIDClients(numClients: 10, samplesPerClient: 100);
        foreach (var client in clients)
            server.RegisterClient(client);

        // Act: Train for 50 rounds
        var config = new FederatedTrainingConfig
        {
            ClientFraction = 0.5,
            LocalEpochs = 5,
            LocalLearningRate = 0.01
        };
        server.Train(numRounds: 50, config);

        // Assert: Model achieves > 90% accuracy
        var accuracy = server.EvaluateGlobalModel(testX, testY);
        Assert.IsTrue(accuracy > 0.9, $"Expected accuracy > 0.9, got {accuracy}");
    }

    [TestMethod]
    public void WeightedAverage_RespectsSampleSizes()
    {
        // Arrange: Two clients with different data sizes
        var client1 = CreateClient(numSamples: 100);
        var client2 = CreateClient(numSamples: 900);

        var update1 = client1.LocalTrain(globalModel, epochs: 1, learningRate: 0.01);
        var update2 = client2.LocalTrain(globalModel, epochs: 1, learningRate: 0.01);

        // Act: Aggregate
        var aggregator = new WeightedAverageAggregation<double>();
        var result = aggregator.Aggregate(new[] { update1, update2 }, globalModel);

        // Assert: Result is closer to client2 (90% weight)
        var distanceToClient1 = ComputeModelDistance(result, update1.Model);
        var distanceToClient2 = ComputeModelDistance(result, update2.Model);
        Assert.IsTrue(distanceToClient2 < distanceToClient1);
    }

    [TestMethod]
    public void DifferentialPrivacy_AddsNoise()
    {
        // Arrange
        var dp = new DifferentialPrivacy<double>(epsilon: 1.0, delta: 1e-5);
        var originalUpdate = CreateModelUpdate();

        // Act
        var noisyUpdate = dp.ApplyPrivacy(originalUpdate);

        // Assert: Weights are different
        var originalWeights = originalUpdate.Model.GetAllWeights().ToArray();
        var noisyWeights = noisyUpdate.Model.GetAllWeights().ToArray();

        bool hasDifference = false;
        for (int i = 0; i < originalWeights.Length; i++)
        {
            if (Math.Abs(originalWeights[i] - noisyWeights[i]) > 1e-10)
            {
                hasDifference = true;
                break;
            }
        }
        Assert.IsTrue(hasDifference, "Differential privacy should add noise");
    }
}
```

### Integration Tests

```csharp
[TestClass]
public class FederatedLearningIntegrationTests
{
    [TestMethod]
    public void EndToEnd_MNIST_FederatedTraining()
    {
        // Arrange: Simulate 100 clients with non-IID MNIST data
        var (trainData, testData) = LoadMNIST();
        var clients = PartitionNonIID(trainData, numClients: 100, alpha: 0.5);

        var server = new FederatedAveragingServer<double>(
            CreateMNISTArchitecture()
        );

        foreach (var client in clients)
            server.RegisterClient(client);

        // Act: Train for 100 rounds
        var config = new FederatedTrainingConfig
        {
            ClientFraction = 0.1,  // 10 clients per round
            LocalEpochs = 5,
            LocalBatchSize = 32,
            LocalLearningRate = 0.01,
            UseDifferentialPrivacy = true,
            DPEpsilon = 1.0
        };

        server.Train(numRounds: 100, config);

        // Assert: Achieves reasonable accuracy despite non-IID data
        var accuracy = server.EvaluateGlobalModel(testData.X, testData.Y);
        Assert.IsTrue(accuracy > 0.85, $"Expected accuracy > 0.85, got {accuracy}");
    }

    [TestMethod]
    public void SecureAggregation_MatchesPlainAggregation()
    {
        // Arrange
        var updates = CreateTestUpdates(numClients: 5);

        // Act
        var plainAgg = new WeightedAverageAggregation<double>();
        var plainResult = plainAgg.Aggregate(updates, globalModel);

        var secureAgg = new SecureAggregation<double>();
        var secureResult = secureAgg.AggregateSecurely(updates);

        // Assert: Results should be identical
        AssertModelsEqual(plainResult, secureResult, tolerance: 1e-6);
    }
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Core Infrastructure (Week 1)

**Step 1: Create Base Interfaces**
```bash
# Create directory structure
mkdir -p src/FederatedLearning/{Server,Client,Aggregation,Privacy}
```

**Files to create**:
1. `src/FederatedLearning/IFederatedServer.cs`
2. `src/FederatedLearning/IFederatedClient.cs`
3. `src/FederatedLearning/ModelUpdate.cs`
4. `src/FederatedLearning/FederatedTrainingConfig.cs`

**Step 2: Implement Basic FedAvg Server**

File: `src/FederatedLearning/Server/FederatedAveragingServer.cs`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.FederatedLearning.Server
{
    public class FederatedAveragingServer<T> : IFederatedServer<T> where T : struct
    {
        private readonly NeuralNetworkArchitecture<T> _architecture;
        private NeuralNetwork<T> _globalModel;
        private readonly List<IFederatedClient<T>> _clients;
        private readonly Random _random;

        public FederatedAveragingServer(NeuralNetworkArchitecture<T> architecture)
        {
            _architecture = architecture;
            _globalModel = new NeuralNetwork<T>(architecture);
            _clients = new List<IFederatedClient<T>>();
            _random = new Random(42);
        }

        public void RegisterClient(IFederatedClient<T> client)
        {
            _clients.Add(client);
        }

        public void Train(int numRounds, FederatedTrainingConfig config)
        {
            for (int round = 0; round < numRounds; round++)
            {
                Console.WriteLine($"=== Round {round + 1}/{numRounds} ===");

                // 1. Select clients
                var selectedClients = SelectClients(config.ClientFraction);
                Console.WriteLine($"Selected {selectedClients.Count} clients");

                // 2. Distribute global model and collect updates
                var updates = new List<ModelUpdate<T>>();
                foreach (var client in selectedClients)
                {
                    var update = client.LocalTrain(
                        _globalModel,
                        config.LocalEpochs,
                        config.LocalLearningRate
                    );
                    update.Round = round;
                    updates.Add(update);
                }

                // 3. Aggregate updates
                _globalModel = AggregateUpdates(updates);

                // 4. Evaluate (optional)
                if (round % 10 == 0)
                {
                    var avgLoss = updates.Average(u => u.LocalLoss);
                    Console.WriteLine($"Average client loss: {avgLoss:F4}");
                }
            }
        }

        public NeuralNetwork<T> GetGlobalModel()
        {
            return _globalModel;
        }

        public double EvaluateGlobalModel(Tensor<T> testX, Tensor<T> testY)
        {
            return _globalModel.Evaluate(testX, testY);
        }

        private List<IFederatedClient<T>> SelectClients(double fraction)
        {
            int numToSelect = Math.Max(1, (int)(_clients.Count * fraction));
            return _clients.OrderBy(_ => _random.Next()).Take(numToSelect).ToList();
        }

        private NeuralNetwork<T> AggregateUpdates(List<ModelUpdate<T>> updates)
        {
            // Weighted average by dataset size
            var totalSamples = updates.Sum(u => u.NumSamples);
            var aggregated = new NeuralNetwork<T>(_architecture);

            // Initialize all weights to zero
            foreach (var layer in aggregated.Layers)
            {
                layer.SetWeights(new Tensor<T>(layer.Weights.Shape));
            }

            // Accumulate weighted updates
            foreach (var update in updates)
            {
                double weight = (double)update.NumSamples / totalSamples;

                foreach (var layer in aggregated.Layers)
                {
                    var currentWeights = layer.Weights;
                    var clientWeights = update.Model.GetLayer(layer.Name).Weights;
                    var weighted = clientWeights.Multiply(NumOps<T>.FromDouble(weight));
                    layer.SetWeights(currentWeights.Add(weighted));
                }
            }

            return aggregated;
        }
    }
}
```

**Step 3: Implement Standard Client**

File: `src/FederatedLearning/Client/StandardClient.cs`

```csharp
namespace AiDotNet.FederatedLearning.Client
{
    public class StandardClient<T> : IFederatedClient<T> where T : struct
    {
        private readonly Tensor<T> _localX;
        private readonly Tensor<T> _localY;
        private readonly ILossFunction<T> _lossFunction;
        private readonly IOptimizer<T> _optimizer;

        public string ClientId { get; }

        public StandardClient(
            string clientId,
            Tensor<T> localX,
            Tensor<T> localY,
            ILossFunction<T> lossFunction,
            IOptimizer<T> optimizer)
        {
            ClientId = clientId;
            _localX = localX;
            _localY = localY;
            _lossFunction = lossFunction;
            _optimizer = optimizer;
        }

        public ModelUpdate<T> LocalTrain(NeuralNetwork<T> globalModel, int epochs, double learningRate)
        {
            // Clone global model for local training
            var localModel = CloneModel(globalModel);
            _optimizer.LearningRate = learningRate;

            // Train locally
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                localModel.Train(_localX, _localY, _optimizer, _lossFunction, batchSize: 32);
            }

            // Compute final loss
            var predictions = localModel.Forward(_localX);
            var loss = _lossFunction.ComputeLoss(predictions, _localY);

            return new ModelUpdate<T>
            {
                Model = localModel,
                NumSamples = _localX.Shape[0],
                LocalLoss = loss,
                ClientId = ClientId
            };
        }

        public int GetDatasetSize() => _localX.Shape[0];

        public double GetLocalLoss(NeuralNetwork<T> model)
        {
            var predictions = model.Forward(_localX);
            return _lossFunction.ComputeLoss(predictions, _localY);
        }

        private NeuralNetwork<T> CloneModel(NeuralNetwork<T> model)
        {
            var clone = new NeuralNetwork<T>(model.Architecture);

            foreach (var layer in model.Layers)
            {
                clone.GetLayer(layer.Name).SetWeights(layer.Weights.Clone());
            }

            return clone;
        }
    }
}
```

### Phase 2: Aggregation Strategies (Week 2)

**Step 4: Implement Aggregation Interface**

File: `src/FederatedLearning/Aggregation/IAggregationStrategy.cs`

```csharp
namespace AiDotNet.FederatedLearning.Aggregation
{
    public interface IAggregationStrategy<T> where T : struct
    {
        NeuralNetwork<T> Aggregate(
            List<ModelUpdate<T>> updates,
            NeuralNetwork<T> previousGlobalModel
        );
    }
}
```

**Step 5: Implement Median Aggregation**

File: `src/FederatedLearning/Aggregation/MedianAggregation.cs`

```csharp
namespace AiDotNet.FederatedLearning.Aggregation
{
    public class MedianAggregation<T> : IAggregationStrategy<T> where T : struct
    {
        public NeuralNetwork<T> Aggregate(
            List<ModelUpdate<T>> updates,
            NeuralNetwork<T> previousGlobalModel)
        {
            var aggregated = new NeuralNetwork<T>(previousGlobalModel.Architecture);

            foreach (var layer in aggregated.Layers)
            {
                var layerName = layer.Name;
                var allWeights = updates
                    .Select(u => u.Model.GetLayer(layerName).Weights.ToArray())
                    .ToList();

                // Compute element-wise median
                int numElements = layer.Weights.Length;
                var medianWeights = new T[numElements];

                for (int i = 0; i < numElements; i++)
                {
                    var values = allWeights
                        .Select(w => Convert.ToDouble(w[i]))
                        .OrderBy(v => v)
                        .ToArray();

                    double median = values.Length % 2 == 0
                        ? (values[values.Length / 2 - 1] + values[values.Length / 2]) / 2
                        : values[values.Length / 2];

                    medianWeights[i] = NumOps<T>.FromDouble(median);
                }

                layer.SetWeights(new Tensor<T>(medianWeights, layer.Weights.Shape));
            }

            return aggregated;
        }
    }
}
```

### Phase 3: Privacy Mechanisms (Week 3)

**Step 6: Implement Differential Privacy**

File: `src/FederatedLearning/Privacy/DifferentialPrivacy.cs`

(See full implementation in Privacy-Preserving Techniques section)

**Step 7: Add Privacy to Server**

Modify `FederatedAveragingServer` to support privacy:

```csharp
public void Train(int numRounds, FederatedTrainingConfig config)
{
    DifferentialPrivacy<T>? dp = null;
    if (config.UseDifferentialPrivacy)
    {
        dp = new DifferentialPrivacy<T>(config.DPEpsilon, config.DPDelta);
    }

    for (int round = 0; round < numRounds; round++)
    {
        var selectedClients = SelectClients(config.ClientFraction);
        var updates = new List<ModelUpdate<T>>();

        foreach (var client in selectedClients)
        {
            var update = client.LocalTrain(_globalModel, config.LocalEpochs, config.LocalLearningRate);

            // Apply differential privacy if enabled
            if (dp != null)
            {
                update = dp.ApplyPrivacy(update);
            }

            updates.Add(update);
        }

        _globalModel = AggregateUpdates(updates);
    }
}
```

### Phase 4: Testing and Validation (Week 4)

**Step 8: Create Test Data Partitioner**

File: `tests/FederatedLearning/TestHelpers/DataPartitioner.cs`

```csharp
public static class DataPartitioner
{
    /// <summary>
    /// Partition data into IID (Independent and Identically Distributed) subsets.
    /// </summary>
    public static List<(Tensor<T> X, Tensor<T> Y)> PartitionIID<T>(
        Tensor<T> X, Tensor<T> Y, int numClients) where T : struct
    {
        int samplesPerClient = X.Shape[0] / numClients;
        var partitions = new List<(Tensor<T>, Tensor<T>)>();

        for (int i = 0; i < numClients; i++)
        {
            int start = i * samplesPerClient;
            int end = (i == numClients - 1) ? X.Shape[0] : start + samplesPerClient;

            var clientX = X.Slice(start, end);
            var clientY = Y.Slice(start, end);
            partitions.Add((clientX, clientY));
        }

        return partitions;
    }

    /// <summary>
    /// Partition data into non-IID subsets using Dirichlet distribution.
    /// Lower alpha = more skewed distribution.
    /// </summary>
    public static List<(Tensor<T> X, Tensor<T> Y)> PartitionNonIID<T>(
        Tensor<T> X, Tensor<T> Y, int numClients, double alpha = 0.5) where T : struct
    {
        // Group samples by label
        var labeledData = GroupByLabel(X, Y);

        // For each class, distribute samples using Dirichlet
        var clientAssignments = new List<List<int>>();
        for (int i = 0; i < numClients; i++)
            clientAssignments.Add(new List<int>());

        foreach (var (label, indices) in labeledData)
        {
            var proportions = SampleDirichlet(numClients, alpha);
            var shuffledIndices = indices.OrderBy(_ => Random.Shared.Next()).ToArray();

            int offset = 0;
            for (int i = 0; i < numClients; i++)
            {
                int count = (int)(proportions[i] * shuffledIndices.Length);
                clientAssignments[i].AddRange(
                    shuffledIndices.Skip(offset).Take(count)
                );
                offset += count;
            }
        }

        // Create tensors for each client
        return clientAssignments.Select(indices =>
            (X.Gather(indices), Y.Gather(indices))
        ).ToList();
    }
}
```

**Step 9: Write Comprehensive Tests**

(See Testing Strategy section for complete test suite)

### Phase 5: Documentation and Examples (Week 5)

**Step 10: Create Usage Examples**

File: `examples/FederatedLearning/MNISTFederated.cs`

```csharp
public class MNISTFederatedExample
{
    public static void Run()
    {
        // 1. Load MNIST data
        var (trainX, trainY, testX, testY) = LoadMNIST();

        // 2. Partition data across 100 clients (non-IID)
        var clientData = DataPartitioner.PartitionNonIID(
            trainX, trainY, numClients: 100, alpha: 0.5
        );

        // 3. Create server
        var architecture = new NeuralNetworkArchitecture<double>
        {
            Layers = new[]
            {
                new DenseLayer<double>(784, 128, new ReLU<double>()),
                new DenseLayer<double>(128, 10, new Softmax<double>())
            }
        };

        var server = new FederatedAveragingServer<double>(architecture);

        // 4. Create clients
        for (int i = 0; i < clientData.Count; i++)
        {
            var (x, y) = clientData[i];
            var client = new StandardClient<double>(
                clientId: $"client_{i}",
                localX: x,
                localY: y,
                lossFunction: new CategoricalCrossEntropyLoss<double>(),
                optimizer: new SGD<double>()
            );
            server.RegisterClient(client);
        }

        // 5. Train federatively
        var config = new FederatedTrainingConfig
        {
            ClientFraction = 0.1,        // 10% of clients per round
            LocalEpochs = 5,             // 5 local epochs per client
            LocalBatchSize = 32,
            LocalLearningRate = 0.01,
            UseDifferentialPrivacy = true,
            DPEpsilon = 1.0              // Privacy budget
        };

        server.Train(numRounds: 100, config);

        // 6. Evaluate
        var accuracy = server.EvaluateGlobalModel(testX, testY);
        Console.WriteLine($"Final test accuracy: {accuracy:F4}");
    }
}
```

---

## Summary

This guide provides:

1. **Theoretical Foundation**: Federated averaging algorithm with mathematical details
2. **Architecture**: Modular design with interfaces for server, client, aggregation, and privacy
3. **Privacy Techniques**: Differential privacy, secure aggregation, homomorphic encryption
4. **Practical Implementation**: Complete code for FedAvg server and client
5. **Testing**: Unit and integration tests for correctness and convergence
6. **Examples**: MNIST federated learning with non-IID data

**Key Implementation Notes**:
- Start with basic FedAvg, then add privacy
- Test on IID data first, then non-IID
- Implement multiple aggregation strategies for robustness
- Always use proper privacy budget management
- Monitor convergence with different client selection strategies

**Expected Timeline**: 5 weeks for full implementation with comprehensive testing
