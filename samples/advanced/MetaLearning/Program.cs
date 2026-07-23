using AiDotNet;
using AiDotNet.MetaLearning;
using AiDotNet.LinearAlgebra;

Console.WriteLine("=== AiDotNet Meta-Learning (MAML) Sample ===");
Console.WriteLine("Model-Agnostic Meta-Learning for Few-Shot Classification\n");

// Overview
Console.WriteLine("What is Meta-Learning?");
Console.WriteLine("  - \"Learning to learn\" - training models that can adapt quickly to new tasks");
Console.WriteLine("  - Few-shot learning: classify new classes with only 1-5 examples each");
Console.WriteLine("  - MAML learns good initial weights that can be fine-tuned rapidly\n");

// Task distribution setup
Console.WriteLine("Task Distribution: Sinusoid Regression");
Console.WriteLine("  - Each task: Regress a sinusoid y = A*sin(x + phi)");
Console.WriteLine("  - Amplitude A ~ U[0.1, 5.0], Phase phi ~ U[0, pi]");
Console.WriteLine("  - 5-shot learning: 5 support points, 10 query points");
Console.WriteLine("  - Goal: Quickly adapt to new sinusoids from few examples\n");

// MAML hyperparameters
Console.WriteLine("MAML Configuration:");
Console.WriteLine("  Inner learning rate (alpha): 0.01");
Console.WriteLine("  Outer learning rate (beta):  0.001");
Console.WriteLine("  Inner loop steps: 5");
Console.WriteLine("  Meta-batch size: 4 tasks");
Console.WriteLine("  Meta-iterations: 1000\n");

try
{
    // Create the sinusoid task distribution
    var taskDistribution = new SinusoidTaskDistribution(seed: 42);

    // Create meta-model (simple neural network for regression)
    var metaModel = new SimpleNeuralNetwork(
        inputSize: 1,
        hiddenSizes: new[] { 40, 40 },
        outputSize: 1
    );

    Console.WriteLine($"Meta-model: Neural Network");
    Console.WriteLine($"  Architecture: 1 -> 40 -> 40 -> 1");
    Console.WriteLine($"  Parameters: {metaModel.ParameterCount}\n");

    // Create MAML algorithm
    var maml = new MAMLTrainer(
        metaModel: metaModel,
        innerLearningRate: 0.01,
        outerLearningRate: 0.001,
        innerLoopSteps: 5,
        useFirstOrderApproximation: true  // FOMAML for efficiency
    );

    // Meta-training parameters
    const int metaIterations = 1000;
    const int metaBatchSize = 4;
    const int reportInterval = 100;
    const int numShots = 5;
    const int numQueryPoints = 10;

    var metaLossHistory = new List<double>();
    var adaptationLossHistory = new List<double>();

    Console.WriteLine("Meta-Training (Bi-Level Optimization)...");
    Console.WriteLine("  Outer loop: Update meta-parameters to improve adaptation");
    Console.WriteLine("  Inner loop: Adapt to each task using gradient descent\n");

    Console.WriteLine("Iteration | Meta-Loss | Avg Adapt Loss | Post-Adapt Loss");
    Console.WriteLine("----------------------------------------------------------");

    var stopwatch = System.Diagnostics.Stopwatch.StartNew();

    for (int iter = 0; iter < metaIterations; iter++)
    {
        // Sample batch of tasks
        var taskBatch = taskDistribution.SampleTasks(metaBatchSize, numShots, numQueryPoints);

        // Perform one meta-training step
        var (metaLoss, avgAdaptLoss, postAdaptLoss) = maml.MetaTrainStep(taskBatch);

        metaLossHistory.Add(metaLoss);
        adaptationLossHistory.Add(postAdaptLoss);

        // Report progress
        if (iter % reportInterval == 0 || iter == metaIterations - 1)
        {
            Console.WriteLine($"{iter,9} | {metaLoss,9:F4} | {avgAdaptLoss,14:F4} | {postAdaptLoss,15:F4}");
        }
    }

    stopwatch.Stop();
    Console.WriteLine($"\nMeta-training complete! Time: {stopwatch.Elapsed.TotalSeconds:F1}s\n");

    // Display learning curves
    Console.WriteLine("--- Meta-Loss Learning Curve ---");
    DisplayLearningCurve(metaLossHistory, reportInterval);

    // Demonstrate few-shot adaptation
    Console.WriteLine("\n--- Few-Shot Adaptation Demonstration ---");
    Console.WriteLine("Adapting to 5 new tasks (never seen during training)...\n");

    // Sample new test tasks
    var testTasks = taskDistribution.SampleTasks(5, numShots, numQueryPoints);

    Console.WriteLine("Task | Amplitude |  Phase  | Pre-Adapt MSE | Post-Adapt MSE | Improvement");
    Console.WriteLine("---------------------------------------------------------------------------");

    var improvements = new List<double>();

    for (int i = 0; i < testTasks.Count; i++)
    {
        var task = testTasks[i];

        // Evaluate BEFORE adaptation (using meta-initialization)
        double preAdaptLoss = maml.EvaluateOnTask(task, adaptSteps: 0);

        // Adapt with just 5 examples (inner loop)
        var adaptedModel = maml.Adapt(task, steps: 5);

        // Evaluate AFTER adaptation
        double postAdaptLoss = EvaluateAdaptedModel(adaptedModel, task.QueryX, task.QueryY);

        double improvement = ((preAdaptLoss - postAdaptLoss) / preAdaptLoss) * 100;
        improvements.Add(improvement);

        Console.WriteLine($"  {i + 1} |    {task.Amplitude,5:F2} | {task.Phase,7:F3} | " +
                          $"{preAdaptLoss,13:F4} | {postAdaptLoss,14:F4} | {improvement,10:F1}%");
    }

    Console.WriteLine($"\nAverage improvement after 5-shot adaptation: {improvements.Average():F1}%");

    // Detailed visualization for one task
    Console.WriteLine("\n--- Detailed Adaptation Visualization ---");
    var demoTask = taskDistribution.SampleTasks(1, numShots, 50)[0];

    Console.WriteLine($"Demo Task: y = {demoTask.Amplitude:F2} * sin(x + {demoTask.Phase:F2})");
    Console.WriteLine($"Support set: {numShots} points\n");

    Console.WriteLine("Step | Loss     | Sample Predictions");
    Console.WriteLine("-------------------------------------");

    // Show adaptation progress step by step
    for (int step = 0; step <= 10; step++)
    {
        var stepModel = maml.Adapt(demoTask, steps: step);
        double stepLoss = EvaluateAdaptedModel(stepModel, demoTask.QueryX, demoTask.QueryY);

        // Sample predictions at a few x values
        var sampleX = new double[] { -3.0, 0.0, 3.0 };
        var predictions = new List<string>();
        foreach (var x in sampleX)
        {
            double pred = stepModel.Predict(x);
            double actual = demoTask.Amplitude * Math.Sin(x + demoTask.Phase);
            predictions.Add($"{pred:F2}({actual:F2})");
        }

        Console.WriteLine($"  {step,2} | {stepLoss:F4} | x=-3: {predictions[0]}, x=0: {predictions[1]}, x=3: {predictions[2]}");
    }

    // Compare with random initialization (no meta-learning)
    Console.WriteLine("\n--- Comparison: MAML vs Random Initialization ---");

    var randomModel = new SimpleNeuralNetwork(1, new[] { 40, 40 }, 1);
    var randomTrainer = new MAMLTrainer(randomModel, 0.01, 0.001, 5, true);

    var comparisonTasks = taskDistribution.SampleTasks(10, numShots, numQueryPoints);

    double mamlAvgLoss = 0;
    double randomAvgLoss = 0;

    foreach (var task in comparisonTasks)
    {
        // MAML-initialized model
        var mamlAdapted = maml.Adapt(task, steps: 5);
        mamlAvgLoss += EvaluateAdaptedModel(mamlAdapted, task.QueryX, task.QueryY);

        // Random-initialized model
        var randomAdapted = randomTrainer.Adapt(task, steps: 5);
        randomAvgLoss += EvaluateAdaptedModel(randomAdapted, task.QueryX, task.QueryY);
    }

    mamlAvgLoss /= comparisonTasks.Count;
    randomAvgLoss /= comparisonTasks.Count;

    Console.WriteLine($"  MAML-initialized (after 5-step adaptation):    {mamlAvgLoss:F4} MSE");
    Console.WriteLine($"  Random-initialized (after 5-step adaptation): {randomAvgLoss:F4} MSE");
    Console.WriteLine($"  MAML advantage: {(randomAvgLoss / mamlAvgLoss):F1}x better\n");

    // Inner vs outer loop explanation
    Console.WriteLine("--- Understanding MAML's Bi-Level Optimization ---\n");
    Console.WriteLine("INNER LOOP (Task Adaptation):");
    Console.WriteLine("  - Clone meta-model for each task");
    Console.WriteLine("  - Perform K gradient steps on support set");
    Console.WriteLine("  - theta_task = theta_meta - alpha * grad(L_support)\n");

    Console.WriteLine("OUTER LOOP (Meta-Update):");
    Console.WriteLine("  - Evaluate adapted models on query sets");
    Console.WriteLine("  - Compute meta-gradient: how to improve adaptation?");
    Console.WriteLine("  - theta_meta = theta_meta - beta * sum(grad(L_query))");
    Console.WriteLine("  - Key insight: Optimizes for POST-adaptation performance!\n");
}
catch (Exception ex)
{
    Console.WriteLine($"\nNote: Full MAML training requires complete neural network implementation.");
    Console.WriteLine($"This sample demonstrates the MAML algorithm pattern and API.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("=== Sample Complete ===");

// Helper functions
static void DisplayLearningCurve(List<double> losses, int smoothWindow)
{
    var smoothed = new List<double>();
    for (int i = 0; i < losses.Count; i += smoothWindow)
    {
        var window = losses.Skip(i).Take(smoothWindow);
        smoothed.Add(window.Average());
    }

    double maxVal = smoothed.DefaultIfEmpty(1).Max();
    double minVal = smoothed.DefaultIfEmpty(0).Min();

    Console.WriteLine($"  Final meta-loss: {losses.Last():F4}");
    Console.WriteLine($"  Best meta-loss:  {losses.Min():F4}");
    Console.WriteLine($"  Loss reduction:  {(1 - losses.Last() / losses.First()) * 100:F1}%");
}

static double EvaluateAdaptedModel(SimpleNeuralNetwork model, double[] queryX, double[] queryY)
{
    double mse = 0;
    for (int i = 0; i < queryX.Length; i++)
    {
        double pred = model.Predict(queryX[i]);
        double error = pred - queryY[i];
        mse += error * error;
    }
    return mse / queryX.Length;
}

/// <summary>
/// Generates sinusoid regression tasks for meta-learning.
/// Each task is defined by y = A*sin(x + phi) with random A and phi.
/// </summary>
public class SinusoidTaskDistribution
{
    private readonly Random _random;

    public SinusoidTaskDistribution(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    public List<SinusoidTask> SampleTasks(int numTasks, int numShots, int numQueryPoints)
    {
        var tasks = new List<SinusoidTask>();

        for (int i = 0; i < numTasks; i++)
        {
            // Sample task parameters
            double amplitude = 0.1 + _random.NextDouble() * 4.9;  // U[0.1, 5.0]
            double phase = _random.NextDouble() * Math.PI;         // U[0, pi]

            // Sample support set (for adaptation)
            var supportX = new double[numShots];
            var supportY = new double[numShots];
            for (int j = 0; j < numShots; j++)
            {
                supportX[j] = -5.0 + _random.NextDouble() * 10.0;  // U[-5, 5]
                supportY[j] = amplitude * Math.Sin(supportX[j] + phase);
            }

            // Sample query set (for meta-loss)
            var queryX = new double[numQueryPoints];
            var queryY = new double[numQueryPoints];
            for (int j = 0; j < numQueryPoints; j++)
            {
                queryX[j] = -5.0 + _random.NextDouble() * 10.0;  // U[-5, 5]
                queryY[j] = amplitude * Math.Sin(queryX[j] + phase);
            }

            tasks.Add(new SinusoidTask(amplitude, phase, supportX, supportY, queryX, queryY));
        }

        return tasks;
    }
}

/// <summary>
/// Represents a single sinusoid regression task.
/// </summary>
public class SinusoidTask
{
    public double Amplitude { get; }
    public double Phase { get; }
    public double[] SupportX { get; }
    public double[] SupportY { get; }
    public double[] QueryX { get; }
    public double[] QueryY { get; }

    public SinusoidTask(double amplitude, double phase,
                        double[] supportX, double[] supportY,
                        double[] queryX, double[] queryY)
    {
        Amplitude = amplitude;
        Phase = phase;
        SupportX = supportX;
        SupportY = supportY;
        QueryX = queryX;
        QueryY = queryY;
    }
}

/// <summary>
/// Simple feedforward neural network for function approximation.
/// </summary>
public class SimpleNeuralNetwork : ICloneable
{
    private readonly int _inputSize;
    private readonly int[] _hiddenSizes;
    private readonly int _outputSize;
    private readonly Random _random;

    // Network weights and biases
    private List<double[,]> _weights;
    private List<double[]> _biases;

    public int ParameterCount { get; }

    public SimpleNeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, int? seed = null)
    {
        _inputSize = inputSize;
        _hiddenSizes = hiddenSizes;
        _outputSize = outputSize;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        _weights = new List<double[,]>();
        _biases = new List<double[]>();

        // Initialize layers
        int prevSize = inputSize;
        int paramCount = 0;

        foreach (var hiddenSize in hiddenSizes)
        {
            _weights.Add(InitializeWeights(prevSize, hiddenSize));
            _biases.Add(new double[hiddenSize]);
            paramCount += prevSize * hiddenSize + hiddenSize;
            prevSize = hiddenSize;
        }

        // Output layer
        _weights.Add(InitializeWeights(prevSize, outputSize));
        _biases.Add(new double[outputSize]);
        paramCount += prevSize * outputSize + outputSize;

        ParameterCount = paramCount;
    }

    private SimpleNeuralNetwork(SimpleNeuralNetwork source)
    {
        _inputSize = source._inputSize;
        _hiddenSizes = source._hiddenSizes;
        _outputSize = source._outputSize;
        _random = new Random();
        ParameterCount = source.ParameterCount;

        // Deep copy weights and biases
        _weights = new List<double[,]>();
        _biases = new List<double[]>();

        foreach (var w in source._weights)
        {
            _weights.Add((double[,])w.Clone());
        }

        foreach (var b in source._biases)
        {
            _biases.Add((double[])b.Clone());
        }
    }

    private double[,] InitializeWeights(int inputSize, int outputSize)
    {
        var weights = new double[inputSize, outputSize];
        double scale = Math.Sqrt(2.0 / inputSize); // He initialization

        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                weights[i, j] = (_random.NextDouble() * 2 - 1) * scale;
            }
        }
        return weights;
    }

    public double Predict(double input)
    {
        return Predict(new[] { input })[0];
    }

    public double[] Predict(double[] input)
    {
        var activation = input;

        for (int layer = 0; layer < _weights.Count; layer++)
        {
            var w = _weights[layer];
            var b = _biases[layer];
            var nextActivation = new double[w.GetLength(1)];

            for (int j = 0; j < w.GetLength(1); j++)
            {
                double sum = b[j];
                for (int i = 0; i < w.GetLength(0); i++)
                {
                    sum += activation[i] * w[i, j];
                }

                // ReLU for hidden layers, linear for output
                nextActivation[j] = (layer < _weights.Count - 1) ? Math.Max(0, sum) : sum;
            }

            activation = nextActivation;
        }

        return activation;
    }

    public double[] GetParameters()
    {
        var parameters = new List<double>();

        foreach (var w in _weights)
        {
            for (int i = 0; i < w.GetLength(0); i++)
            {
                for (int j = 0; j < w.GetLength(1); j++)
                {
                    parameters.Add(w[i, j]);
                }
            }
        }

        foreach (var b in _biases)
        {
            parameters.AddRange(b);
        }

        return parameters.ToArray();
    }

    public void SetParameters(double[] parameters)
    {
        int index = 0;

        for (int layer = 0; layer < _weights.Count; layer++)
        {
            var w = _weights[layer];
            for (int i = 0; i < w.GetLength(0); i++)
            {
                for (int j = 0; j < w.GetLength(1); j++)
                {
                    w[i, j] = parameters[index++];
                }
            }
        }

        for (int layer = 0; layer < _biases.Count; layer++)
        {
            for (int i = 0; i < _biases[layer].Length; i++)
            {
                _biases[layer][i] = parameters[index++];
            }
        }
    }

    public (double[] gradients, double loss) ComputeGradientsAndLoss(double[] inputX, double[] targetY)
    {
        // Forward pass with activations stored for backprop
        var activations = new List<double[]> { inputX.Length == 1 ? inputX : inputX };
        var preActivations = new List<double[]>();
        var currentActivation = inputX;

        for (int layer = 0; layer < _weights.Count; layer++)
        {
            var w = _weights[layer];
            var b = _biases[layer];
            var preActivation = new double[w.GetLength(1)];
            var postActivation = new double[w.GetLength(1)];

            for (int j = 0; j < w.GetLength(1); j++)
            {
                double sum = b[j];
                for (int i = 0; i < w.GetLength(0); i++)
                {
                    sum += currentActivation[i] * w[i, j];
                }
                preActivation[j] = sum;
                postActivation[j] = (layer < _weights.Count - 1) ? Math.Max(0, sum) : sum;
            }

            preActivations.Add(preActivation);
            activations.Add(postActivation);
            currentActivation = postActivation;
        }

        // Compute loss (MSE)
        var output = activations.Last();
        double loss = 0;
        var outputGrad = new double[output.Length];

        for (int i = 0; i < output.Length; i++)
        {
            double error = output[i] - targetY[i];
            loss += error * error;
            outputGrad[i] = 2 * error / output.Length;
        }
        loss /= output.Length;

        // Backward pass
        var gradients = new List<double>();
        var delta = outputGrad;

        for (int layer = _weights.Count - 1; layer >= 0; layer--)
        {
            var w = _weights[layer];
            var prevActivation = activations[layer];
            var preAct = preActivations[layer];

            // Gradient for weights
            var wGrad = new double[w.GetLength(0), w.GetLength(1)];
            var bGrad = new double[delta.Length];

            for (int j = 0; j < w.GetLength(1); j++)
            {
                bGrad[j] = delta[j];
                for (int i = 0; i < w.GetLength(0); i++)
                {
                    wGrad[i, j] = prevActivation[i] * delta[j];
                }
            }

            // Store gradients
            var layerGrad = new List<double>();
            for (int i = 0; i < w.GetLength(0); i++)
            {
                for (int j = 0; j < w.GetLength(1); j++)
                {
                    layerGrad.Add(wGrad[i, j]);
                }
            }
            layerGrad.AddRange(bGrad);
            gradients.InsertRange(0, layerGrad);

            // Compute delta for previous layer
            if (layer > 0)
            {
                var newDelta = new double[w.GetLength(0)];
                for (int i = 0; i < w.GetLength(0); i++)
                {
                    double sum = 0;
                    for (int j = 0; j < w.GetLength(1); j++)
                    {
                        sum += w[i, j] * delta[j];
                    }
                    // ReLU derivative
                    newDelta[i] = preActivations[layer - 1][i] > 0 ? sum : 0;
                }
                delta = newDelta;
            }
        }

        return (gradients.ToArray(), loss);
    }

    public object Clone()
    {
        return new SimpleNeuralNetwork(this);
    }
}

/// <summary>
/// MAML (Model-Agnostic Meta-Learning) trainer implementation.
/// Implements bi-level optimization for few-shot learning.
/// </summary>
public class MAMLTrainer
{
    private readonly SimpleNeuralNetwork _metaModel;
    private readonly double _innerLearningRate;
    private readonly double _outerLearningRate;
    private readonly int _innerLoopSteps;
    private readonly bool _useFirstOrderApproximation;

    public MAMLTrainer(
        SimpleNeuralNetwork metaModel,
        double innerLearningRate,
        double outerLearningRate,
        int innerLoopSteps,
        bool useFirstOrderApproximation = true)
    {
        _metaModel = metaModel;
        _innerLearningRate = innerLearningRate;
        _outerLearningRate = outerLearningRate;
        _innerLoopSteps = innerLoopSteps;
        _useFirstOrderApproximation = useFirstOrderApproximation;
    }

    /// <summary>
    /// Performs one meta-training step on a batch of tasks.
    /// </summary>
    public (double metaLoss, double avgAdaptLoss, double postAdaptLoss) MetaTrainStep(List<SinusoidTask> taskBatch)
    {
        var metaParams = _metaModel.GetParameters();
        var metaGradients = new double[metaParams.Length];

        double totalMetaLoss = 0;
        double totalAdaptLoss = 0;
        double totalPostAdaptLoss = 0;

        foreach (var task in taskBatch)
        {
            // Clone meta-model for this task
            var taskModel = (SimpleNeuralNetwork)_metaModel.Clone();

            // Inner loop: Adapt to task using support set
            double adaptLoss = 0;
            for (int step = 0; step < _innerLoopSteps; step++)
            {
                // Compute gradients on support set
                double stepLoss = 0;
                var stepGradients = new double[metaParams.Length];

                for (int i = 0; i < task.SupportX.Length; i++)
                {
                    var (grad, loss) = taskModel.ComputeGradientsAndLoss(
                        new[] { task.SupportX[i] },
                        new[] { task.SupportY[i] });

                    for (int g = 0; g < grad.Length; g++)
                    {
                        stepGradients[g] += grad[g];
                    }
                    stepLoss += loss;
                }

                adaptLoss = stepLoss / task.SupportX.Length;

                // Average gradients
                for (int g = 0; g < stepGradients.Length; g++)
                {
                    stepGradients[g] /= task.SupportX.Length;
                }

                // Update task model parameters
                var params_ = taskModel.GetParameters();
                for (int p = 0; p < params_.Length; p++)
                {
                    params_[p] -= _innerLearningRate * stepGradients[p];
                }
                taskModel.SetParameters(params_);
            }

            totalAdaptLoss += adaptLoss;

            // Evaluate adapted model on query set (meta-loss)
            double queryLoss = 0;
            var queryGradients = new double[metaParams.Length];

            for (int i = 0; i < task.QueryX.Length; i++)
            {
                var (grad, loss) = taskModel.ComputeGradientsAndLoss(
                    new[] { task.QueryX[i] },
                    new[] { task.QueryY[i] });

                for (int g = 0; g < grad.Length; g++)
                {
                    queryGradients[g] += grad[g];
                }
                queryLoss += loss;
            }

            queryLoss /= task.QueryX.Length;
            totalMetaLoss += queryLoss;
            totalPostAdaptLoss += queryLoss;

            // Average query gradients
            for (int g = 0; g < queryGradients.Length; g++)
            {
                queryGradients[g] /= task.QueryX.Length;
            }

            // Accumulate meta-gradients
            // For FOMAML, we use the query gradients directly
            for (int g = 0; g < queryGradients.Length; g++)
            {
                metaGradients[g] += queryGradients[g];
            }
        }

        // Average meta-gradients across tasks
        for (int g = 0; g < metaGradients.Length; g++)
        {
            metaGradients[g] /= taskBatch.Count;
        }

        // Outer loop: Update meta-parameters
        for (int p = 0; p < metaParams.Length; p++)
        {
            metaParams[p] -= _outerLearningRate * metaGradients[p];
        }
        _metaModel.SetParameters(metaParams);

        return (
            totalMetaLoss / taskBatch.Count,
            totalAdaptLoss / taskBatch.Count,
            totalPostAdaptLoss / taskBatch.Count
        );
    }

    /// <summary>
    /// Adapts the meta-model to a new task using the support set.
    /// </summary>
    public SimpleNeuralNetwork Adapt(SinusoidTask task, int steps)
    {
        var adaptedModel = (SimpleNeuralNetwork)_metaModel.Clone();

        for (int step = 0; step < steps; step++)
        {
            var stepGradients = new double[adaptedModel.ParameterCount];

            for (int i = 0; i < task.SupportX.Length; i++)
            {
                var (grad, _) = adaptedModel.ComputeGradientsAndLoss(
                    new[] { task.SupportX[i] },
                    new[] { task.SupportY[i] });

                for (int g = 0; g < grad.Length; g++)
                {
                    stepGradients[g] += grad[g];
                }
            }

            // Average and apply gradients
            var params_ = adaptedModel.GetParameters();
            for (int p = 0; p < params_.Length; p++)
            {
                params_[p] -= _innerLearningRate * stepGradients[p] / task.SupportX.Length;
            }
            adaptedModel.SetParameters(params_);
        }

        return adaptedModel;
    }

    /// <summary>
    /// Evaluates performance on a task without or with adaptation.
    /// </summary>
    public double EvaluateOnTask(SinusoidTask task, int adaptSteps)
    {
        var model = adaptSteps > 0 ? Adapt(task, adaptSteps) : (SimpleNeuralNetwork)_metaModel.Clone();

        double mse = 0;
        for (int i = 0; i < task.QueryX.Length; i++)
        {
            double pred = model.Predict(task.QueryX[i]);
            double error = pred - task.QueryY[i];
            mse += error * error;
        }
        return mse / task.QueryX.Length;
    }
}
