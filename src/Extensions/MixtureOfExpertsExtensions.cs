using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Extensions;

/// <summary>
/// Extension methods for integrating Mixture-of-Experts layers with PredictionModelBuilder.
/// </summary>
/// <remarks>
/// <para>
/// These extensions provide convenient methods for creating neural network architectures with
/// Mixture-of-Experts layers and integrating them with the standard PredictionModelBuilder workflow.
/// </para>
/// <para>
/// <b>For Beginners:</b> These methods make it easy to use MoE layers with AiDotNet's standard model building workflow.
///
/// Instead of manually creating complex architectures, these extensions let you:
/// - Quickly create neural networks with MoE layers
/// - Use the familiar PredictionModelBuilder pattern
/// - Get automatically trained models ready for predictions
///
/// This follows AiDotNet's philosophy: configure components, call Build(), and get a trained model.
/// </para>
/// </remarks>
public static class MixtureOfExpertsExtensions
{
    /// <summary>
    /// Creates a neural network architecture with a Mixture-of-Experts layer.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations (typically float or double).</typeparam>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="outputSize">The size of the output (number of classes for classification, 1 for regression).</param>
    /// <param name="numExperts">The number of expert networks (default: 4).</param>
    /// <param name="topK">Number of experts to activate per input (0 = all experts, default: 0).</param>
    /// <param name="useLoadBalancing">Whether to enable load balancing (default: true).</param>
    /// <param name="loadBalancingWeight">Weight for load balancing loss (default: 0.01).</param>
    /// <param name="taskType">The type of task (classification or regression).</param>
    /// <returns>A NeuralNetworkArchitecture configured with MoE layer.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a simple feedforward architecture with one MoE layer, suitable for
    /// many common tasks. The architecture consists of:
    /// - Input layer
    /// - Mixture-of-Experts layer with configurable experts
    /// - Output layer
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Creates a complete neural network blueprint with MoE for use with PredictionModelBuilder.
    ///
    /// This is the easiest way to use MoE in AiDotNet:
    ///
    /// <code>
    /// // 1. Create MoE architecture
    /// var architecture = MixtureOfExpertsExtensions.CreateMoEArchitecture&lt;float&gt;(
    ///     inputSize: 10,           // 10 input features
    ///     outputSize: 3,           // 3 output classes
    ///     numExperts: 8,           // 8 specialist networks
    ///     topK: 2,                 // Use top 2 experts per input
    ///     useLoadBalancing: true,  // Ensure balanced expert usage
    ///     taskType: NeuralNetworkTaskType.MultiClassClassification
    /// );
    ///
    /// // 2. Wrap in a model
    /// var model = new NeuralNetworkModel&lt;float&gt;(architecture);
    ///
    /// // 3. Use with PredictionModelBuilder
    /// var builder = new PredictionModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;();
    /// var result = builder
    ///     .ConfigureModel(model)
    ///     .Build(trainingData, trainingLabels);
    ///
    /// // 4. Make predictions
    /// var predictions = builder.Predict(newData, result);
    /// </code>
    ///
    /// Parameters explained:
    /// - <b>inputSize:</b> How many features in your data (e.g., 10 attributes)
    /// - <b>outputSize:</b> How many outputs (e.g., 3 classes to predict)
    /// - <b>numExperts:</b> How many specialist networks (more = more capacity)
    /// - <b>topK:</b> How many experts to use per input (0 = all, 2 = top 2 only)
    /// - <b>useLoadBalancing:</b> Ensures all experts are used equally
    /// - <b>taskType:</b> What kind of problem (classification or regression)
    ///
    /// Default values are research-backed and work well for most cases.
    /// </para>
    /// </remarks>
    public static NeuralNetworkArchitecture<T> CreateMoEArchitecture<T>(
        int inputSize,
        int outputSize,
        int numExperts = 4,
        int topK = 0,
        bool useLoadBalancing = true,
        double loadBalancingWeight = 0.01,
        NeuralNetworkTaskType taskType = NeuralNetworkTaskType.MultiClassClassification)
    {
        // Use the builder to create a well-configured MoE layer
        var moeLayer = new MixtureOfExpertsBuilder<T>()
            .WithExperts(numExperts)
            .WithDimensions(inputSize, inputSize) // Keep same dimensionality
            .WithTopK(topK)
            .WithLoadBalancing(useLoadBalancing, loadBalancingWeight)
            .Build();

        // Create output layer (type depends on task)
        var outputLayer = taskType switch
        {
            NeuralNetworkTaskType.MultiClassClassification =>
                new DenseLayer<T>(inputSize, outputSize, new SoftmaxActivation<T>()),
            NeuralNetworkTaskType.BinaryClassification =>
                new DenseLayer<T>(inputSize, 1, new SigmoidActivation<T>()),
            NeuralNetworkTaskType.Regression =>
                new DenseLayer<T>(inputSize, outputSize, new IdentityActivation<T>()),
            _ => throw new ArgumentException($"Unsupported task type: {taskType}", nameof(taskType))
        };

        // Assemble layers
        var layers = new List<ILayer<T>>
        {
            moeLayer,
            outputLayer
        };

        // Create architecture
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: taskType,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers
        );
    }

    /// <summary>
    /// Creates a deep neural network architecture with multiple MoE layers.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations (typically float or double).</typeparam>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="hiddenSize">The size of hidden representations.</param>
    /// <param name="outputSize">The size of the output.</param>
    /// <param name="numMoELayers">Number of MoE layers to stack (default: 2).</param>
    /// <param name="numExperts">The number of experts per MoE layer (default: 4).</param>
    /// <param name="topK">Number of experts to activate per input (0 = all experts, default: 0).</param>
    /// <param name="useLoadBalancing">Whether to enable load balancing (default: true).</param>
    /// <param name="taskType">The type of task (classification or regression).</param>
    /// <returns>A deep NeuralNetworkArchitecture with stacked MoE layers.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a deeper architecture with multiple stacked MoE layers, similar to
    /// modern transformer architectures. Each MoE layer can learn different levels of abstraction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Creates a powerful multi-layer MoE architecture for complex tasks.
    ///
    /// Use this when:
    /// - You have a complex problem that needs deep learning
    /// - You want multiple levels of expert specialization
    /// - You have enough data to train a larger model
    ///
    /// <code>
    /// // Create a 3-layer MoE architecture for complex classification
    /// var architecture = MixtureOfExpertsExtensions.CreateDeepMoEArchitecture&lt;float&gt;(
    ///     inputSize: 128,          // 128 input features
    ///     hiddenSize: 256,         // 256-dimensional hidden representations
    ///     outputSize: 10,          // 10 output classes
    ///     numMoELayers: 3,         // 3 MoE layers stacked
    ///     numExperts: 8,           // 8 experts per layer
    ///     topK: 2                  // Use top 2 experts per input per layer
    /// );
    ///
    /// var model = new NeuralNetworkModel&lt;float&gt;(architecture);
    /// </code>
    ///
    /// Architecture structure:
    /// - Input projection: input → hidden
    /// - MoE Layer 1: hidden → hidden (with 8 experts)
    /// - MoE Layer 2: hidden → hidden (with 8 experts)
    /// - ...
    /// - MoE Layer N: hidden → hidden (with 8 experts)
    /// - Output projection: hidden → output
    ///
    /// This is similar to how large language models like GPT use MoE!
    /// </para>
    /// </remarks>
    public static NeuralNetworkArchitecture<T> CreateDeepMoEArchitecture<T>(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numMoELayers = 2,
        int numExperts = 4,
        int topK = 0,
        bool useLoadBalancing = true,
        NeuralNetworkTaskType taskType = NeuralNetworkTaskType.MultiClassClassification)
    {
        if (numMoELayers < 1)
        {
            throw new ArgumentException("Must have at least 1 MoE layer.", nameof(numMoELayers));
        }

        var layers = new List<ILayer<T>>();

        // Input projection if input size doesn't match hidden size
        if (inputSize != hiddenSize)
        {
            layers.Add(new DenseLayer<T>(inputSize, hiddenSize, new ReLUActivation<T>()));
        }

        // Stack MoE layers
        for (int i = 0; i < numMoELayers; i++)
        {
            var moeLayer = new MixtureOfExpertsBuilder<T>()
                .WithExperts(numExperts)
                .WithDimensions(hiddenSize, hiddenSize)
                .WithTopK(topK)
                .WithLoadBalancing(useLoadBalancing, 0.01)
                .Build();

            layers.Add(moeLayer);
        }

        // Output layer
        var outputLayer = taskType switch
        {
            NeuralNetworkTaskType.MultiClassClassification =>
                new DenseLayer<T>(hiddenSize, outputSize, new SoftmaxActivation<T>()),
            NeuralNetworkTaskType.BinaryClassification =>
                new DenseLayer<T>(hiddenSize, 1, new SigmoidActivation<T>()),
            NeuralNetworkTaskType.Regression =>
                new DenseLayer<T>(hiddenSize, outputSize, new IdentityActivation<T>()),
            _ => throw new ArgumentException($"Unsupported task type: {taskType}", nameof(taskType))
        };

        layers.Add(outputLayer);

        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: taskType,
            complexity: NetworkComplexity.Complex,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers
        );
    }

    /// <summary>
    /// Convenience method to create and configure a NeuralNetworkModel with MoE for use with PredictionModelBuilder.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations (typically float or double).</typeparam>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="outputSize">The size of the output.</param>
    /// <param name="numExperts">The number of expert networks (default: 4).</param>
    /// <param name="topK">Number of experts to activate per input (0 = all experts, default: 0).</param>
    /// <param name="useLoadBalancing">Whether to enable load balancing (default: true).</param>
    /// <param name="taskType">The type of task (classification or regression).</param>
    /// <returns>A ready-to-use NeuralNetworkModel with MoE.</returns>
    /// <remarks>
    /// <para>
    /// This is the most convenient method for using MoE with PredictionModelBuilder. It creates both
    /// the architecture and the model in one call.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The quickest way to create an MoE model for PredictionModelBuilder.
    ///
    /// One-line MoE model creation:
    ///
    /// <code>
    /// // Create MoE model in one line
    /// var moeModel = MixtureOfExpertsExtensions.CreateMoEModel&lt;float&gt;(
    ///     inputSize: 10,
    ///     outputSize: 3,
    ///     numExperts: 8,
    ///     topK: 2
    /// );
    ///
    /// // Use immediately with PredictionModelBuilder
    /// var result = new PredictionModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(moeModel)
    ///     .Build(trainingData, trainingLabels);
    /// </code>
    ///
    /// This combines CreateMoEArchitecture() + NeuralNetworkModel creation in one call.
    /// Perfect for quick experimentation or when you want sensible defaults.
    /// </para>
    /// </remarks>
    public static NeuralNetworkModel<T> CreateMoEModel<T>(
        int inputSize,
        int outputSize,
        int numExperts = 4,
        int topK = 0,
        bool useLoadBalancing = true,
        NeuralNetworkTaskType taskType = NeuralNetworkTaskType.MultiClassClassification)
    {
        var architecture = CreateMoEArchitecture<T>(
            inputSize,
            outputSize,
            numExperts,
            topK,
            useLoadBalancing,
            0.01,
            taskType);

        return new NeuralNetworkModel<T>(architecture);
    }

    /// <summary>
    /// Convenience method to create a deep MoE model for use with PredictionModelBuilder.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations (typically float or double).</typeparam>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="hiddenSize">The size of hidden representations.</param>
    /// <param name="outputSize">The size of the output.</param>
    /// <param name="numMoELayers">Number of MoE layers to stack (default: 2).</param>
    /// <param name="numExperts">The number of experts per MoE layer (default: 4).</param>
    /// <param name="topK">Number of experts to activate per input (0 = all experts, default: 0).</param>
    /// <param name="useLoadBalancing">Whether to enable load balancing (default: true).</param>
    /// <param name="taskType">The type of task (classification or regression).</param>
    /// <returns>A ready-to-use deep NeuralNetworkModel with MoE.</returns>
    /// <remarks>
    /// <para>
    /// Combines architecture creation and model wrapping for deep MoE networks.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> One-line creation of powerful multi-layer MoE models.
    ///
    /// <code>
    /// // Create a 3-layer deep MoE model
    /// var deepMoE = MixtureOfExpertsExtensions.CreateDeepMoEModel&lt;float&gt;(
    ///     inputSize: 128,
    ///     hiddenSize: 256,
    ///     outputSize: 10,
    ///     numMoELayers: 3,
    ///     numExperts: 8,
    ///     topK: 2
    /// );
    ///
    /// // Train with PredictionModelBuilder
    /// var result = builder.ConfigureModel(deepMoE).Build(data, labels);
    /// </code>
    ///
    /// This is perfect for complex tasks that need deep learning with MoE specialization.
    /// </para>
    /// </remarks>
    public static NeuralNetworkModel<T> CreateDeepMoEModel<T>(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numMoELayers = 2,
        int numExperts = 4,
        int topK = 0,
        bool useLoadBalancing = true,
        NeuralNetworkTaskType taskType = NeuralNetworkTaskType.MultiClassClassification)
    {
        var architecture = CreateDeepMoEArchitecture<T>(
            inputSize,
            hiddenSize,
            outputSize,
            numMoELayers,
            numExperts,
            topK,
            useLoadBalancing,
            taskType);

        return new NeuralNetworkModel<T>(architecture);
    }
}
