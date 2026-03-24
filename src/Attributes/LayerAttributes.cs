using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Attributes;

/// <summary>
/// Declares architectural properties of a neural network layer for cataloging and test generation.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class LayerPropertyAttribute : Attribute
{
    /// <summary>Whether the layer has trainable parameters (weights/biases). Default: true.</summary>
    public bool IsTrainable { get; set; } = true;

    /// <summary>Whether the layer supports backpropagation gradient computation. Default: true.</summary>
    public bool SupportsBackpropagation { get; set; } = true;

    /// <summary>Whether the layer behaves differently during training vs inference (e.g., Dropout, BatchNorm). Default: false.</summary>
    public bool HasTrainingMode { get; set; }

    /// <summary>Expected input rank (number of dimensions). 0 means any rank. Default: 0.</summary>
    public int ExpectedInputRank { get; set; }

    /// <summary>Whether the layer changes the shape of its input. Default: false.</summary>
    public bool ChangesShape { get; set; }

    /// <summary>Whether the layer supports in-place operation (output can alias input). Default: false.</summary>
    public bool SupportsInPlace { get; set; }

    /// <summary>Whether the layer is stateful across forward passes (RNN hidden state, running mean). Default: false.</summary>
    public bool IsStateful { get; set; }

    /// <summary>Whether the layer normalizes input (LayerNorm, BatchNorm, etc.) so uniform-value inputs
    /// produce identical outputs regardless of the actual value. Default: false.</summary>
    public bool NormalizesInput { get; set; }

    /// <summary>Relative computational cost. Default: Medium.</summary>
    public ComputeCost Cost { get; set; } = ComputeCost.Medium;

    /// <summary>
    /// The Forward method signature shape this layer uses.
    /// The test scaffold generator uses this to select the correct test base class.
    /// Default: SingleTensor (standard Forward(Tensor) interface).
    /// </summary>
    public LayerApiShape ApiShape { get; set; } = LayerApiShape.SingleTensor;

    /// <summary>
    /// The input tensor shape to use for auto-generated tests, as a comma-separated string.
    /// Examples: "1,4" for [batch=1, features=4], "1,3,8,8" for [batch=1, channels=3, h=8, w=8].
    /// When empty (default), the generator uses the LayerTestBase default of [1, 4].
    /// This maps to the <c>InputShape</c> override in LayerTestBase.
    /// </summary>
    public string TestInputShape { get; set; } = "";

    /// <summary>
    /// C# code to call on the layer after construction to set up graph data, adjacency
    /// matrices, or other domain-specific initialization required before Forward.
    /// The code is emitted inside a SetupLayer(ILayer&lt;double&gt; layer) override.
    /// The variable 'layer' is the ILayer&lt;double&gt; to configure.
    /// Example: "((DiffusionConvLayer&lt;double&gt;)layer).SetLaplacian(laplacian);"
    /// When empty (default), no setup is emitted (standard layers).
    /// </summary>
    public string TestSetupCode { get; set; } = "";

    /// <summary>
    /// Constructor arguments as a comma-separated string for auto-generated tests.
    /// Examples: "4,8" for DenseLayer(inputSize=4, outputSize=8), "3,8,8" for Conv2D(channels=3, h=8, w=8).
    /// When empty (default), the generator assumes a parameterless or all-default-args constructor.
    /// The generator emits these as integer literal arguments: new LayerType&lt;double&gt;(4, 8).
    /// </summary>
    public string TestConstructorArgs { get; set; } = "";
}

/// <summary>
/// Specifies the architectural category of a neural network layer.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class LayerCategoryAttribute : Attribute
{
    /// <summary>The category.</summary>
    public LayerCategory Category { get; }
    /// <summary>Creates a new LayerCategory attribute.</summary>
    public LayerCategoryAttribute(LayerCategory category) => Category = category;
}

/// <summary>
/// Specifies a task that a neural network layer performs.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class LayerTaskAttribute : Attribute
{
    /// <summary>The task.</summary>
    public LayerTask Task { get; }
    /// <summary>Creates a new LayerTask attribute.</summary>
    public LayerTaskAttribute(LayerTask task) => Task = task;
}
