namespace AiDotNet.Enums;

/// <summary>
/// Defines the complexity level of a neural network architecture.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Neural networks are AI models inspired by the human brain. They consist of 
/// interconnected "neurons" organized in layers. The complexity of a neural network refers to 
/// how many layers it has and how many neurons are in each layer. More complex networks can 
/// learn more sophisticated patterns but require more data and computing power. This enum helps 
/// you choose an appropriate complexity level for your task without needing to understand all 
/// the technical details.
/// </para>
/// </remarks>
public enum NetworkComplexity
{
    /// <summary>
    /// Simple network with minimal layers, suitable for basic tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Simple network typically has just 1-2 hidden layers with few neurons. 
    /// It's like a small team of people working on a problem. These networks train quickly, 
    /// require less data, and work well for straightforward problems like linear relationships 
    /// or basic classification. Use this when you have limited data or when your problem isn't 
    /// very complex.
    /// </para>
    /// </remarks>
    Simple,

    /// <summary>
    /// Medium complexity network with a moderate number of layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Medium complexity network usually has 3-5 hidden layers with more neurons. 
    /// It's like a medium-sized team with specialists. These networks can handle more complex 
    /// patterns than Simple networks while still training in a reasonable time. They're a good 
    /// balance between power and efficiency for many common problems.
    /// </para>
    /// </remarks>
    Medium,

    /// <summary>
    /// Deep network with many layers, suitable for complex tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Deep network has many layers (typically 6-10) with numerous neurons. 
    /// It's like a large organization with multiple departments working together. These networks 
    /// can learn complex patterns and relationships in data, making them suitable for challenging 
    /// tasks like image recognition or natural language processing. They require more data and 
    /// computing power but can achieve higher accuracy on difficult problems.
    /// </para>
    /// </remarks>
    Deep,

    /// <summary>
    /// Very deep network with extensive layers and connections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A VeryDeep network has a large number of layers (10+ layers) with many neurons 
    /// and connections. It's like a massive organization with highly specialized departments and 
    /// complex communication channels. These networks can solve extremely complex problems and learn 
    /// subtle patterns, but they require substantial data, computing power, and time to train. They 
    /// might be overkill for simpler problems and can be prone to overfitting (memorizing data rather 
    /// than learning patterns) if not properly managed.
    /// </para>
    /// </remarks>
    VeryDeep,

    /// <summary>
    /// Custom complexity defined by the user.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Custom option allows you to define your own network structure rather than 
    /// using the predefined complexity levels. This gives you complete control over how many layers 
    /// your network has and how many neurons are in each layer. This option is useful when you have 
    /// specific requirements or when you're experimenting to find the optimal network structure for 
    /// your particular problem.
    /// </para>
    /// </remarks>
    Custom
}
