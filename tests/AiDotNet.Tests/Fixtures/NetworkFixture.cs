using AiDotNet.NeuralNetworks;

namespace AiDotNet.Tests.Fixtures;

/// <summary>
/// Shared test fixture for neural network instances.
/// Constructs networks once and reuses them across tests to avoid repeated initialization overhead.
/// </summary>
/// <remarks>
/// Use this fixture with xUnit's IClassFixture pattern:
/// <code>
/// public class MyTests : IClassFixture&lt;NetworkFixture&lt;float&gt;&gt;
/// {
///     private readonly NetworkFixture&lt;float&gt; _fixture;
///
///     public MyTests(NetworkFixture&lt;float&gt; fixture)
///     {
///         _fixture = fixture;
///     }
///
///     [Fact]
///     public void Test_Something()
///     {
///         var network = _fixture.MiniDenseNet;
///         // Use network...
///     }
/// }
/// </code>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NetworkFixture<T> : IDisposable
{
    private readonly object _lock = new();
    private DenseNetNetwork<T>? _miniDenseNet;
    private EfficientNetNetwork<T>? _miniEfficientNet;
    private ResNetNetwork<T>? _miniResNet;
    private bool _disposed;

    /// <summary>
    /// Gets a minimal DenseNet network for testing.
    /// Uses [2,2,2,2] block config with growth rate 8 and 32x32 input.
    /// Thread-safe and lazily initialized.
    /// </summary>
    public DenseNetNetwork<T> MiniDenseNet
    {
        get
        {
            if (_miniDenseNet == null)
            {
                lock (_lock)
                {
                    _miniDenseNet ??= DenseNetNetwork<T>.ForTesting(numClasses: 10);
                }
            }
            return _miniDenseNet;
        }
    }

    /// <summary>
    /// Gets a minimal EfficientNet network for testing.
    /// Uses 32x32 input with 1.0 width/depth multipliers.
    /// Thread-safe and lazily initialized.
    /// </summary>
    public EfficientNetNetwork<T> MiniEfficientNet
    {
        get
        {
            if (_miniEfficientNet == null)
            {
                lock (_lock)
                {
                    _miniEfficientNet ??= EfficientNetNetwork<T>.ForTesting(numClasses: 10);
                }
            }
            return _miniEfficientNet;
        }
    }

    /// <summary>
    /// Gets a minimal ResNet network for testing.
    /// Uses ResNet18 with 32x32 input.
    /// Thread-safe and lazily initialized.
    /// </summary>
    public ResNetNetwork<T> MiniResNet
    {
        get
        {
            if (_miniResNet == null)
            {
                lock (_lock)
                {
                    _miniResNet ??= ResNetNetwork<T>.ForTesting(numClasses: 10);
                }
            }
            return _miniResNet;
        }
    }

    /// <summary>
    /// Disposes of the fixture and its resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the fixture and its resources.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                // Networks don't implement IDisposable, but we clear references
                _miniDenseNet = null;
                _miniEfficientNet = null;
                _miniResNet = null;
            }
            _disposed = true;
        }
    }
}
