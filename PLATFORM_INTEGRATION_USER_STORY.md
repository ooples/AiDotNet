# AiDotNet Platform Integration - Comprehensive User Story

## Epic Overview

**Title:** Model Metadata System, License Verification, Model Hub, and Platform API Integration

**Epic Goal:** Transform AiDotNet from a library-only solution into a complete platform ecosystem that enables web-based model creation, deployment, and monetization through a "Lovable for AI Models" experience.

**Business Value:**
- Enable non-technical users to create AI models through natural language
- Monetize pre-trained models through license verification
- Create recurring revenue through hosted API inference
- Build a model marketplace ecosystem
- Lower barrier to entry for ML adoption

**Target Users:**
1. **Platform Users** - Non-technical users creating models via web interface
2. **Library Users** - Developers using AiDotNet library directly
3. **Enterprise Users** - Organizations requiring custom models and SLAs
4. **Model Contributors** - Developers publishing models to the hub

---

## Phase 1: Model Metadata Foundation

### User Story 1.1: Serialization Format with Type Metadata

**As a** library user
**I want** to save my trained models with complete type metadata
**So that** I can load them later without manually specifying generic type parameters

**Acceptance Criteria:**

1. **Model Header Structure**
   - ✅ Header contains format version number (starting at 1)
   - ✅ Header contains model type identifier (e.g., "SuperNet", "AutoMLModel")
   - ✅ Header contains numeric type (Double, Single, Decimal)
   - ✅ Header contains input/output type descriptors (Tensor, Vector, Matrix)
   - ✅ Header contains input/output shape information
   - ✅ Header contains model unique identifier (GUID)
   - ✅ Header contains publisher information
   - ✅ Header contains license requirements
   - ✅ Header contains SHA256 checksum for integrity verification
   - ✅ Header is JSON-serialized for human readability and extensibility

2. **File Format Specification**
   ```
   [4 bytes: Magic number 0x41444E4D "ADNM" - AiDotNet Model]
   [4 bytes: Header length (N)]
   [N bytes: JSON header]
   [Remaining: Binary model data]
   ```

3. **Backward Compatibility**
   - ✅ Can read legacy models without headers (defaults to manual type specification)
   - ✅ Provides clear error messages for unsupported format versions
   - ✅ Includes migration utility to add headers to legacy models

4. **Performance Requirements**
   - ✅ Header read/write adds < 1ms overhead
   - ✅ Total file size increase < 1KB for metadata
   - ✅ Binary data format unchanged (no performance regression)

**Technical Specifications:**

```csharp
/// <summary>
/// Metadata header for serialized AiDotNet models.
/// This header enables automatic model type detection and loading.
/// </summary>
public class ModelMetadataHeader
{
    /// <summary>
    /// Format version for backward compatibility (current: 1)
    /// </summary>
    public int Version { get; set; } = 1;

    /// <summary>
    /// Model type identifier for factory pattern
    /// Examples: "SuperNet", "AutoMLModel", "CustomModel"
    /// </summary>
    public string ModelType { get; set; } = string.Empty;

    /// <summary>
    /// Numeric type used for calculations
    /// Values: "Double", "Single", "Decimal"
    /// </summary>
    public string NumericType { get; set; } = "Double";

    /// <summary>
    /// Input tensor type
    /// Values: "Tensor", "Vector", "Matrix", "Custom"
    /// </summary>
    public string InputType { get; set; } = "Tensor";

    /// <summary>
    /// Output tensor type
    /// Values: "Tensor", "Vector", "Matrix", "Custom"
    /// </summary>
    public string OutputType { get; set; } = "Tensor";

    /// <summary>
    /// Input shape/dimensions [batch, channels, height, width] or [features]
    /// </summary>
    public int[] InputShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Output shape/dimensions [classes] or [features]
    /// </summary>
    public int[] OutputShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Unique model identifier (GUID)
    /// </summary>
    public string ModelId { get; set; } = string.Empty;

    /// <summary>
    /// Publisher/creator identifier
    /// Examples: "OoplesFinance", "CommunityUser123"
    /// </summary>
    public string Publisher { get; set; } = string.Empty;

    /// <summary>
    /// Model name/title
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// Model description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// License type
    /// Values: "MIT", "Apache-2.0", "Commercial", "Evaluation", "Enterprise"
    /// </summary>
    public string LicenseType { get; set; } = "MIT";

    /// <summary>
    /// Whether this model requires license verification
    /// </summary>
    public bool RequiresLicense { get; set; } = false;

    /// <summary>
    /// SHA256 checksum of binary model data for integrity verification
    /// </summary>
    public string ChecksumSHA256 { get; set; } = string.Empty;

    /// <summary>
    /// UTC timestamp when model was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Framework version used to create this model
    /// </summary>
    public string FrameworkVersion { get; set; } = string.Empty;

    /// <summary>
    /// Custom metadata key-value pairs
    /// </summary>
    public Dictionary<string, string> CustomMetadata { get; set; } = new();
}
```

**Implementation Tasks:**

1. **Create ModelMetadataHeader class** (src/Interfaces/ModelMetadataHeader.cs)
   - Define all properties with XML documentation
   - Add JSON serialization attributes
   - Include validation methods

2. **Extend IModelSerializer interface** (src/Interfaces/IModelSerializer.cs)
   - Add `ModelMetadataHeader GetMetadata()` method
   - Add `void SaveModelWithMetadata(string filePath, ModelMetadataHeader header)` method
   - Maintain backward compatibility with existing methods

3. **Implement ModelSerializer helper class** (src/Serialization/ModelSerializer.cs)
   - `WriteHeader(Stream stream, ModelMetadataHeader header)` - Write header to stream
   - `ReadHeader(Stream stream)` - Read and validate header from stream
   - `ValidateChecksum(Stream stream, string expectedChecksum)` - Verify data integrity
   - `DetectLegacyFormat(Stream stream)` - Check if file has header or is legacy format

4. **Update SuperNet<T> serialization** (src/NeuralNetworks/SuperNet.cs)
   - Implement new `SaveModelWithMetadata()` method
   - Populate header with SuperNet-specific metadata
   - Compute and store checksum
   - Update existing `SaveModel()` to call new method with default header

5. **Update AutoMLModelBase<T> serialization** (src/AutoML/AutoMLModelBase.cs)
   - Delegate to BestModel's metadata-aware serialization
   - Preserve AutoML-specific metadata (search results, best config)

6. **Create migration utility** (tools/MigrateModels.cs)
   - Scan directory for legacy models
   - Add headers to models without metadata
   - Preserve original files with .backup extension
   - Generate migration report

7. **Add comprehensive tests** (tests/Serialization/ModelMetadataTests.cs)
   - Test header serialization/deserialization
   - Test checksum verification
   - Test legacy model loading
   - Test format version compatibility
   - Test corrupted file detection

**Testing Strategy:**

```csharp
[Fact]
public void SaveModelWithMetadata_CreatesValidHeader()
{
    // Arrange
    var model = new SuperNet<double>(inputSize: 10, outputSize: 5);
    var tempFile = Path.GetTempFileName();

    // Act
    model.SaveModel(tempFile);

    // Assert
    using var stream = File.OpenRead(tempFile);
    var header = ModelSerializer.ReadHeader(stream);

    Assert.Equal(1, header.Version);
    Assert.Equal("SuperNet", header.ModelType);
    Assert.Equal("Double", header.NumericType);
    Assert.Equal(new[] { 10 }, header.InputShape);
    Assert.Equal(new[] { 5 }, header.OutputShape);
    Assert.NotEmpty(header.ModelId);
    Assert.NotEmpty(header.ChecksumSHA256);
}

[Fact]
public void LoadModel_WithInvalidChecksum_ThrowsSecurityException()
{
    // Arrange
    var tempFile = CreateModelWithTamperedData();

    // Act & Assert
    var ex = Assert.Throws<SecurityException>(() =>
        ModelSerializer.LoadModel<double>(tempFile));
    Assert.Contains("integrity check failed", ex.Message);
}

[Fact]
public void LoadModel_LegacyFormat_LoadsSuccessfully()
{
    // Arrange
    var legacyFile = CreateLegacyModelFile();

    // Act
    var model = new SuperNet<double>();
    model.LoadModel(legacyFile);

    // Assert
    Assert.NotNull(model);
    Assert.Equal(10, model.InputSize);
}
```

**Documentation Requirements:**

1. **Migration Guide** (docs/migration/adding-model-metadata.md)
   - Step-by-step instructions for migrating legacy models
   - Code examples for both library and CLI approaches
   - Troubleshooting common issues

2. **API Documentation** (docs/api/model-serialization.md)
   - Complete API reference for ModelMetadataHeader
   - Examples of custom metadata usage
   - Best practices for model versioning

3. **Format Specification** (docs/specifications/model-file-format.md)
   - Binary format specification
   - Header JSON schema
   - Version compatibility matrix

---

### User Story 1.2: Model Type Registry Pattern

**As a** developer
**I want** to register custom model factories
**So that** the LoadModel endpoint can automatically instantiate my custom model types

**Acceptance Criteria:**

1. **Factory Interface**
   - ✅ IModelFactory interface defines contract for model loading
   - ✅ Factory can validate if it handles a specific model type
   - ✅ Factory can create IServableModel<T> from file path and header
   - ✅ Factory supports async loading for large models

2. **Registry System**
   - ✅ ModelTypeRegistry maintains factory registrations
   - ✅ Registry supports priority-based factory selection
   - ✅ Registry provides factory discovery by model type
   - ✅ Registry logs factory registration and usage

3. **Built-in Factories**
   - ✅ SuperNetFactory for SuperNet models
   - ✅ AutoMLModelFactory for AutoML models
   - ✅ Factories auto-register during startup

4. **Extension Points**
   - ✅ Custom factories can be registered via DI
   - ✅ Clear documentation for implementing custom factories
   - ✅ Sample custom factory in examples

**Technical Specifications:**

```csharp
/// <summary>
/// Factory interface for creating servable models from serialized files.
/// Implement this interface to support custom model types.
/// </summary>
public interface IModelFactory
{
    /// <summary>
    /// Model type identifier this factory handles (e.g., "SuperNet", "CustomModel")
    /// </summary>
    string ModelType { get; }

    /// <summary>
    /// Priority for factory selection when multiple factories support a type.
    /// Higher priority factories are tried first. Default: 0
    /// </summary>
    int Priority { get; }

    /// <summary>
    /// Determines if this factory can handle the given model metadata.
    /// </summary>
    /// <param name="header">Model metadata from file</param>
    /// <returns>True if this factory can load the model</returns>
    bool CanHandle(ModelMetadataHeader header);

    /// <summary>
    /// Loads a model from file and wraps it as IServableModel.
    /// </summary>
    /// <typeparam name="T">Numeric type (double, float, etc.)</typeparam>
    /// <param name="filePath">Absolute path to model file</param>
    /// <param name="header">Pre-read model metadata</param>
    /// <param name="options">Optional loading options</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Servable model ready for inference</returns>
    Task<IServableModel<T>> LoadModelAsync<T>(
        string filePath,
        ModelMetadataHeader header,
        ModelLoadOptions? options = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Options for model loading behavior.
/// </summary>
public class ModelLoadOptions
{
    /// <summary>
    /// Whether to verify model checksum (default: true)
    /// </summary>
    public bool VerifyChecksum { get; set; } = true;

    /// <summary>
    /// Whether to load model in read-only mode (default: true)
    /// </summary>
    public bool ReadOnly { get; set; } = true;

    /// <summary>
    /// Optional license key for premium models
    /// </summary>
    public string? LicenseKey { get; set; }

    /// <summary>
    /// Custom metadata to attach to loaded model
    /// </summary>
    public Dictionary<string, string> CustomMetadata { get; set; } = new();
}

/// <summary>
/// Registry for model factories supporting extensible model loading.
/// </summary>
public interface IModelTypeRegistry
{
    /// <summary>
    /// Registers a model factory.
    /// </summary>
    /// <param name="factory">Factory to register</param>
    void Register(IModelFactory factory);

    /// <summary>
    /// Unregisters a factory by model type.
    /// </summary>
    /// <param name="modelType">Model type identifier</param>
    /// <returns>True if factory was found and removed</returns>
    bool Unregister(string modelType);

    /// <summary>
    /// Gets all factories that can handle the given model type, ordered by priority.
    /// </summary>
    /// <param name="header">Model metadata</param>
    /// <returns>List of compatible factories in priority order</returns>
    IReadOnlyList<IModelFactory> GetFactories(ModelMetadataHeader header);

    /// <summary>
    /// Attempts to load a model using registered factories.
    /// </summary>
    /// <typeparam name="T">Numeric type</typeparam>
    /// <param name="filePath">Path to model file</param>
    /// <param name="options">Loading options</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Loaded servable model</returns>
    /// <exception cref="ModelLoadException">No factory could load the model</exception>
    Task<IServableModel<T>> LoadModelAsync<T>(
        string filePath,
        ModelLoadOptions? options = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Exception thrown when model loading fails.
/// </summary>
public class ModelLoadException : Exception
{
    public string? FilePath { get; init; }
    public ModelMetadataHeader? Header { get; init; }
    public List<Exception> AttemptedFactoryErrors { get; init; } = new();

    public ModelLoadException(string message, string? filePath = null)
        : base(message)
    {
        FilePath = filePath;
    }
}
```

**Implementation Tasks:**

1. **Create IModelFactory interface** (src/Interfaces/IModelFactory.cs)
   - Define interface with XML documentation
   - Include examples in documentation

2. **Implement ModelTypeRegistry** (src/ModelLoading/ModelTypeRegistry.cs)
   - Thread-safe factory registration
   - Priority-based factory ordering
   - Comprehensive error handling and logging
   - Fallback mechanisms for factory failures

3. **Create SuperNetFactory** (src/ModelLoading/Factories/SuperNetFactory.cs)
   ```csharp
   public class SuperNetFactory : IModelFactory
   {
       public string ModelType => "SuperNet";
       public int Priority => 100;

       public bool CanHandle(ModelMetadataHeader header)
       {
           return header.ModelType == "SuperNet" &&
                  header.InputType == "Tensor" &&
                  header.OutputType == "Tensor";
       }

       public async Task<IServableModel<T>> LoadModelAsync<T>(
           string filePath,
           ModelMetadataHeader header,
           ModelLoadOptions? options = null,
           CancellationToken cancellationToken = default)
       {
           options ??= new ModelLoadOptions();

           // Verify checksum if requested
           if (options.VerifyChecksum)
           {
               await VerifyChecksumAsync(filePath, header.ChecksumSHA256, cancellationToken);
           }

           // Load SuperNet model
           var superNet = new SuperNet<T>();
           await Task.Run(() => superNet.LoadModel(filePath), cancellationToken);

           // Wrap as servable
           return new ServableModelWrapper<T>(
               modelName: header.ModelName,
               inputDimension: header.InputShape[0],
               outputDimension: header.OutputShape[0],
               predictFunc: input => superNet.Predict(input),
               predictBatchFunc: inputs => superNet.PredictBatch(inputs)
           );
       }
   }
   ```

4. **Create AutoMLModelFactory** (src/ModelLoading/Factories/AutoMLModelFactory.cs)
   - Similar structure to SuperNetFactory
   - Handle AutoML-specific metadata

5. **Update ModelsController** (src/AiDotNet.Serving/Controllers/ModelsController.cs)
   ```csharp
   [HttpPost]
   public async Task<IActionResult> LoadModel(
       [FromBody] LoadModelRequest request,
       CancellationToken cancellationToken)
   {
       // ... security checks ...

       try
       {
           var options = new ModelLoadOptions
           {
               LicenseKey = request.LicenseKey,
               VerifyChecksum = true
           };

           IServableModel<double> model = request.NumericType?.ToLower() switch
           {
               "double" => await _registry.LoadModelAsync<double>(
                   candidatePath, options, cancellationToken),
               "float" => await _registry.LoadModelAsync<float>(
                   candidatePath, options, cancellationToken),
               _ => throw new NotSupportedException(
                   $"Numeric type '{request.NumericType}' is not supported")
           };

           _modelRepository.LoadModel(request.Name, model, candidatePath);

           return Ok(new LoadModelResponse
           {
               Success = true,
               ModelInfo = _modelRepository.GetModelInfo(request.Name)
           });
       }
       catch (ModelLoadException ex)
       {
           _logger.LogError(ex, "Failed to load model from {Path}", candidatePath);
           return BadRequest(new LoadModelResponse
           {
               Success = false,
               Error = $"Failed to load model: {ex.Message}"
           });
       }
   }
   ```

6. **Add DI registration** (src/AiDotNet.Serving/Program.cs)
   ```csharp
   // Register model type registry
   builder.Services.AddSingleton<IModelTypeRegistry, ModelTypeRegistry>();

   // Auto-register built-in factories
   builder.Services.AddSingleton<IModelFactory, SuperNetFactory>();
   builder.Services.AddSingleton<IModelFactory, AutoMLModelFactory>();

   // Auto-register all factories on startup
   var app = builder.Build();
   var registry = app.Services.GetRequiredService<IModelTypeRegistry>();
   var factories = app.Services.GetServices<IModelFactory>();
   foreach (var factory in factories)
   {
       registry.Register(factory);
   }
   ```

7. **Create example custom factory** (examples/CustomModelFactory/)
   - Complete example showing custom model integration
   - Documentation for third-party model support

8. **Add comprehensive tests** (tests/ModelLoading/ModelTypeRegistryTests.cs)
   - Test factory registration
   - Test priority ordering
   - Test fallback behavior
   - Test error handling
   - Test concurrent loading

**Testing Strategy:**

```csharp
[Fact]
public async Task LoadModelAsync_WithSuperNet_LoadsSuccessfully()
{
    // Arrange
    var registry = new ModelTypeRegistry(NullLogger<ModelTypeRegistry>.Instance);
    registry.Register(new SuperNetFactory());

    var modelFile = CreateSuperNetModelFile();

    // Act
    var model = await registry.LoadModelAsync<double>(modelFile);

    // Assert
    Assert.NotNull(model);
    Assert.Equal(10, model.InputDimension);
}

[Fact]
public async Task LoadModelAsync_NoCompatibleFactory_ThrowsModelLoadException()
{
    // Arrange
    var registry = new ModelTypeRegistry(NullLogger<ModelTypeRegistry>.Instance);
    var unknownModelFile = CreateUnknownModelFile();

    // Act & Assert
    var ex = await Assert.ThrowsAsync<ModelLoadException>(
        () => registry.LoadModelAsync<double>(unknownModelFile));
    Assert.Contains("No factory could load", ex.Message);
}

[Fact]
public void GetFactories_OrdersByPriority()
{
    // Arrange
    var registry = new ModelTypeRegistry(NullLogger<ModelTypeRegistry>.Instance);
    var factory1 = new MockFactory { Priority = 10 };
    var factory2 = new MockFactory { Priority = 100 };

    registry.Register(factory1);
    registry.Register(factory2);

    // Act
    var factories = registry.GetFactories(CreateMockHeader());

    // Assert
    Assert.Equal(factory2, factories[0]); // Higher priority first
    Assert.Equal(factory1, factories[1]);
}
```

**Documentation Requirements:**

1. **Factory Development Guide** (docs/extending/custom-model-factories.md)
   - Step-by-step guide for implementing IModelFactory
   - Best practices for error handling
   - Performance considerations

2. **Registry API Documentation** (docs/api/model-type-registry.md)
   - Complete API reference
   - Usage examples
   - Configuration options

---

## Phase 2: License Verification System

### User Story 2.1: License Key Validation Service

**As a** platform owner
**I want** to verify license keys before allowing premium model access
**So that** I can monetize pre-trained models and prevent unauthorized usage

**Acceptance Criteria:**

1. **License Service Interface**
   - ✅ ILicenseVerificationService defines verification contract
   - ✅ Supports both online and offline verification
   - ✅ Returns detailed license information (tier, expiration, limits)
   - ✅ Includes rate limiting to prevent abuse

2. **License Key Format**
   - ✅ Cryptographically signed license keys
   - ✅ Embedded license metadata (model ID, tier, expiration)
   - ✅ Cannot be forged or modified
   - ✅ Human-readable format (e.g., ADNET-XXXX-XXXX-XXXX-XXXX)

3. **Verification Modes**
   - ✅ Online verification via API call to license server
   - ✅ Offline verification using embedded signature
   - ✅ Cached verification results (configurable TTL)
   - ✅ Graceful degradation when license server unavailable

4. **Security Requirements**
   - ✅ License keys encrypted in transit (HTTPS)
   - ✅ License verification failures logged
   - ✅ Rate limiting prevents brute-force attacks
   - ✅ License key not stored in model files or logs

**Technical Specifications:**

```csharp
/// <summary>
/// Service for verifying model license keys.
/// </summary>
public interface ILicenseVerificationService
{
    /// <summary>
    /// Verifies a license key for a specific model.
    /// </summary>
    /// <param name="licenseKey">License key to verify</param>
    /// <param name="modelId">Model unique identifier</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>License verification result</returns>
    Task<LicenseVerificationResult> VerifyLicenseAsync(
        string licenseKey,
        string modelId,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if a license is cached and still valid.
    /// </summary>
    /// <param name="licenseKey">License key</param>
    /// <param name="modelId">Model identifier</param>
    /// <returns>Cached license info if available and valid, null otherwise</returns>
    LicenseInfo? GetCachedLicense(string licenseKey, string modelId);
}

/// <summary>
/// Result of license verification.
/// </summary>
public class LicenseVerificationResult
{
    /// <summary>
    /// Whether the license is valid
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// License information if valid
    /// </summary>
    public LicenseInfo? License { get; set; }

    /// <summary>
    /// Error message if invalid
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Whether verification was performed online or offline
    /// </summary>
    public VerificationMode Mode { get; set; }
}

/// <summary>
/// License information.
/// </summary>
public class LicenseInfo
{
    /// <summary>
    /// License key (masked for security)
    /// </summary>
    public string LicenseKey { get; set; } = string.Empty;

    /// <summary>
    /// Model ID this license is valid for
    /// </summary>
    public string ModelId { get; set; } = string.Empty;

    /// <summary>
    /// License tier
    /// </summary>
    public LicenseTier Tier { get; set; }

    /// <summary>
    /// UTC expiration date (null for perpetual licenses)
    /// </summary>
    public DateTime? ExpiresAt { get; set; }

    /// <summary>
    /// Licensed to (organization or user)
    /// </summary>
    public string LicensedTo { get; set; } = string.Empty;

    /// <summary>
    /// Maximum inferences per month (null for unlimited)
    /// </summary>
    public long? MonthlyInferenceLimit { get; set; }

    /// <summary>
    /// Current inference count this month
    /// </summary>
    public long CurrentMonthInferences { get; set; }

    /// <summary>
    /// License metadata
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// License tier levels.
/// </summary>
public enum LicenseTier
{
    Free = 0,
    Standard = 1,
    Professional = 2,
    Enterprise = 3
}

/// <summary>
/// Verification mode used.
/// </summary>
public enum VerificationMode
{
    Online,
    Offline,
    Cached
}

/// <summary>
/// Configuration for license verification service.
/// </summary>
public class LicenseVerificationOptions
{
    /// <summary>
    /// License server URL (for online verification)
    /// </summary>
    public string LicenseServerUrl { get; set; } = "https://licenses.aidotnet.com/api/verify";

    /// <summary>
    /// Public key for offline verification (RSA or ECDSA)
    /// </summary>
    public string? PublicKey { get; set; }

    /// <summary>
    /// Cache duration for verified licenses (default: 1 hour)
    /// </summary>
    public TimeSpan CacheDuration { get; set; } = TimeSpan.FromHours(1);

    /// <summary>
    /// Whether to allow offline verification
    /// </summary>
    public bool AllowOfflineVerification { get; set; } = true;

    /// <summary>
    /// Timeout for online verification requests (default: 5 seconds)
    /// </summary>
    public TimeSpan OnlineTimeout { get; set; } = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Maximum verification attempts per hour per IP (rate limiting)
    /// </summary>
    public int MaxVerificationsPerHour { get; set; } = 100;
}
```

**Implementation Tasks:**

1. **Create ILicenseVerificationService interface** (src/Interfaces/ILicenseVerificationService.cs)

2. **Implement LicenseVerificationService** (src/Licensing/LicenseVerificationService.cs)
   - Online verification via HTTPS POST to license server
   - Offline verification using RSA/ECDSA signature validation
   - In-memory cache with configurable TTL
   - Rate limiting using sliding window algorithm
   - Comprehensive logging and telemetry

3. **Create license key generation utility** (tools/GenerateLicenseKey/)
   - Command-line tool for generating license keys
   - Signs license data with private key
   - Outputs human-readable license key format
   - Admin tool only (not distributed)

4. **Implement license server API** (services/LicenseServer/)
   - ASP.NET Core API for license verification
   - PostgreSQL database for license storage
   - API endpoints:
     - POST /api/verify - Verify license key
     - POST /api/usage - Report inference usage
     - GET /api/licenses/{key}/info - Get license details
   - Authentication via API keys
   - Rate limiting and DDoS protection

5. **Update ModelFactory to check licenses** (src/ModelLoading/Factories/LicensedModelFactory.cs)
   ```csharp
   public class LicensedModelFactory : IModelFactory
   {
       private readonly ILicenseVerificationService _licenseService;

       public async Task<IServableModel<T>> LoadModelAsync<T>(
           string filePath,
           ModelMetadataHeader header,
           ModelLoadOptions? options = null,
           CancellationToken cancellationToken = default)
       {
           // Check if model requires license
           if (header.RequiresLicense)
           {
               if (string.IsNullOrEmpty(options?.LicenseKey))
               {
                   throw new LicenseRequiredException(
                       $"Model '{header.ModelName}' requires a valid license. " +
                       $"Visit https://aidotnet.com/licenses to purchase.");
               }

               // Verify license
               var verificationResult = await _licenseService.VerifyLicenseAsync(
                   options.LicenseKey,
                   header.ModelId,
                   cancellationToken);

               if (!verificationResult.IsValid)
               {
                   throw new InvalidLicenseException(
                       $"License verification failed: {verificationResult.ErrorMessage}");
               }

               // Check usage limits
               if (verificationResult.License!.MonthlyInferenceLimit.HasValue)
               {
                   var remaining = verificationResult.License.MonthlyInferenceLimit.Value -
                                   verificationResult.License.CurrentMonthInferences;

                   if (remaining <= 0)
                   {
                       throw new LicenseLimitExceededException(
                           $"Monthly inference limit exceeded. Upgrade your license at " +
                           $"https://aidotnet.com/upgrade");
                   }
               }
           }

           // Delegate to base factory
           return await _baseFactory.LoadModelAsync(filePath, header, options, cancellationToken);
       }
   }
   ```

6. **Add license exceptions** (src/Exceptions/LicenseExceptions.cs)
   - LicenseRequiredException
   - InvalidLicenseException
   - LicenseExpiredException
   - LicenseLimitExceededException

7. **Add comprehensive tests** (tests/Licensing/LicenseVerificationTests.cs)
   - Test online verification success/failure
   - Test offline verification with valid/invalid signatures
   - Test cache behavior
   - Test rate limiting
   - Test license expiration
   - Test usage limits

**Testing Strategy:**

```csharp
[Fact]
public async Task VerifyLicenseAsync_ValidLicense_ReturnsSuccess()
{
    // Arrange
    var service = CreateLicenseService();
    var licenseKey = "ADNET-1234-5678-9ABC-DEF0";
    var modelId = "model-12345";

    // Act
    var result = await service.VerifyLicenseAsync(licenseKey, modelId);

    // Assert
    Assert.True(result.IsValid);
    Assert.NotNull(result.License);
    Assert.Equal(LicenseTier.Professional, result.License.Tier);
}

[Fact]
public async Task VerifyLicenseAsync_ExpiredLicense_ReturnsFalse()
{
    // Arrange
    var service = CreateLicenseService();
    var expiredLicenseKey = CreateExpiredLicense();

    // Act
    var result = await service.VerifyLicenseAsync(expiredLicenseKey, "model-123");

    // Assert
    Assert.False(result.IsValid);
    Assert.Contains("expired", result.ErrorMessage, StringComparison.OrdinalIgnoreCase);
}

[Fact]
public async Task LoadModel_WithoutRequiredLicense_ThrowsException()
{
    // Arrange
    var factory = CreateLicensedFactory();
    var premiumModelFile = CreatePremiumModelFile();
    var options = new ModelLoadOptions(); // No license key

    // Act & Assert
    await Assert.ThrowsAsync<LicenseRequiredException>(
        () => factory.LoadModelAsync<double>(premiumModelFile, options));
}
```

**Documentation Requirements:**

1. **License System Overview** (docs/licensing/overview.md)
   - How licensing works
   - License tiers and limits
   - Purchasing and activation

2. **Integration Guide** (docs/licensing/integration.md)
   - How to use license keys in applications
   - Handling license errors
   - Best practices for key storage

---

## Phase 3: Model Hub Integration

### User Story 3.1: Model Hub Client

**As a** platform user
**I want** to download pre-trained models from the AiDotNet Model Hub
**So that** I can use state-of-the-art models without training from scratch

**Acceptance Criteria:**

1. **Hub Client Interface**
   - ✅ IModelHubClient defines hub interaction contract
   - ✅ Supports model search by category, task, size
   - ✅ Downloads models with progress tracking
   - ✅ Validates model integrity after download

2. **Hub API Integration**
   - ✅ REST API client for hub.aidotnet.com
   - ✅ Authentication via API keys
   - ✅ Supports both free and premium models
   - ✅ Handles rate limiting and retries

3. **Model Discovery**
   - ✅ Browse models by category (vision, NLP, audio)
   - ✅ Filter by license type (free, commercial)
   - ✅ Search by task (classification, detection, generation)
   - ✅ Sort by popularity, rating, size

4. **Download Management**
   - ✅ Resume interrupted downloads
   - ✅ Verify checksums after download
   - ✅ Cache downloaded models locally
   - ✅ Update models when new versions available

**Technical Specifications:**

```csharp
/// <summary>
/// Client for interacting with AiDotNet Model Hub.
/// </summary>
public interface IModelHubClient
{
    /// <summary>
    /// Searches for models in the hub.
    /// </summary>
    /// <param name="query">Search query</param>
    /// <param name="filter">Optional filter criteria</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of matching models</returns>
    Task<ModelSearchResult> SearchModelsAsync(
        string? query = null,
        ModelSearchFilter? filter = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets detailed information about a specific model.
    /// </summary>
    /// <param name="modelId">Model identifier</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Model details</returns>
    Task<HubModelInfo> GetModelInfoAsync(
        string modelId,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Downloads a model from the hub.
    /// </summary>
    /// <param name="modelId">Model to download</param>
    /// <param name="destinationPath">Where to save the model</param>
    /// <param name="options">Download options</param>
    /// <param name="progress">Progress reporter</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to downloaded model</returns>
    Task<string> DownloadModelAsync(
        string modelId,
        string destinationPath,
        DownloadOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if a newer version of a model is available.
    /// </summary>
    /// <param name="modelId">Model identifier</param>
    /// <param name="currentVersion">Currently installed version</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Update info if available, null otherwise</returns>
    Task<ModelUpdateInfo?> CheckForUpdatesAsync(
        string modelId,
        string currentVersion,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Model search filter criteria.
/// </summary>
public class ModelSearchFilter
{
    /// <summary>
    /// Filter by category (e.g., "vision", "nlp", "audio")
    /// </summary>
    public string? Category { get; set; }

    /// <summary>
    /// Filter by task (e.g., "classification", "detection", "generation")
    /// </summary>
    public string? Task { get; set; }

    /// <summary>
    /// Filter by license type
    /// </summary>
    public LicenseTier? LicenseTier { get; set; }

    /// <summary>
    /// Maximum model size in MB
    /// </summary>
    public long? MaxSizeMB { get; set; }

    /// <summary>
    /// Minimum accuracy/performance score (0-100)
    /// </summary>
    public int? MinScore { get; set; }

    /// <summary>
    /// Sort order
    /// </summary>
    public ModelSortOrder SortBy { get; set; } = ModelSortOrder.Popularity;
}

/// <summary>
/// Model information from hub.
/// </summary>
public class HubModelInfo
{
    public string ModelId { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Category { get; set; } = string.Empty;
    public string Task { get; set; } = string.Empty;
    public string Publisher { get; set; } = string.Empty;
    public LicenseTier LicenseTier { get; set; }
    public string Version { get; set; } = string.Empty;
    public long SizeBytes { get; set; }
    public DateTime PublishedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
    public int Downloads { get; set; }
    public double Rating { get; set; }
    public string[] Tags { get; set; } = Array.Empty<string>();
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// Download progress information.
/// </summary>
public class DownloadProgress
{
    public long BytesDownloaded { get; set; }
    public long TotalBytes { get; set; }
    public double PercentComplete => TotalBytes > 0 ? (BytesDownloaded * 100.0 / TotalBytes) : 0;
    public TimeSpan ElapsedTime { get; set; }
    public TimeSpan? EstimatedTimeRemaining { get; set; }
    public double SpeedBytesPerSecond { get; set; }
}

/// <summary>
/// Options for model download.
/// </summary>
public class DownloadOptions
{
    /// <summary>
    /// License key for premium models
    /// </summary>
    public string? LicenseKey { get; set; }

    /// <summary>
    /// Whether to overwrite existing file
    /// </summary>
    public bool Overwrite { get; set; } = false;

    /// <summary>
    /// Whether to verify checksum after download
    /// </summary>
    public bool VerifyChecksum { get; set; } = true;

    /// <summary>
    /// Timeout for download operation
    /// </summary>
    public TimeSpan Timeout { get; set; } = TimeSpan.FromMinutes(30);
}
```

**Implementation Tasks:**

1. **Create IModelHubClient interface** (src/Interfaces/IModelHubClient.cs)

2. **Implement ModelHubClient** (src/ModelHub/ModelHubClient.cs)
   - HttpClient-based REST API integration
   - Authentication header injection
   - Retry logic with exponential backoff
   - Download resumption support
   - Progress tracking
   - Checksum verification

3. **Implement Hub API endpoints** (services/ModelHub/)
   - ASP.NET Core API for model hub
   - Endpoints:
     - GET /api/models/search
     - GET /api/models/{id}
     - GET /api/models/{id}/download
     - POST /api/models/{id}/report-download
   - CDN integration for model downloads
   - Analytics tracking

4. **Update ModelsController to support hub downloads** (src/AiDotNet.Serving/Controllers/ModelsController.cs)
   ```csharp
   [HttpPost("load-from-hub")]
   public async Task<IActionResult> LoadModelFromHub(
       [FromBody] LoadFromHubRequest request,
       CancellationToken cancellationToken)
   {
       try
       {
           // Download from hub
           var downloadPath = Path.Combine(_servingOptions.ModelDirectory, $"{request.ModelId}.bin");

           var downloadOptions = new DownloadOptions
           {
               LicenseKey = request.LicenseKey,
               VerifyChecksum = true
           };

           var progress = new Progress<DownloadProgress>(p =>
           {
               _logger.LogInformation("Download progress: {Percent}% ({Downloaded}/{Total} bytes)",
                   p.PercentComplete, p.BytesDownloaded, p.TotalBytes);
           });

           var modelPath = await _hubClient.DownloadModelAsync(
               request.ModelId,
               downloadPath,
               downloadOptions,
               progress,
               cancellationToken);

           // Load the downloaded model
           var loadOptions = new ModelLoadOptions
           {
               LicenseKey = request.LicenseKey
           };

           var model = await _registry.LoadModelAsync<double>(modelPath, loadOptions, cancellationToken);

           _modelRepository.LoadModel(request.Name, model, modelPath);

           return Ok(new LoadModelResponse
           {
               Success = true,
               ModelInfo = _modelRepository.GetModelInfo(request.Name)
           });
       }
       catch (Exception ex)
       {
           _logger.LogError(ex, "Failed to load model from hub: {ModelId}", request.ModelId);
           return BadRequest(new LoadModelResponse
           {
               Success = false,
               Error = ex.Message
           });
       }
   }
   ```

5. **Add CLI tool for model hub** (tools/ModelHubCLI/)
   - Command-line interface for browsing/downloading models
   - Commands:
     - `aidotnet-hub search <query>`
     - `aidotnet-hub download <model-id>`
     - `aidotnet-hub info <model-id>`
     - `aidotnet-hub update <model-id>`

6. **Add comprehensive tests** (tests/ModelHub/ModelHubClientTests.cs)
   - Test model search
   - Test download with progress
   - Test resume interrupted downloads
   - Test checksum verification
   - Test authentication

**Documentation Requirements:**

1. **Model Hub Guide** (docs/model-hub/overview.md)
   - How to browse and search models
   - Downloading models programmatically
   - Using CLI tool

2. **Publishing Models** (docs/model-hub/publishing.md)
   - How to publish models to hub
   - Licensing options
   - Versioning and updates

---

## Phase 4: Platform API for Model Creation

### User Story 4.1: Web-Based Model Creation API

**As a** platform user
**I want** to create AI models via natural language through a web interface
**So that** I can build models without coding or ML expertise

**Acceptance Criteria:**

1. **Model Creation Endpoint**
   - ✅ POST /api/platform/models/create accepts natural language description
   - ✅ Returns job ID for async training
   - ✅ Validates input requirements (dataset, task type)
   - ✅ Estimates training time and cost

2. **Training Job Management**
   - ✅ GET /api/platform/jobs/{id}/status returns training progress
   - ✅ POST /api/platform/jobs/{id}/cancel cancels training
   - ✅ WebSocket for real-time progress updates
   - ✅ Automatic cleanup of failed/cancelled jobs

3. **Model Deployment**
   - ✅ POST /api/platform/models/{id}/deploy creates serving endpoint
   - ✅ Returns unique API endpoint URL
   - ✅ Configures autoscaling based on tier
   - ✅ Tracks usage metrics

4. **Inference API**
   - ✅ POST /api/inference/{user}/{model}/predict handles predictions
   - ✅ Applies rate limiting based on tier
   - ✅ Tracks usage for billing
   - ✅ Returns predictions with confidence scores

**Technical Specifications:**

```csharp
/// <summary>
/// API for platform model creation and management.
/// </summary>
[ApiController]
[Route("api/platform")]
public class PlatformController : ControllerBase
{
    private readonly IModelCreationService _modelCreation;
    private readonly ITrainingJobService _trainingJobs;
    private readonly IDeploymentService _deployment;

    /// <summary>
    /// Creates a new model from natural language description.
    /// </summary>
    [HttpPost("models/create")]
    [Authorize]
    public async Task<IActionResult> CreateModel(
        [FromBody] CreateModelRequest request,
        CancellationToken cancellationToken)
    {
        // Validate user tier and limits
        var user = await GetCurrentUserAsync();
        if (!await CheckUserLimitsAsync(user))
        {
            return StatusCode(429, new ErrorResponse
            {
                Error = "Monthly model creation limit reached. Upgrade your plan."
            });
        }

        // Parse natural language description
        var modelConfig = await _modelCreation.ParseDescriptionAsync(
            request.Description,
            request.TaskType,
            cancellationToken);

        // Validate dataset
        if (!await ValidateDatasetAsync(request.DatasetId))
        {
            return BadRequest(new ErrorResponse
            {
                Error = "Invalid or inaccessible dataset"
            });
        }

        // Estimate cost and time
        var estimate = await _modelCreation.EstimateTrainingAsync(
            modelConfig,
            request.DatasetId,
            cancellationToken);

        // Create training job
        var job = await _trainingJobs.CreateJobAsync(
            userId: user.Id,
            modelConfig: modelConfig,
            datasetId: request.DatasetId,
            tier: user.Tier,
            cancellationToken);

        return Accepted(new CreateModelResponse
        {
            JobId = job.Id,
            EstimatedDuration = estimate.Duration,
            EstimatedCost = estimate.Cost,
            StatusUrl = $"/api/platform/jobs/{job.Id}/status"
        });
    }

    /// <summary>
    /// Gets training job status.
    /// </summary>
    [HttpGet("jobs/{jobId}/status")]
    [Authorize]
    public async Task<IActionResult> GetJobStatus(
        string jobId,
        CancellationToken cancellationToken)
    {
        var job = await _trainingJobs.GetJobAsync(jobId, cancellationToken);

        if (job == null)
        {
            return NotFound();
        }

        // Verify ownership
        var user = await GetCurrentUserAsync();
        if (job.UserId != user.Id)
        {
            return Forbid();
        }

        return Ok(new JobStatusResponse
        {
            JobId = job.Id,
            Status = job.Status,
            Progress = job.Progress,
            CurrentEpoch = job.CurrentEpoch,
            TotalEpochs = job.TotalEpochs,
            TrainingMetrics = job.Metrics,
            CreatedAt = job.CreatedAt,
            StartedAt = job.StartedAt,
            CompletedAt = job.CompletedAt,
            ErrorMessage = job.ErrorMessage
        });
    }

    /// <summary>
    /// Deploys a trained model.
    /// </summary>
    [HttpPost("models/{modelId}/deploy")]
    [Authorize]
    public async Task<IActionResult> DeployModel(
        string modelId,
        [FromBody] DeployModelRequest request,
        CancellationToken cancellationToken)
    {
        var user = await GetCurrentUserAsync();
        var model = await GetModelAsync(modelId);

        if (model == null || model.UserId != user.Id)
        {
            return NotFound();
        }

        if (model.Status != ModelStatus.Trained)
        {
            return BadRequest(new ErrorResponse
            {
                Error = "Model must be fully trained before deployment"
            });
        }

        // Create deployment
        var deployment = await _deployment.DeployModelAsync(
            model,
            user.Tier,
            request.Environment,
            cancellationToken);

        return Ok(new DeployModelResponse
        {
            DeploymentId = deployment.Id,
            EndpointUrl = deployment.EndpointUrl,
            ApiKey = deployment.ApiKey, // Newly generated API key
            Environment = deployment.Environment,
            Status = deployment.Status
        });
    }
}

/// <summary>
/// Request to create a new model.
/// </summary>
public class CreateModelRequest
{
    /// <summary>
    /// Natural language description of desired model
    /// Example: "Classify customer support tickets into urgent/normal/low priority"
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Task type hint
    /// Values: "classification", "regression", "detection", "generation"
    /// </summary>
    public string TaskType { get; set; } = string.Empty;

    /// <summary>
    /// Dataset identifier
    /// </summary>
    public string DatasetId { get; set; } = string.Empty;

    /// <summary>
    /// Training preferences
    /// </summary>
    public TrainingPreferences? Preferences { get; set; }
}

/// <summary>
/// Training preferences for model creation.
/// </summary>
public class TrainingPreferences
{
    /// <summary>
    /// Maximum training time in minutes (null for unlimited)
    /// </summary>
    public int? MaxTrainingMinutes { get; set; }

    /// <summary>
    /// Maximum cost in USD (null for unlimited)
    /// </summary>
    public decimal? MaxCostUSD { get; set; }

    /// <summary>
    /// Target accuracy (0-1, null for best effort)
    /// </summary>
    public double? TargetAccuracy { get; set; }

    /// <summary>
    /// Optimization goal: "accuracy", "speed", "size", "balanced"
    /// </summary>
    public string OptimizationGoal { get; set; } = "balanced";
}

/// <summary>
/// Training job status.
/// </summary>
public enum JobStatus
{
    Pending,
    Preprocessing,
    Training,
    Validating,
    Completed,
    Failed,
    Cancelled
}
```

**Implementation Tasks:**

1. **Create Platform API project** (src/AiDotNet.Platform/)
   - ASP.NET Core Web API
   - Authentication with JWT tokens
   - Authorization with role-based access
   - Rate limiting middleware

2. **Implement Model Creation Service** (src/AiDotNet.Platform/Services/ModelCreationService.cs)
   - NLP parser for model descriptions
   - Model architecture selection
   - Hyperparameter recommendation
   - Training configuration generation

3. **Implement Training Job Service** (src/AiDotNet.Platform/Services/TrainingJobService.cs)
   - Job queue management (using Hangfire or similar)
   - Distributed training coordination
   - Progress tracking and metrics collection
   - Failure recovery and retry logic

4. **Implement Deployment Service** (src/AiDotNet.Platform/Services/DeploymentService.cs)
   - Container orchestration (Kubernetes)
   - Load balancer configuration
   - Auto-scaling setup
   - Health monitoring

5. **Create WebSocket hub for real-time updates** (src/AiDotNet.Platform/Hubs/TrainingHub.cs)
   ```csharp
   public class TrainingHub : Hub
   {
       public async Task SubscribeToJob(string jobId)
       {
           await Groups.AddToGroupAsync(Context.ConnectionId, $"job-{jobId}");
       }

       public async Task UnsubscribeFromJob(string jobId)
       {
           await Groups.RemoveFromGroupAsync(Context.ConnectionId, $"job-{jobId}");
       }
   }
   ```

6. **Implement Inference API** (src/AiDotNet.Platform/Controllers/InferenceController.cs)
   - Multi-tenant routing
   - Usage tracking
   - Rate limiting per tier
   - Response caching

7. **Add comprehensive tests**
   - Integration tests for model creation flow
   - Load tests for inference API
   - Security tests for authorization

**Documentation Requirements:**

1. **Platform API Documentation** (docs/platform/api-reference.md)
   - Complete API reference with examples
   - Authentication guide
   - Rate limits and quotas

2. **Getting Started Guide** (docs/platform/getting-started.md)
   - Step-by-step tutorial
   - Example model creations
   - Best practices

---

## Cross-Cutting Concerns

### Security Requirements

1. **Authentication & Authorization**
   - JWT-based authentication for Platform API
   - API key authentication for Inference API
   - Role-based access control (RBAC)
   - Multi-factor authentication for premium tiers

2. **Data Protection**
   - Encryption at rest for models and datasets
   - Encryption in transit (TLS 1.3)
   - Secure key management (Azure Key Vault, AWS KMS)
   - GDPR compliance for user data

3. **API Security**
   - Rate limiting per tier
   - DDoS protection (Cloudflare, AWS Shield)
   - Input validation and sanitization
   - SQL injection prevention
   - XSS prevention

### Performance Requirements

1. **Latency**
   - Model metadata read: < 1ms
   - Model loading: < 5 seconds for typical models
   - License verification: < 100ms (online), < 1ms (cached)
   - Inference API: < 100ms for small models

2. **Throughput**
   - Support 10,000+ concurrent model loads
   - Handle 1M+ inferences per second (distributed)
   - Process 100+ training jobs concurrently

3. **Scalability**
   - Horizontal scaling for inference API
   - Distributed training across multiple GPUs/nodes
   - Auto-scaling based on load

### Monitoring & Observability

1. **Metrics**
   - Model load success/failure rates
   - License verification rates
   - Training job duration
   - Inference latency (p50, p95, p99)
   - API error rates

2. **Logging**
   - Structured logging (JSON format)
   - Correlation IDs for request tracing
   - Centralized log aggregation (ELK, Splunk)
   - Security event logging

3. **Alerting**
   - License verification failures
   - Model load failures
   - Training job failures
   - API error rate spikes
   - Resource exhaustion

### Testing Strategy

1. **Unit Tests**
   - 80%+ code coverage
   - Test all business logic
   - Mock external dependencies

2. **Integration Tests**
   - End-to-end model creation flow
   - License verification with test server
   - Model hub download

3. **Performance Tests**
   - Load testing inference API
   - Stress testing model loading
   - Benchmark training performance

4. **Security Tests**
   - Penetration testing
   - License bypass attempts
   - API authentication tests

---

## Migration & Rollout Plan

### Phase 1: Foundation (Weeks 1-2)
- Implement model metadata system
- Add backward compatibility for legacy models
- Create migration utility

### Phase 2: Registry (Weeks 3-4)
- Implement model type registry
- Create SuperNet and AutoML factories
- Update LoadModel endpoint

### Phase 3: Licensing (Weeks 5-6)
- Implement license verification service
- Create license server
- Integrate with factories

### Phase 4: Model Hub (Weeks 7-8)
- Implement hub client
- Create hub API
- Deploy hub infrastructure

### Phase 5: Platform API (Weeks 9-12)
- Implement model creation service
- Build training job management
- Create deployment service
- Launch web interface

### Phase 6: Production Hardening (Weeks 13-14)
- Performance optimization
- Security audit
- Load testing
- Documentation finalization

---

## Success Metrics

1. **Adoption Metrics**
   - Number of models created via platform
   - Number of active API users
   - Model hub downloads per month

2. **Revenue Metrics**
   - Monthly recurring revenue (MRR)
   - Average revenue per user (ARPU)
   - License conversion rate

3. **Technical Metrics**
   - API uptime (target: 99.9%)
   - Average inference latency
   - Training job success rate

4. **User Experience Metrics**
   - Time from description to deployed model
   - User satisfaction score
   - Support ticket volume

---

## Dependencies

1. **Infrastructure**
   - Kubernetes cluster for serving
   - PostgreSQL for license/user data
   - Redis for caching
   - CDN for model distribution
   - GPU nodes for training

2. **Third-Party Services**
   - Stripe for billing
   - SendGrid for emails
   - Auth0 for authentication (optional)
   - Cloudflare for DDoS protection

3. **Internal Dependencies**
   - AiDotNet core library
   - AiDotNet.Serving framework
   - Existing serialization code

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| License server downtime | High | Medium | Implement offline verification, caching |
| Training job failures | High | Medium | Retry logic, checkpointing, user notifications |
| API abuse | Medium | High | Rate limiting, DDoS protection, monitoring |
| Model IP theft | High | Low | Encryption, checksums, legal terms |
| Backward compatibility breaks | High | Low | Thorough testing, migration utilities |

---

## Open Questions

1. **Pricing Strategy**
   - What should be the pricing for each tier?
   - How to handle compute costs for training?
   - Free tier limits?

2. **Model Marketplace**
   - Should third-party developers publish models?
   - Revenue sharing model?
   - Quality control process?

3. **Enterprise Features**
   - On-premise deployment requirements?
   - Custom SLAs?
   - White-label options?

4. **International Expansion**
   - Multi-region deployment strategy?
   - Data residency requirements?
   - Currency support?

---

## Appendices

### Appendix A: API Examples

**Creating a Model:**
```bash
curl -X POST https://api.aidotnet.com/platform/models/create \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Classify customer feedback as positive, negative, or neutral",
    "taskType": "classification",
    "datasetId": "dataset-12345",
    "preferences": {
      "maxTrainingMinutes": 60,
      "optimizationGoal": "accuracy"
    }
  }'
```

**Loading a Model:**
```bash
curl -X POST https://api.aidotnet.com/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-sentiment-model",
    "path": "models/sentiment.bin",
    "numericType": "double",
    "licenseKey": "ADNET-1234-5678-9ABC-DEF0"
  }'
```

**Making Predictions:**
```bash
curl -X POST https://api.aidotnet.com/inference/myuser/sentiment-model/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0]],
    "requestId": "req-12345"
  }'
```

### Appendix B: Database Schema

**License Table:**
```sql
CREATE TABLE licenses (
    id UUID PRIMARY KEY,
    license_key VARCHAR(100) UNIQUE NOT NULL,
    model_id UUID NOT NULL,
    user_id UUID NOT NULL,
    tier VARCHAR(50) NOT NULL,
    expires_at TIMESTAMP,
    monthly_inference_limit BIGINT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_licenses_key ON licenses(license_key);
CREATE INDEX idx_licenses_model ON licenses(model_id);
CREATE INDEX idx_licenses_user ON licenses(user_id);
```

**Training Jobs Table:**
```sql
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    model_config JSONB NOT NULL,
    dataset_id UUID NOT NULL,
    status VARCHAR(50) NOT NULL,
    progress DECIMAL(5,2),
    current_epoch INT,
    total_epochs INT,
    metrics JSONB,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_jobs_user ON training_jobs(user_id);
CREATE INDEX idx_jobs_status ON training_jobs(status);
```

### Appendix C: License Key Format

**Structure:**
```
ADNET-XXXX-XXXX-XXXX-XXXX
  |     |    |    |    |
  |     |    |    |    +-- Checksum (4 chars)
  |     |    |    +------- License data (4 chars)
  |     |    +------------ License data (4 chars)
  |     +----------------- License data (4 chars)
  +----------------------- Prefix
```

**Encoding:**
- Base32 encoding (exclude 0, O, I, L for clarity)
- Embedded: model ID hash, tier, expiration date, signature
- Total: 29 characters (including hyphens)

### Appendix D: Model File Format Specification

**Binary Layout:**
```
Offset  | Size    | Description
--------|---------|------------------------------------------
0x0000  | 4 bytes | Magic number: 0x41444E4D ("ADNM")
0x0004  | 4 bytes | Header length N (little-endian uint32)
0x0008  | N bytes | JSON header (UTF-8 encoded)
0x0008+N| Rest    | Binary model data (format varies by type)
```

**JSON Header Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "modelType", "numericType", "inputType", "outputType"],
  "properties": {
    "version": { "type": "integer", "minimum": 1 },
    "modelType": { "type": "string" },
    "numericType": { "type": "string", "enum": ["Double", "Single", "Decimal"] },
    "inputType": { "type": "string" },
    "outputType": { "type": "string" },
    "inputShape": { "type": "array", "items": { "type": "integer" } },
    "outputShape": { "type": "array", "items": { "type": "integer" } },
    "modelId": { "type": "string", "format": "uuid" },
    "checksumSHA256": { "type": "string", "pattern": "^[a-f0-9]{64}$" }
  }
}
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-07
**Author:** AiDotNet Team
**Status:** Draft - Pending Gap Analysis
