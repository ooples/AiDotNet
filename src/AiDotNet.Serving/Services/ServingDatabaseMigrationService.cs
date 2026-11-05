using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Persistence;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Applies EF Core migrations for the Serving persistence database at startup, when
/// <see cref="ServingPersistenceOptions.MigrateOnStartup"/> is set.
/// </summary>
/// <remarks>
/// Hosted services start before the server accepts requests, so no request reaches an unmigrated database.
/// </remarks>
public sealed class ServingDatabaseMigrationService : IHostedService
{
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly ServingPersistenceOptions _persistence;
    private readonly ILogger<ServingDatabaseMigrationService> _logger;

    public ServingDatabaseMigrationService(
        IServiceScopeFactory scopeFactory,
        IOptions<ServingPersistenceOptions> persistence,
        ILogger<ServingDatabaseMigrationService> logger)
    {
        Guard.NotNull(scopeFactory);
        _scopeFactory = scopeFactory;
        Guard.NotNull(persistence);
        _persistence = persistence.Value;
        Guard.NotNull(logger);
        _logger = logger;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        if (!_persistence.MigrateOnStartup)
        {
            return;
        }

        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<ServingDbContext>();

        _logger.LogInformation("Applying Serving DB migrations (Provider={Provider}).", _persistence.Provider);
        await db.Database.MigrateAsync(cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("Serving DB migrations applied.");
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}

