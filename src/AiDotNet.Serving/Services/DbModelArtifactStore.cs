using System.Security.Cryptography;
using AiDotNet.Serving.Persistence;
using AiDotNet.Serving.Persistence.Entities;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.EntityFrameworkCore;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Database-backed store for protected model artifacts.
/// </summary>
public sealed class DbModelArtifactStore : IModelArtifactStore
{
    private const string ProtectorPurpose = "AiDotNet.Serving.ProtectedArtifacts.v1";

    private readonly IServiceScopeFactory _scopeFactory;
    private readonly IDataProtector _protector;

    public DbModelArtifactStore(IServiceScopeFactory scopeFactory, IDataProtectionProvider dataProtectionProvider)
    {
        _scopeFactory = scopeFactory ?? throw new ArgumentNullException(nameof(scopeFactory));
        _protector = (dataProtectionProvider ?? throw new ArgumentNullException(nameof(dataProtectionProvider)))
            .CreateProtector(ProtectorPurpose);
    }

    public bool TryGet(string modelName, out ProtectedModelArtifact? artifact)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            artifact = null;
            return false;
        }

        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<ServingDbContext>();
        var entity = db.ProtectedArtifacts
            .AsNoTracking()
            .SingleOrDefault(e => e.ArtifactName == modelName);

        if (entity == null)
        {
            artifact = null;
            return false;
        }

        if (string.IsNullOrWhiteSpace(entity.EncryptedPath) || !File.Exists(entity.EncryptedPath))
        {
            TryRemove(db, entity.ArtifactName);
            artifact = null;
            return false;
        }

        if (!TryToArtifact(entity, out var parsed))
        {
            TryRemove(db, entity.ArtifactName);
            artifact = null;
            return false;
        }

        artifact = parsed;
        return true;
    }

    public ProtectedModelArtifact GetOrCreate(string modelName, Func<ProtectedModelArtifact> factory)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Artifact name is required.", nameof(modelName));
        }

        if (factory == null)
        {
            throw new ArgumentNullException(nameof(factory));
        }

        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<ServingDbContext>();
        var existing = db.ProtectedArtifacts
            .AsNoTracking()
            .SingleOrDefault(e => e.ArtifactName == modelName);

        if (existing != null &&
            !string.IsNullOrWhiteSpace(existing.EncryptedPath) &&
            File.Exists(existing.EncryptedPath) &&
            TryToArtifact(existing, out var artifact))
        {
            return artifact;
        }

        var created = factory();
        return Persist(db, modelName, created, recordExists: existing != null);
    }

    public void Remove(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            return;
        }

        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<ServingDbContext>();
        var entity = db.ProtectedArtifacts.SingleOrDefault(e => e.ArtifactName == modelName);
        if (entity == null)
        {
            return;
        }

        db.ProtectedArtifacts.Remove(entity);
        db.SaveChanges();
    }

    private ProtectedModelArtifact Persist(ServingDbContext db, string artifactName, ProtectedModelArtifact artifact, bool recordExists)
    {
        var entity = new ProtectedArtifactEntity
        {
            ArtifactName = artifactName,
            EncryptedPath = artifact.EncryptedPath,
            KeyId = artifact.KeyId,
            Algorithm = artifact.Algorithm,
            ProtectedKey = _protector.Protect(artifact.Key),
            ProtectedNonce = _protector.Protect(artifact.Nonce),
            CreatedAt = DateTimeOffset.UtcNow
        };

        if (recordExists)
        {
            db.ProtectedArtifacts.Update(entity);
            db.SaveChanges();
            return artifact;
        }

        db.ProtectedArtifacts.Add(entity);

        try
        {
            db.SaveChanges();
            return artifact;
        }
        catch (DbUpdateException)
        {
            var concurrent = db.ProtectedArtifacts
                .AsNoTracking()
                .SingleOrDefault(e => e.ArtifactName == artifactName);

            if (concurrent == null)
            {
                throw;
            }

            if (TryToArtifact(concurrent, out var stored))
            {
                TryDeleteFile(artifact.EncryptedPath, concurrent.EncryptedPath);
                return stored;
            }

            throw;
        }
    }

    private static void TryDeleteFile(string candidatePath, string preservedPath)
    {
        if (string.IsNullOrWhiteSpace(candidatePath) || string.IsNullOrWhiteSpace(preservedPath))
        {
            return;
        }

        if (candidatePath.Equals(preservedPath, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        try
        {
            if (File.Exists(candidatePath))
            {
                File.Delete(candidatePath);
            }
        }
        catch
        {
            // Best-effort cleanup; ignore.
        }
    }

    private bool TryToArtifact(ProtectedArtifactEntity entity, out ProtectedModelArtifact artifact)
    {
        try
        {
            var key = _protector.Unprotect(entity.ProtectedKey);
            var nonce = _protector.Unprotect(entity.ProtectedNonce);

            try
            {
                artifact = new ProtectedModelArtifact(
                    entity.ArtifactName,
                    entity.EncryptedPath,
                    entity.KeyId,
                    key,
                    nonce,
                    entity.Algorithm);
            }
            finally
            {
                CryptographicOperations.ZeroMemory(key);
                CryptographicOperations.ZeroMemory(nonce);
            }

            return true;
        }
        catch (CryptographicException)
        {
            artifact = null!;
            return false;
        }
    }

    private static void TryRemove(ServingDbContext db, string artifactName)
    {
        try
        {
            db.ProtectedArtifacts.Remove(new ProtectedArtifactEntity { ArtifactName = artifactName });
            db.SaveChanges();
        }
        catch
        {
            // Best-effort cleanup; ignore.
        }
    }
}
