using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Exceptions;
using Newtonsoft.Json;

namespace AiDotNet.Helpers;

/// <summary>
/// Manages the AiDotNet free trial state persisted in <c>~/.aidotnet/trial.json</c>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> AiDotNet provides a free trial for model save/load operations.
/// This class tracks when you first used save/load and how many times you've used them.
/// The trial allows 30 days OR 10 operations (saves + loads), whichever comes first.</para>
///
/// <para>Training and inference are <b>never</b> restricted — only <c>SaveModel()</c> and
/// <c>LoadModel()</c> operations are tracked by the trial.</para>
///
/// <para>The trial state is stored locally in your home directory at <c>~/.aidotnet/trial.json</c>.
/// This file is HMAC-signed to prevent tampering. If the file is corrupted or tampered with,
/// the trial is treated as expired.</para>
/// </remarks>
internal sealed class TrialStateManager
{
    /// <summary>
    /// Maximum number of days the free trial is active.
    /// </summary>
    internal const int TrialDurationDays = 30;

    /// <summary>
    /// Maximum number of save/load operations allowed during the free trial.
    /// </summary>
    internal const int TrialOperationLimit = 10;

    private static readonly object FileLock = new();

    private readonly string _trialFilePath;

    /// <summary>
    /// Creates a new <see cref="TrialStateManager"/> using the default trial file location.
    /// </summary>
    public TrialStateManager()
        : this(GetDefaultTrialFilePath())
    {
    }

    /// <summary>
    /// Creates a new <see cref="TrialStateManager"/> with a custom trial file path.
    /// Used for testing.
    /// </summary>
    /// <param name="trialFilePath">Full path to the trial state JSON file.</param>
    internal TrialStateManager(string trialFilePath)
    {
        _trialFilePath = trialFilePath ?? throw new ArgumentNullException(nameof(trialFilePath));
    }

    /// <summary>
    /// Checks whether the trial is still active and records an operation.
    /// If the trial has expired, throws <see cref="LicenseRequiredException"/>.
    /// If the trial is active, increments the operation counter and writes a
    /// console info message.
    /// </summary>
    /// <exception cref="LicenseRequiredException">
    /// Thrown when the free trial has expired (either by time or operation count).
    /// </exception>
    public void RecordOperationOrThrow()
    {
        lock (FileLock)
        {
            var state = LoadOrCreateState();

            // Check time expiration
            int daysElapsed = (int)(DateTimeOffset.UtcNow - state.FirstUseUtc).TotalDays;
            if (daysElapsed > TrialDurationDays)
            {
                throw new LicenseRequiredException(
                    TrialExpirationReason.TimeExpired,
                    trialDaysElapsed: daysElapsed,
                    operationsPerformed: state.OperationCount);
            }

            // Check operation limit
            if (state.OperationCount >= TrialOperationLimit)
            {
                throw new LicenseRequiredException(
                    TrialExpirationReason.OperationLimitReached,
                    trialDaysElapsed: daysElapsed,
                    operationsPerformed: state.OperationCount);
            }

            // Trial is active — increment counter and persist
            state.OperationCount++;
            state.LastOperationUtc = DateTimeOffset.UtcNow;
            SaveState(state);

            // Console info message during trial
            int daysRemaining = TrialDurationDays - daysElapsed;
            int opsRemaining = TrialOperationLimit - state.OperationCount;
            Console.WriteLine(
                $"AiDotNet Community — {daysRemaining} day(s) and {opsRemaining} operation(s) remaining in free trial. " +
                "Register for a free license at https://aidotnet.dev");
        }
    }

    /// <summary>
    /// Returns the current trial status without recording an operation or throwing.
    /// Useful for displaying trial information to the user.
    /// </summary>
    public LicenseTrialStatus GetStatus()
    {
        lock (FileLock)
        {
            var state = LoadOrCreateState();
            int daysElapsed = (int)(DateTimeOffset.UtcNow - state.FirstUseUtc).TotalDays;

            bool timeExpired = daysElapsed > TrialDurationDays;
            bool opsExpired = state.OperationCount >= TrialOperationLimit;

            return new LicenseTrialStatus(
                IsExpired: timeExpired || opsExpired,
                DaysElapsed: daysElapsed,
                DaysRemaining: Math.Max(0, TrialDurationDays - daysElapsed),
                OperationsUsed: state.OperationCount,
                OperationsRemaining: Math.Max(0, TrialOperationLimit - state.OperationCount),
                FirstUseUtc: state.FirstUseUtc);
        }
    }

    /// <summary>
    /// Resets the trial state. Intended for testing and license key provisioning only.
    /// </summary>
    internal void Reset()
    {
        lock (FileLock)
        {
            if (File.Exists(_trialFilePath))
            {
                File.Delete(_trialFilePath);
            }
        }
    }

    private TrialState LoadOrCreateState()
    {
        if (!File.Exists(_trialFilePath))
        {
            var newState = new TrialState
            {
                FirstUseUtc = DateTimeOffset.UtcNow,
                OperationCount = 0,
                LastOperationUtc = DateTimeOffset.UtcNow,
                MachineId = MachineFingerprint.GetMachineId()
            };
            SaveState(newState);
            return newState;
        }

        try
        {
            string json = File.ReadAllText(_trialFilePath, Encoding.UTF8);
            var envelope = JsonConvert.DeserializeObject<TrialFileEnvelope>(json);

            if (envelope is null || string.IsNullOrEmpty(envelope.Data) || string.IsNullOrEmpty(envelope.Signature))
            {
                // Corrupted file — treat as expired trial to prevent tampering
                return CreateExpiredState();
            }

            // Verify HMAC signature
            string expectedSignature = ComputeSignature(envelope.Data);
            if (!ConstantTimeEquals(expectedSignature, envelope.Signature))
            {
                // Tampered file — treat as expired
                return CreateExpiredState();
            }

            byte[] dataBytes = Convert.FromBase64String(envelope.Data);
            string dataJson = Encoding.UTF8.GetString(dataBytes);
            var state = JsonConvert.DeserializeObject<TrialState>(dataJson);

            if (state is null)
            {
                return CreateExpiredState();
            }

            // Sanity check: first use can't be in the future (clock manipulation)
            if (state.FirstUseUtc > DateTimeOffset.UtcNow.AddMinutes(5))
            {
                return CreateExpiredState();
            }

            return state;
        }
        catch (Exception ex) when (ex is JsonException or FormatException or IOException)
        {
            // Corrupted or unreadable — treat as expired
            return CreateExpiredState();
        }
    }

    private void SaveState(TrialState state)
    {
        string stateJson = JsonConvert.SerializeObject(state);
        string dataBase64 = Convert.ToBase64String(Encoding.UTF8.GetBytes(stateJson));
        string signature = ComputeSignature(dataBase64);

        var envelope = new TrialFileEnvelope
        {
            Version = 1,
            Data = dataBase64,
            Signature = signature
        };

        string envelopeJson = JsonConvert.SerializeObject(envelope, Formatting.Indented);

        // Ensure directory exists
        string directory = Path.GetDirectoryName(_trialFilePath) ?? string.Empty;
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(_trialFilePath, envelopeJson, Encoding.UTF8);
    }

    /// <summary>
    /// Creates a state that appears expired. Used when the trial file is tampered with
    /// or corrupted, to prevent users from gaining additional trial time by deleting
    /// or modifying the file.
    /// </summary>
    private static TrialState CreateExpiredState()
    {
        return new TrialState
        {
            // Set first use far enough in the past to exceed the trial duration
            FirstUseUtc = DateTimeOffset.UtcNow.AddDays(-(TrialDurationDays + 1)),
            OperationCount = TrialOperationLimit,
            LastOperationUtc = DateTimeOffset.UtcNow,
            MachineId = MachineFingerprint.GetMachineId()
        };
    }

    /// <summary>
    /// Computes an HMAC-SHA256 signature over the given data using a key derived from
    /// the machine fingerprint. This binds the trial state to the current machine and
    /// prevents simple copy-paste attacks.
    /// </summary>
    private static string ComputeSignature(string data)
    {
        // Derive signing key from machine fingerprint + a domain separator.
        // This makes the signature machine-specific without requiring a server.
        string machineId = MachineFingerprint.GetMachineId();
        byte[] keyMaterial = Encoding.UTF8.GetBytes("AiDotNet.Trial.v1:" + machineId);

        using var sha = SHA256.Create();
        byte[] signingKey = sha.ComputeHash(keyMaterial);

        byte[] dataBytes = Encoding.UTF8.GetBytes(data);
        using var hmac = new HMACSHA256(signingKey);
        byte[] hash = hmac.ComputeHash(dataBytes);

        return Convert.ToBase64String(hash);
    }

    /// <summary>
    /// Constant-time string comparison to prevent timing attacks on signature verification.
    /// </summary>
    private static bool ConstantTimeEquals(string a, string b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        int result = 0;
        for (int i = 0; i < a.Length; i++)
        {
            result |= a[i] ^ b[i];
        }

        return result == 0;
    }

    private static string GetDefaultTrialFilePath()
    {
        string homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(homeDir, ".aidotnet", "trial.json");
    }

    /// <summary>
    /// Internal trial state persisted to disk.
    /// </summary>
    private sealed class TrialState
    {
        [JsonProperty("firstUseUtc")]
        public DateTimeOffset FirstUseUtc { get; set; }

        [JsonProperty("operationCount")]
        public int OperationCount { get; set; }

        [JsonProperty("lastOperationUtc")]
        public DateTimeOffset LastOperationUtc { get; set; }

        [JsonProperty("machineId")]
        public string MachineId { get; set; } = string.Empty;
    }

    /// <summary>
    /// Signed envelope that wraps the trial state. The data field is base64-encoded JSON
    /// and the signature is an HMAC-SHA256 of the data field.
    /// </summary>
    private sealed class TrialFileEnvelope
    {
        [JsonProperty("version")]
        public int Version { get; set; }

        [JsonProperty("data")]
        public string Data { get; set; } = string.Empty;

        [JsonProperty("signature")]
        public string Signature { get; set; } = string.Empty;
    }
}

/// <summary>
/// Represents the current status of the AiDotNet free trial for model persistence operations.
/// </summary>
/// <param name="IsExpired">Whether the trial has expired.</param>
/// <param name="DaysElapsed">Number of days since the trial started.</param>
/// <param name="DaysRemaining">Number of days remaining in the trial (0 if expired).</param>
/// <param name="OperationsUsed">Number of save/load operations performed.</param>
/// <param name="OperationsRemaining">Number of save/load operations remaining (0 if expired).</param>
/// <param name="FirstUseUtc">When the trial started.</param>
internal sealed record LicenseTrialStatus(
    bool IsExpired,
    int DaysElapsed,
    int DaysRemaining,
    int OperationsUsed,
    int OperationsRemaining,
    DateTimeOffset FirstUseUtc);
