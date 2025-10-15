# Quick Recovery Guide - Multi-Agent Build Fix System

## If Your Computer Shuts Down and You Lose Everything

### Option 1: Use the Recovery Prompt (Recommended)
Copy and paste this prompt to Claude:

```
I need to recreate the multi-agent build error fix system for my AiDotNet C# project. Please read the file RECOVERY_PROMPT.md in my project directory and implement the complete system as described. If that file doesn't exist, create a 4-agent coordination system with:

1. Build Checker Agent (supervisor)
2. Agent 1: Duplicate Resolution (CS0101, CS0111 errors)  
3. Agent 2: Constraints Specialist (CS8377, CS0104 errors)
4. Agent 3: Inheritance Specialist (CS0462, CS0115 errors)

The system should have file locking, progress tracking, .NET Framework 4.6.2+ compatibility, and be able to handle computer shutdowns. Create all necessary scripts and coordination files.
```

### Option 2: Use the Bootstrap Script
If you still have the bootstrap script:

```bash
./bootstrap_multi_agent_system.sh
```

### Option 3: Manual Recreation
If you have nothing, run these commands:

```bash
# Download the bootstrap script (if available in git)
git checkout bootstrap_multi_agent_system.sh

# Or create it manually and run
chmod +x bootstrap_multi_agent_system.sh
./bootstrap_multi_agent_system.sh
```

## After Recovery

1. **Verify the system**:
   ```bash
   ./state_manager.sh check-integrity
   ```

2. **Test in simulation mode**:
   ```bash
   ./multi_agent_coordinator.sh simulation
   ```

3. **Check current build status**:
   ```bash
   dotnet build > current_build_errors.txt
   grep -c "error CS" current_build_errors.txt
   ```

4. **Resume work**:
   ```bash
   ./multi_agent_coordinator.sh execute
   ```

## Key Files Created by Recovery

- `AGENT_COORDINATION.md` - System state and coordination
- `multi_agent_coordinator.sh` - Main coordination script
- `build_checker_agent.sh` - Build validation agent
- `agent1_duplicate_resolver.sh` - Duplicate resolution agent
- `agent2_constraints_specialist.sh` - Constraints specialist agent  
- `agent3_inheritance_specialist.sh` - Inheritance specialist agent
- `state_manager.sh` - State persistence and recovery
- `RECOVERY_PROMPT.md` - Complete recovery instructions

## Emergency Recovery Commands

```bash
# Check what files exist
ls -la *.sh *.md

# See current build errors
dotnet build 2>&1 | grep "error CS" | wc -l

# Initialize state management
./state_manager.sh init

# Create backup before starting work
./state_manager.sh backup

# Get recovery information
./state_manager.sh recovery-info
```

## Success Indicators

After recovery, you should see:
- All agent scripts are executable (chmod +x applied)
- AGENT_COORDINATION.md exists and contains current status
- Multi-agent coordinator runs without errors
- Build checker can count and categorize errors
- System can run in both simulation and execute modes

## Contact Information

If recovery fails, provide Claude with:
1. Current directory listing: `ls -la`
2. Current build errors: `dotnet build 2>&1 | head -20`
3. Project structure: `find src -name "*.cs" | head -10`
4. Any existing agent files: `ls -la *agent*.sh multi_agent*.sh`

The system is designed to be fully recoverable from this documentation alone.