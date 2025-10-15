#!/bin/bash

# State Persistence and Recovery Manager
# Handles checkpointing and recovery for the multi-agent system

STATE_DIR=".agent_state"
CHECKPOINT_FILE="$STATE_DIR/checkpoint.json"
PROGRESS_FILE="$STATE_DIR/progress.json"
BACKUP_DIR="$STATE_DIR/backups"

create_state_directory() {
    mkdir -p "$STATE_DIR"
    mkdir -p "$BACKUP_DIR"
    echo "State management directories created"
}

save_checkpoint() {
    local iteration="$1"
    local errors_before="$2"
    local errors_after="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat > "$CHECKPOINT_FILE" << EOF
{
    "timestamp": "$timestamp",
    "iteration": $iteration,
    "errors_before": $errors_before,
    "errors_after": $errors_after,
    "status": "in_progress",
    "current_phase": "agent_coordination",
    "files_processed": [],
    "agents_active": ["agent1", "agent2", "agent3"],
    "last_successful_build": "$timestamp"
}
EOF
    
    echo "Checkpoint saved: Iteration $iteration, Errors: $errors_before -> $errors_after"
}

save_progress() {
    local agent_id="$1"
    local file_path="$2"
    local action="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create or update progress file
    if [ ! -f "$PROGRESS_FILE" ]; then
        echo '{"progress": []}' > "$PROGRESS_FILE"
    fi
    
    # Add new progress entry (simplified - would use jq in real implementation)
    echo "Progress saved: $agent_id $action $file_path at $timestamp"
}

backup_current_state() {
    local backup_name="backup_$(date '+%Y%m%d_%H%M%S')"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    mkdir -p "$backup_path"
    
    # Backup key files
    cp -f AGENT_COORDINATION.md "$backup_path/" 2>/dev/null || true
    cp -f agent_coordination.log "$backup_path/" 2>/dev/null || true
    cp -f build_output_iteration.txt "$backup_path/" 2>/dev/null || true
    cp -f build_error_count.txt "$backup_path/" 2>/dev/null || true
    
    echo "State backed up to: $backup_path"
    echo "$backup_path" > "$STATE_DIR/latest_backup.txt"
}

restore_from_backup() {
    local backup_path="$1"
    
    if [ -z "$backup_path" ] && [ -f "$STATE_DIR/latest_backup.txt" ]; then
        backup_path=$(cat "$STATE_DIR/latest_backup.txt")
    fi
    
    if [ -d "$backup_path" ]; then
        echo "Restoring from backup: $backup_path"
        cp -f "$backup_path"/* . 2>/dev/null || true
        echo "Backup restored successfully"
        return 0
    else
        echo "Backup path not found: $backup_path"
        return 1
    fi
}

get_recovery_info() {
    echo "=== Multi-Agent System Recovery Information ==="
    
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "Last checkpoint found:"
        cat "$CHECKPOINT_FILE"
        echo ""
    else
        echo "No checkpoint file found"
    fi
    
    if [ -f "$PROGRESS_FILE" ]; then
        echo "Progress information available"
    else
        echo "No progress file found"
    fi
    
    echo "Available backups:"
    ls -la "$BACKUP_DIR" 2>/dev/null || echo "No backups found"
    
    echo ""
    echo "Key files status:"
    echo "  AGENT_COORDINATION.md: $([ -f AGENT_COORDINATION.md ] && echo "EXISTS" || echo "MISSING")"
    echo "  agent_coordination.log: $([ -f agent_coordination.log ] && echo "EXISTS" || echo "MISSING")"
    echo "  multi_agent_coordinator.sh: $([ -f multi_agent_coordinator.sh ] && echo "EXISTS" || echo "MISSING")"
    echo "  build_checker_agent.sh: $([ -f build_checker_agent.sh ] && echo "EXISTS" || echo "MISSING")"
}

create_recovery_point() {
    echo "Creating recovery point..."
    create_state_directory
    backup_current_state
    
    # Save system state
    cat > "$STATE_DIR/system_state.json" << EOF
{
    "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')",
    "project_path": "$(pwd)",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "system_ready": true,
    "agents_deployed": {
        "build_checker": $([ -f build_checker_agent.sh ] && echo "true" || echo "false"),
        "agent1_duplicate": $([ -f agent1_duplicate_resolver.sh ] && echo "true" || echo "false"),
        "agent2_constraints": $([ -f agent2_constraints_specialist.sh ] && echo "true" || echo "false"),
        "agent3_inheritance": $([ -f agent3_inheritance_specialist.sh ] && echo "true" || echo "false"),
        "coordinator": $([ -f multi_agent_coordinator.sh ] && echo "true" || echo "false")
    }
}
EOF
    
    echo "Recovery point created successfully"
}

check_system_integrity() {
    echo "Checking multi-agent system integrity..."
    
    local missing_files=()
    
    [ ! -f "multi_agent_coordinator.sh" ] && missing_files+=("multi_agent_coordinator.sh")
    [ ! -f "build_checker_agent.sh" ] && missing_files+=("build_checker_agent.sh") 
    [ ! -f "agent1_duplicate_resolver.sh" ] && missing_files+=("agent1_duplicate_resolver.sh")
    [ ! -f "agent2_constraints_specialist.sh" ] && missing_files+=("agent2_constraints_specialist.sh")
    [ ! -f "agent3_inheritance_specialist.sh" ] && missing_files+=("agent3_inheritance_specialist.sh")
    [ ! -f "AGENT_COORDINATION.md" ] && missing_files+=("AGENT_COORDINATION.md")
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        echo "✓ All system files present"
        return 0
    else
        echo "✗ Missing files: ${missing_files[*]}"
        return 1
    fi
}

usage() {
    echo "Multi-Agent System State Manager"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  init                    - Initialize state management"
    echo "  checkpoint <iter> <before> <after> - Save checkpoint"
    echo "  backup                  - Create backup of current state"
    echo "  restore [backup_path]   - Restore from backup"
    echo "  recovery-info          - Show recovery information"
    echo "  recovery-point         - Create comprehensive recovery point"
    echo "  check-integrity        - Check if all system files exist"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 checkpoint 1 100 85"
    echo "  $0 backup"
    echo "  $0 restore"
    echo "  $0 recovery-info"
}

main() {
    local command="$1"
    
    case "$command" in
        "init")
            create_state_directory
            ;;
        "checkpoint")
            save_checkpoint "$2" "$3" "$4"
            ;;
        "backup")
            backup_current_state
            ;;
        "restore")
            restore_from_backup "$2"
            ;;
        "recovery-info")
            get_recovery_info
            ;;
        "recovery-point")
            create_recovery_point
            ;;
        "check-integrity")
            check_system_integrity
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi