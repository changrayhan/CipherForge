#!/usr/bin/env bash
# run_trec_full_background.sh — detach the TREC-QC runner via setsid+nohup.
#
# Usage:
#   ./run_trec_full_background.sh start [PHASES=1,4] [SEEDS=42,123,2025]
#   ./run_trec_full_background.sh status
#   ./run_trec_full_background.sh stop
#   ./run_trec_full_background.sh tail
#
# Same pattern as v2/run_phase4_full_background.sh.
set -euo pipefail

TREC_ROOT="${TREC_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/TrecAATestData}"
SCRIPT_DIR="${TREC_ROOT}/scripts"
RUNNER="${SCRIPT_DIR}/_trec_internal_runner.sh"
LOG_ROOT="${TREC_ROOT}/runs/baseline/_runner_logs"
mkdir -p "${LOG_ROOT}"

PIDFILE="${LOG_ROOT}/runner.pid"

start() {
    local phases="${1:-1,4}"
    local seeds="${2:-42,123,2025}"
    local exp_set="${3:-full}"

    if [[ -f "${PIDFILE}" ]]; then
        local pid
        pid=$(cat "${PIDFILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "RUNNING: PID=${pid}"
            return 0
        fi
    fi

    # Make scripts executable
    chmod +x "${SCRIPT_DIR}"/*.sh "${SCRIPT_DIR}"/*.py 2>/dev/null || true

    echo "STARTING TREC-QC runner: PHASES=${phases} SEEDS=${seeds} EXP=${exp_set}"
    PHASES="${phases}" SEEDS="${seeds}" EXPERIMENT_SET="${exp_set}" \
        setsid nohup bash "${RUNNER}" \
        > "${LOG_ROOT}/runner.stdout.log" 2>&1 < /dev/null &
    local pid=$!
    echo "${pid}" > "${PIDFILE}"
    sleep 2
    echo "STARTED: PID=${pid}"
    echo "Logs: ${LOG_ROOT}/runner.log"
    echo "Tail: ${LOG_ROOT}/runner.stdout.log"
}

status() {
    if [[ ! -f "${PIDFILE}" ]]; then
        echo "NOT RUNNING (no pidfile)"
        return 0
    fi
    local pid
    pid=$(cat "${PIDFILE}")
    if kill -0 "${pid}" 2>/dev/null; then
        echo "RUNNING: PID=${pid}"
        ps -p "${pid}" -o pid,etime,cmd || true
    else
        echo "DEAD (pid ${pid} gone)"
    fi
    echo
    echo "--- log tail (last 15 lines) ---"
    tail -n 15 "${LOG_ROOT}/runner.log" 2>/dev/null || true
    echo
    echo "--- completed experiments ---"
    find "${TREC_ROOT}/runs" -name DONE.flag 2>/dev/null | sort
}

stop() {
    if [[ ! -f "${PIDFILE}" ]]; then
        echo "NOT RUNNING"
        return 0
    fi
    local pid
    pid=$(cat "${PIDFILE}")
    echo "Stopping PID ${pid} and children ..."
    pkill -P "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 3
    kill -KILL "${pid}" 2>/dev/null || true
    rm -f "${PIDFILE}"
    echo "STOPPED"
}

tail_log() {
    tail -f "${LOG_ROOT}/runner.log"
}

case "${1:-status}" in
    start)  start "${2:-1,4}" "${3:-42,123,2025}" "${4:-full}" ;;
    status) status ;;
    stop)   stop ;;
    tail)   tail_log ;;
    *)      echo "Usage: $0 {start|status|stop|tail}" ; exit 1 ;;
esac