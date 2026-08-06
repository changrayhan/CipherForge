#!/usr/bin/env bash
# monitor_phase_runs.sh — Periodic status snapshot for Phase 1.5 / Phase 1.5-B.
#
# Usage:
#   bash scripts/monitor_phase_runs.sh [INTERVAL_SEC] [MAX_ITERATIONS]
# Defaults: every 300s, up to 60 iterations (~5 h).

INTERVAL="${1:-300}"
MAXITER="${2:-60}"

BIO_ROOT="${BIO_ROOT:-/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/BioTriplex1BTestData}"
LOG="${BIO_ROOT}/runs/_monitor.log"

mkdir -p "$(dirname "$LOG")"

snapshot() {
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    {
        echo "============================================================"
        echo "[$ts] === STATUS SNAPSHOT ==="
        echo

        echo "[GPU]"
        nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null
        echo

        echo "[GPU procs]"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
        echo

        echo "[Trainer procs]"
        # Show only processes that have all of {--scale, --g_H_dtype, --seed} on the
        # cmdline (this filters out monitors, bash glue, etc).
        ps -eo pid,etimes,cmd | grep bio_baseline_trainer | grep -- '--scale ' \
            | grep -- '--g_H_dtype ' | grep -- '--seed ' \
            | while read pid etime rest; do
            scale=$(echo "$rest" | grep -oE -- '--scale [0-9]+' | awk '{print $2}')
            dtype=$(echo "$rest" | grep -oE -- '--g_H_dtype [a-z0-9]+' | awk '{print $2}')
            seed=$(echo "$rest" | grep -oE -- '--seed [0-9]+' | awk '{print $2}')
            dpa=$(echo "$rest" | grep -oE -- '--dp_alpha [0-9.]+' | awk '{print $2}')
            em=$((etime/60)); es=$((etime%60))
            printf "  pid=%-7s etime=%dm%-2ds  scale=%-7s dtype=%-5s dp_alpha=%-5s seed=%s\n" "$pid" "$em" "$es" "${scale:-?}" "${dtype:-?}" "${dpa:-?}" "${seed:-?}"
        done
        echo

        echo "[Phase 1.5 (B-q-*) DONE]"
        local done15=$(find "$BIO_ROOT/runs/quant" -maxdepth 3 -name 'DONE.flag' 2>/dev/null | wc -l)
        echo "  $done15 / 24"
        echo

        echo "[Phase 1.5-B DONE]"
        local done15b=$(find "$BIO_ROOT/runs/quant_dp15" -maxdepth 3 -name 'DONE.flag' 2>/dev/null | wc -l)
        echo "  $done15b / 18"
        echo

        echo "[Phase 1.5 latest epoch metrics]"
        local f15=$(ls -t "$BIO_ROOT"/runs/quant/B-q-*/seed_*/logs/epoch_*_bio_metrics.json 2>/dev/null | head -1)
        if [ -n "$f15" ]; then
            python3 -c "
import json,sys
d=json.load(open('$f15'))
print(f'  file=$f15'.split('/')[-1])
print(f'  macro_f1={d.get(\"macro_f1\",0):.4f} accuracy={d.get(\"accuracy\",0):.4f}')
"
        else
            echo "  (none)"
        fi
        echo

        echo "[Phase 1.5-B latest epoch metrics]"
        local f15b=$(ls -t "$BIO_ROOT"/runs/quant_dp15/B-dpa15-*/seed_*/logs/epoch_*_bio_metrics.json 2>/dev/null | head -1)
        if [ -n "$f15b" ]; then
            python3 -c "
import json,sys
d=json.load(open('$f15b'))
print(f'  file=$f15b'.split('/')[-1])
print(f'  macro_f1={d.get(\"macro_f1\",0):.4f} accuracy={d.get(\"accuracy\",0):.4f}')
"
        else
            echo "  (none)"
        fi
        echo

        echo "[DISK]"
        df -h /root/autodl-tmp | tail -1
        echo

    } | tee -a "$LOG"
}

for i in $(seq 1 "$MAXITER"); do
    snapshot
    sleep "$INTERVAL"
done
