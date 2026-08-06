#!/usr/bin/env bash
# worker.sh — a single worker in the chained_runner pool.
#
# This is a separate file (not chained_runner.sh) so that spawning is
# just `bash worker.sh` (no export -f complications across nested
# bash -c boundaries).
#
# We source chained_runner.sh to get all the helpers (PHASE_QUEUE,
# pick_next_phase, validate_phase, etc.) and the worker_loop function,
# then call worker_loop. CHAINED_WORKER env var must be set by the caller
# (this is what triggers the dispatch to call worker_loop instead of main).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/chained_runner.sh"

# If we reach here without dispatching, fall through to worker_loop.
# (The dispatch block at the end of chained_runner.sh will call main()
# when CHAINED_WORKER is unset, or worker_loop when set.)
