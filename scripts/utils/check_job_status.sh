#!/bin/bash
# Check status of submitted jobs
# Usage: bash scripts/check_job_status.sh

echo "=========================================="
echo "Job Status Summary"
echo "=========================================="
echo ""

# Show queue status for your jobs
echo "Your jobs in queue:"
squeue -u $USER --format="%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R"

echo ""
echo "=========================================="
echo "Recent job logs:"
echo "=========================================="

# Show recent output files
ls -lt experiments/logs/*.out | head -5

echo ""
echo "To view a specific job log:"
echo "  tail -f experiments/logs/fno-training-<JOB_ID>.out"
echo ""
echo "To cancel a job:"
echo "  scancel <JOB_ID>"
