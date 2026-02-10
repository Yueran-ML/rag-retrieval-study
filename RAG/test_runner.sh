#!/usr/bin/env bash

# Fix default conda environment to NLPA2
source /conda/etc/profile.d/conda.sh
conda activate NLPA2
export PATH=/conda/envs/NLPA2/bin:$PATH

PYTHON="/conda/envs/NLPA2/bin/python"
WORKING_DIR="$HOME/A2-v2.0"
LOGFILE="${WORKING_DIR}/run.log"

if [ -f ${LOGFILE} ];
then
  OLD=$(mktemp -d ${LOGFILE}.XXXX)
  mv ${LOGFILE} ${OLD}
fi

echo "Running example retrieval"
${PYTHON} example_retrieval.py 2>&1 >> ${LOGFILE} 2>&1
echo "Running example reranker"
${PYTHON} example_reranker.py 2>&1 >> ${LOGFILE} 2>&1
echo "Running example rag"
${PYTHON} example_rag.py 2>&1 >> ${LOGFILE} 2>&1
