#!/bin/bash

mvn package

# These variables can be set also in IntelliJ run configuration
export FLINK_BIN_PATH="/home/michal/Documents/mgr/flink-1.19.1/bin"
export RESULTS_DIRECTORY="/home/michal/Documents/mgr/flink-classifiers/results"
export FLINK_ADDRESS="localhost"
export FLINK_PORT="8081"

if [ -z "$FLINK_BIN_PATH" ]; then
  echo "FLINK_BIN_PATH env variable not set" >&2
  exit 1
fi

if [ -z "$RESULTS_DIRECTORY" ]; then
  echo "RESULTS_DIRECTORY env variable not set" >&2
  exit 1
fi

if [ -z "$FLINK_ADDRESS" ]; then
  export FLINK_ADDRESS="localhost"
fi

if [ -z "$FLINK_PORT" ]; then
  export FLINK_PORT="8081"
fi

#if [ "$FLINK_ADDRESS" = "localhost" ]; then ## if flink cluster is not running on cluster start it
#  http_status=$(curl --write-out "%{http_code}" --silent --output /dev/null "http://${FLINK_ADDRESS}:${FLINK_PORT}/taskmanagers")
#
#  if [ "${http_status}" != "200" ]; then
#    "${FLINK_BIN_PATH}"/start-cluster.sh
#  fi
#fi

# Always start a new cluster to limit local machine memory usage when it's not running any jobs
${FLINK_BIN_PATH}/start-cluster.sh

jar_dir="./target"
jar_path=$(ls -t $jar_dir/*.jar | head -n 1 )
if [ -z "$jar_path" ]; then
  echo "Error: No JAR file found in $jar_dir" >&2
  exit 1
fi

echo "tu doszedłem i żyję, jarpath to będzie $(realpath $jar_path)}"

jar_absolute_path=$(realpath $jar_path)

export EXPERIMENT_ID=$(date +%Y-%m-%dT%H:%M:%S)

"$FLINK_BIN_PATH"/flink run -m "${FLINK_ADDRESS}:${FLINK_PORT}" "$jar_absolute_path" --env EXPERIMENT_ID="$EXPERIMENT_ID" --env RESULTS_DIRECTORY="$RESULTS_DIRECTORY"

#export EXPERIMENT_ID=ea3e596c-5da2-42da-8572-88bd038469ab

${FLINK_BIN_PATH}/stop-cluster.sh
