#!/usr/bin/env bash
# All execution roles use the same Ultra image; no worker images are pulled.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d ' \r\n' < "$ROOT/docker/VERSION")"
IMAGE="${IMAGE:-vllm-sm75:v${VERSION/-beta/-ultra-beta}}"
NETWORK="${BENCHLOCAL_NETWORK:-sm75-benchlocal-native}"
ACTION="${1:-start}"
case "$ACTION" in start|status|stop|connect) ;; *) echo 'Usage: benchlocal.sh start|status|stop|connect' >&2; exit 2;; esac
if [[ "$ACTION" == status ]]; then
  for name in sm75-bench-bugfind sm75-bench-cli sm75-bench-hermes; do docker inspect --format '{{.Name}} {{.State.Status}} {{.Config.Image}}' "$name"; done
  exit
fi
if [[ "$ACTION" == connect ]]; then
  name="${CONTAINER_NAME:-vllm-sm75-ultra}"
  docker network inspect "$NETWORK" >/dev/null
  if ! docker inspect --format '{{json .NetworkSettings.Networks}}' "$name" | grep -Fq "\"$NETWORK\""; then
    docker network connect --alias sm75-bench-controller "$NETWORK" "$name"
  fi
  exit
fi
if [[ "$ACTION" == start ]]; then
  : "${ULTRA_DATA_ROOT:?Set ULTRA_DATA_ROOT to your persistent Ultra directory}"
  [[ "$ULTRA_DATA_ROOT" == /* && "$ULTRA_DATA_ROOT" != / ]] || exit 2
  KEY_FILE="${BENCHLOCAL_ENV_FILE:-$ULTRA_DATA_ROOT/benchlocal.env}"
  [[ "$KEY_FILE" == /* && "$KEY_FILE" != / ]] || exit 2
  [[ "$(docker image inspect --format '{{index .Config.Labels "io.sm75.benchlocal.unified"}}' "$IMAGE")" == 1 ]] || { echo 'Build the beta2 Ultra image first' >&2; exit 2; }
  mkdir -p "$(dirname -- "$KEY_FILE")"
  if [[ ! -s "$KEY_FILE" ]]; then
    umask 077
    printf 'SM75_BENCH_VERIFIER_KEY=%s\n' "$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')" > "$KEY_FILE"
  fi
  chmod 600 "$KEY_FILE"
  grep -Eq '^SM75_BENCH_VERIFIER_KEY=[a-fA-F0-9]{64}$' "$KEY_FILE" || { echo 'Invalid BenchLocal credential file' >&2; exit 2; }
  docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create --internal --driver bridge --label io.sm75.benchlocal.role=network "$NETWORK" >/dev/null
fi
for pair in bugfind:bugfind-15 cli:cli-40 hermes:hermesagent-20; do
  name=sm75-bench-${pair%%:*}; role=${pair#*:}
  if docker container inspect "$name" >/dev/null 2>&1; then
    [[ "$(docker inspect --format '{{index .Config.Labels "io.sm75.benchlocal.role"}}' "$name")" == verifier ]] || { echo "Unmanaged container: $name" >&2; exit 2; }
    if [[ "$(docker inspect --format '{{.State.Running}}' "$name")" == true ]]; then
      docker exec "$name" node -e 'fetch("http://127.0.0.1:4200/health").then(r=>r.json()).then(r=>{if(r.busy)process.exit(2)}).catch(()=>process.exit(2))' || { echo "$name is busy or unavailable; retry after the test completes" >&2; exit 2; }
    fi
    if [[ "$ACTION" == stop ]]; then docker stop "$name" >/dev/null; continue; fi
    if [[ "$(docker inspect --format '{{.Image}}' "$name")" == "$(docker image inspect --format '{{.Id}}' "$IMAGE")" ]]; then docker start "$name" >/dev/null; continue; fi
    docker stop "$name" >/dev/null
    docker rm "$name" >/dev/null
  elif [[ "$ACTION" == stop ]]; then continue
  fi
  docker run -d --name "$name" --network "$NETWORK" --runtime runc --restart unless-stopped \
    --init --read-only --pids-limit 512 --memory 4g --cpus 2 \
    --cap-drop ALL --cap-add SETUID --cap-add SETGID --cap-add KILL --security-opt no-new-privileges \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g,mode=1777 \
    --tmpfs /home/verifier:rw,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=700 \
    --tmpfs /workspace:rw,exec,nosuid,nodev,size=512m,uid=10001,gid=10001,mode=700 \
    --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= \
    --env SM75_BENCHLOCAL_ROLE="$role" --env-file "$KEY_FILE" \
    --env SM75_BENCH_PROXY_ORIGIN="${SM75_BENCH_PROXY_ORIGIN:-http://sm75-bench-controller:1615}" \
    --label io.sm75.benchlocal.role=verifier --label io.sm75.benchlocal.unified=1 \
    --label net.unraid.docker.managed=dockerman --label net.unraid.docker.icon=https://github.com/stevibe.png \
    --entrypoint python3 "$IMAGE" /opt/vllm-sm75/benchlocal-entrypoint.py >/dev/null
  printf '%s started (%s)\n' "$name" "$IMAGE"
done
