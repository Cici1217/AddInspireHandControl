#!/usr/bin/env bash
# G1-local SONIC; PICO and the SSH tunnel run on the computer.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

input_type=zmq_manager
hand_type=inspire
check=false
ready_pose=""
motion_data=reference/example/
while (($#)); do
    case "$1" in
        --keyboard) input_type=keyboard ;;
        --check) check=true ;;
        --hand)
            [[ $# -ge 2 ]] || { echo '--hand needs a native transport: dex3, external, or legacy inspire' >&2; exit 2; }
            case "$2" in dex3|external|inspire) hand_type=$2 ;; *) echo 'Unsupported native hand transport' >&2; exit 2 ;; esac
            shift ;;
        --ready-pose)
            [[ $# -ge 2 && -n "$2" ]] || { echo "--ready-pose needs 29 quoted angles" >&2; exit 2; }
            ready_pose=$2; shift ;;
        --motion-data)
            [[ $# -ge 2 && -n "$2" ]] || { echo '--motion-data needs a directory' >&2; exit 2; }
            motion_data=$2; shift ;;
        *) echo 'Usage: ./run_body.sh [--keyboard] [--hand dex3|external|inspire] [--motion-data DIR] [--ready-pose "29 angles"] [--check]' >&2; exit 2 ;;
    esac
    shift
done

interface=$(ip -4 -o addr show | awk '$4 ~ /^192\.168\.123\./ {print $2; exit}')
[[ -n "$interface" ]] || { echo 'G1 internal DDS interface (192.168.123.x) not found' >&2; exit 1; }

export LD_LIBRARY_PATH="$PWD/thirdparty/unitree_sdk2/thirdparty/lib/aarch64:/opt/onnxruntime/lib:$HOME/TensorRT/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
binary=target/release/g1_deploy_onnx_ref
policy=policy/low_latency
planner=planner/target_vel/V2/planner_sonic.onnx
[[ -x "$binary" ]] || { echo "Missing executable: $binary" >&2; exit 1; }
for file in "$policy/model_decoder.onnx" "$policy/model_encoder.onnx" "$policy/observation_config.yaml" "$planner"; do
    [[ -s "$file" ]] || { echo "Missing model/configuration: $file" >&2; exit 1; }
done
[[ -d "$motion_data" ]] || { echo "Missing motion directory: $motion_data" >&2; exit 1; }

command=("./$binary" "$interface" "$policy/model_decoder.onnx" "$motion_data"
    --obs-config "$policy/observation_config.yaml"
    --encoder-file "$policy/model_encoder.onnx"
    --planner-file "$planner"
    --hand "$hand_type" --input-type "$input_type" --output-type zmq
    --zmq-host 127.0.0.1 --zmq-port 5556)
if [[ -n "$ready_pose" ]]; then
    validation=$("./$binary" --check-ready-pose "$ready_pose") || exit 1
    [[ "$validation" == READY_POSE_VALID_NO_CONTROL ]] || {
        echo 'SONIC executable needs the ready-pose update; rebuild on G1 before using a custom pose.' >&2; exit 1;
    }
    command+=(--ready-pose "$ready_pose")
fi
if "$check"; then
    printf 'G1 body configuration OK: interface=%s input=%s\n' "$interface" "$input_type"
    printf '%q ' "${command[@]}"
    printf '\nCheck only; no robot commands sent.\n'
    exit 0
fi
exec "${command[@]}"
