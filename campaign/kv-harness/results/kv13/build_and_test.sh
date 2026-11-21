#!/usr/bin/env bash
set -x
export PATH=/home/volmarr/.nix-profile/bin:$PATH
DRV=/nix/store/p0lg8y71b44dschxdz5kbf8bdp7l4sh8-gufo-0-unstable-4938016.drv
LOG=/home/volmarr/src/gemma4-gufo-extension/campaign/kv-harness/results/kv13
nix --extra-experimental-features 'nix-command flakes' develop "$DRV" --command bash -c '
set -x
cd /home/volmarr/src/gufo-dev
rm -rf build/gpu-test
cmake --preset gpu-test -DGUFO_EXTENSION_DIRS=/home/volmarr/src/gemma4-gufo-extension -DGUFO_BUILD_TOOLS=OFF || exit 11
cmake --build build/gpu-test --target gemma4_kv13_movekey_test gemma4_kv13_machinery_test gemma4_kv_cache_format_test gemma4_target_test -j 8 || exit 12
GUFO_GEMMA4_MODEL=/home/volmarr/models/gufo/gemma-4-26B-A4B-it-heretic.Q8_0.gguf ctest --test-dir build/gpu-test -R "gemma4.(kv13_movekey|kv13_machinery|kv_cache_format|target)" --output-on-failure || exit 13
' > $LOG/build.log 2>&1
echo EXIT=$? > $LOG/build.done
