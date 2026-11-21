#!/usr/bin/env bash
export PATH=/home/volmarr/.nix-profile/bin:$PATH
DRV=/nix/store/p0lg8y71b44dschxdz5kbf8bdp7l4sh8-gufo-0-unstable-4938016.drv
LOG=/home/volmarr/src/gemma4-gufo-extension/campaign/kv-harness/results/kv13
nix --extra-experimental-features 'nix-command flakes' develop "$DRV" --command bash -c '
export PKG_CONFIG_PATH=/nix/store/qi0whsif4mi6a0d88d51qk4fa6bgns89-libsndfile-1.2.2-dev/lib/pkgconfig:$PKG_CONFIG_PATH
cd /home/volmarr/src/gufo-dev
cmake --build build/gpu-test --target gemma4_continuation_test -j 8 || exit 12
export GUFO_GEMMA4_MODEL=/home/volmarr/models/gufo/gemma-4-26B-A4B-it-heretic.Q8_0.gguf
ctest --test-dir build/gpu-test -R "gemma4.continuation" --output-on-failure || exit 13
' > $LOG/continuation.log 2>&1
echo EXIT=$? > $LOG/continuation.done
