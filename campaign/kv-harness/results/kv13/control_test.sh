#!/usr/bin/env bash
set -x
export PATH=/home/volmarr/.nix-profile/bin:$PATH
DRV=/nix/store/p0lg8y71b44dschxdz5kbf8bdp7l4sh8-gufo-0-unstable-4938016.drv
LOG=/home/volmarr/src/gemma4-gufo-extension/campaign/kv-harness/results/kv13
cd /home/volmarr/src/gemma4-gufo-extension && git stash push -- engine.cpp engine.hpp && git diff --stat
nix --extra-experimental-features 'nix-command flakes' develop "$DRV" --command bash -c '
set -x
export PKG_CONFIG_PATH=/nix/store/qi0whsif4mi6a0d88d51qk4fa6bgns89-libsndfile-1.2.2-dev/lib/pkgconfig:$PKG_CONFIG_PATH
cd /home/volmarr/src/gufo-dev
cmake --build build/gpu-test --target gemma4_target_test -j 8 || exit 12
GUFO_GEMMA4_MODEL=/home/volmarr/models/gufo/gemma-4-26B-A4B-it-heretic.Q8_0.gguf ctest --test-dir build/gpu-test -R "gemma4.target" --output-on-failure || exit 13
' > $LOG/control.log 2>&1
echo EXIT=$? > $LOG/control.done
cd /home/volmarr/src/gemma4-gufo-extension && git stash pop && git diff --stat
