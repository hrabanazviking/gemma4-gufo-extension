#!/usr/bin/env bash
export PATH=/home/volmarr/.nix-profile/bin:$PATH
DRV=/nix/store/p0lg8y71b44dschxdz5kbf8bdp7l4sh8-gufo-0-unstable-4938016.drv
LOG=/home/volmarr/src/gemma4-gufo-extension/campaign/kv-harness/results/kv13
nix --extra-experimental-features 'nix-command flakes' develop "$DRV" --command bash -c '
export PKG_CONFIG_PATH=/nix/store/qi0whsif4mi6a0d88d51qk4fa6bgns89-libsndfile-1.2.2-dev/lib/pkgconfig:$PKG_CONFIG_PATH
cd /home/volmarr/src/gufo-dev
cmake --build build/gpu-test --target gemma4_kv13_movekey_test gemma4_kv13_machinery_test -j 8 || exit 12
export GUFO_GEMMA4_MODEL=/home/volmarr/models/gufo/gemma-4-26B-A4B-it-heretic.Q8_0.gguf
./build/gpu-test/extensions/build/gemma4/build/gemma4_kv13_movekey_test || exit 14
find build/gpu-test -name gemma4_kv13_machinery_test -type f
./build/gpu-test/extensions/build/gemma4/build/gemma4_kv13_machinery_test || exit 15
ctest --test-dir build/gpu-test -R "gemma4.kv13" --output-on-failure || exit 16
' > $LOG/rerun.log 2>&1
echo EXIT=$? > $LOG/rerun.done
