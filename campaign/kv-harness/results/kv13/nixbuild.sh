#!/usr/bin/env bash
export PATH=/home/volmarr/.nix-profile/bin:$PATH
LOG=/home/volmarr/src/gemma4-gufo-extension/campaign/kv-harness/results/kv13
cd /home/volmarr/src/gemma4-gufo-extension
nix --extra-experimental-features 'nix-command flakes' build --no-link --print-out-paths . > $LOG/nixbuild.log 2>&1
echo EXIT=$? >> $LOG/nixbuild.log
