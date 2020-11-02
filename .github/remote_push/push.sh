#!/usr/bin/env sh
set -eu

mkdir ~/.ssh
echo "$INPUT_PRIVATE_SSH_KEY" > ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa

export GIT_SSH_COMMAND="ssh -v -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -l $INPUT_REMOTE_USERNAME"

echo "Adding remote repo $INPUT_REMOTE_URL"
git remote add repo "$INPUT_REMOTE_URL"

echo "Current branch: "`git rev-parse --abbrev-ref HEAD`

echo "Pushing to $INPUT_REMOTE_BRANCH"
git push repo "$INPUT_REMOTE_BRANCH"
