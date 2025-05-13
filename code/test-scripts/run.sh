#!/bin/bash
REPO="path-ro-repo"
MONITOR="path-to-rcssmonitor"
SERVER="path-to-rcssserver"

$SERVER/src/rcssserver server::coach=on & (
sleep 2; $MONITOR/src/rcssmonitor ) & (
sleep 1; julia $REPO/code/test-scripts/trainer_and_players.jl ) & (
sleep 60; pkill rcssserver; pkill trainer_and_players.jl; pkill rcssmonitor )
