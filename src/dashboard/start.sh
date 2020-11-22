#!/bin/bash

killbg() {
    for p in "${pids[@]}" ; do
        kill "$p";
    done
}
trap killbg EXIT
pids=()
cd backend/chart_view
./lsl-app.py &
pids+=($!)
echo $pids
cd ../../prod-frontend
npm start
