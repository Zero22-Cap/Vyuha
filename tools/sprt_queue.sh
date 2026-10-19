#!/bin/bash
# Run SPRT matches one after another: a candidate setting against the default, same exe and net.
# Usage: nohup bash tools/sprt_queue.sh <exe> <queue file> > <log> 2>&1 &
# Queue file: one test per line, "<name> <option.X=v> [option.Y=w ...]"; lines starting with # are
# skipped; a test whose result file exists is skipped (the queue can be restarted). Before each test
# the script waits while <queue>.PAUSE exists and ends if <queue>.STOP exists. Results:
# <queue dir>/sprt-<name>.log; summary lines are appended to <queue>.results.
EXE=$1; Q=$2; DIR=$(dirname "$Q")
FC=E:/Chess/tools/fastchess/fastchess-windows-x86-64/fastchess.exe
BOOK=E:/Chess/lab/book/balanced-openings.epd
TC=${TC:-10+0.1}; CONC=${CONC:-12}; ROUNDS=${ROUNDS:-10000}; ELO0=${ELO0:-0}; ELO1=${ELO1:-4}
while read -r name opts; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    out=$DIR/sprt-$name.log
    [ -f "$out" ] && grep -q "SPRT.*\(H0\|H1\)\|Finished match" "$out" && continue
    while [ -f "$Q.PAUSE" ] && [ ! -f "$Q.STOP" ]; do sleep 30; done
    [ -f "$Q.STOP" ] && { echo "$(date '+%m-%d %H:%M') STOP before $name" >> "$Q.results"; exit 0; }
    echo "$(date '+%m-%d %H:%M') start $name: $opts" >> "$Q.results"
    "$FC" -engine cmd=$EXE name=$name $opts -engine cmd=$EXE name=base \
        -each tc=$TC option.Threads=1 option.Hash=32 "option.SyzygyPath=<empty>" \
        -openings file=$BOOK format=epd order=random -games 2 -rounds $ROUNDS -repeat -concurrency $CONC -recover \
        -sprt elo0=$ELO0 elo1=$ELO1 alpha=0.05 beta=0.05 -pgnout file=$DIR/sprt-$name.pgn > "$out" 2>&1
    echo "$(date '+%m-%d %H:%M') $name: $(grep -E '^Elo' "$out" | tail -1) | $(grep -E '^Games' "$out" | tail -1) | $(grep -iE 'H0 was accepted|H1 was accepted|LLR' "$out" | tail -1)" >> "$Q.results"
done < "$Q"
echo "$(date '+%m-%d %H:%M') queue done" >> "$Q.results"
