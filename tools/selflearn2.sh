#!/bin/bash
# Self-learning loop driver, pipelined: the CPU starts the next generation's self-play as soon as
# one generation has its positions, while the GPU trains on them and the matches run. Self-play
# runs as spgen.exe at below-normal priority; matches use sp.exe at normal priority.
# Control files: $SP/PAUSE halts everything (self-play at once, other steps before they start)
# until removed; $SP/STOP ends the run. Everything is logged to $SP/driver.log.
# Usage: START_GEN=8 BEST_NET=<net> nohup bash tools/selflearn2.sh > $SP/driver.out 2>&1 &
SP=${SP:-C:/scdata/sp}
NETS=${NETS:-E:/Chess/Vyuha/nets/sp}
FC=E:/Chess/tools/fastchess/fastchess-windows-x86-64/fastchess.exe
SF=E:/Chess/lab/src/sf-lab2/src/stockfish.exe
TRAIN=E:/Chess/Vyuha/train/train.py
MIX=E:/Chess/Vyuha/tools/mix_records.py
WINDOW=${WINDOW:-8}              # generations in the training data
PASSES=${PASSES:-10}             # passes over that data
MAXPOS=${MAXPOS:-1200000000}     # cap on positions seen per training
LR=${LR:-5e-4}
BOOK=E:/Chess/lab/book/balanced-openings.epd
TARGET=${TARGET:-12000000}       # positions per generation
ROUNDS=${ROUNDS:-100}            # gating match rounds (2 games each)
GEN_ANCHOR=${GEN_ANCHOR:-3000}   # every new net plays Stockfish at this UCI_Elo (0 = off)
ANCHOR_ROUNDS=${ANCHOR_ROUNDS:-50}
MAX_GENS=${MAX_GENS:-0}          # 0 = no limit
MATCH_EXE=${MATCH_EXE:-$SP/sp.exe}   # engine binary for matches (not spgen.exe)
LOG=$SP/driver.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
recs() { echo $(( $(stat -c %s "$1" 2>/dev/null || echo 0) / 72 )); }
sp_running() { tasklist //FI "IMAGENAME eq spgen.exe" 2>/dev/null | grep -qi spgen.exe; }
sp_stop() { taskkill //IM spgen.exe //F > /dev/null 2>&1; sleep 3; }

sp_start() {  # gen net
    (cd $SP && printf 'setoption name Threads value 16\nsetoption name Hash value 4096\nsetoption name SyzygyPath value <empty>\nsetoption name EvalFile value %s\nisready\nselfplay %s/g%s.bin 5000000 5000 12 8\nquit\n' \
        "$2" "$SP" "$1" | ./spgen.exe >> $SP/g$1.log 2>&1 &)
    sleep 5
    powershell -NoProfile -Command "Get-Process spgen -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" > /dev/null 2>&1
}

check_stop() { if [ -f $SP/STOP ]; then sp_stop; log "STOP file: run ended ($1)"; exit 0; fi; }

# PAUSE: stop self-play, wait until the file is removed, then resume self-play of generation $1
pause_check() {
    check_stop "$2"
    if [ -f $SP/PAUSE ]; then
        sp_stop; log "PAUSE file: paused ($2)"
        while [ -f $SP/PAUSE ] && [ ! -f $SP/STOP ]; do sleep 30; done
        check_stop "during pause"
        log "PAUSE removed: resuming"
        sp_start $1 "$best"; log "gen $1 self-play resumed at $(recs $SP/g$1.bin) records with $(basename $best)"
    fi
}

cp $SP/sp.exe $SP/spgen.exe
g=${START_GEN:-1}
best=${BEST_NET:-$NETS/g0-latest.nnue}
done_gens=0
log "pipelined driver start: gen $g, best $(basename $best), target $TARGET"
sp_start $g "$best"; log "gen $g self-play started with $(basename $best)"
while true; do
    prev=$((g - 1)); next=$((g + 1))
    until [ "$(recs $SP/g$g.bin)" -ge "$TARGET" ]; do
        sleep 30
        pause_check $g "gen $g self-play at $(recs $SP/g$g.bin) records"
        if ! sp_running; then sp_start $g "$best"; log "gen $g self-play restarted at $(recs $SP/g$g.bin) records"; fi
    done
    sp_stop
    log "gen $g self-play stopped at $(recs $SP/g$g.bin) records"
    sp_start $next "$best"; log "gen $next self-play started with $(basename $best) (pipelined)"

    pause_check $next "before training gen $g"
    # the last WINDOW generations, mixed so every shard holds a slice of each (measured 2026-10-19:
    # 8 generations, 10 passes, lr 5e-4 gave +111 over the two-generation recipe)
    files=""; n=0
    for ((i = g; i >= 1 && i > g - WINDOW; i--)); do
        if [ -f $SP/g$i.bin ]; then files="$SP/g$i.bin,$files"; n=$((n + $(recs $SP/g$i.bin))); fi
    done
    rm -f $SP/t${g}_*.bin $SP/t$g.done
    python $MIX ${files%,} $SP/t$g 10 >> $LOG
    total=$((n * PASSES)); [ $total -gt $MAXPOS ] && total=$MAXPOS
    python -u $TRAIN --data $SP/t$g --out $NETS --tag g$g --total $total --export_every $total --lr $LR \
        --resume ${best%.nnue}.pt >> $SP/train-g$g.log 2>&1
    log "gen $g trained on $n positions, $total seen ($(tail -1 $SP/train-g$g.log))"
    if [ ! -f $NETS/g$g-latest.nnue ]; then sp_stop; log "gen $g: training produced no net, stopping"; exit 1; fi

    pause_check $next "before match gen $g"
    "$FC" -engine cmd=$MATCH_EXE name=g$g option.EvalFile=$NETS/g$g-latest.nnue -engine cmd=$MATCH_EXE name=best option.EvalFile=$best \
        -each tc=5+0.05 option.Threads=1 option.Hash=32 "option.SyzygyPath=<empty>" \
        -openings file=$BOOK format=epd order=random -games 2 -rounds $ROUNDS -repeat -concurrency 14 -recover \
        -pgnout file=$SP/m-g$g-vs-best.pgn > $SP/m-g$g-vs-best.log 2>&1
    if grep -q "FAILED to load" $SP/m-g$g-vs-best.log; then sp_stop; log "gen $g: an engine could not load its net, stopping"; exit 1; fi
    res=$(grep -E "Elo:" $SP/m-g$g-vs-best.log | tail -1)
    log "gen $g vs $(basename $best): $res"
    elo=$(echo "$res" | sed -n 's/^Elo: *\(-\?[0-9.]*\).*/\1/p')
    if [ -n "$elo" ] && awk "BEGIN{exit !($elo >= -10)}"; then
        best=$NETS/g$g-latest.nnue; log "best is now g$g"
        sp_stop; sp_start $next "$best"; log "gen $next self-play switched to $(basename $best) at $(recs $SP/g$next.bin) records"
    fi

    if [ "$GEN_ANCHOR" != 0 ]; then
        pause_check $next "before anchor gen $g"
        "$FC" -engine cmd=$MATCH_EXE name=g$g option.EvalFile=$NETS/g$g-latest.nnue \
            -engine cmd=$SF name=sf$GEN_ANCHOR option.UCI_LimitStrength=true option.UCI_Elo=$GEN_ANCHOR option.MateSingular=false \
            -each tc=5+0.05 option.Threads=1 option.Hash=32 "option.SyzygyPath=<empty>" \
            -openings file=$BOOK format=epd order=random -games 2 -rounds $ANCHOR_ROUNDS -repeat -concurrency 14 -recover \
            -pgnout file=$SP/anchor-g$g.pgn > $SP/anchor-g$g.log 2>&1
        log "gen $g vs Stockfish UCI_Elo $GEN_ANCHOR: $(grep -E 'Elo:' $SP/anchor-g$g.log | tail -1) | $(grep -E 'Games:' $SP/anchor-g$g.log | tail -1)"
    fi
    check_stop "after gen $g"
    done_gens=$((done_gens + 1))
    if [ "$MAX_GENS" != 0 ] && [ "$done_gens" -ge "$MAX_GENS" ]; then sp_stop; log "MAX_GENS reached, driver done"; exit 0; fi
    g=$next
done
