#!/bin/bash
# DFPS vs alpha-beta, same exe, 1 thread each, tablebases off, balanced openings, paired colours.
# Usage: tools/dfps_match.sh <name> <tc> <rounds> [extra DFPS options as option.Name=Value ...]
# Example: tools/dfps_match.sh v1 10+0.1 500 option.DfpsMaxTreePly=1
set -e
NAME=$1; TC=$2; ROUNDS=$3; shift 3
FC=E:/Chess/tools/fastchess/fastchess-windows-x86-64/fastchess.exe
EXE=${EXE:-E:/Chess/Vyuha/vyuha-dev.exe}
OUT=C:/scdata/dfps/match-$NAME
"$FC" \
  -engine cmd=$EXE name=DFPS option.SearchMode=DFPS "$@" \
  -engine cmd=$EXE name=AB option.SearchMode=AB \
  -each tc=$TC option.Threads=1 option.Hash=64 "option.SyzygyPath=<empty>" \
  -openings file=E:/Chess/lab/book/balanced-openings.epd format=epd order=random \
  -games 2 -rounds "$ROUNDS" -repeat -concurrency ${CONC:-14} -recover \
  -pgnout file=$OUT.pgn > $OUT.log 2>&1
grep -E "Elo|Games|Ptnml|LOS" $OUT.log | tail -6
