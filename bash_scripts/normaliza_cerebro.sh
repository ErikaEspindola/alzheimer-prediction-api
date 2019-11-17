#!/bin/bash

caminho_flirt=$1
caminho_inp=$2
caminho_out=$3
caminho_ref=$4
$1 -in $caminho_inp -ref $caminho_ref -out $caminho_out -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear
