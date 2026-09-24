# EVAL_DIR=$SAMPLE_DIR/eval${START}_${END}
eval "$(conda shell.bash hook)"
conda activate $1

"${@:2}"