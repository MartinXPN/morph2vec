#!/bin/bash

script_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
input_file_url="https://raw.githubusercontent.com/UniversalDependencies/UD_Armenian-ArmTDP/master/hy_armtdp-ud-train.conllu"
input_file=${script_dir}/${input_file_url##*/}

if [ ! -f ${input_file} ]
then
  wget -c ${input_file_url}
fi


# The contents of
# ./preprocess_train_evaluate.sh $input_file
corpus_conll=${input_file}_filtered
grep -Ev '^#' ${input_file} > ${corpus_conll}
corpus_processed=${corpus_conll}_processed

python preprocess.py --input_path ${corpus_conll} --output_path ${corpus_processed}


# The contents of
# ./train_evaluate.sh $corpus_processed
props="w+m"
model_path=${corpus_processed}_${props}
model_full_path=${model_path}.vec

script_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
training_dir=${script_dir}/fasttext
exe_name=fasttext

(cd ${training_dir}; [ -e ${exe_name} ] && rm ${exe_name}; make; ./${exe_name} skipgram -input ${corpus_processed} -output ${model_path} -props ${props} -lr 0.025 -dim 200 -ws 2 -epoch 5 -minCount 5 -neg 5 -loss ns -bucket 2000000 -thread 1 -t 1e-3 -lrUpdateRate 100)

# Do not evaluate the armenian model as there is no data yet
# ./evaluate.sh $model_full_path