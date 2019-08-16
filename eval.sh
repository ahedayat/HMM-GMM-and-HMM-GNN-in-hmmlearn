analysis=$1
optimizer='adam'
learning_rate=1e-3
criterion='cross_entropy'
num_epochs=20
start_epoch=0
num_workers=2
dataset='./datasets/spoken_arabic_digits/test'
preprocessing_path='./dnn_hmm_preprocess'
one_layer_fnn_neuron='200'
two_layer_fnn_neuron='100,200'
data_size=13
num_derivative=2
n_iter=10
states_num=3
gmm_mix_num=3
covariance_type='diag'
hmm_type='gaussian'


if [ $analysis -eq 'gmm_hmm' ]
then
    echo "GMM-HMM Analysis"
    !python hmm_eval.py --analysis $analysis \
                        --dataset $dataset \
                        --num_derivative $num_derivative

elif [ $analysis -eq 'dnn_hmm_1' ]
then
    echo "DNN-HMM ( 1-layer Feed-Forward )"
    !python dnn_hmm_eval.py --analysis $analysis \
                            --optimizer $optimizer \
                            --learning_rate $learning_rate \
                            --criterion $criterion \
                            --num_epochs $num_epochs \
                            --start_epoch $start_epoch \
                            --num_workers $num_workers \
                            --gpu \
                            --dataset $dataset \
                            --preprocessing_path $preprocessing_path \
                            --fnn $one_layer_fnn_neuron \
                            --preprocess
elif [ $analysis -eq 'dnn_hmm_2' ]
then
    echo "DNN-HMM ( 2-layer Feed-Forward )"
    !python dnn_hmm_eval.py --analysis $analysis \
                            --optimizer $optimizer \
                            --learning_rate $learning_rate \
                            --criterion $criterion \
                            --num_epochs $num_epochs \
                            --start_epoch $start_epoch \
                            --num_workers $num_workers \
                            --gpu \
                            --dataset $dataset \
                            --preprocessing_path $preprocessing_path \
                            --fnn $two_layer_fnn_neuron \
                            --preprocess
else
    echo "Not OK"
fi