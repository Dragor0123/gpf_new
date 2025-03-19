
# Parameter settings for full-shot scenarios.
#----------------------------------------------------------------------------------------------------------------------------------------
#### EdgePred
## BBBP
#### GPF
python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset bbbp --tuning_type gpf --num_layers 1
#### GPF-plus
python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset bbbp --tuning_type gpf-plus --num_layers 1 --pnum 5
#----------------------------------------------------------------------------------------------------------------------------------------
## Tox21
### EdgePred
#### GPF
python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset tox21 --tuning_type gpf --num_layers 1

#### GPF-plus
python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset tox21 --tuning_type gpf-plus --num_layers 1 --pnum 10
#----------------------------------------------------------------------------------------------------------------------------------------
## ToxCast
### EdgePred
#### GPF
python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset toxcast --tuning_type gpf --num_layers 2
#### GPF-plus
python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset toxcast --tuning_type gpf-plus --num_layers 3 --pnum 5

#----------------------------------------------------------------------------------------------------------------------------------------
## SIDER
### EdgePred

#### GPF

python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset sider --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset sider --tuning_type gpf-plus --num_layers 3 --pnum 10

#----------------------------------------------------------------------------------------------------------------------------------------
## ClinTox
### EdgePred
#### GPF

python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset clintox --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset clintox --tuning_type gpf-plus --num_layers 3 --pnum 20

#----------------------------------------------------------------------------------------------------------------------------------------
## MUV
#
### EdgePred

#### GPF

python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset muv --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset muv --tuning_type gpf-plus --num_layers 1 --pnum 20

#----------------------------------------------------------------------------------------------------------------------------------------
## HIV

### EdgePred

#### GPF

python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset hiv --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_full_shot.py --model_file pretrained_models/edgepred.pth --dataset hiv --tuning_type gpf-plus --num_layers 2 --pnum 20

#----------------------------------------------------------------------------------------------------------------------------------------
