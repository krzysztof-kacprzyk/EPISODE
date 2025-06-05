# A bash script to run all baselines on all datasets 

for dataset in "SIR" "real_pharma" "pk" "bike" "tumor"
do
    for model in "SINDy-5" "SINDy-20" "WSINDy-5" "WSINDy-20"
    do
        python run_scripts/run_baselines.py $dataset $model 0 30 --trial --name Main
    done
    
    for model in "NODE" "ANODE" "LatentODE"
    do
        python run_scripts/run_baselines.py $dataset $model 0 20 --trial --name Main
    done

    python run_scripts/generate_composition_scores.py $dataset 0 --trial
    python run_scripts/run_psode.py $dataset 0 --no-dtw --name Main --trial
    python run_scripts/run_psode.py $dataset 0 --no-dtw --name Main --trial --biases
done


for dataset in "SIR" "real_pharma" "pk" "bike" "tumor"
do
    for model in "SINDy-5" "SINDy-20" "WSINDy-5" "WSINDy-20"
    do
        python run_scripts/run_baselines.py $dataset $model 0 30 --name Main
    done
    
    for model in "NODE" "ANODE" "LatentODE"
    do
        python run_scripts/run_baselines.py $dataset $model 0 20 --name Main
    done

    python run_scripts/generate_composition_scores.py $dataset 0
    python run_scripts/run_psode.py $dataset 0 --no-dtw --name Main
    python run_scripts/run_psode.py $dataset 0 --no-dtw --name Main --biases
done
