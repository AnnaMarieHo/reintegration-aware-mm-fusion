# for missing_rate in 0.1 0.2 0.3 0.4 0.5; do
for missing_rate in 0.3; do
    # for fed_alg in fed_opt; do
    for fed_alg in fed_avg; do
        # taskset 100 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.01 --global_learning_rate 0.005 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --en_missing_modality --missing_modailty_rate $missing_rate
        # taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.01 --global_learning_rate 0.005 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --en_missing_modality --missing_modailty_rate $missing_rate
    # taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.01 --global_learning_rate 0.005 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --en_missing_modality --missing_modailty_rate $missing_rate --availability_process markov --availability_sidecar_path ../../output/simulation_feature/meld/availability_markov_005_005_seed42.pkl 
    # taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.01 --global_learning_rate 0.005 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --en_missing_modality --missing_modailty_rate $missing_rate --availability_process markov --availability_sidecar_path ../../output/simulation_feature/meld/availability_markov_005_005_seed42_reint.pkl 
    # taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.01 --global_learning_rate 0.005 --num_epochs 5 --en_att --att_name fuse_base --fed_alg $fed_alg --en_missing_modality --missing_modailty_rate $missing_rate --availability_process markov --availability_sidecar_path ../../output/simulation_feature/meld/availability_markov_005_005_seed42_reint.pkl 
    taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.01 --global_learning_rate 0.005 --num_epochs 5 --en_att --att_name fuse_base --fed_alg $fed_alg --en_missing_modality --missing_modailty_rate $missing_rate --availability_process markov --availability_sidecar_path ../../output/simulation_feature/meld/availability_markov_005_005_seed42_stable.pkl 
    done
done

#2026-02-25 20:21:56