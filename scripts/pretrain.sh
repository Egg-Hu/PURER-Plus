python main.py \
--multigpu 5 \
--gpu 0 \
--dataset cifar100 \
--testdataset cifar100 \
--val_interval 2000 \
--save_interval 2000 \
--episode_batch 4 \
--way_train 5 \
--num_sup_train 1 \
--num_qur_train 15 \
--way_test 5 \
--num_sup_test 1 \
--num_qur_test 15 \
--backbone conv4 \
--episode_train 240000 \
--episode_test 600 \
--start_id 1 \
--inner_update_num 5 \
--test_inner_update_num 10 \
--inner_lr 0.01 \
--outer_lr 0.001 \
--candidate_size 13 \
--method purer \
--teacherMethod maml \
--inversionMethod deepinv \
--way_pretrain 5 \
--APInum 100 \
--pre_backbone conv4 \
--generate_interval 200 \
--generate_iterations 200 \
--Glr 0.01 \
--pretrain