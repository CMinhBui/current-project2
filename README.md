# A2C_sharing_experience
Multi-task learning with Advantage Actor Critic  and sharing experience 

## To run the code

- Recommend running command: `python train.py --logname log_1 --map_index 4 --num_task 2 --share_exp 0 1 --num_episode 24 --num_iters 100 --num_epochs 3000`
- List args:
```
	--logname [LOGNAME]				Name of the log folder
	--num_tests [NUM_TESTS]			Number of test to run
	--map_index [MAP_INDEX]			Index of map
	--num_task [NUM_TASK]			Number of tasks to train on
	--share_exp SHARE_EXP [SHARE_EXP ...]		Whether to turn on sharing samples on training
	--num_episode [NUM_EPISODE]		Number of episodes to sample in each epoch
	--num_iters [NUM_ITERS]			Number of steps to be sampled in each episode
	--lr [LR]             			Learning rate
	--use_laser [USE_LASER]			Whether to use laser as input observation instead of
									one-hot vector
	--use_gae [USE_GAE]   			Whether to use generalized advantage estimate
	--num_epochs [NUM_EPOCHS]		Number of epochs to train
	--plot_model [PLOT_MODEL]		Plot interval
	--noise_argmax [NOISE_ARGMAX]	Whether touse noise argmax in action sampling
	--save_model [SAVE_MODEL]		Saving interval
	--gpu_frac [GPU_FRAC]			Fraction of gpu usage
```
### Adding new map
To add new map, add code in `env/map.py` and run `python generate_start_positions.py --map_indexs <list of new map indexs>`, ex: `python generate_start_positions.py --map_indexs 3 4 5`
## Network:
- Sử dụng network NewA2C với một số thay đổi về sharing experience:
	- Bỏ cách tính important weight cũ, tính important weight thông qua network: `self.ratio = tf.exp(self.task_neg_log_prob - self.actor.neg_log_prob)`
		- Với `self.task_neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.task_logits, labels = self.actor.actions)`
		- `self.task_logits` là policy logits của task gốc đã generate ra sample đầu vào
		- Từ đó nếu một actor sử dụng sample do chính nó generate thì `ratio = 1` còn sử dụng share_sample của task khác thì `ratio != 1`
	- Hàm loss của actor được tính bằng: `self.policy_loss = tf.reduce_mean(tf.maximum(self.policy_loss1, self.policy_loss2))`
		- Với `self.policy_loss1 = self.actor.policy_loss * self.ratio`
		- Và `self.policy_loss2 = self.actor.policy_loss * tf.clip_by_value(self.ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)`
		- Và `policy_loss` gốc của actor được tính bằng `actor.policy_loss = tf.reduce_mean(actor.neg_log_prob * actor.advantages)`
	- Cách tính này cho kết quả sharing experience luôn hội tụ nhanh hơn, và hội tụ ổn định hơn
	- Có thể là do với cách tính này, important weight có đóng góp vào gradient của hàm loss policy chứ không chỉ là một hằng số nhân với advantage như cũ nữa
	- Các thông tin được 2 task share với nhau:
		- `share_states`
		- `share_actions`
		- `share_advantanges` (giữ nguyên advantages chứ không thêm trọng số iw như trước)
		- `share_logits` (policy logits của task gốc để tính iw trong network)
## Experiments:
- Logs của các thí nghiệm được lưu trong `logs/test_newa2c_ver2`
- Plot policy được lưu trong `plot/test_newa2c_ver2`
- Giải thích về tên các đường plot:
	- `num_task_2-num_episode_24-num_iters_50-lr_0.005-use_gae`: 2 task, 24 thread sample, rollout length 50, learning rate 0.005, sử dụng GAE, không sử dụng share experiences
	- `num_task_2-share_exp-num_episode_24-num_iters_50-lr_0.005-use_gae`: như trên nhưng sử dụng sharing

- Các thí nghiệm đã chạy:
	- Thí nghiệm share experience và không share, mỗi thí nghiệm chạy 5 test với:
		- 24 thread, mỗi thread 100 và 50 steps
		- 16 thread, mỗi thread 100 và 50 steps
		- 10 thread, mỗi thread 100 và 50 steps
		- 8 thread, mỗi thread 100 và 50 steps
	- Tất cả các thí nghiệm sharing đều cho kết quả tốt hơn không share về tốc độ hội tụ và độ ổn định khi hội tụ


