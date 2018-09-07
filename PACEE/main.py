import gym
from config import load_config
from modules import DDPG, OrnsteinUhlenbeckNoise, to_scalar
import numpy as np
from itertools import count
import csv

cf = load_config('config/baseline.py')
env = gym.make('InvertedPendulum-v2') #Humanoid
out = open('InvertedPendulum(2).csv', 'a', newline='')
csv_writer = csv.writer(out, dialect='excel')
cf.state_dim = env.observation_space.shape[0]
cf.action_dim = env.action_space.shape[0]
cf.scale = float(env.action_space.high[0])

print(' State Dimensions: ', cf.state_dim)
print(' Action Dimensions: ', cf.action_dim)
print('Action low: ', env.action_space.low)
print('Action high: ', env.action_space.high)

noise_process = OrnsteinUhlenbeckNoise(cf)
model = DDPG(cf)
for i in range(cf.num_agent):
    model.copy_weights(model.actor[i], model.actor_target[i])
model.copy_weights(model.critic, model.critic_target)

losses = []
total_timesteps = 0

print('num_agent:',cf.num_agent)
for epi in range(cf.max_episodes):
    s_t = env.reset()
    #noise_process.reset()
    avg_reward = 0
    trace=[]
    for t in range(1000):
        T = int(t / cf.stage)
        signal,a_t = model.sample_action(s_t,T)
        ex_at=model.sample_ex(signal)
        a_t = a_t + (ex_at-a_t)*1e-5 #(a_t-ex_at)*1e-5,还要设置一下200,600,1000,还要考虑是否每个阶段要一直训练，当其这个loss（critic也分开）基本不变之后可以不训练这个ａｃｔｏｒ了
        #由于第一个ａｃｔｏｒ可能会过拟合，因此，我们给他的一个更慢的学习绿，而后面的两个ａｃｔｏｒ给其越来越大的学习率（都可以很小），ｃｒｉｔｉｃ的学习绿也可以小一点
        #网络是否可以参数多一点,有时不训练，因为训练效果并不佳啊,当比如Ｔ＝２，那么０，１两个ａｃｔｏｒ可以选择不训练
        s_tp1, r_t, done, info = env.step(a_t)
        trace.append((signal, a_t, r_t, s_tp1, float(done == False)))
        if model.buffer[0].len <= cf.replay_start_size:
            for i in range(cf.num_agent):
                model.buffer[i].add(s_t, a_t, r_t, s_tp1, float(done == False))
        else:
            model.buffer[T].add(s_t, a_t, r_t, s_tp1, float(done == False))
        avg_reward += r_t

        if done:
            break
        else:
            s_t = s_tp1

        if model.buffer[0].len >= cf.replay_start_size:
            _loss_a, _loss_c = model.train_batch(T)
            losses.append(to_scalar([_loss_a, _loss_c]))
    if epi<=10:
        model.Buffer.add_episode(trace,avg_reward)
    elif avg_reward>=model.Buffer.aver_R():
        model.Buffer.add_episode(trace,avg_reward)
    if len(losses) > 0:
        total_timesteps += t
        avg_loss_a, avg_loss_c = np.asarray(losses)[-100:].mean(0)
        print(
            'Episode {}: actor loss: {} critic loss: {}\
            episode_reward: {} episode_num_step: {} tot_timesteps: {}'.format(
             epi, avg_loss_a, avg_loss_c, avg_reward, t, total_timesteps
            ))
        csv_writer.writerow([epi, avg_loss_a, avg_loss_c, avg_reward, t, total_timesteps])

    if (epi + 1) % 1000 == 0:
        model.save_models()
print('Completed training!')
