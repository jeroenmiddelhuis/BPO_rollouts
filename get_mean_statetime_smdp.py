import smdp
import mdp
import numpy as np

for config in ['slow_server', 'low_utilization', 'high_utilization', 'n_system', 'parallel', 'down_stream', 'single_activity']:
    nr_replications = 500
    times = []
    for i in range(nr_replications):
        env = smdp.SMDP(3000, config)
        done = False
        prev_time = 0
        while not done:     
            action = mdp.random_policy(env)
            
            state, reward, done, _, _ = env.step(action)
            time = env.total_time
            times.append(time - prev_time)
            prev_time = time

    print(config, np.mean(times), np.std(times))
