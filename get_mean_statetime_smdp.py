import smdp, smdp_composite
import mdp
import numpy as np

for config in ['low_utilization']:
    nr_replications = 10
    times = []
    for i in range(nr_replications):
        print(i)
        if config == 'composite':
            env = smdp_composite.SMDP_composite(2500)
        else:
            env = smdp.SMDP(5000, config)
        done = False
        prev_time = 0
        while not done:     
            action = smdp.greedy_policy(env)
            
            state, reward, done, _, _ = env.step(action)
            time = env.total_time
            #print(np.round(time, 2), np.round(time - prev_time, 2))
            times.append(time - prev_time)
            prev_time = time
    print(config, np.mean(times), np.std(times))
