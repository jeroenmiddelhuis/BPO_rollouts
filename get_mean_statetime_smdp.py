import smdp, smdp_composite
from heuristic_policies import greedy_policy
import mdp
import numpy as np

for config in ['composite']:
    nr_replications = 100
    times = []
    for i in range(nr_replications):
        print(i)
        if config == 'composite':
            env = smdp_composite.SMDP_composite(2500)
        else:
            env = smdp.SMDP(2500, config)
        done = False
        prev_time = 0
        while not done:     
            action = greedy_policy(env)
            
            state, reward, done, _, _ = env.step(action)
            time = env.total_time
            #print(np.round(time, 2), np.round(time - prev_time, 2))
            times.append(time - prev_time)
            prev_time = time
    print(config, np.mean(times), np.std(times))
