import smdp, smdp_composite
import mdp
import numpy as np

for config in ['composite']:
    nr_replications = 500
    times = []
    for i in range(nr_replications):
        if config == 'composite':
            env = smdp_composite.SMDP_composite(2500)
        else:
            env = smdp.SMDP(2500, config)
        done = False
        prev_time = 0
        while not done:     
            action = smdp.random_policy(env)
            
            state, reward, done, _, _ = env.step(action)
            time = env.total_time
            times.append(time - prev_time)
            prev_time = time

    print(config, np.mean(times), np.std(times))
