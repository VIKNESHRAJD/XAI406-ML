# XAI406-ML
def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n

    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []
     
    for ep in range(num_episodes):
        state = env.reset()
        tot_rew = 0

        if eps > 0.01:
            eps -= eps_decay

        done = False
        while not done:
            action = eps_greedy(Q, state, eps)
            next_state, rew, done, _ = env.step(action)
        
            Q[state][action] = Q[state][action] + lr * (rew + gamma * np.max(Q[next_state]) - Q[state][action])

            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episode:{:5d} Eps:{:2.4f} Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)

    return Q

    if __name__ == '__main__':
   env = gym.make('Taxi-v3')
   print("Q-Learning")
   Q_learning = Q_learning(env, lr=.1, num_episodes= 5000, eps= 0.4 , gamma = 0.95, eps_decay=0.001)
