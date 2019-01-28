import gym
import numpy as np
Gamma = 0.9 
def max_dict(d):
	# returns the argmax (key) and max (value) from a dictionary
	# put this into a function since we are using it so often
	max_key = None
	max_val = float('-inf')
	for k, v in d.items():
		if v > max_val:
			max_val = v
			max_key = k
	return max_key, max_val
def epsilon_action(a,eps,env):
	p = np.random.random()
	if p < (1-eps):
		return a
	else:
		return np.random.choice(env.env.nA)
def play_game(env,policy,eps):
	s = env.reset()
	a = epsilon_action(policy[s],eps,env)
	state_actions_rewards = [(s,a,0)]
	while True:
		state, reward, done, _ = env.step(a)
		#print ("inside loop")
		if done:
			state_actions_rewards.append((state,None,reward))
			break
		else:
			a=epsilon_action(policy[state],eps,env)
			state_actions_rewards.append((state,a,reward))
	G = 0
	state_actions_returns = []
	first = True
	for s, a, r in reversed(state_actions_rewards):
		if first:
			first = False
		else:
			state_actions_returns.append((s,a,G))
		G = r + Gamma*G
	state_actions_returns.reverse()
	return state_actions_returns
def monte_carlo(env):
	policy = {}
	for s in range(env.env.nS):
		policy[s] = np.random.choice(env.env.nA)
	Q = {}
	returns = {}
	states= [a for a in range(env.env.nS)]
	for s in states:
		Q[s] = {}
		for a in range(env.env.nA):
			Q[s][a] = 0
			returns[(s,a)] = []
	deltas = []
	eps = 0.4
	for epi in range(10000):
		biggest_change = 0
		state_actions_returns = play_game(env,policy,eps)
		seen_state_action_pairs = set()
		for s,a,g in state_actions_returns:
			sa = (s, a)
			if sa not in seen_state_action_pairs:
				returns[sa].append(g)
				old_q = Q[s][a]
				Q[s][a] = np.mean(returns[sa])
				biggest_change = max(biggest_change, np.abs(old_q-Q[s][a]))
				seen_state_action_pairs.add(sa)
		deltas.append(biggest_change)
		for s in policy.keys():
			a, _ =max_dict(Q[s])
			policy[s]=a
	V = {}
	for s in policy.keys():
		V[s] = max_dict(Q[s])[1]
	return V,policy,deltas


	return state_actions_returns
def run_prime_policy(policy,env):
	s = env.reset()
	while True:
		a = policy[s] 
		env.render()
		s,r,done,_ = env.step(a)
		
		#print (s,policy[s])
		if done:
			env.render()
			break
env = gym.make('FrozenLake-v0')
V, policy, deltas = monte_carlo(env)
#print ("deltas")
#print (deltas)
#print ("policy")
#print (policy)
run_prime_policy(policy,env)
