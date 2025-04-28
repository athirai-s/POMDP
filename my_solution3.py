import shlex

def read_state_weights(filename):
    
    with open(filename, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    parts = lines[1].split()
    num_states = int(parts[0])
    state_weights = {}
    total_weight = 0
    
    
    for line in lines[2:2 + num_states]:
        tokens = shlex.split(line)
        state = tokens[0]
        weight = float(tokens[1])
        state_weights[state] = weight
        total_weight += weight
        
    for s in state_weights:
        state_weights[s] /= total_weight
        
        
    return state_weights

def read_state_action_weights(filename, valid_states):
    
    with open(filename, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
                
    parts = lines[1].split()
    default_weight = float(parts[3])
    raw = {}
    
    for line in lines[2:]:
        tokens = shlex.split(line)
        s1, a, s2, w = tokens[0], tokens[1], tokens[2], float(tokens[3])
        if s1 not in valid_states or s2 not in valid_states:
            continue
        if (s1, a) not in raw:
            raw[(s1, a)] = {}
        raw[(s1, a)][s2] = w
        
    trans_prob = {}
    trans_default_prob = {}
    
    for (s1, a), dests in raw.items():
        total = sum(dests.values()) + default_weight * (len(valid_states) - len(dests))
        if s1 not in trans_prob:
            trans_prob[s1] = {}
        trans_prob[s1][a] = {}
        
        for s2 in dests:
            trans_prob[s1][a][s2] = dests[s2] / total
        trans_default_prob[(s1, a)] = default_weight / total if total > 0 else 0
        
        
    return trans_prob, trans_default_prob, default_weight


def read_state_observation_weights(filename, valid_states):
    
    with open(filename, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
                
    parts = lines[1].split()
    num_obs = int(parts[2])
    default_weight = float(parts[3])
    raw = {s: {} for s in valid_states}
    
    
    for line in lines[2:]:
        tokens = shlex.split(line)
        s, o, w = tokens[0], tokens[1], float(tokens[2])
        if s in valid_states:
            raw[s][o] = w
            
    obs_prob = {}
    obs_default_prob = {}
    
    for s in valid_states:
        total = sum(raw[s].values()) + default_weight * (num_obs - len(raw[s]))
        obs_prob[s] = {o: w / total for o, w in raw[s].items()}
        obs_default_prob[s] = default_weight / total if total > 0 else 0
        
        
    return obs_prob, obs_default_prob

def read_observation_actions(filename):
    
    with open(filename, 'r') as f:
        f.readline()
        count = int(f.readline().strip())
        obs, act = [], []
        
        for _ in range(count):
            tokens = shlex.split(f.readline())
            obs.append(tokens[0])
            act.append(tokens[1] if len(tokens) > 1 else None)
            
    return obs, act

def viterbi(obs, actions, states, init_p, trans_p, trans_def_p, trans_def_w, obs_p, obs_def_p):
    
    N = len(obs)
    V = [{} for _ in range(N)]
    back = [{} for _ in range(N)]
    
    for s in states:
        V[0][s] = init_p.get(s, 0.0) * obs_p.get(s, {}).get(obs[0], obs_def_p.get(s, 0.0))
        back[0][s] = None
        
        
    for t in range(1, N):
        act = actions[t - 1] if t - 1 < len(actions) and actions[t - 1] is not None else "N"
        for s in states:
            max_p, prev_st = 0, None
            for ps in states:
                if act in trans_p.get(ps, {}):
                    tp = trans_p[ps][act].get(s, trans_def_p.get((ps, act), 0.0))
                else:
                    tp = 1.0 / len(states) if trans_def_w > 0 else 0.0
                prob = V[t - 1].get(ps, 0) * tp
                if prob > max_p:
                    max_p = prob
                    prev_st = ps
            emit_p = obs_p.get(s, {}).get(obs[t], obs_def_p.get(s, 0.0))
            V[t][s] = max_p * emit_p
            back[t][s] = prev_st
            
    last = max(V[N - 1], key=V[N - 1].get)
    path = [last]
    
    for t in range(N - 1, 0, -1):
        path.append(back[t][path[-1]] or "")
        
        
    return list(reversed(path))

def write_output(states, filename="states.txt"):
    
    with open(filename, 'w') as f:
        f.write("states\n")
        f.write(f"{len(states)}\n")
        
        for i, s in enumerate(states):
            f.write(f'"{s}"')
            if i != len(states) - 1:
                f.write("\n")

if __name__ == '__main__':
    init_p = read_state_weights("state_weights.txt")
    states = list(init_p.keys())
    
    trans_p, trans_def_p, trans_def_w = read_state_action_weights("state_action_state_weights.txt", set(states))
    
    obs_p, obs_def_p = read_state_observation_weights("state_observation_weights.txt", set(states))
    
    observations, actions = read_observation_actions("observation_actions.txt")
    
    if actions and actions[-1] is None and len(actions) >= len(observations):
        actions = actions[:-1]
    result = viterbi(observations, actions, states, init_p, trans_p, trans_def_p, trans_def_w, obs_p, obs_def_p)
    write_output(result)