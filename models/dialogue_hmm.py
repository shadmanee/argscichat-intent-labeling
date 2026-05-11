import numpy as np

class DialogueHMM:
    def __init__(self, states, observations, smoothing=1e-6):
        self.states = list(states)
        self.observations = list(observations)
        self.smoothing = smoothing # why

        self.n_states = len(self.states)
        self.n_obs = len(self.observations)

        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.obs_to_idx = {obs: i for i, obs in enumerate(self.observations)}

        self.log_pi = None
        self.log_A = None
        self.log_B = None

    def _normalize(self, arr):
        if arr.ndim == 1:
            total = arr.sum()
            if total == 0:
                return np.ones_like(arr) / len(arr)
            return arr / total

        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return arr / row_sums

    def fit(self, observation_sequences, tag_sequences):
        pi = np.zeros(self.n_states)
        A = np.zeros((self.n_states, self.n_states))
        B = np.zeros((self.n_states, self.n_obs))

        for obs_seq, tag_seq in zip(observation_sequences, tag_sequences):
            if len(obs_seq) == 0:
                continue

            first_tag = tag_seq[0]
            pi[self.state_to_idx[first_tag]] += 1

            for t in range(len(obs_seq)):
                tag = tag_seq[t]
                obs = obs_seq[t]

                tag_idx = self.state_to_idx[tag]
                obs_idx = self.obs_to_idx.get(obs, None)

                if obs_idx is not None:
                    B[tag_idx, obs_idx] += 1

                if t > 0:
                    prev_tag = tag_seq[t - 1]
                    prev_tag_idx = self.state_to_idx[prev_tag]

                    A[prev_tag_idx, tag_idx] += 1

        self.log_pi = np.log(self._normalize(pi + self.smoothing))
        self.log_A = np.log(self._normalize(A + self.smoothing))
        self.log_B = np.log(self._normalize(B + self.smoothing))

        return self

    def _log_emission(self, state_idx, obs):
        obs_idx = self.obs_to_idx.get(obs, None)

        if obs_idx is None:
            # Unknown observation at test time
            return -np.log(self.n_obs)

        return self.log_B[state_idx, obs_idx]

    def viterbi(self, obs_seq):
        T = len(obs_seq)

        if T == 0:
            return []

        dp = np.full((self.n_states, T), -np.inf)
        backpointer = np.zeros((self.n_states, T), dtype=int)

        # Initialization
        for s in range(self.n_states):
            dp[s, 0] = self.log_pi[s] + self._log_emission(s, obs_seq[0])

        # Recursion
        for t in range(1, T):
            for curr_s in range(self.n_states):
                scores = dp[:, t - 1] + self.log_A[:, curr_s]
                best_prev_s = np.argmax(scores)

                dp[curr_s, t] = scores[best_prev_s] + self._log_emission(curr_s, obs_seq[t])
                backpointer[curr_s, t] = best_prev_s

        # Termination
        best_last_s = np.argmax(dp[:, T - 1])

        # Backtracking
        best_path_idx = [best_last_s]

        for t in range(T - 1, 0, -1):
            best_path_idx.append(backpointer[best_path_idx[-1], t])

        best_path_idx.reverse()

        best_path = [self.states[i] for i in best_path_idx]

        return best_path

    def predict(self, observation_sequences):
        return [self.viterbi(obs_seq) for obs_seq in observation_sequences]