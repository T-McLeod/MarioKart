-- PPO-specific reward function
-- Checkpoint progress + time bleed penalty, no speed reward

function getRewardPPO()
	-- Checkpoint progress reward (from base script.lua)
	-- Time bleed: -0.025 per frame to punish dawdling
	return getCheckpointReward() + getExperimentalReward() - 0.025
end
