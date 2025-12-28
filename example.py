from agent_manager import AgentManager
from glob import glob
import os

if __name__ == "__main__":
    os.environ["USER_AGENT"] = "AutoML-Agent/1.0"
    # Check if datasets exist
    datasets = sorted(glob('agent_workspace/datasets/*'))
    if len(datasets) > 2:
        data_path = glob(datasets[2] + '/*')
    else:
        # Use a simple test without data for now
        print("No datasets found in agent_workspace/datasets/")
        print("Running without data path for testing...")
        data_path = None

    user_prompt = "Summary reviews. The reviews are stored in ./local-test-data"
    manager = AgentManager(llm='qwen', task="ts_forecasting", interactive=False, data_path=data_path, n_revise=3)

    manager.initiate_chat(user_prompt)