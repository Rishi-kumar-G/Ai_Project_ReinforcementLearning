import game
import numpy as np

 
g = game.Game("dqn_agent", "cpu")


hyperparameter = {
  "lr_start": 1e-4,
  "lr_end": 1e-4,
  "batch_size": 128,
  "gamma": 0.9,
  "eps_start": 0.9,
  "eps_end": 1e-2
}
    

g.train_agent(True, 100, 100, hyperparameter)  
# g.main(draw = "True")
score = g.main(True)
print("Score: ", score)
