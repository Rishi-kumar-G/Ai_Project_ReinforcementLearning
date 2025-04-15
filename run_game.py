import game
import sys
import os
import pygame
import torch

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def main():
    # Initialize pygame
    pygame.init()
    
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create game instance with dqn_agent
    g = game.Game("dqn_agent", device)
    
    print("Welcome to Flappy Bird with Reinforcement Learning!")
    print("1: Play as user (Space to jump)")
    print("2: Watch trained DQN agent")
    print("3: Watch random agent")
    print("4: Train a new DQN agent")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        g = game.Game("user_agent", device)
        score = g.main(True)
        print(f"Game Over! Your score: {score}")
    
    elif choice == "2":
        score = g.main(True)
        print(f"Game Over! Agent score: {score}")
    
    elif choice == "3":
        g = game.Game("random_agent", device)
        score = g.main(True)
        print(f"Game Over! Random agent score: {score}")
    
    elif choice == "4":
        hyperparameter = {
            "lr_start": 1e-4,
            "lr_end": 1e-4,
            "batch_size": 128,
            "gamma": 0.9,
            "eps_start": 0.9,
            "eps_end": 1e-2
        }
        
        g.train_agent(True, 100, 100, hyperparameter)
        score = g.main(True)
        print(f"Training complete! Agent scored: {score}")
    
    else:
        print("Invalid choice. Exiting.")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()